"""Minimal read-only Zarr v2 reader over HTTP with a persistent on-disk chunk cache.

Scroll volumes are tens of TB; we only ever touch small crops, so each chunk is
fetched once, stored verbatim (still compressed) under ``cache_dir``, and decoded
on read. Missing chunks (HTTP 403/404, i.e. masked-out regions) decode to
``fill_value`` and are remembered with an empty marker file.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import numcodecs
import numpy as np

S3_OPEN_DATA = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"


def _http_get(url: str, timeout: float = 120.0, retries: int = 4) -> bytes | None:
    """Return the body, or None when the object does not exist."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code in (403, 404):
                return None
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
    raise AssertionError("unreachable")


class ZarrArray:
    """One resolution level of a Zarr v2 array served over HTTP."""

    def __init__(self, url: str, cache_dir: Path = DEFAULT_CACHE, workers: int = 16):
        self.url = url.rstrip("/")
        self.workers = workers
        key = hashlib.sha1(self.url.encode()).hexdigest()[:16]
        self.cache = Path(cache_dir) / key
        self.cache.mkdir(parents=True, exist_ok=True)
        meta_path = self.cache / ".zarray"
        if not meta_path.exists():
            body = _http_get(f"{self.url}/.zarray")
            if body is None:
                raise FileNotFoundError(f"{self.url}/.zarray")
            meta_path.write_bytes(body)
            (self.cache / "SOURCE").write_text(self.url + "\n")
        meta = json.loads(meta_path.read_text())
        if meta.get("zarr_format") != 2:
            raise ValueError(f"only zarr v2 is supported, got {meta.get('zarr_format')}")
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.order = meta.get("order", "C")
        self.fill_value = 0 if meta.get("fill_value") is None else meta["fill_value"]
        self.separator = meta.get("dimension_separator", ".")
        self.compressor = numcodecs.get_codec(meta["compressor"]) if meta.get("compressor") else None
        self.filters = [numcodecs.get_codec(f) for f in meta.get("filters") or []]

    def _chunk_path(self, idx: tuple[int, ...], suffix: str = "") -> Path:
        return self.cache / ("_".join(map(str, idx)) + suffix)

    def _fetch(self, idx: tuple[int, ...]) -> None:
        path = self._chunk_path(idx)
        missing = self._chunk_path(idx, ".missing")
        if path.exists() or missing.exists():
            return
        body = _http_get(f"{self.url}/{self.separator.join(map(str, idx))}")
        if body is None:
            missing.touch()
            return
        tmp = self._chunk_path(idx, ".part")
        tmp.write_bytes(body)
        tmp.rename(path)

    def _decode(self, idx: tuple[int, ...]) -> np.ndarray | None:
        path = self._chunk_path(idx)
        if not path.exists():
            return None
        buf = path.read_bytes()
        if self.compressor is not None:
            buf = self.compressor.decode(buf)
        for flt in reversed(self.filters):
            buf = flt.decode(buf)
        return np.frombuffer(buf, dtype=self.dtype).reshape(self.chunks, order=self.order)

    def read(self, box: tuple[tuple[int, int], ...]) -> np.ndarray:
        """Read the half-open box ((z0, z1), (y0, y1), (x0, x1)); out-of-bounds reads as fill."""
        lo = np.array([b[0] for b in box])
        hi = np.array([b[1] for b in box])
        if np.any(hi <= lo):
            raise ValueError(f"empty box {box}")
        out = np.full(tuple(hi - lo), self.fill_value, dtype=self.dtype)
        shape = np.array(self.shape)
        clo = np.maximum(lo, 0)
        chi = np.minimum(hi, shape)
        if np.any(chi <= clo):
            return out
        csize = np.array(self.chunks)
        ranges = [range(a // c, (b - 1) // c + 1) for a, b, c in zip(clo, chi, csize)]
        indices = list(product(*ranges))
        with ThreadPoolExecutor(self.workers) as pool:
            list(pool.map(self._fetch, indices))
        for idx in indices:
            chunk = self._decode(idx)
            if chunk is None:
                continue
            c0 = np.array(idx) * csize
            s0 = np.maximum(c0, clo)
            s1 = np.minimum(c0 + csize, chi)
            dst = tuple(slice(a, b) for a, b in zip(s0 - lo, s1 - lo))
            src = tuple(slice(a, b) for a, b in zip(s0 - c0, s1 - c0))
            out[dst] = chunk[src]
        return out
