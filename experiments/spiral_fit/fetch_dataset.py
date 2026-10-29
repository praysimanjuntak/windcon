#!/usr/bin/env python3
"""Download the subset of the PHerc. Paris 4 `spiral-input` dataset a z-window fit needs.

Stdlib only. Source: https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/

* small files and directories: umbilicus, winding JSONs, outer_shell/, winding_inference/
* lasagna normals sidecar (`las_008_nx.ome.zarr.respool_g4_pair`): only the brick rows whose
  z-layer intersects the window are fetched, written into full-size sparse files. The fitter
  memory-maps the channel files and reads exactly those rows (sparse_cuda_cache.ResidentBrickPool),
  so the untouched rows can stay holes.
* verified_patches/: every meta.json is fetched, patches whose bbox can intersect the window are
  kept, and their folders are mirrored completely.

Re-running resumes: finished files are skipped, fetched sidecar row runs are recorded.
"""

from __future__ import annotations

import argparse
import array
import ast
import json
import math
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4"
SMALL_FILES = ["umbilicus.json", "abs_winding.json", "relative_windings.json", "same_windings.json"]
SMALL_DIRS = ["outer_shell", "winding_inference", "eval_fibers"]
NORMALS = "lasagna_inputs/las_008_nx.ome.zarr.respool_g4_pair"
LASAGNA_SCALE = 4  # sidecar array_shape is ceil(fit frame / 4); see the sidecar meta.json

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def http(url: str, rng: tuple[int, int] | None = None, retries: int = 6) -> bytes | None:
    """GET a URL (optionally the half-open byte range ``rng``); None on 404."""
    headers = {"Range": f"bytes={rng[0]}-{rng[1] - 1}"} if rng else {}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
                status = resp.status
            if rng and status == 200:  # server ignored Range
                body = body[rng[0] : rng[1]]
            return body
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def quote(path: str) -> str:
    return urllib.parse.quote(path)


def listing(rel_dir: str) -> list[str]:
    """Entries of an autoindex page; directories end with '/'."""
    body = http(f"{BASE}/{quote(rel_dir.strip('/'))}/")
    if body is None:
        raise FileNotFoundError(rel_dir)
    names = re.findall(r'href="([^"?]+)"', body.decode())
    return [urllib.parse.unquote(n) for n in names if not n.startswith(("../", "/", "http"))]


def fetch_file(rel: str, out_root: Path) -> int:
    dest = out_root / rel
    if dest.exists():
        return 0
    body = http(f"{BASE}/{quote(rel)}")
    if body is None:
        raise FileNotFoundError(rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(body)
    tmp.rename(dest)
    return len(body)


def walk(rel_dir: str) -> list[str]:
    files = []
    for name in listing(rel_dir):
        rel = f"{rel_dir.rstrip('/')}/{name}"
        files.extend(walk(rel) if name.endswith("/") else [rel])
    return files


def fetch_small(out_root: Path, pool: ThreadPoolExecutor) -> None:
    rels = list(SMALL_FILES)
    for d in SMALL_DIRS:
        rels.extend(walk(d))
    total = sum(f.result() for f in [pool.submit(fetch_file, r, out_root) for r in rels])
    log(f"small inputs: {len(rels)} files, {total / 2**20:.1f} MiB fetched")


def runs(ids: list[int]) -> list[tuple[int, int]]:
    """Contiguous [lo, hi) runs of sorted integer ids."""
    out: list[tuple[int, int]] = []
    for i in ids:
        if out and out[-1][1] == i:
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


def read_npy_int32(path: Path) -> array.array:
    raw = path.read_bytes()
    assert raw[:6] == b"\x93NUMPY", path
    major = raw[6]
    if major == 1:
        hlen, start = struct.unpack("<H", raw[8:10])[0], 10
    else:
        hlen, start = struct.unpack("<I", raw[8:12])[0], 12
    header = ast.literal_eval(raw[start : start + hlen].decode("latin1"))
    assert header["descr"] in ("<i4", "|i4") and not header["fortran_order"], header
    values = array.array("i")
    values.frombytes(raw[start + hlen :])
    return values


def fetch_normals(out_root: Path, z_begin: int, z_end: int, pool: ThreadPoolExecutor) -> None:
    side = out_root / NORMALS
    for name in ("meta.json", "table.npy", "brick_coords.npy"):
        fetch_file(f"{NORMALS}/{name}", out_root)
    meta = json.loads((side / "meta.json").read_text())
    bz = int(meta["brick_shape"][0])
    brick_bytes = math.prod(int(v) for v in meta["brick_shape"])  # one uint8 channel
    rows = int(meta["rows"])
    coords = read_npy_int32(side / "brick_coords.npy")
    assert len(coords) == 3 * rows, (len(coords), rows)
    z_lo = math.floor(z_begin / LASAGNA_SCALE)
    z_hi = math.ceil(z_end / LASAGNA_SCALE)
    # The fitter keeps bricks with brick_z*bz < z_hi and (brick_z+1)*bz > z_lo; fetch one
    # extra brick layer on each side. Row 0 is the reserved all-zero brick.
    keep = [0] + [r for r in range(1, rows)
                  if (coords[3 * r] - 1) * bz < z_hi and (coords[3 * r] + 2) * bz > z_lo]
    row_runs = runs(sorted(set(keep)))
    done_path = side / "fetched_runs.json"
    done = {tuple(x) for x in json.loads(done_path.read_text())} if done_path.exists() else set()
    channels = [f"channel_{i}.u8" for i in range(len(meta["channels"]))]
    part = 8 << 20
    jobs = []
    for ch in channels:
        path = side / ch
        if not path.exists():
            with open(path, "wb") as f:
                f.truncate(rows * brick_bytes)  # sparse: unfetched rows stay holes
        for lo, hi in row_runs:
            if (ch, lo, hi) in done:
                continue
            a, b = lo * brick_bytes, hi * brick_bytes
            jobs.append((ch, lo, hi, path, [(s, min(s + part, b)) for s in range(a, b, part)]))
    total = sum(b - a for *_, pieces in jobs for a, b in pieces)
    log(f"normals: {len(keep)}/{rows} brick rows in {len(row_runs)} runs, "
        f"{total / 2**30:.2f} GiB to fetch over {len(channels)} channels")

    def piece(path: Path, rel: str, a: int, b: int) -> int:
        body = http(f"{BASE}/{quote(rel)}", (a, b))
        assert body is not None and len(body) == b - a, (rel, a, b)
        fd = os.open(path, os.O_WRONLY)
        try:
            os.pwrite(fd, body, a)
        finally:
            os.close(fd)
        return b - a

    futs = [pool.submit(piece, path, f"{NORMALS}/{ch}", a, b)
            for ch, lo, hi, path, pieces in jobs for a, b in pieces]
    fetched = sum(f.result() for f in futs)
    done.update((ch, lo, hi) for ch, lo, hi, *_ in jobs)
    done_path.write_text(json.dumps(sorted(done)))
    log(f"normals: fetched {fetched / 2**30:.2f} GiB")


def bbox_hits_window(bbox, z_begin: int, z_end: int) -> bool:
    """True if the patch may intersect [z_begin, z_end) under either axis order.

    meta.json bboxes are usually [[x,y,z],[x,y,z]], but producers have emitted other orders
    (tifxyz.load_tifxyz ignores bboxes for that reason); keep the patch if the z interval read
    as the last OR the first axis intersects the window. Over-inclusion is harmless: the
    fitter's own z prefilter drops non-intersecting patches.
    """
    try:
        lo, hi = bbox
        cands = [(lo[2], hi[2]), (lo[0], hi[0])]
    except (TypeError, ValueError, IndexError):
        return True
    return any(min(a, b) < z_end and max(a, b) >= z_begin for a, b in cands)


def fetch_patches(out_root: Path, z_begin: int, z_end: int, pool: ThreadPoolExecutor) -> None:
    vp = out_root / "verified_patches"
    meta_dir = out_root / "_patch_meta"  # every patch's meta.json, kept outside the fit folder
    meta_dir.mkdir(parents=True, exist_ok=True)
    names = [n.rstrip("/") for n in listing("verified_patches") if n.endswith("/")]
    log(f"patches: {len(names)} folders listed")

    def get_meta(name: str):
        cached = meta_dir / f"{name}.json"
        if cached.exists():
            return name, json.loads(cached.read_text())
        body = http(f"{BASE}/{quote(f'verified_patches/{name}/meta.json')}")
        if body is None:
            return name, None
        cached.write_bytes(body)
        return name, json.loads(body)

    selected, missing = [], 0
    t0 = time.time()
    futs = [pool.submit(get_meta, n) for n in names]
    for i, fut in enumerate(as_completed(futs), 1):
        name, meta = fut.result()
        if meta is None:
            missing += 1
        elif bbox_hits_window(meta.get("bbox"), z_begin, z_end):
            selected.append(name)
        if i % 5000 == 0:
            log(f"patches: {i}/{len(names)} meta.json ({time.time() - t0:.0f}s)")
    selected.sort()
    (out_root / "selected_patches.json").write_text(json.dumps(
        {"z_begin": z_begin, "z_end": z_end, "selected": selected}, indent=0))
    log(f"patches: {len(selected)} may intersect z [{z_begin}, {z_end}); {missing} without meta.json")

    def mirror(name: str) -> int:
        marker = out_root / "_patch_complete" / name
        if marker.exists():
            return 0
        rels = [f"verified_patches/{name}/{f}" for f in listing(f"verified_patches/{name}")
                if not f.endswith("/")]
        n = sum(fetch_file(r, out_root) for r in rels)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return n

    t0 = time.time()
    total = 0
    futs = [pool.submit(mirror, n) for n in selected]
    for i, fut in enumerate(as_completed(futs), 1):
        total += fut.result()
        if i % 2000 == 0:
            log(f"patches: mirrored {i}/{len(selected)} ({total / 2**20:.0f} MiB, {time.time() - t0:.0f}s)")
    log(f"patches: mirrored {len(selected)} folders into {vp}, {total / 2**20:.0f} MiB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--z-begin", type=int, required=True)
    ap.add_argument("--z-end", type=int, required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--parts", nargs="+", default=["small", "normals", "patches"],
                    choices=["small", "normals", "patches"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        if "small" in args.parts:
            fetch_small(args.out, pool)
        if "normals" in args.parts:
            fetch_normals(args.out, args.z_begin, args.z_end, pool)
        if "patches" in args.parts:
            fetch_patches(args.out, args.z_begin, args.z_end, pool)
    log("done")


if __name__ == "__main__":
    sys.exit(main())
