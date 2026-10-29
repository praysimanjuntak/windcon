"""Read and write VC3D point-collection documents (``vc_pointcollections_json_version`` "1").

This is the format of the spiral-fit inputs ``same_windings.json``,
``relative_windings.json`` and ``abs_winding.json``. Points are stored as
``p = [x, y, z]`` in the coordinate frame of the annotated volume level. A
collection whose points carry no ``wind_a`` is a same-winding collection: an
ordered strip of points asserted to lie on one winding.
"""

from __future__ import annotations

import colorsys
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = "1"


@dataclass
class Collection:
    id: int
    name: str
    xyz: np.ndarray  # (N, 3) in point-id order
    wind: np.ndarray  # (N,) NaN where the point has no winding annotation
    metadata: dict

    @property
    def is_same_winding(self) -> bool:
        return bool(np.all(np.isnan(self.wind)))


def load_collections(path: str | Path) -> list[Collection]:
    doc = json.loads(Path(path).read_text())
    if doc.get("vc_pointcollections_json_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported point-collection version")
    out = []
    for cid, col in doc["collections"].items():
        items = sorted(col.get("points", {}).items(), key=lambda kv: int(kv[0]))
        xyz = np.array([pt["p"] for _, pt in items], dtype=np.float64).reshape(-1, 3)
        wind = np.array(
            [np.nan if pt.get("wind_a") is None else pt["wind_a"] for _, pt in items],
            dtype=np.float64,
        )
        out.append(Collection(int(cid), col.get("name", cid), xyz, wind, col.get("metadata", {})))
    return out


def point_document(collections: list[tuple[str, np.ndarray, np.ndarray | None]]) -> dict:
    """Build a document from named (N, 3) xyz collections, points in collection order. ``wind``
    holds each point's winding annotation (a relative-winding collection), or is None (a
    same-winding collection). Point ids are unique across the document, as VC3D writes them."""
    now_ms = int(time.time() * 1000)
    out = {}
    next_point = 0
    for cid, (name, xyz, wind) in enumerate(collections):
        hue = (cid * 0.6180339887) % 1.0
        points = {}
        for k, p in enumerate(np.asarray(xyz, dtype=np.float64)):
            points[str(next_point)] = {
                "p": [float(p[0]), float(p[1]), float(p[2])],
                "wind_a": None if wind is None else float(wind[k]),
                "creation_time": now_ms,
            }
            next_point += 1
        out[str(cid)] = {
            "name": name,
            "color": list(colorsys.hsv_to_rgb(hue, 0.9, 1.0)),
            "metadata": {"winding_is_absolute": False},
            "points": points,
        }
    return {"vc_pointcollections_json_version": FORMAT_VERSION, "collections": out}


def same_winding_document(strips: list[tuple[str, np.ndarray]]) -> dict:
    """Build a same-winding document from named (N, 3) xyz strips, points in strip order."""
    return point_document([(name, xyz, None) for name, xyz in strips])


def write_same_winding(path: str | Path, strips: list[tuple[str, np.ndarray]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(same_winding_document(strips), indent=1))


def write_relative_winding(path: str | Path, collections: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    """Write named (N, 3) xyz collections whose points carry winding annotations ``wind``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(point_document(collections), indent=1))
