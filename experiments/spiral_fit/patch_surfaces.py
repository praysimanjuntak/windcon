"""Verified-patch surfaces as labelled vertices with normals, for point-to-surface offsets.

A tifxyz patch is a grid of 3D vertices (20 or 12.5 px apart). The offset of a point from a patch
surface is approximated with the nearest patch vertex: the signed distance along that vertex's
normal (oriented toward the scroll axis, + = the point lies on the axis side) and the remaining
lateral distance, which must stay within about half a grid diagonal for the point to be over the
surface. Reads the TIFFs with PIL, like the fitter (masks are LZW-compressed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

LATERAL = 14.5  # half the diagonal of a 20 px grid cell, plus a margin


def imread(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def umbilicus_xy(path: Path):
    """Function z -> (x, y) of the scroll axis (vectorised over z)."""
    pts = sorted((p["z"], p["x"], p["y"]) for p in json.loads(Path(path).read_text())["control_points"])
    z, x, y = (np.array(v, float) for v in zip(*pts))
    return lambda zq: np.stack([np.interp(zq, z, x), np.interp(zq, z, y)], axis=-1)


def patch_vertices(patch_dir: Path, z_lo: float, z_hi: float):
    """Valid vertices of one tifxyz patch with z in [z_lo, z_hi) and unit grid normals there."""
    zz = imread(patch_dir / "z.tif")
    valid = zz != -1
    mask_path = patch_dir / "mask.tif"
    if mask_path.exists():
        m = imread(mask_path)
        valid &= (m[..., 0] if m.ndim == 3 else m) != 0
    if not (valid & (zz >= z_lo) & (zz < z_hi)).any():
        return None
    P = np.stack([imread(patch_dir / "x.tif"), imread(patch_dir / "y.tif"), zz], axis=-1).astype(np.float64)
    valid &= (P != -1).all(axis=-1)
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1] = P[2:] - P[:-2]
    ok = valid.copy()
    ok[:, 0] = ok[:, -1] = ok[0] = ok[-1] = False
    ok[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
    ok[1:-1] &= valid[2:] & valid[:-2]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1)
    ok &= norm > 0
    ok &= (P[..., 2] >= z_lo) & (P[..., 2] < z_hi)
    return P[ok], n[ok] / norm[ok][:, None]


class PatchSurfaces:
    """All verified-patch vertices intersecting a z range, with labels and axis-facing normals."""

    def __init__(self, dataset: Path, z_lo: float, z_hi: float):
        dataset = Path(dataset)
        verts, normals, labels, self.names = [], [], [], []
        for d in sorted((dataset / "verified_patches").iterdir()):
            got = patch_vertices(d, z_lo, z_hi) if d.is_dir() else None
            if got is None:
                continue
            verts.append(got[0])
            normals.append(got[1])
            labels.append(np.full(len(got[0]), len(self.names), dtype=np.int32))
            self.names.append(d.name)
        self.V = np.concatenate(verts)
        self.N = np.concatenate(normals)
        self.label = np.concatenate(labels)
        axis = umbilicus_xy(dataset / "umbilicus.json")(self.V[:, 2])
        flip = np.einsum("ij,ij->i", self.N[:, :2], axis - self.V[:, :2]) < 0
        self.N[flip] *= -1
        self.tree = cKDTree(self.V)
        self._order = np.argsort(self.label, kind="stable")
        self._bounds = np.searchsorted(self.label[self._order], np.arange(len(self.names) + 1))
        self._patch_trees: dict[int, cKDTree] = {}

    def nearest(self, points: np.ndarray, max_normal: float, max_lateral: float = LATERAL, k: int = 8):
        """Per point: (label, signed normal offset, lateral) of the candidate vertex with the smallest
        lateral distance among the k nearest that satisfy both limits; label -1 if none does."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        dist, idx = self.tree.query(points, k=k, distance_upper_bound=max_normal + max_lateral, workers=-1)
        return self._pick(points, dist, idx, np.arange(len(self.V)), max_normal, max_lateral)

    def nearest_on(self, patch: int, points: np.ndarray, max_normal: float, max_lateral: float = LATERAL, k: int = 4):
        """Like nearest(), restricted to one patch."""
        ids = self._order[self._bounds[patch]:self._bounds[patch + 1]]
        tree = self._patch_trees.get(patch)
        if tree is None:
            tree = self._patch_trees[patch] = cKDTree(self.V[ids])
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        dist, idx = tree.query(points, k=min(k, len(ids)), distance_upper_bound=max_normal + max_lateral)
        return self._pick(points, dist, idx, ids, max_normal, max_lateral)

    def _pick(self, points, dist, idx, ids, max_normal, max_lateral):
        dist = dist.reshape(len(points), -1)
        idx = idx.reshape(len(points), -1)
        found = np.isfinite(dist)
        safe = np.where(found, idx, 0)
        vid = ids[safe]
        delta = points[:, None, :] - self.V[vid]
        normal = np.einsum("pkc,pkc->pk", delta, self.N[vid])
        lateral = np.linalg.norm(delta - normal[..., None] * self.N[vid], axis=-1)
        ok = found & (np.abs(normal) <= max_normal) & (lateral <= max_lateral)
        lateral_masked = np.where(ok, lateral, np.inf)
        best = np.argmin(lateral_masked, axis=1)
        rows = np.arange(len(points))
        has = np.isfinite(lateral_masked[rows, best])
        label = np.where(has, self.label[vid[rows, best]], -1)
        return label, np.where(has, normal[rows, best], np.nan), np.where(has, lateral[rows, best], np.nan)
