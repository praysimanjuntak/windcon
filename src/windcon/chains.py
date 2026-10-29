"""Turn per-slice traced paths into longer same-winding constraints.

* ``smooth_path`` removes the vertex-to-vertex jitter of the tracer's re-centring.
* ``shift_inward`` moves a path across the sheet toward the scroll axis by a constant, so it sits on
  the sheet face the human annotations and verified patches follow rather than on the middle of the
  bright band.
* ``link_slices`` pairs paths of two nearby slices that trace the same sheet: enough vertices of
  each lie within ``r_match`` of the other with no competing path within ``r_sep``, and each is the
  other's best partner. ``build_chains`` follows those one-to-one links through the slices, so a
  chain is one sheet followed through z and every path belongs to exactly one chain.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def smooth_path(xy: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a polyline's vertices along the path (``sigma`` in vertices)."""
    xy = np.asarray(xy, dtype=np.float64)
    if sigma <= 0 or len(xy) < 3:
        return xy.copy()
    return gaussian_filter1d(xy, sigma, axis=0, mode="nearest")


def inward_normals(xy: np.ndarray, center_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit normals of a polyline, all on the side facing ``center_xy`` (the scroll axis).

    The side is chosen once per path by the summed dot product with the direction to the axis, so
    a path never flips side; the second value is the mean |cos| between normal and that direction
    (near 0 when the path runs toward the axis and the side is ill-defined).
    """
    xy = np.asarray(xy, dtype=np.float64)
    tangent = np.gradient(xy, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    to_axis = np.asarray(center_xy, dtype=np.float64) - xy
    to_axis /= np.linalg.norm(to_axis, axis=1, keepdims=True).clip(1e-9)
    cos = np.einsum("ij,ij->i", normal, to_axis)
    if cos.sum() < 0:
        normal, cos = -normal, -cos
    return normal, float(np.abs(cos).mean())


def shift_inward(xy: np.ndarray, center_xy: np.ndarray, delta: float, min_cos: float = 0.3) -> np.ndarray:
    """Move a path ``delta`` px toward the scroll axis; paths running toward the axis stay put."""
    normal, cos = inward_normals(xy, center_xy)
    if cos < min_cos:
        return np.asarray(xy, dtype=np.float64).copy()
    return np.asarray(xy, dtype=np.float64) + delta * normal


def link_slices(
    paths_a: list[np.ndarray],
    paths_b: list[np.ndarray],
    *,
    r_match: float = 2.5,
    r_sep: float = 5.0,
    min_votes: int = 8,
) -> list[tuple[int, int]]:
    """One-to-one links (i, j) between paths of two slices that trace the same sheet.

    A vertex of one path votes for the nearest path of the other slice when that path is within
    ``r_match`` and every other path is farther than ``r_sep``. A pair links when both directions
    give it at least ``min_votes`` votes and each path is the other's best-voted partner.
    """
    if not paths_a or not paths_b:
        return []
    votes_ab = _votes(paths_a, paths_b, r_match, r_sep)
    votes_ba = _votes(paths_b, paths_a, r_match, r_sep)
    best_a = {i: max(row, key=row.get) for i, row in votes_ab.items()}
    best_b = {j: max(row, key=row.get) for j, row in votes_ba.items()}
    links = [
        (i, j) for i, j in best_a.items()
        if best_b.get(j) == i and min(votes_ab[i][j], votes_ba[j][i]) >= min_votes
    ]
    return sorted(links)


def _votes(src: list[np.ndarray], dst: list[np.ndarray], r_match: float, r_sep: float, k: int = 16) -> dict[int, dict[int, int]]:
    """votes[i][j] = vertices of src path i whose only nearby dst path is j.

    The k nearest dst vertices within r_sep decide: k = 16 exceeds the vertices one 2 px-spaced
    path can have within 5 px, so a competing path within r_sep is never crowded out.
    """
    dst_label = np.concatenate([np.full(len(p), k_, dtype=np.int64) for k_, p in enumerate(dst)] + [[-1]])
    tree = cKDTree(np.concatenate(dst))
    src_label = np.concatenate([np.full(len(p), i, dtype=np.int64) for i, p in enumerate(src)])
    dist, idx = tree.query(np.concatenate(src), k=k, distance_upper_bound=r_sep, workers=-1)
    labels = dst_label[idx]  # -1 where fewer than k neighbours lie within r_sep
    nearest = labels[:, 0]
    clean = (dist[:, 0] <= r_match) & np.all((labels == nearest[:, None]) | (labels < 0), axis=1)
    pairs, counts = np.unique(np.stack([src_label[clean], nearest[clean]], axis=1), axis=0, return_counts=True)
    votes: dict[int, dict[int, int]] = {}
    for (i, j), n in zip(pairs.tolist(), counts.tolist()):
        votes.setdefault(i, {})[j] = n
    return votes


def build_chains(num_paths: list[int], links: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    """Chains of (slice index, path index), following one-to-one links between consecutive slices.

    ``num_paths[s]`` is the number of paths in slice s; ``links[s]`` the links between slices s and
    s + 1. Every path is in exactly one chain (unlinked paths form single-path chains).
    """
    nxt: dict[tuple[int, int], tuple[int, int]] = {}
    has_prev: set[tuple[int, int]] = set()
    for s, pairs in enumerate(links):
        for i, j in pairs:
            nxt[(s, i)] = (s + 1, j)
            has_prev.add((s + 1, j))
    chains = []
    for s, n in enumerate(num_paths):
        for i in range(n):
            if (s, i) in has_prev:
                continue
            chain = [(s, i)]
            while chain[-1] in nxt:
                chain.append(nxt[chain[-1]])
            chains.append(chain)
    return chains
