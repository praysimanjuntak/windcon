"""Relative-winding constraints between the traced paths of one slice: adjacent windings.

Walking outward from a traced path along its normal (away from the scroll axis), the next sheet is
one winding further out. From every few vertices of every path a ray is cast both ways along the
normal to the first other traced path it meets. The CT profile along the ray then decides:

* *clean*: after the first path's band and a real dark gap, the next band peak is the path that
  was hit, so no sheet lies between them;
* *blocked*: a band peak appears before the path that was hit, so an untraced sheet lies between;
* otherwise the ray is unclear and does not count.

A pair (p, q), q one winding outward of p, is adjacent when enough rays vote for it cleanly and no
ray between them is blocked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from windcon.chains import inward_normals
from windcon.tracer import TraceParams, _next_peak, _walk_valley

CLEAN, BLOCKED, NO_GAP, UNCLEAR = "clean", "blocked", "no_gap", "unclear"
VERDICTS = (CLEAN, BLOCKED, NO_GAP, UNCLEAR)  # index = stored verdict code


@dataclass(frozen=True)
class AdjacencyParams:
    """Pixel units are slice pixels (9.6 um at level 2)."""

    anchor_step: int = 4  # cast rays from every 4th path vertex (8 px at the tracer's 2 px)
    max_reach: float = 60.0  # how far along the normal to look for the next traced path
    hit_radius: float = 1.2  # a ray sample this close to a path vertex hits that path
    spacing: float = 0.5  # ray sample spacing
    thickness: float = 3.0  # +/- extent along the tangent averaged into each profile sample
    capture: float = 3.0  # band peaks must lie within this of where the paths are
    min_cos: float = 0.87  # the two paths must run within 30 degrees of parallel at the ray
    min_axis_cos: float = 0.3  # paths running toward the axis have no outward side: skipped
    batch: int = 4096  # rays per vectorised batch


@dataclass
class Rays:
    """Every ray that reached another traced path. Index arrays refer to the input paths."""

    inner: np.ndarray  # (R,) path on the axis side
    outer: np.ndarray  # (R,) path one winding outward if the ray is clean
    inner_vertex: np.ndarray  # (R,) vertex index within the inner path
    outer_vertex: np.ndarray  # (R,) vertex index within the outer path
    verdict: np.ndarray  # (R,) CLEAN, BLOCKED, NO_GAP or UNCLEAR

    def pairs(self, min_votes: int) -> dict[tuple[int, int], np.ndarray]:
        """Adjacent pairs (inner, outer) -> indices of their clean rays: at least ``min_votes``
        clean rays and no blocked ray between the two paths."""
        out: dict[tuple[int, int], list[int]] = {}
        blocked: set[tuple[int, int]] = set()
        for r, (p, q, v) in enumerate(zip(self.inner.tolist(), self.outer.tolist(), self.verdict.tolist())):
            if v == CLEAN:
                out.setdefault((p, q), []).append(r)
            elif v == BLOCKED:
                blocked.add((p, q))
        return {k: np.asarray(v) for k, v in out.items() if len(v) >= min_votes and k not in blocked}


def classify_profile(prof: np.ndarray, s: np.ndarray, t_hit: float, tp: TraceParams, capture: float) -> str:
    """Verdict for one ray whose profile ``prof`` is sampled at distances ``s`` from the anchor (on
    the first path's band centre) and meets the other traced path at ``t_hit``."""
    zero = int(np.argmin(np.abs(s)))
    reach = int(round(capture / (s[1] - s[0])))
    lo = max(zero - reach, 0)
    b = lo + int(np.argmax(prof[lo:zero + reach + 1]))
    peak = float(prof[b])
    span = peak - tp.air_level
    if peak < tp.min_peak or span <= 0:
        return UNCLEAR
    rise = tp.valley_rise * span
    out = prof[b:]
    v = _walk_valley(out, rise)
    nxt = _next_peak(out, v, rise)
    if nxt is None:
        return UNCLEAR
    s_next = float(s[b + nxt])
    if s_next < t_hit - capture:
        return BLOCKED
    if s_next > t_hit + capture:
        return UNCLEAR
    if min(peak, float(out[nxt])) - float(out[v]) < tp.min_gap_contrast * span:
        return NO_GAP
    return CLEAN


def _tangents(xy: np.ndarray) -> np.ndarray:
    t = np.gradient(xy, axis=0) if len(xy) > 1 else np.array([[1.0, 0.0]])
    return t / np.linalg.norm(t, axis=1, keepdims=True).clip(1e-9)


def find_rays(
    smooth: np.ndarray,
    origin: np.ndarray,
    paths: list[np.ndarray],
    center_xy: np.ndarray,
    tp: TraceParams = TraceParams(),
    ap: AdjacencyParams = AdjacencyParams(),
) -> Rays:
    """Cast the rays of every path in one slice and classify them.

    ``smooth``: the slice image (smoothed like the tracer's), indexed [y, x], covering volume xy
    from ``origin``; ``paths``: band-centre polylines in volume xy (the tracer's 2 px vertices);
    ``center_xy``: the scroll axis in this slice.
    """
    verts = np.concatenate(paths)
    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(paths)])
    first = np.cumsum([0] + [len(p) for p in paths])[:-1]
    tangents = np.concatenate([_tangents(p) for p in paths])
    tree = cKDTree(verts)

    anchors, directions, sources = [], [], []
    for i, p in enumerate(paths):
        if len(p) < 2:
            continue
        normal, axis_cos = inward_normals(p, center_xy)
        if axis_cos < ap.min_axis_cos:
            continue
        idx = np.arange(0, len(p), ap.anchor_step)
        for sign in (-1.0, 1.0):  # -1: outward (toward the next winding), +1: inward
            anchors.append(first[i] + idx)
            directions.append(sign * normal[idx])
            sources.append(np.full(len(idx), sign))
    if not anchors:
        empty = np.zeros(0, dtype=np.int64)
        return Rays(empty, empty, empty, empty, np.zeros(0, dtype=object))
    anchors = np.concatenate(anchors)
    directions = np.concatenate(directions)
    signs = np.concatenate(sources)

    s_hit = np.arange(1, int(ap.max_reach / ap.spacing) + 1) * ap.spacing
    u = np.arange(-ap.thickness, ap.thickness + 1e-9, 1.0)
    inner, outer, inner_v, outer_v, verdicts = [], [], [], [], []
    for b0 in range(0, len(anchors), ap.batch):
        a = anchors[b0:b0 + ap.batch]
        d = directions[b0:b0 + ap.batch]
        sg = signs[b0:b0 + ap.batch]
        pts = verts[a][:, None, :] + s_hit[None, :, None] * d[:, None, :]
        dist, idx = tree.query(pts.reshape(-1, 2), distance_upper_bound=ap.hit_radius)
        dist, idx = dist.reshape(len(a), -1), idx.reshape(len(a), -1)
        hit_owner = np.where(np.isfinite(dist), owner[np.minimum(idx, len(owner) - 1)], -1)
        other = (hit_owner >= 0) & (hit_owner != owner[a][:, None])
        has = other.any(axis=1)
        k_first = np.argmax(other, axis=1)
        for r in np.flatnonzero(has):
            q = hit_owner[r, k_first[r]]
            run = np.flatnonzero(hit_owner[r, k_first[r]:] == q) + k_first[r]
            breaks = np.flatnonzero(np.diff(run) > 1)
            run = run[: breaks[0] + 1] if breaks.size else run  # the first crossing only
            k = run[np.argmin(dist[r, run])]
            qv = int(idx[r, k])
            if abs(float(np.dot(tangents[a[r]], tangents[qv]))) < ap.min_cos:
                continue
            t_hit = float(s_hit[k])
            s = np.arange(-ap.capture - 1.0, t_hit + ap.capture + 2.0, ap.spacing)
            t = tangents[a[r]]
            grid = (verts[a[r]][None, None, :] + s[None, :, None] * d[r][None, None, :]
                    + u[:, None, None] * t[None, None, :]) - origin
            prof = ndimage.map_coordinates(smooth, [grid[..., 1].ravel(), grid[..., 0].ravel()],
                                           order=1, mode="nearest").reshape(len(u), len(s)).mean(axis=0)
            verdict = classify_profile(prof, s, t_hit, tp, ap.capture)
            p_path, p_vert = int(owner[a[r]]), int(a[r] - first[owner[a[r]]])
            q_path, q_vert = int(q), int(qv - first[q])
            if sg[r] < 0:  # outward ray: the hit path is the outer one
                inner.append(p_path); inner_v.append(p_vert); outer.append(q_path); outer_v.append(q_vert)
            else:
                inner.append(q_path); inner_v.append(q_vert); outer.append(p_path); outer_v.append(p_vert)
            verdicts.append(verdict)
    return Rays(np.asarray(inner, dtype=np.int64), np.asarray(outer, dtype=np.int64),
                np.asarray(inner_v, dtype=np.int64), np.asarray(outer_v, dtype=np.int64),
                np.asarray(verdicts, dtype=object))
