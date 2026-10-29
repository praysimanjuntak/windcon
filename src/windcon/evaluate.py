"""Score slice-traced paths against human same-winding arcs.

Ground truth: same-winding collections drawn within a single XY slice
(``same_windings.json``). Each is an ordered polyline that a human asserts stays
on one winding. A traced path seeded on such an arc is walked outward from its
seed in both directions:

* while the path stays within ``tau`` of the arc it is *correct*;
* once it stays farther than ``tau`` for ``persist`` consecutive vertices while
  still inside the arc's extent, it has *departed* (a sheet switch or drift) and
  every later vertex inside the extent counts as *wrong*;
* vertices whose projection falls beyond either end of the arc are unscored and
  end the walk, since the human made no claim there.

Precision is correct / (correct + wrong) path length; coverage is the arc length
spanned by correct vertices over the total arc length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .annotations import Collection
from .tracer import Path, SliceField, TraceParams, trace, trace_region
from .zarr_io import ZarrArray


@dataclass
class Arc:
    name: str
    z: int
    xy: np.ndarray  # (N, 2) volume x, y in drawing order

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum())


def flat_z_arcs(collections: list[Collection], min_points: int = 3) -> list[Arc]:
    """Same-winding collections drawn within one XY slice, with repeated points removed."""
    arcs = []
    for col in collections:
        if not col.is_same_winding or len(col.xyz) < min_points:
            continue
        if np.ptp(col.xyz[:, 2]) > 0.5:
            continue
        xy = col.xyz[:, :2]
        keep = np.concatenate([[True], np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6])
        if keep.sum() >= min_points:
            arcs.append(Arc(col.name, int(round(col.xyz[0, 2])), xy[keep]))
    return arcs


def project_to_polyline(q: np.ndarray, poly: np.ndarray, edge: float = 1.0):
    """Distance, arc-length position and inside-extent flag of each point of ``q``."""
    a = poly[:-1]
    ab = poly[1:] - a
    seg_len2 = (ab**2).sum(axis=1)
    seg_len = np.sqrt(seg_len2)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    t = ((q[:, None, :] - a[None]) * ab[None]).sum(axis=-1) / seg_len2[None]
    t = np.clip(t, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(q[:, None, :] - proj, axis=-1)
    j = np.argmin(d, axis=1)
    rows = np.arange(len(q))
    s = cum[j] + t[rows, j] * seg_len[j]
    inside = (s > edge) & (s < cum[-1] - edge)
    return d[rows, j], s, inside


@dataclass
class SeedScore:
    arc: str
    z: int
    seed_xy: tuple[float, float]
    arc_length: float
    traced: bool
    correct: float = 0.0
    wrong: float = 0.0
    covered: float = 0.0
    departures: int = 0
    stop_reasons: tuple[str, str] = ("", "")
    path_xy: np.ndarray | None = field(default=None, repr=False)  # volume coords
    wrong_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class PathScore:
    correct: float  # path length that stayed on the arc
    wrong: float  # path length inside the arc's extent after a departure
    s_lo: float  # arc-length interval spanned by correct vertices
    s_hi: float
    departures: int
    wrong_mask: np.ndarray


def score_path(xy: np.ndarray, anchor: int, arc: np.ndarray, tau: float, persist: int) -> PathScore:
    """Walk outward from vertex ``anchor`` (assumed on the arc) in both directions."""
    dist, s, inside = project_to_polyline(xy, arc)
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    wrong_mask = np.zeros(len(xy), dtype=bool)
    correct = wrong = 0.0
    departures = 0
    s_lo = s_hi = s[anchor]
    for direction in (1, -1):
        idx = range(anchor + 1, len(xy)) if direction > 0 else range(anchor - 1, -1, -1)
        pending: list[int] = []
        pending_len = 0.0
        departed = False
        for i in idx:
            step = steps[i - 1] if direction > 0 else steps[i]
            if not inside[i]:
                break
            if departed:
                wrong += step
                wrong_mask[i] = True
                continue
            if dist[i] > tau:
                pending.append(i)
                pending_len += step
                if len(pending) >= persist:
                    departed = True
                    departures += 1
                    wrong += pending_len
                    wrong_mask[pending] = True
                continue
            correct += pending_len + step
            pending, pending_len = [], 0.0
            s_lo, s_hi = min(s_lo, s[i]), max(s_hi, s[i])
    return PathScore(correct, wrong, float(s_lo), float(s_hi), departures, wrong_mask)


def read_crop(vol: ZarrArray, z: int, x0: int, x1: int, y0: int, y1: int, z_half: int = 0) -> np.ndarray:
    """XY crop at slice ``z`` (optionally the mean of ``z +/- z_half``), indexed [y, x]."""
    block = vol.read(((z - z_half, z + z_half + 1), (y0, y1), (x0, x1)))
    return block.mean(axis=0) if z_half else block[0].astype(np.float32)


def arc_box(arc: Arc, margin: int) -> tuple[int, int, int, int]:
    lo = np.floor(arc.xy.min(axis=0)).astype(int) - margin
    hi = np.ceil(arc.xy.max(axis=0)).astype(int) + margin
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def evaluate_seeded(
    vol: ZarrArray,
    arcs: list[Arc],
    params: TraceParams = TraceParams(),
    *,
    seed_stride: int = 8,
    tau: float = 6.0,
    persist: int = 3,
    margin: int = 128,
    z_half: int = 0,
) -> list[SeedScore]:
    """Seed the tracer on every ``seed_stride``-th point of each arc and score each trace."""
    scores = []
    for arc in arcs:
        x0, x1, y0, y1 = arc_box(arc, margin)
        img = read_crop(vol, arc.z, x0, x1, y0, y1, z_half)
        fld = SliceField(img, params)
        origin = np.array([x0, y0], dtype=np.float64)
        local_arc = arc.xy - origin
        n = len(arc.xy)
        seed_ids = range(seed_stride // 2, n, seed_stride) if n > seed_stride else [n // 2]
        for k in seed_ids:
            path: Path | None = trace(fld, local_arc[k])
            sc = SeedScore(arc.name, arc.z, tuple(arc.xy[k]), arc.length, path is not None)
            if path is not None:
                ps = score_path(path.xy, path.seed_index, local_arc, tau, persist)
                sc.correct, sc.wrong, sc.departures = ps.correct, ps.wrong, ps.departures
                sc.covered = ps.s_hi - ps.s_lo
                sc.stop_reasons = path.stop_reasons
                sc.path_xy = path.xy + origin
                sc.wrong_mask = ps.wrong_mask
            scores.append(sc)
    return scores


def summarize(scores: list[SeedScore]) -> dict:
    traced = [s for s in scores if s.traced]
    correct = sum(s.correct for s in traced)
    wrong = sum(s.wrong for s in traced)
    return {
        "seeds": len(scores),
        "traced": len(traced),
        "seed_acceptance": len(traced) / max(len(scores), 1),
        "precision": correct / max(correct + wrong, 1e-9),
        "mean_coverage": float(np.mean([s.covered / s.arc_length for s in traced])) if traced else 0.0,
        "departure_rate": sum(s.departures > 0 for s in traced) / max(len(traced), 1),
        "correct_px": correct,
        "wrong_px": wrong,
    }


@dataclass
class ArcScore:
    """All automatically traced paths that claim one ground-truth arc."""

    arc: str
    z: int
    arc_length: float
    claims: int = 0
    correct: float = 0.0
    wrong: float = 0.0
    covered: float = 0.0  # union of the arc-length intervals of correct vertices
    departures: int = 0
    paths: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list, repr=False)  # (xy, wrong)
    crop_paths_px: float = 0.0  # all traced length in the crop, claiming or not


def claim_anchor(xy: np.ndarray, arc: np.ndarray, tau: float, min_run: int) -> int | None:
    """Vertex where a path most convincingly sits on the arc: the closest vertex of the
    longest run of at least ``min_run`` consecutive vertices within ``tau`` of it."""
    d, _, inside = project_to_polyline(xy, arc)
    ok = inside & (d <= tau)
    best, best_len, start = None, 0, None
    for i, flag in enumerate(np.append(ok, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best_len:
                best, best_len = (start, i), i - start
            start = None
    if best is None or best_len < min_run:
        return None
    lo, hi = best
    return lo + int(np.argmin(d[lo:hi]))


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in sorted(intervals):
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        total += cur_hi - cur_lo
    return total


def evaluate_auto(
    vol: ZarrArray,
    arcs: list[Arc],
    params: TraceParams = TraceParams(),
    *,
    tau: float = 6.0,
    persist: int = 3,
    claim_tau: float = 3.0,
    claim_run: int = 5,
    margin: int = 128,
    z_half: int = 0,
    grid_step: float = 6.0,
    min_length: float = 40.0,
) -> list[ArcScore]:
    """Trace every band in each arc's crop with no knowledge of the arc, then score the
    paths that claim it (sit within ``claim_tau`` of it for ``claim_run`` vertices)."""
    scores = []
    for arc in arcs:
        x0, x1, y0, y1 = arc_box(arc, margin)
        fld = SliceField(read_crop(vol, arc.z, x0, x1, y0, y1, z_half), params)
        origin = np.array([x0, y0], dtype=np.float64)
        local_arc = arc.xy - origin
        paths = trace_region(fld, grid_step=grid_step, min_length=min_length)
        sc = ArcScore(arc.name, arc.z, arc.length, crop_paths_px=sum(p.length for p in paths))
        intervals = []
        for path in paths:
            anchor = claim_anchor(path.xy, local_arc, claim_tau, claim_run)
            if anchor is None:
                continue
            ps = score_path(path.xy, anchor, local_arc, tau, persist)
            sc.claims += 1
            sc.correct += ps.correct
            sc.wrong += ps.wrong
            sc.departures += ps.departures
            intervals.append((ps.s_lo, ps.s_hi))
            sc.paths.append((path.xy + origin, ps.wrong_mask))
        sc.covered = _union_length(intervals)
        scores.append(sc)
    return scores


def summarize_auto(scores: list[ArcScore]) -> dict:
    correct = sum(s.correct for s in scores)
    wrong = sum(s.wrong for s in scores)
    claims = sum(s.claims for s in scores)
    return {
        "arcs": len(scores),
        "claims": claims,
        "precision": correct / max(correct + wrong, 1e-9),
        "coverage": sum(s.covered for s in scores) / max(sum(s.arc_length for s in scores), 1e-9),
        "departure_rate": sum(s.departures for s in scores) / max(claims, 1),
        "correct_px": correct,
        "wrong_px": wrong,
    }
