"""Conservative same-winding path tracing in a single XY slice of a scroll CT volume.

In an XY slice a papyrus winding is a bright, textured band separated from its
neighbours by dark gaps. Starting from a seed on a band, the tracer repeatedly

1. steps along the local band tangent (Hessian eigenvector at a coarse scale),
2. samples a *thick* intensity profile across the band (the normal profile
   averaged over a short stretch along the tangent, which washes out fibre
   lumens and other in-band texture),
3. delimits the band by its first flanking valley on each side and re-centres
   on the midpoint of its half-level edges,
4. checks that the band is still unambiguous: bright enough, a real gap on
   *both* sides, no neighbouring band closer than ``min_neighbor_dist`` (the
   signature of a fork, a merge or a delaminating sheet), not wider than one
   sheet, and only a small lateral correction and heading change.

Fork and merge signatures (``too_wide``, ``close_neighbor``) stop the trace at
once. Other failures (a faint gap, a dark spot, crossing fibre texture) may be
bridged by coasting straight for a few steps, but the bridge is kept only if a
clean band is re-acquired where predicted and then holds for several steps.
Every stop trims the last few steps, so a path is the longest stretch over
which the local evidence never allowed a second interpretation. Precision is
preferred over length: a stopped path is a smaller constraint, a switched path
is a wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class TraceParams:
    """Pixel units are slice pixels of the volume level being traced (9.6 um at L2)."""

    sigma_image: float = 1.0  # isotropic pre-smoothing of the slice
    sigma_orient: float = 3.0  # Hessian scale used for the band tangent
    step: float = 2.0  # advance per iteration
    window: float = 16.0  # +/- extent of the normal profile
    profile_spacing: float = 0.5  # sampling interval along the normal
    thickness: float = 3.0  # +/- extent along the tangent averaged into each profile
    capture: float = 3.0  # the band peak is searched within +/- capture of the prediction
    air_level: float = 40.0  # intensity of empty space, the zero of contrast (uint8 units)
    min_peak: float = 70.0  # band peak brightness (uint8 units)
    valley_rise: float = 0.15  # climb, as a fraction of (peak - air), that ends a valley
    min_gap_contrast: float = 0.3  # (peak - valley) / (peak - air) required on both sides
    min_neighbor_dist: float = 8.0  # a neighbouring band closer than this suggests a fork or merge
    max_width: float = 14.0  # a wider band is taken to be fused sheets
    max_shift: float = 2.0  # max lateral re-centring per step
    turn_baseline: int = 5  # steps per chord in the heading-change test
    max_turn_deg: float = 30.0  # max heading change between consecutive chords
    trim_steps: int = 3  # steps removed before every stop (joined ends excepted)
    max_steps: int = 5000  # per direction
    max_bridge_steps: int = 4  # failed steps the tracer may coast across before stopping
    bridge_confirm_steps: int = 3  # clean steps required after a bridge to keep it
    max_step_turn_deg: float = 15.0  # max change of the band tangent between consecutive steps
    join_steps: int = 3  # consecutive steps on already-traced territory that end a trace
    self_approach: float = 1.5  # stop when re-entering the path's own trail (in steps)


STOP_REASONS = (
    "weak_peak",
    "close_neighbor",
    "no_gap",
    "too_wide",
    "lateral_jump",
    "tangent_turn",
    "sharp_turn",
    "left_image",
    "joined",
    "self_intersection",
    "max_steps",
)


@dataclass
class Band:
    """Cross-section of one band along a normal; offsets are signed distances along the normal."""

    center: float
    width: float
    peak: float
    gap_contrast: float  # min over both sides of (peak - valley) / (peak - air)
    neighbor_dist: float  # centre-to-peak distance of the nearest neighbouring band (inf if none)


@dataclass
class Path:
    """An ordered polyline in slice pixel coordinates (x, y) with per-vertex evidence.

    Evidence arrays are NaN at vertices that were coasted across (bridged)."""

    xy: np.ndarray  # (N, 2) float64
    width: np.ndarray  # (N,) band width
    gap_contrast: np.ndarray  # (N,)
    neighbor_dist: np.ndarray  # (N,)
    seed_index: int
    stop_reasons: tuple[str, str]  # (start end, finish end)

    @property
    def length(self) -> float:
        if len(self.xy) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum())

    def resampled(self, spacing: float) -> np.ndarray:
        """Vertices at (about) ``spacing`` arc-length intervals, both end points included."""
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))])
        n = max(int(round(cum[-1] / spacing)), 1)
        s = np.linspace(0.0, cum[-1], n + 1)
        return np.stack([np.interp(s, cum, self.xy[:, 0]), np.interp(s, cum, self.xy[:, 1])], axis=1)


class SliceField:
    """Smoothed intensity and Hessian of one slice, sampled at sub-pixel points."""

    def __init__(self, image: np.ndarray, params: TraceParams = TraceParams()):
        img = np.asarray(image, dtype=np.float32)
        self.params = params
        self.shape = img.shape
        self.smooth = ndimage.gaussian_filter(img, params.sigma_image)
        s = params.sigma_orient
        self.hxx = ndimage.gaussian_filter(img, s, order=(0, 2))
        self.hyy = ndimage.gaussian_filter(img, s, order=(2, 0))
        self.hxy = ndimage.gaussian_filter(img, s, order=(1, 1))
        p = params
        self._v = np.arange(-p.window, p.window + 1e-9, p.profile_spacing)
        self._u = np.arange(-p.thickness, p.thickness + 1e-9, 1.0)

    def inside(self, xy: np.ndarray, margin: float) -> bool:
        x, y = xy
        return margin <= x < self.shape[1] - 1 - margin and margin <= y < self.shape[0] - 1 - margin

    @staticmethod
    def _sample(arr: np.ndarray, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        return ndimage.map_coordinates(arr, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")

    def tangent(self, xy: np.ndarray) -> np.ndarray:
        """Unit vector along the band: eigenvector of the larger Hessian eigenvalue."""
        hxx, hyy, hxy = (float(self._sample(a, xy)[0]) for a in (self.hxx, self.hyy, self.hxy))
        theta = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)
        return np.array([np.cos(theta), np.sin(theta)])

    def thick_profile(self, xy: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        """Normal profile at offsets ``self._v``, averaged over ``self._u`` along the tangent."""
        normal = np.array([-tangent[1], tangent[0]])
        grid = (
            xy[None, None, :]
            + self._v[None, :, None] * normal[None, None, :]
            + self._u[:, None, None] * tangent[None, None, :]
        )
        vals = self._sample(self.smooth, grid.reshape(-1, 2))
        return vals.reshape(len(self._u), len(self._v)).mean(axis=0)

    def band(self, xy: np.ndarray, tangent: np.ndarray) -> Band | None:
        return analyse_profile(self._v, self.thick_profile(xy, tangent), self.params)


def _walk_valley(outward: np.ndarray, rise: float) -> int:
    """Index of the first valley walking outward from index 0: the running minimum up to
    the first climb of ``rise`` above it (or the overall minimum if the profile never climbs)."""
    running = np.minimum.accumulate(outward)
    climbed = np.flatnonzero(outward > running + rise)
    end = climbed[0] if climbed.size else len(outward)
    return int(np.argmin(outward[:end]))


def _half_level_edge(outward: np.ndarray, level: float, spacing: float) -> float:
    """Distance from index 0 to the first sub-sample crossing below ``level``."""
    below = np.flatnonzero(outward < level)
    if below.size == 0:
        return len(outward) * spacing
    i = int(below[0])
    if i == 0:
        return 0.0
    a, b = outward[i - 1], outward[i]
    frac = (a - level) / (a - b) if a != b else 0.0
    return (i - 1 + frac) * spacing


def _next_peak(outward: np.ndarray, start: int, rise: float) -> int | None:
    """Index of the first peak beyond the valley at ``start`` that climbs ``rise`` above it:
    the running maximum up to its first drop of ``rise``. None if the profile never climbs."""
    seg = outward[start:]
    if seg.size < 2:
        return None
    running = np.maximum.accumulate(seg)
    dropped = np.flatnonzero(seg < running - rise)
    end = dropped[0] if dropped.size else len(seg)
    j = int(np.argmax(seg[:end]))
    return start + j if seg[j] >= seg[0] + rise else None


def analyse_profile(v: np.ndarray, prof: np.ndarray, p: TraceParams) -> Band | None:
    zero = int(np.argmin(np.abs(v)))
    reach = int(round(p.capture / p.profile_spacing))
    lo, hi = max(zero - reach, 0), min(zero + reach + 1, len(prof))
    b = lo + int(np.argmax(prof[lo:hi]))
    peak = float(prof[b])
    if peak < p.min_peak:
        return None
    span = peak - p.air_level
    if span <= 0:
        return None
    rise = p.valley_rise * span
    left_out, right_out = prof[b::-1], prof[b:]
    lv, rv = _walk_valley(left_out, rise), _walk_valley(right_out, rise)
    left_valley, right_valley = float(left_out[lv]), float(right_out[rv])
    gap = min(peak - left_valley, peak - right_valley) / span
    e_left = _half_level_edge(left_out, 0.5 * (peak + left_valley), p.profile_spacing)
    e_right = _half_level_edge(right_out, 0.5 * (peak + right_valley), p.profile_spacing)
    center = float(v[b]) + 0.5 * (e_right - e_left)
    neighbor = np.inf
    ln, rn = _next_peak(left_out, lv, rise), _next_peak(right_out, rv, rise)
    if ln is not None:
        neighbor = min(neighbor, center - float(v[b - ln]))
    if rn is not None:
        neighbor = min(neighbor, float(v[b + rn]) - center)
    return Band(center, e_left + e_right, peak, float(gap), float(neighbor))


# Failures that indicate a fork or a merge: never bridged across.
HARD_FAILURES = frozenset({"too_wide", "close_neighbor"})


def band_failure(band: Band | None, p: TraceParams) -> str | None:
    if band is None:
        return "weak_peak"
    if band.width > p.max_width:
        return "too_wide"
    if band.neighbor_dist < p.min_neighbor_dist:
        return "close_neighbor"
    if band.gap_contrast < p.min_gap_contrast:
        return "no_gap"
    return None


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _trace_direction(
    field_: SliceField,
    start: np.ndarray,
    heading: np.ndarray,
    occupied: np.ndarray | None = None,
    prior: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[Band | None], str]:
    """Follow the band from ``start``.

    A failed check opens a bridge: the tracer coasts along its last trusted heading
    for up to ``max_bridge_steps``. The coasted points are kept only if a clean band
    is re-acquired where the extrapolation predicts it *and* the trace then survives
    ``bridge_confirm_steps`` clean steps; otherwise the trace is rolled back to where
    the bridge began and stops there.
    """
    p = field_.params
    margin = p.window + p.thickness + 2.0
    pos, head = start.copy(), heading / np.linalg.norm(heading)
    pts: list[np.ndarray] = [start]
    bands: list[Band | None] = []
    coast: list[np.ndarray] = []
    first_failure = ""
    rollback: int | None = None  # len(pts) before the last bridge, until it is confirmed
    since_bridge = 0
    joined_run = 0
    k = p.turn_baseline
    reason = "max_steps"
    for _ in range(p.max_steps):
        pred = pos + p.step * head
        if not field_.inside(pred, margin):
            reason = "left_image"
            break
        t = field_.tangent(pred)
        if np.dot(t, head) < 0:
            t = -t
        failure = None
        band = None
        if _angle_deg(t, head) > p.max_step_turn_deg:
            failure = "tangent_turn"
        else:
            band = field_.band(pred, t)
            failure = band_failure(band, p)
            if failure is None and abs(band.center) > p.max_shift:
                failure = "lateral_jump"
        if failure is not None:
            if rollback is not None:
                reason = failure
                del pts[rollback:], bands[rollback - 1 :]
                coast = []
                break
            if failure in HARD_FAILURES:
                reason = failure
                break
            if not coast:
                first_failure = failure
            if len(coast) >= p.max_bridge_steps:
                reason = first_failure
                break
            coast.append(pred)
            pos = pred
            continue
        new = pred + band.center * np.array([-t[1], t[0]])
        chain = pts + coast + [new]
        if len(chain) > 2 * k and _angle_deg(new - chain[-1 - k], chain[-1 - k] - chain[-1 - 2 * k]) > p.max_turn_deg:
            reason = "sharp_turn"
            break
        trail = np.asarray(chain[:-11]) if len(chain) > 11 else np.zeros((0, 2))
        if prior is not None:  # the other direction's trail, e.g. when tracing round a closed band
            trail = np.vstack([trail, prior])
        if len(trail) and np.min(np.linalg.norm(trail - new, axis=1)) < p.self_approach * p.step:
            reason = "self_intersection"
            break
        if coast:
            rollback, since_bridge = len(pts), 0
            pts.extend(coast)
            bands.extend([None] * len(coast))
            coast = []
        elif rollback is not None:
            since_bridge += 1
            if since_bridge >= p.bridge_confirm_steps:
                rollback = None
        pts.append(new)
        bands.append(band)
        nt = field_.tangent(new)
        head = nt if np.dot(nt, t) >= 0 else -nt
        pos = new
        if occupied is not None:
            on_other = occupied[int(round(new[1])), int(round(new[0]))]
            joined_run = joined_run + 1 if on_other else 0
            if joined_run >= p.join_steps:
                reason = "joined"
                break
    if rollback is not None:  # stopped (or left the image) before the last bridge was confirmed
        del pts[rollback:], bands[rollback - 1 :]
    pts = pts[1:]  # drop the shared start point
    keep = len(pts) if reason in ("left_image", "joined") else max(0, len(pts) - p.trim_steps)
    return pts[:keep], bands[:keep], reason


def trace(field_: SliceField, seed_xy: np.ndarray, occupied: np.ndarray | None = None) -> Path | None:
    """Trace both ways from a seed. Returns None when the seed is not on a clean band.

    ``occupied`` is an optional boolean raster of already-traced territory; a trace that
    runs onto it for ``join_steps`` consecutive steps stops there with reason ``joined``.
    """
    p = field_.params
    seed_xy = np.asarray(seed_xy, dtype=np.float64)
    if not field_.inside(seed_xy, p.window + p.thickness + 2.0):
        return None
    t = field_.tangent(seed_xy)
    band = field_.band(seed_xy, t)
    if band_failure(band, p) is not None or abs(band.center) > p.capture:
        return None
    start = seed_xy + band.center * np.array([-t[1], t[0]])
    t = field_.tangent(start)
    fwd, fwd_bands, fwd_reason = _trace_direction(field_, start, t, occupied)
    prior = np.asarray(fwd[10:]) if len(fwd) > 10 else None  # skip the stretch next to the start
    bwd, bwd_bands, bwd_reason = _trace_direction(field_, start, -t, occupied, prior)
    xy = np.array(bwd[::-1] + [start] + fwd, dtype=np.float64).reshape(-1, 2)
    bands = bwd_bands[::-1] + [band] + fwd_bands
    return Path(
        xy=xy,
        width=np.array([b.width if b else np.nan for b in bands]),
        gap_contrast=np.array([b.gap_contrast if b else np.nan for b in bands]),
        neighbor_dist=np.array([b.neighbor_dist if b else np.nan for b in bands]),
        seed_index=len(bwd),
        stop_reasons=(bwd_reason, fwd_reason),
    )


def _mark(occupied: np.ndarray, xy: np.ndarray, radius: float) -> None:
    """Set every pixel within ``radius`` of the polyline vertices ``xy``."""
    r = int(np.ceil(radius))
    dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = dx**2 + dy**2 <= radius**2
    offs_y, offs_x = dy[disk], dx[disk]
    centers = np.round(xy).astype(int)
    ys = (centers[:, 1, None] + offs_y[None]).ravel()
    xs = (centers[:, 0, None] + offs_x[None]).ravel()
    ok = (ys >= 0) & (ys < occupied.shape[0]) & (xs >= 0) & (xs < occupied.shape[1])
    occupied[ys[ok], xs[ok]] = True


def find_seeds(field_: SliceField, grid_step: float) -> np.ndarray:
    """Band-centred seed candidates on a regular grid, best gap contrast first. (K, 2) xy."""
    p = field_.params
    margin = p.window + p.thickness + 2.0
    ys = np.arange(margin, field_.shape[0] - margin, grid_step)
    xs = np.arange(margin, field_.shape[1] - margin, grid_step)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    # Cheap pre-filter: only bright pixels can be band centres.
    grid = grid[field_._sample(field_.smooth, grid) >= p.min_peak]
    seeds, quality = [], []
    for xy in grid:
        t = field_.tangent(xy)
        band = field_.band(xy, t)
        if band_failure(band, p) is None and abs(band.center) <= 0.5 * grid_step:
            seeds.append(xy + band.center * np.array([-t[1], t[0]]))
            quality.append(band.gap_contrast)
    if not seeds:
        return np.zeros((0, 2))
    order = np.argsort(quality)[::-1]
    return np.asarray(seeds)[order]


def trace_region(
    field_: SliceField,
    *,
    grid_step: float = 6.0,
    min_length: float = 40.0,
    dedup_radius: float = 4.0,
) -> list[Path]:
    """Trace every clean band in the slice once: seeds are visited best-first and skipped
    when they fall on territory an earlier path already claimed."""
    occupied = np.zeros(field_.shape, dtype=bool)
    paths = []
    for seed in find_seeds(field_, grid_step):
        if occupied[int(round(seed[1])), int(round(seed[0]))]:
            continue
        path = trace(field_, seed, occupied)
        if path is None or path.length < min_length:
            continue
        paths.append(path)
        _mark(occupied, path.xy, dedup_radius)
    return paths
