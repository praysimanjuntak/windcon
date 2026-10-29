import numpy as np
from scipy import ndimage

from windcon.relwind import BLOCKED, CLEAN, classify_profile, find_rays
from windcon.tracer import TraceParams

CENTER = np.array([150.0, 150.0])
RADII = [50.0 + 15.0 * k for k in range(6)]  # six windings 15 px apart


def _scroll_image() -> np.ndarray:
    yy, xx = np.mgrid[0:300, 0:300].astype(np.float64)
    r = np.hypot(xx - CENTER[0], yy - CENTER[1])
    img = 30.0 + sum(170.0 * np.exp(-((r - rk) ** 2) / (2 * 2.0**2)) for rk in RADII)
    return ndimage.gaussian_filter(img, 1.0)


def _arc(radius: float) -> np.ndarray:
    theta = np.arange(0.2, 1.4, 2.0 / radius)  # 2 px vertex spacing, like the tracer
    return CENTER + radius * np.column_stack([np.cos(theta), np.sin(theta)])


def test_traced_neighbours_pair_up_but_an_untraced_sheet_between_blocks_the_pair():
    paths = [_arc(RADII[0]), _arc(RADII[1]), _arc(RADII[3])]  # winding 2 is in the scan, untraced
    rays = find_rays(_scroll_image(), np.zeros(2), paths, CENTER)
    pairs = rays.pairs(min_votes=3)
    assert (0, 1) in pairs  # path 1 is one winding outward of path 0
    assert (1, 0) not in pairs
    assert (1, 2) not in pairs  # two windings apart: never adjacent
    assert np.any((rays.inner == 1) & (rays.outer == 2) & (rays.verdict == BLOCKED))


def test_profile_verdicts():
    tp = TraceParams()
    s = np.arange(-4.0, 30.0, 0.5)

    def bands(*centres):
        return 30.0 + sum(170.0 * np.exp(-((s - c) ** 2) / (2 * 2.0**2)) for c in centres)

    assert classify_profile(bands(0.0, 15.0), s, 15.0, tp, 3.0) == CLEAN
    assert classify_profile(bands(0.0, 12.0, 24.0), s, 24.0, tp, 3.0) == BLOCKED
    bridged = np.maximum(bands(0.0, 15.0), np.where((s > 0) & (s < 15), 175.0, 0.0))  # no dark gap
    assert classify_profile(bridged, s, 15.0, tp, 3.0) != CLEAN
