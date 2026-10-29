import numpy as np

from windcon.tracer import SliceField, trace, trace_region

AIR, PAPYRUS, HALF_THICKNESS = 40.0, 120.0, 3.0


def _render(distance_to_centerline: np.ndarray, seed: int = 0) -> np.ndarray:
    """Bright bands where the distance to a centreline is below the half thickness."""
    img = np.where(distance_to_centerline <= HALF_THICKNESS, PAPYRUS, AIR)
    return img + np.random.default_rng(seed).normal(0.0, 3.0, img.shape)


def _rings(radii, size=420, center=210.0):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    r = np.hypot(xx - center, yy - center)
    return np.min([np.abs(r - ri) for ri in radii], axis=0)


def test_stays_on_its_ring_among_parallel_rings():
    radii = [60.0, 80.0, 100.0, 120.0, 140.0]
    field = SliceField(_render(_rings(radii)))
    path = trace(field, np.array([210.0 + 100.0, 210.0]))
    assert path is not None
    radius = np.hypot(path.xy[:, 0] - 210.0, path.xy[:, 1] - 210.0)
    assert np.max(np.abs(radius - 100.0)) < 1.5  # never drifts towards the rings 20 px away
    assert path.length > 0.8 * 2 * np.pi * 100.0  # follows most of the circumference


def test_stops_before_a_merging_band_instead_of_switching():
    size = 400
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    # Band A: horizontal line y = 200. Band B converges onto A from 30 px below and merges at x = 220.
    dist_a = np.abs(yy - 200.0)
    b_center = np.where(xx < 220.0, 200.0 + 30.0 * (220.0 - xx) / 200.0, 200.0)
    dist_b = np.abs(yy - b_center)
    field = SliceField(_render(np.minimum(dist_a, dist_b)))
    path = trace(field, np.array([60.0, 200.0 + 30.0 * 160.0 / 200.0]))
    assert path is not None and path.length > 40.0
    b_at_x = np.where(path.xy[:, 0] < 220.0, 200.0 + 30.0 * (220.0 - path.xy[:, 0]) / 200.0, 200.0)
    assert np.all(np.abs(path.xy[:, 1] - b_at_x) < 1.5)  # on B throughout
    assert np.all(np.abs(path.xy[:, 1] - 200.0) > 4.0)  # never on A, i.e. stopped before the merge


def test_region_tracing_claims_each_ring_once():
    radii = [80.0, 110.0, 140.0]
    field = SliceField(_render(_rings(radii)))
    paths = trace_region(field, grid_step=6.0)
    owner = []
    for path in paths:
        radius = np.hypot(path.xy[:, 0] - 210.0, path.xy[:, 1] - 210.0)
        nearest = np.array(radii)[np.argmin(np.abs(radius[:, None] - np.array(radii)[None]), axis=1)]
        assert np.all(nearest == nearest[0])  # every path lies on a single ring
        owner.append(nearest[0])
    for r in radii:
        traced = sum(p.length for p, o in zip(paths, owner) if o == r)
        assert 0.8 * 2 * np.pi * r < traced < 1.2 * 2 * np.pi * r  # covered, without duplicates
