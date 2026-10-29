import numpy as np
import pytest

from windcon.evaluate import claim_anchor, score_path

ARC = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])


def _line(x0, x1, y, step=2.0):
    xs = np.arange(x0, x1 + 1e-9, step)
    return np.column_stack([xs, np.full_like(xs, y)])


def test_path_on_the_arc_is_fully_correct():
    path = _line(4.0, 96.0, 1.0)
    s = score_path(path, anchor=len(path) // 2, arc=ARC, tau=6.0, persist=3)
    assert s.wrong == 0.0 and s.departures == 0
    assert s.correct == pytest.approx(92.0)
    assert s.s_hi - s.s_lo == pytest.approx(92.0)


def test_switch_to_the_neighbouring_sheet_counts_as_wrong():
    path = np.vstack([_line(10.0, 50.0, 0.0), _line(52.0, 90.0, 12.0)])  # jumps 12 px sideways
    s = score_path(path, anchor=5, arc=ARC, tau=6.0, persist=3)
    assert s.departures == 1
    assert s.wrong == pytest.approx(np.hypot(2.0, 12.0) + 2.0 * 19)
    assert s.s_hi == pytest.approx(50.0)


def test_brief_excursion_shorter_than_persist_is_not_a_departure():
    path = _line(10.0, 90.0, 0.0)
    path[20:22, 1] = 7.0  # two vertices just beyond tau
    s = score_path(path, anchor=5, arc=ARC, tau=6.0, persist=3)
    assert s.departures == 0 and s.wrong == 0.0


def test_vertices_beyond_the_arc_end_are_not_scored():
    path = np.vstack([_line(60.0, 98.0, 0.0), _line(100.0, 140.0, 15.0)])
    s = score_path(path, anchor=5, arc=ARC, tau=6.0, persist=3)
    assert s.wrong == 0.0 and s.departures == 0


def test_neighbouring_sheet_path_does_not_claim_the_arc():
    assert claim_anchor(_line(0.0, 100.0, 12.0), ARC, tau=3.0, min_run=5) is None
    assert claim_anchor(_line(0.0, 100.0, 1.0), ARC, tau=3.0, min_run=5) is not None
