import numpy as np

from windcon.chains import build_chains, link_slices, shift_inward


def _line(x0, x1, y, step=2.0):
    xs = np.arange(x0, x1 + 1e-9, step)
    return np.column_stack([xs, np.full_like(xs, float(y))])


def test_links_each_sheet_to_itself_across_a_small_drift():
    a = [_line(0, 200, 100), _line(0, 200, 120), _line(0, 200, 140)]  # three sheets, 20 px pitch
    b = [_line(10, 210, 141.5), _line(10, 210, 101.5), _line(10, 210, 121.5)]  # drifted 1.5 px, shuffled
    assert link_slices(a, b) == [(0, 1), (1, 2), (2, 0)]


def test_refuses_to_link_between_sheets_too_close_to_tell_apart():
    a = [_line(0, 200, 100), _line(0, 200, 104)]  # two sheets 4 px apart (< r_sep)
    b = [_line(0, 200, 101)]
    assert link_slices(a, b) == []


def test_a_path_spanning_two_fragments_links_only_to_its_best_partner():
    a = [_line(0, 60, 100), _line(70, 200, 100)]  # one sheet broken into two fragments
    b = [_line(0, 200, 100.5)]
    assert link_slices(a, b) == [(1, 0)]  # one-to-one: the longer overlap wins


def test_chains_follow_links_and_keep_every_path_once():
    chains = build_chains([2, 2, 1], [[(0, 1), (1, 0)], [(1, 0)]])
    assert sorted(chains) == [[(0, 0), (1, 1), (2, 0)], [(0, 1), (1, 0)]]


def test_shift_moves_toward_the_axis_whichever_way_the_path_was_traced():
    axis = np.array([100.0, 0.0])  # the scroll axis lies at smaller y
    forward = _line(0, 200, 50)
    backward = forward[::-1]
    for path in (forward, backward):
        moved = shift_inward(path, np.repeat(axis[None], len(path), axis=0), 1.5)
        np.testing.assert_allclose(moved[:, 1], 48.5)
