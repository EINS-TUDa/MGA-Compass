import numpy as np
import pytest

from mga_surveyor.common import linearized_objective_cut, select_by_min_angle


def test_linearized_objective_cut_collapses_to_x0_when_no_slack_left():
    """K == 0 (obj_value already equals the (1+slack)*obj_optimal
    threshold exactly) -- the cut must pass through x0 itself, same as
    any other cut that touches its own generating point."""
    x0 = np.array([3.0, -1.0])
    duals = np.array([2.0, 5.0])
    obj_optimal = 100.0
    slack = 0.1
    obj_value = (1 + slack) * obj_optimal  # exactly at the threshold -> K = 0

    cut = linearized_objective_cut(x0, obj_value, duals, obj_optimal, slack)
    assert cut is not None
    direction, point = cut
    np.testing.assert_allclose(direction, duals)
    np.testing.assert_allclose(point, x0)


def test_linearized_objective_cut_returns_none_for_zero_duals():
    """duals ~ 0 means the linearized constraint is `0 <= K`, always true
    -- a vacuous cut, not worth adding."""
    x0 = np.array([1.0, 2.0])
    duals = np.array([0.0, 0.0])
    cut = linearized_objective_cut(x0, obj_value=50.0, duals=duals, obj_optimal=100.0, slack=0.1)
    assert cut is None


def test_linearized_objective_cut_reproduces_the_exact_linear_case():
    """obj(x) = 2x + 1 is exactly linear (not just convex), so the tangent
    at any point reproduces the function itself exactly, with zero
    approximation error -- lets this be checked precisely, the same
    numeric example worked through by hand."""
    x0 = np.array([3.0])
    duals = np.array([2.0])
    obj_value = 2 * x0[0] + 1  # = 7
    obj_optimal = 50.0
    slack = 0.2
    threshold = (1 + slack) * obj_optimal

    direction, point = linearized_objective_cut(x0, obj_value, duals, obj_optimal, slack)

    # The halfspace {x : direction . (x - point) <= 0} must be exactly
    # {x : 2x + 1 <= threshold}, i.e. direction . point == threshold - 1.
    np.testing.assert_allclose(direction @ point, threshold - 1.0)

    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(-50, 50, size=1)
        in_halfspace = direction @ (x - point) <= 1e-9
        obj_x = 2 * x[0] + 1
        assert in_halfspace == (obj_x <= threshold + 1e-9)


def test_linearized_objective_cut_clips_negative_k_to_zero():
    """obj_value slightly over the threshold (numerical noise from an
    earlier solve) must not send K negative and the point off in the
    wrong direction -- K is clipped at 0, same as the "no slack left"
    case."""
    x0 = np.array([1.0, 1.0])
    duals = np.array([3.0, -1.0])
    obj_optimal = 10.0
    slack = 0.0
    obj_value = obj_optimal * 1.0001  # slightly over (1+slack)*obj_optimal

    direction, point = linearized_objective_cut(x0, obj_value, duals, obj_optimal, slack)
    np.testing.assert_allclose(point, x0)
    np.testing.assert_allclose(direction, duals)


def test_linearized_objective_cut_point_choice_does_not_affect_the_halfspace():
    """Any point actually on the constructed hyperplane defines the exact
    same halfspace as the canonical one returned -- add_cut only cares
    about the scalar direction @ point, not which point achieves it."""
    x0 = np.array([2.0, -3.0, 0.5])
    duals = np.array([1.0, 4.0, -2.0])
    obj_optimal = 200.0
    slack = 0.15
    obj_value = 150.0

    direction, point = linearized_objective_cut(x0, obj_value, duals, obj_optimal, slack)

    # A different, non-canonical point on the same hyperplane (nudge along
    # a direction orthogonal to `duals`).
    orthogonal = np.array([4.0, -1.0, 0.0])  # duals . orthogonal == 0
    assert abs(duals @ orthogonal) < 1e-9
    other_point = point + orthogonal

    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.uniform(-20, 20, size=3)
        via_canonical = direction @ (x - point) <= 1e-9
        via_other = direction @ (x - other_point) <= 1e-9
        assert via_canonical == via_other


def test_linearized_objective_cut_avoids_extreme_points_under_mismatched_duals_scale():
    """duals mixing very different real-world-unit magnitudes (e.g. $/MW
    vs $/MWh) -- the scenario that produced a real ~187,000-vertex OOM
    crash in this project's (now-removed) incremental edge-walking
    construction, traced back to this function's point construction --
    must not send `point` to a
    numerically extreme value. Picking the largest-magnitude axis instead
    of the Euclidean projection keeps the shift confined to a single,
    well-conditioned dimension, never dividing by the tiny one."""
    x0 = np.array([10.0, 10.0])
    duals = np.array([1e-6, 100.0])  # dimension 0 barely matters, dimension 1 dominates
    obj_optimal = 1000.0
    slack = 0.1
    obj_value = 900.0  # K = 1100 - 900 = 200

    direction, point = linearized_objective_cut(x0, obj_value, duals, obj_optimal, slack)

    np.testing.assert_allclose(direction, duals)
    np.testing.assert_allclose(point[0], x0[0])  # untouched -- not the tiny-denominator axis
    np.testing.assert_allclose(point[1], x0[1] + 200.0 / 100.0)  # shifted along the dominant axis only
    assert abs(point[1] - x0[1]) < 1e3  # nowhere near the ~1e8-scale blowup the old formula gave here


def _rot2d(deg: float) -> np.ndarray:
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)])


def test_select_by_min_angle_skips_points_too_close_to_an_already_selected_one():
    """0/30/90/180 degrees, threshold 45: 30 is too close to 0 (30 <= 45)
    and gets skipped; 90 and 180 both clear 45 degrees from every point
    kept so far and are kept."""
    points = np.array([_rot2d(0), _rot2d(30), _rot2d(90), _rot2d(180)])

    selected = select_by_min_angle(points, n=4, min_angle_deg=45.0)

    assert len(selected) == 3
    np.testing.assert_allclose(selected[0], _rot2d(0), atol=1e-9)
    np.testing.assert_allclose(selected[1], _rot2d(90), atol=1e-9)
    np.testing.assert_allclose(selected[2], _rot2d(180), atol=1e-9)


def test_select_by_min_angle_caps_at_n_even_when_more_would_qualify():
    points = np.array([_rot2d(0), _rot2d(90), _rot2d(180), _rot2d(270)])
    selected = select_by_min_angle(points, n=2, min_angle_deg=45.0)
    assert len(selected) == 2


def test_select_by_min_angle_is_greedy_not_globally_optimal():
    """Walking order matters: with 0/44/90 and threshold 45, starting from
    0 skips 44 (44 <= 45) but keeps 90 (90 > 45) -- even though 44 and 90
    are themselves more than 45 degrees apart and could have coexisted had
    0 not been picked first."""
    points = np.array([_rot2d(0), _rot2d(44), _rot2d(90)])
    selected = select_by_min_angle(points, n=3, min_angle_deg=45.0)
    assert len(selected) == 2
    np.testing.assert_allclose(selected[0], _rot2d(0), atol=1e-9)
    np.testing.assert_allclose(selected[1], _rot2d(90), atol=1e-9)


def test_select_by_min_angle_skips_zero_vectors():
    """A point with no well-defined direction (norm ~0) is skipped
    outright, not treated as compatible with everything."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    selected = select_by_min_angle(points, n=3, min_angle_deg=45.0)
    assert len(selected) == 2
    np.testing.assert_allclose(selected, [[1.0, 0.0], [0.0, 1.0]])


def test_select_by_min_angle_empty_input_returns_empty_output():
    selected = select_by_min_angle(np.empty((0, 3)), n=2, min_angle_deg=10.0)
    assert selected.shape == (0, 3)


def test_select_by_min_angle_rejects_out_of_range_threshold():
    points = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="min_angle_deg"):
        select_by_min_angle(points, n=1, min_angle_deg=200.0)
    with pytest.raises(ValueError, match="min_angle_deg"):
        select_by_min_angle(points, n=1, min_angle_deg=-1.0)
