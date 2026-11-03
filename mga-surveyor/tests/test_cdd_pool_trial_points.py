"""CddMethod.find_trial_points_above_tol / find_pool_trial_points -- the
batched candidate-selection building blocks for solving several
closest-point targets in parallel per drive() iteration instead of just
one (see explore.drive's own docstring for the full picture).
"""
import numpy as np
import pytest

from mga_surveyor.methods.cdd import CddMethod


def _box_halfspaces(k: int, half_width: float = 1.0) -> list[tuple[np.ndarray, np.ndarray]]:
    halfspaces = []
    for axis in range(k):
        for sign in (1.0, -1.0):
            direction = np.zeros(k)
            direction[axis] = sign
            halfspaces.append((direction, direction * half_width))
    return halfspaces


def _asymmetric_method() -> CddMethod:
    """A 2D box whose inner hull isn't centered, so outer-vertex gaps
    aren't tied -- ties would make find_trial_point's own argmax pick an
    implementation-defined winner among equals, which breaks the "index 0
    always matches find_trial_point" checks below for reasons that have
    nothing to do with the functions actually under test."""
    method = CddMethod(k=2, norm="linf", backend="float")
    inner_points = [np.array([0.9, 0.0]), np.array([0.0, 0.85]), np.array([-0.6, -0.6])]
    method.bootstrap(inner_points, _box_halfspaces(2))
    return method


def test_find_trial_points_above_tol_matches_find_trial_point_at_index_0():
    method = _asymmetric_method()
    trial_point, gap = method.find_trial_point()

    points, gaps = method.find_trial_points_above_tol(tol=0.0)

    np.testing.assert_allclose(points[0], trial_point)
    assert gaps[0] == pytest.approx(gap)
    assert list(gaps) == sorted(gaps, reverse=True)


def test_find_trial_points_above_tol_returns_every_outer_vertex_at_tol_zero():
    method = _asymmetric_method()
    points, gaps = method.find_trial_points_above_tol(tol=0.0)
    assert len(points) == 4  # a 2D box has 4 outer vertices
    assert len(gaps) == 4


def test_find_trial_points_above_tol_boundary_is_inclusive():
    """gap == tol exactly must be KEPT, not dropped -- matches
    explore.drive's own convergence test (`gap < tol` means converged), so
    a candidate sitting exactly at tol is one drive() itself would still
    have dispatched a solve for."""
    method = _asymmetric_method()
    _points, gaps = method.find_trial_points_above_tol(tol=0.0)
    boundary_tol = float(gaps.min())

    kept_points, kept_gaps = method.find_trial_points_above_tol(tol=boundary_tol)

    assert kept_gaps.min() == pytest.approx(boundary_tol)


def test_find_trial_points_above_tol_excludes_gaps_below_tol():
    method = _asymmetric_method()
    _points, all_gaps = method.find_trial_points_above_tol(tol=0.0)
    threshold = float(all_gaps[len(all_gaps) // 2])

    _points, gaps = method.find_trial_points_above_tol(tol=threshold + 1e-9)

    assert all(g >= threshold for g in gaps)
    assert len(gaps) < len(all_gaps)


def test_find_trial_points_above_tol_empty_when_tol_exceeds_every_gap():
    method = _asymmetric_method()
    points, gaps = method.find_trial_points_above_tol(tol=1e6)
    assert points.shape == (0, 2)
    assert gaps.shape == (0,)


def test_find_pool_trial_points_matches_find_trial_point_at_index_0():
    method = _asymmetric_method()
    trial_point, gap = method.find_trial_point()

    points, gaps = method.find_pool_trial_points(max_parallel=4, tol=0.0)

    np.testing.assert_allclose(points[0], trial_point)
    assert gaps[0] == pytest.approx(gap)


def test_find_pool_trial_points_caps_at_max_parallel():
    method = _asymmetric_method()
    points, gaps = method.find_pool_trial_points(max_parallel=2, tol=0.0)
    assert len(points) <= 2
    assert len(gaps) == len(points)


def test_find_pool_trial_points_applies_tol():
    method = _asymmetric_method()
    _points, all_gaps = method.find_trial_points_above_tol(tol=0.0)
    threshold = float(all_gaps[len(all_gaps) // 2])

    points, gaps = method.find_pool_trial_points(max_parallel=10, tol=threshold + 1e-9)

    assert all(g >= threshold for g in gaps)


def test_find_pool_trial_points_requires_max_parallel_and_tol():
    method = _asymmetric_method()
    with pytest.raises(TypeError):
        method.find_pool_trial_points(max_parallel=2)
    with pytest.raises(TypeError):
        method.find_pool_trial_points(tol=0.0)
