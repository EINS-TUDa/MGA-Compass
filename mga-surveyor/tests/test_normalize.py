import numpy as np
import pytest

from mga_surveyor.normalize import AffineMap, compute_affine_map, normalize_oracle_fns


def _box_points_2d():
    """k=2 box points, ordered like explore.drive's own box_directions
    (+e_0, +e_1, -e_0, -e_1): dimension 0 ranges [-5, 5], dimension 1
    ranges [-3, 3] -- deliberately different scales, the whole point of
    normalizing at all."""
    return np.array([
        [5.0, 1.0],   # +e_0 solve: x_0 = 5 is dimension 0's max
        [0.0, 3.0],   # +e_1 solve: x_1 = 3 is dimension 1's max
        [-5.0, -2.0],  # -e_0 solve: x_0 = -5 is dimension 0's min
        [0.0, -3.0],  # -e_1 solve: x_1 = -3 is dimension 1's min
    ])


def test_compute_affine_map_reads_min_max_off_each_dimensions_own_coordinate():
    affine_map = compute_affine_map(_box_points_2d())
    np.testing.assert_allclose(affine_map.center, [0.0, 0.0])
    np.testing.assert_allclose(affine_map.half_range, [5.0, 3.0])


def test_compute_affine_map_handles_asymmetric_ranges():
    # dimension 0: [2, 10] -> center 6, half_range 4
    box_points = np.array([[10.0], [2.0]])
    affine_map = compute_affine_map(box_points)
    np.testing.assert_allclose(affine_map.center, [6.0])
    np.testing.assert_allclose(affine_map.half_range, [4.0])


def test_compute_affine_map_rejects_a_fixed_dimension():
    """A dimension whose box max == min (no achievable range under the
    current slack budget) must raise, not silently produce inf/nan via
    1/half_range."""
    box_points = np.array([
        [5.0, 3.0],  # +e_0: x_0 = 5 (real range)
        [1.0, 3.0],  # +e_1: x_1 = 3 -- same as -e_1 below, fixed
        [-5.0, 3.0],  # -e_0: x_0 = -5
        [1.0, 3.0],  # -e_1: x_1 = 3 -- dimension 1 never moves
    ])
    with pytest.raises(ValueError, match="fixed at a single value"):
        compute_affine_map(box_points)


def test_compute_affine_map_names_the_fixed_dimension_when_keys_given():
    box_points = np.array([
        [5.0, 3.0],
        [1.0, 3.0],
        [-5.0, 3.0],
        [1.0, 3.0],
    ])
    with pytest.raises(ValueError, match="solar"):
        compute_affine_map(box_points, keys=["wind", "solar"])


def test_to_normalized_maps_the_box_onto_the_unit_cube():
    affine_map = compute_affine_map(_box_points_2d())
    np.testing.assert_allclose(affine_map.to_normalized([0.0, 0.0]), [0.0, 0.0])
    np.testing.assert_allclose(affine_map.to_normalized([5.0, 3.0]), [1.0, 1.0])
    np.testing.assert_allclose(affine_map.to_normalized([-5.0, -3.0]), [-1.0, -1.0])


def test_to_normalized_and_to_real_are_inverses():
    affine_map = compute_affine_map(_box_points_2d())
    rng = np.random.default_rng(0)
    for _ in range(20):
        point = rng.uniform(-10, 10, size=2)
        np.testing.assert_allclose(affine_map.to_real(affine_map.to_normalized(point)), point)


def test_normal_to_normalized_preserves_the_halfspace():
    """A halfspace {x : normal . (x - point) <= 0} (Method.add_cut's own
    convention) must contain/exclude the same points before and after
    transforming both the halfspace and the points into normalized space
    -- that's the whole point of transforming the normal by half_range
    (not by weights, i.e. not the same way a point transforms)."""
    affine_map = compute_affine_map(_box_points_2d())
    normal = np.array([1.0, 2.0])
    point_on_plane = np.array([1.0, 1.0])

    def in_halfspace_real(x):
        return normal @ (x - point_on_plane) <= 1e-9

    normal_n = affine_map.normal_to_normalized(normal)
    point_on_plane_n = affine_map.to_normalized(point_on_plane)

    def in_halfspace_normalized(x_n):
        return normal_n @ (x_n - point_on_plane_n) <= 1e-9

    rng = np.random.default_rng(1)
    for _ in range(30):
        x = rng.uniform(-8, 8, size=2)
        assert in_halfspace_real(x) == in_halfspace_normalized(affine_map.to_normalized(x))


def test_normalize_oracle_fns_solves_box_directions_exactly_once():
    calls = []

    def furthest_point(directions):
        calls.append(np.asarray(directions).copy())
        return _box_points_2d(), np.vstack([np.eye(2), -np.eye(2)])

    def make_closest_point(weights):
        make_closest_point.weights_seen = weights
        return lambda target: (target, np.array([1.0, 0.0]))

    def optimal_fn():
        return np.array([1.0, 1.0])

    furthest_point_n, closest_point_n, optimal_fn_n, affine_map = normalize_oracle_fns(
        furthest_point, make_closest_point, optimal_fn, k=2,
    )

    box_directions = np.vstack([np.eye(2), -np.eye(2)])
    points_n, normals_n = furthest_point_n(box_directions)
    points_n2, _ = furthest_point_n(box_directions)  # called again by drive() in principle -- still no re-solve
    assert len(calls) == 1  # the real furthest_point was solved exactly once
    np.testing.assert_allclose(points_n, points_n2)

    # weights handed to the closest-point factory are 1/half_range
    np.testing.assert_allclose(make_closest_point.weights_seen, affine_map.weights)
    np.testing.assert_allclose(affine_map.weights, [1 / 5.0, 1 / 3.0])

    np.testing.assert_allclose(optimal_fn_n(), affine_map.to_normalized([1.0, 1.0]))


def test_normalize_oracle_fns_closest_point_round_trips_through_real_space():
    def furthest_point(directions):
        del directions
        return _box_points_2d(), np.zeros((4, 2))

    real_calls = []

    def real_closest_point(target):
        real_calls.append(np.asarray(target).copy())
        return target * 2, np.array([3.0, 4.0])

    def make_closest_point(weights):
        del weights
        return real_closest_point

    def optimal_fn():
        return np.array([0.0, 0.0])

    _f, closest_point_n, _o, affine_map = normalize_oracle_fns(
        furthest_point, make_closest_point, optimal_fn, k=2,
    )

    target_n = np.array([0.5, -0.5])
    point_n, normal_n = closest_point_n(target_n)

    # the real closest_point was called with the denormalized target
    np.testing.assert_allclose(real_calls[0], affine_map.to_real(target_n))
    # and the result came back normalized again
    np.testing.assert_allclose(point_n, affine_map.to_normalized(real_calls[0] * 2))
    np.testing.assert_allclose(normal_n, affine_map.normal_to_normalized([3.0, 4.0]))
