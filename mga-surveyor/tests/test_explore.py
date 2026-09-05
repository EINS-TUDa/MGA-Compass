import numpy as np

from mga_surveyor import drive


class _RecordingMethod:
    """Bare-minimum Method stub recording call order/arguments, for testing
    drive()'s control flow without needing a real geometry/solver."""

    def __init__(self, k: int, norm: str) -> None:
        self.k = k
        self.norm = norm
        self.calls = []

    def bootstrap(self, initial_points, initial_halfspaces) -> None:
        self.calls.append(("bootstrap", [tuple(p) for p in initial_points], list(initial_halfspaces)))

    def add_point(self, point) -> bool:
        self.calls.append(("add_point", tuple(point)))
        return True

    def add_cut(self, direction, point) -> None:
        self.calls.append(("add_cut", tuple(direction), tuple(point)))

    def find_trial_point(self):
        self.calls.append(("find_trial_point",))
        return None, 0.0  # converge immediately -- no closest_point call needed

    def max_gap(self) -> float:
        return 0.0


def _stub_furthest_point(directions):
    directions = np.asarray(directions, dtype=float)
    return directions, directions


def _stub_optimal_fn():
    return np.zeros(2)


def _stub_closest_point(_target):
    raise AssertionError("closest_point must not be called -- find_trial_point converges immediately")


def test_drive_bootstraps_from_box_directions_and_optimal_point():
    """With no extra_points/extra_halfspaces, bootstrap gets exactly the
    2k box points plus optimal_fn's point."""
    method = _RecordingMethod(k=2, norm="linf")

    drive(method, _stub_furthest_point, _stub_closest_point, _stub_optimal_fn, max_iter=None, tol=1e-6, verbose=False)

    assert [c[0] for c in method.calls] == ["bootstrap", "find_trial_point"]
    _, points, halfspaces = method.calls[0]
    assert points == [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 0.0)]
    assert len(halfspaces) == 4


def test_drive_folds_extra_points_and_halfspaces_into_the_one_bootstrap_call():
    """extra_points/extra_halfspaces are appended to the box/optimal seed
    and passed to a single method.bootstrap call -- not added afterward
    via add_point/add_cut (Method.bootstrap must be called exactly once,
    before any add_point/add_cut/find_trial_point call)."""
    method = _RecordingMethod(k=2, norm="linf")

    extra_points = [np.array([0.5, 0.5])]
    extra_halfspaces = [(np.array([1.0, 0.0]), np.array([0.5, 0.5]))]

    drive(
        method, _stub_furthest_point, _stub_closest_point, _stub_optimal_fn, max_iter=None, tol=1e-6,
        extra_points=extra_points, extra_halfspaces=extra_halfspaces, verbose=False,
    )

    assert [c[0] for c in method.calls] == ["bootstrap", "find_trial_point"]
    _, points, halfspaces = method.calls[0]
    assert points[-1] == (0.5, 0.5)
    direction, point = halfspaces[-1]
    np.testing.assert_allclose(direction, [1.0, 0.0])
    np.testing.assert_allclose(point, [0.5, 0.5])
