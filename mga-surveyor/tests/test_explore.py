import numpy as np
import pytest

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


class _RecordingPoolMethod(_RecordingMethod):
    """Extends _RecordingMethod with find_pool_trial_points, for testing
    drive()'s batched-path control flow -- pool_rounds is consumed one
    round per call, each a (points, gaps) pair; an empty round signals
    convergence, same as find_trial_point returning (None, gap)."""

    def __init__(self, k: int, norm: str, pool_rounds: list) -> None:
        super().__init__(k, norm)
        self._pool_rounds = list(pool_rounds)

    def find_pool_trial_points(self, pool_num, tol, min_angle_deg):
        self.calls.append(("find_pool_trial_points", pool_num, tol, min_angle_deg))
        return self._pool_rounds.pop(0)


def _stub_batch_closest_point(targets):
    """Shape-preserving stand-in for a real ClosestPointFn's batch path:
    the "closest point" is always the target nudged by 0.1 along every
    axis (so it differs from the target -> cut_added True for every
    result), with the target itself echoed back as the normal."""
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    return targets + 0.1, targets


def test_drive_uses_find_pool_trial_points_when_method_supports_it():
    round_1 = (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.5, 0.4]))
    round_2 = (np.empty((0, 2)), np.empty(0))  # converged: no more candidates
    method = _RecordingPoolMethod(k=2, norm="linf", pool_rounds=[round_1, round_2])

    history = drive(
        method, _stub_furthest_point, _stub_batch_closest_point, _stub_optimal_fn,
        max_iter=5, tol=1e-6, verbose=False, pool_num=2, min_angle_deg=10.0,
    )

    call_names = [c[0] for c in method.calls]
    pool_calls = [c for c in method.calls if c[0] == "find_pool_trial_points"]
    assert call_names.count("find_pool_trial_points") == 2  # one per round, including the converging one
    assert pool_calls == [("find_pool_trial_points", 2, 1e-6, 10.0)] * 2
    # find_trial_point (the single-target path) must never be called once
    # the batched path is in use.
    assert "find_trial_point" not in call_names


def test_drive_folds_every_point_of_a_batched_round_into_history_and_the_method():
    round_1 = (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.5, 0.4]))
    round_2 = (np.empty((0, 2)), np.empty(0))
    method = _RecordingPoolMethod(k=2, norm="linf", pool_rounds=[round_1, round_2])

    history = drive(
        method, _stub_furthest_point, _stub_batch_closest_point, _stub_optimal_fn,
        max_iter=5, tol=1e-6, verbose=False, pool_num=2, min_angle_deg=10.0,
    )

    assert len(history) == 2  # one dict per point in the one productive round
    assert history[0]["iter"] == history[1]["iter"] == 1  # both from the same round
    assert all(h["cut_added"] for h in history)  # _stub_batch_closest_point always disagrees with its target
    assert history[0]["gap"] == pytest.approx(0.5)
    assert history[1]["gap"] == pytest.approx(0.4)
    # trial_point_time is attributed to the round's first point only, so
    # summing it across history still gives the round's one real cost.
    assert history[1]["trial_point_time"] == 0.0

    add_point_calls = [c for c in method.calls if c[0] == "add_point"]
    add_cut_calls = [c for c in method.calls if c[0] == "add_cut"]
    assert len(add_point_calls) == 2
    assert len(add_cut_calls) == 2


def test_drive_falls_back_to_find_trial_point_when_method_lacks_pool_support():
    """pool_num > 1 against a Method with no find_pool_trial_points (e.g.
    BilevelMethod) must silently keep the original single-target path,
    not raise AttributeError."""
    method = _RecordingMethod(k=2, norm="linf")

    drive(
        method, _stub_furthest_point, _stub_closest_point, _stub_optimal_fn,
        max_iter=None, tol=1e-6, verbose=False, pool_num=4,
    )

    assert [c[0] for c in method.calls] == ["bootstrap", "find_trial_point"]
