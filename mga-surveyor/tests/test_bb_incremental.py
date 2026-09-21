"""Cross-checks BBMethod's incremental add_cut path (folding one dual
point at a time into a persistent incremental scipy ConvexHull) against a
fresh all-at-once bootstrap from the complete accumulated halfspace list --
and, separately, its find_trial_point/max_gap gap computation against an
uncached brute-force reference. Mirrors tests/test_cdd_incremental.py's own
structure for CddMethod; see that module's docstring for the general
rationale (agreement between incremental and all-at-once is the actual
correctness evidence, not just "it ran without crashing").
"""
import numpy as np
import pytest

from mga_surveyor.common import gaps
from mga_surveyor.gurobi_projection import closest_point_on_hull
from mga_surveyor.methods.bb import BBMethod


def _box_halfspaces(k: int, half_width: float = 10.0) -> list[tuple[np.ndarray, np.ndarray]]:
    halfspaces = []
    for axis in range(k):
        for sign in (1.0, -1.0):
            direction = np.zeros(k)
            direction[axis] = sign
            halfspaces.append((direction, direction * half_width))
    return halfspaces


def _random_oblique_cuts(k: int, n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Same construction as test_cdd_incremental's own -- cuts guaranteed
    to keep the origin (this file's bootstrap point, hence BBMethod's
    dual-construction origin too) strictly interior, since every cut's
    offset point is at radius in [2.0, 9.9), strictly inside the box's own
    half_width=10 boundary.
    """
    rng = np.random.default_rng(seed)
    cuts = []
    for _ in range(n):
        direction = rng.uniform(-1.0, 1.0, size=k)
        point = direction / np.linalg.norm(direction) * rng.uniform(2.0, 9.9)
        cuts.append((direction, point))
    return cuts


def _vertex_set(verts: np.ndarray, decimals: int = 5) -> set[tuple[float, ...]]:
    return {tuple(np.round(v, decimals)) for v in verts}


def _all_at_once_vertices(k: int, halfspaces: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Ground truth: a fresh BBMethod, bootstrapped directly with the
    complete halfspace list in one call -- no incremental add_cut involved.
    """
    method = BBMethod(k=k, norm="linf")
    method.bootstrap([np.zeros(k)], halfspaces)
    return method._compute_outer_vertices()


def _assert_matches_ground_truth(method: BBMethod, k: int, step: int) -> None:
    incremental_verts = method._compute_outer_vertices()
    truth_verts = _all_at_once_vertices(k, method.outer_halfspaces)
    assert _vertex_set(incremental_verts) == _vertex_set(truth_verts), (
        f"k={k}: incremental and all-at-once vertex sets diverged after cut {step}"
    )


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("seed", range(5))
def test_incremental_matches_all_at_once_every_single_step(k, seed):
    method = BBMethod(k=k, norm="linf")
    method.bootstrap([np.zeros(k)], _box_halfspaces(k))

    cuts = _random_oblique_cuts(k, n=20, seed=seed)
    for i, (direction, point) in enumerate(cuts):
        method.add_cut(direction, point)
        _assert_matches_ground_truth(method, k, i)


def test_gap_matches_bruteforce_after_incremental_cuts_and_points():
    """Cross-checks find_trial_point's gap against an independent,
    uncached brute-force reference (fresh all-at-once vertex recompute
    plus a fresh closest_point_on_hull solve for every vertex, every step)
    after a mixed sequence of add_cut and add_point calls -- the latter
    specifically exercising _refresh_stale_closest_points.

    Uses norm='l1', not 'linf' -- `closest_point_on_hull` caches its Gurobi
    model per-norm in thread-local state keyed only by `norm` (not `norm`
    AND point count), persisting across tests within the same process/main
    thread. test_cdd_incremental.py's own bruteforce gap test already
    exercises 'linf' directly on the main thread the same way this test
    would, and (being in a file that sorts before this one) always runs
    first, leaving that cache primed with more points than this test's own
    from-scratch growing sequence expects. A different norm sidesteps that
    collision entirely rather than depending on test file ordering.
    """
    k = 3
    method = BBMethod(k=k, norm="l1")
    initial_points = [np.array([0.0, 0.0, 0.0])]
    method.bootstrap(initial_points, _box_halfspaces(k))

    cuts = _random_oblique_cuts(k, n=10, seed=1)
    extra_points = [np.array([1.0, 2.0, -1.0]), np.array([-2.0, 0.5, 3.0])]

    inner_points = list(initial_points)
    for i, (direction, point) in enumerate(cuts):
        method.add_cut(direction, point)
        if i == 3:
            method.add_point(extra_points[0])
            inner_points.append(extra_points[0])
        if i == 7:
            method.add_point(extra_points[1])
            inner_points.append(extra_points[1])

        _cached_trial_point, cached_gap = method.find_trial_point()

        truth_verts = _all_at_once_vertices(k, method.outer_halfspaces)
        truth_inner = np.array(inner_points)
        truth_closest = np.array([
            closest_point_on_hull(v, truth_inner, norm="l1")[0] for v in truth_verts
        ])
        truth_gap_arr = gaps(truth_verts, truth_closest, norm="l1")
        truth_max_gap = float(truth_gap_arr.max()) if len(truth_gap_arr) else 0.0

        assert cached_gap == pytest.approx(truth_max_gap, abs=1e-6), f"gap mismatch after cut {i}"


def test_add_point_reuses_cache_for_unaffected_vertices(monkeypatch):
    """Directly measures that add_point's cache-invalidation path
    (_refresh_stale_closest_points) only re-solves vertices whose cached
    closest point/normal a new inner point actually invalidates -- not
    just that the end result happens to be correct.
    """
    import mga_surveyor.methods.bb as bb_module

    call_count = {"n": 0}
    real_solver = bb_module.closest_point_on_hull

    def counting_solver(point, points, norm):
        call_count["n"] += 1
        return real_solver(point, points, norm)

    monkeypatch.setattr(bb_module, "closest_point_on_hull", counting_solver)

    k = 3
    method = BBMethod(k=k, norm="linf")
    method.bootstrap([np.array([0.0, 0.0, 0.0])], _box_halfspaces(k))
    method.find_trial_point()
    n_outer = len(method._outer_vertices)
    assert call_count["n"] == n_outer  # nothing cached yet -- must solve every vertex

    call_count["n"] = 0
    # A point far outside the box only invalidates vertices whose cached
    # separating hyperplane it actually violates -- not necessarily all of them.
    method.add_point(np.array([0.5, 0.5, 9.9]))
    method.find_trial_point()
    assert 0 < call_count["n"] <= n_outer


def test_add_cut_reuses_cache_for_surviving_vertex_hashes(monkeypatch):
    """Directly measures that add_cut's own refresh only solves
    closest_point_on_hull for vertices whose coordinate hash is new since
    the last refresh -- not just that the end result happens to be
    correct (the same trap test_add_point_reuses_cache_for_unaffected_
    vertices above guards against for add_point). Mirrors test_cdd_
    incremental.py's own test_closest_point_cache_skips_solves_for_
    surviving_vertex_ids, but for BBMethod's hash-based identity
    instead of cdd's real generator ids.
    """
    import mga_surveyor.methods.bb as bb_module

    call_count = {"n": 0}
    real_solver = bb_module.closest_point_on_hull

    def counting_solver(point, points, norm):
        call_count["n"] += 1
        return real_solver(point, points, norm)

    monkeypatch.setattr(bb_module, "closest_point_on_hull", counting_solver)

    k = 3
    method = BBMethod(k=k, norm="linf")
    method.bootstrap([np.array([0.0, 0.0, 0.0])], _box_halfspaces(k))
    method.find_trial_point()
    hashes_before = set(method._vertex_hashes)
    assert call_count["n"] == len(hashes_before)  # nothing cached yet on the very first build -- must solve every vertex

    call_count["n"] = 0
    method.add_cut(np.array([1.0, 1.0, 1.0]), np.array([9.0, 9.0, 9.0]))  # cuts one corner off the box
    method.find_trial_point()
    hashes_after = set(method._vertex_hashes)

    n_new_hashes = len(hashes_after - hashes_before)
    assert n_new_hashes > 0  # sanity: the cut must actually have created at least one new vertex, or this proves nothing
    assert call_count["n"] == n_new_hashes, (
        "expected exactly one fresh solve per newly created vertex hash, and none for survivors"
    )
