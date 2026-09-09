"""Cross-check CddMethod's incremental add_cut path (polyhedron_append_halfspaces,
folding one cut at a time into a persistent cdd.Polyhedron) against the
all-at-once mode it used to always use (a fresh cdd_vertices_of recompute
from the complete accumulated halfspace list) -- for both cdd arithmetic
backends. A mismatch here would mean the incremental path has drifted from
ground truth; agreement across many successive cuts, in both backends, is
the actual correctness evidence for wiring polyhedron_append_halfspaces
into CddMethod at all.

Two usage patterns, both real:
- test_incremental_matches_all_at_once_every_single_step: refreshes after
  EVERY add_cut, exactly how explore.py's drive() actually drives
  CddMethod in production (find_trial_point, i.e. _refresh, exactly once
  per add_cut -- never more than one pending row at a time). Run across
  several seeds and dimensions for real reliability evidence, not just a
  single lucky trial.
- test_incremental_matches_all_at_once_with_batched_appends: refreshes
  only at a few checkpoints, so several cuts accumulate in
  self._pending_rows and get folded in via one multi-row
  polyhedron_append_halfspaces call. Nothing in this codebase currently
  drives CddMethod this way, but the function accepts multi-row input and
  is documented to support it, so it needs its own coverage -- this exact
  pattern is what found a real bug (a stale WeaklyFeasibleRayCount causing
  dd_DDMain's early-exit to fire before every row in the batch had been
  processed; see mycode/cdd project notes) that every-single-step checking
  structurally cannot catch, since it never creates a multi-row batch.

The vertex-set tests above bypass find_trial_point/max_gap (and so
gurobi_projection's closest_point_on_hull) entirely -- they compare vertex
SETS, which CddMethod._vertices_with_ids() produces without ever touching
inner_vertices, so there's no need for a real solver there, only a real
cdd computation. test_closest_point_cache_reused_for_surviving_vertex_ids
and test_gap_matches_bruteforce_after_incremental_cuts below instead
target _refresh()'s id-keyed closest-point cache specifically (see
CddMethod's module docstring): the first checks the cache is actually
reused, not just correct by coincidence of always resolving; the second
cross-checks find_trial_point/max_gap's cached-aware numbers against an
uncached brute-force reference, including add_point's cache invalidation
path. Both need a real solver (gurobipy, via closest_point_on_hull).
"""
import numpy as np
import pytest

from mga_surveyor.common import gaps
from mga_surveyor.gurobi_projection import closest_point_on_hull
from mga_surveyor.methods.cdd import CddMethod, cdd_vertices_of


def _box_halfspaces(k: int, half_width: float = 10.0) -> list[tuple[np.ndarray, np.ndarray]]:
    halfspaces = []
    for axis in range(k):
        for sign in (1.0, -1.0):
            direction = np.zeros(k)
            direction[axis] = sign
            halfspaces.append((direction, direction * half_width))
    return halfspaces


def _random_oblique_cuts(k: int, n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``n`` halfspaces with random unit-ish directions and offsets chosen
    to slice corners off the box (sometimes redundant, sometimes not) --
    the same shape of scenario that found the dd_DDAppendHalfspaces
    stale-Iteration crash (see mycode/cdd project notes): a plain
    axis-aligned slab cut touches many vertices uniformly, but an oblique
    corner cut removes exactly one vertex and creates several new ones,
    which is what actually exercises ray creation during a resumed run.
    """
    rng = np.random.default_rng(seed)
    cuts = []
    for _ in range(n):
        direction = rng.uniform(-1.0, 1.0, size=k)
        point = direction / np.linalg.norm(direction) * rng.uniform(2.0, 9.9)
        cuts.append((direction, point))
    return cuts


def _vertex_set(verts: np.ndarray, decimals: int = 6) -> set[tuple[float, ...]]:
    return {tuple(np.round(v, decimals)) for v in verts}


def _assert_matches_ground_truth(method: CddMethod, backend: str, step: int) -> None:
    incremental_verts, _ids, _fresh = method._vertices_with_ids()
    all_at_once_verts = cdd_vertices_of(method.outer_halfspaces, backend=backend)
    assert _vertex_set(incremental_verts) == _vertex_set(all_at_once_verts), (
        f"backend={backend!r}: incremental and all-at-once vertex sets diverged after cut {step}"
    )


@pytest.mark.parametrize("backend", ["float", "gmp"])
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("seed", range(5))
def test_incremental_matches_all_at_once_every_single_step(backend, k, seed):
    """The real usage pattern (see module docstring): one add_cut, then
    immediately one refresh, repeated -- self._pending_rows never holds
    more than a single row when polyhedron_append_halfspaces is called.
    """
    method = CddMethod(k=k, norm="linf", backend=backend)
    method.bootstrap([np.zeros(k)], _box_halfspaces(k))

    n_cuts = 8 if backend == "gmp" else 20  # gmp is much slower per cut; still several seeds x dims for coverage
    cuts = _random_oblique_cuts(k, n=n_cuts, seed=seed)

    for i, (direction, point) in enumerate(cuts):
        method.add_cut(direction, point)
        _assert_matches_ground_truth(method, backend, i)


@pytest.mark.parametrize("backend", ["float", "gmp"])
def test_incremental_matches_all_at_once_with_batched_appends(backend):
    method = CddMethod(k=3, norm="linf", backend=backend)
    method.bootstrap([np.zeros(3)], _box_halfspaces(3))

    cuts = _random_oblique_cuts(k=3, n=50, seed=0)
    checkpoints = {2, 5, 15, 30, 49}  # varying batch sizes: 3, 3, 10, 15, 19 pending rows at each checkpoint

    for i, (direction, point) in enumerate(cuts):
        method.add_cut(direction, point)
        if i in checkpoints:
            _assert_matches_ground_truth(method, backend, i)


@pytest.mark.parametrize("backend", ["float", "gmp"])
def test_incremental_actually_used_polyhedron_append_halfspaces(backend):
    """Guards against the comparisons above passing vacuously because the
    incremental path silently fell back to full_rebuild() every time (e.g.
    if polyhedron_append_halfspaces were missing or always raised) -- that
    would still "match" all-at-once trivially, without ever exercising the
    thing this test module is actually meant to check.
    """
    method = CddMethod(k=3, norm="linf", backend=backend)
    method.bootstrap([np.zeros(3)], _box_halfspaces(3))
    method._vertices_with_ids()  # first-ever build: always a full rebuild, establishes self._poly
    poly_after_bootstrap = method._poly

    method.add_cut(np.array([0.5, 0.5, 0.5]), np.array([5.0, 5.0, 5.0]))
    method._vertices_with_ids()

    # A real incremental append mutates self._poly in place and keeps the
    # same object; a fallback full rebuild replaces it with a new one.
    assert method._poly is poly_after_bootstrap


def test_closest_point_cache_skips_solves_for_surviving_vertex_ids(monkeypatch):
    """Directly measures that _refresh only calls closest_point_on_hull for
    vertices whose generator id is new -- not just that the end result
    happens to be correct (which caching-that-secretly-never-triggers
    would also produce, same trap test_incremental_actually_used_polyhedron_
    append_halfspaces above guards against for the vertex-enumeration
    side).
    """
    import mga_surveyor.methods.cdd as cdd_module

    call_count = {"n": 0}
    real_solver = cdd_module.closest_point_on_hull

    def counting_solver(point, points, norm):
        call_count["n"] += 1
        return real_solver(point, points, norm)

    monkeypatch.setattr(cdd_module, "closest_point_on_hull", counting_solver)

    method = CddMethod(k=3, norm="linf", backend="float")
    method.bootstrap([np.array([0.0, 0.0, 0.0])], _box_halfspaces(3))
    method.find_trial_point()
    ids_before = set(method._vertex_ids)
    assert call_count["n"] == len(ids_before)  # nothing cached yet on the very first build -- must solve every vertex

    call_count["n"] = 0
    method.add_cut(np.array([1.0, 1.0, 1.0]), np.array([9.0, 9.0, 9.0]))  # cuts one corner off the box
    method.find_trial_point()
    ids_after = set(method._vertex_ids)

    n_new_ids = len(ids_after - ids_before)
    assert n_new_ids > 0  # sanity: the cut must actually have created at least one new vertex, or this proves nothing
    assert call_count["n"] == n_new_ids, (
        "expected exactly one fresh solve per newly created vertex id, and none for survivors"
    )


def test_gap_matches_bruteforce_after_incremental_cuts_and_points():
    """Cross-checks find_trial_point's cached-aware gap against an
    independent, fully uncached brute-force reference (fresh cdd_vertices_of
    plus a fresh closest_point_on_hull solve for every vertex, every step)
    after a mixed sequence of add_cut and add_point calls -- the latter
    specifically to exercise _refresh_stale_closest_points' cache
    invalidation, not just the id-survives-unchanged path the test above
    covers.
    """
    k = 3
    method = CddMethod(k=k, norm="linf", backend="float")
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

        truth_verts = cdd_vertices_of(method.outer_halfspaces, backend="float")
        truth_inner = np.array(inner_points)
        truth_closest = np.array([
            closest_point_on_hull(v, truth_inner, norm="linf")[0] for v in truth_verts
        ])
        truth_gap_arr = gaps(truth_verts, truth_closest, norm="linf")
        truth_max_gap = float(truth_gap_arr.max()) if len(truth_gap_arr) else 0.0

        assert cached_gap == pytest.approx(truth_max_gap, abs=1e-6), f"gap mismatch after cut {i}"
