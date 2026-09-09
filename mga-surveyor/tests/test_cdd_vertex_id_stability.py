"""Checks the load-bearing invariant CddMethod._refresh's closest-point
cache relies on (see cdd.py's module docstring and _refresh's own): once
cddlib assigns a generator id, that id must always refer to the SAME
geometric vertex for as long as it's alive. If id 7 pointed at vertex A
during one refresh and silently pointed at vertex B during a later one,
_refresh's id-diffing would reuse A's stale closest-point/normal for B --
a real vertex, wrongly believed unchanged -- corrupting the gap without
ever raising an error.

Drives CddMethod through a real convergence run against one of the
existing example test regions (mga_surveyor.tests.truncated_icosahedron),
recording (id -> vertex coordinates) after every refresh, then asserts no
id's recorded coordinates ever drifted beyond a tight numerical tolerance
across the whole run -- not just that the final answer was right (which a
mid-run id collision could still coincidentally produce).
"""
import numpy as np
import pytest

from mga_surveyor import CddMethod, drive
from mga_surveyor.tests.lp_region import closest_point_in_region, furthest_point
from mga_surveyor.tests.truncated_icosahedron import build_truncated_icosahedron

pytestmark = pytest.mark.slow  # real Gurobi/HiGHS LP solves each iteration, real convergence run


@pytest.mark.parametrize("backend", ["float", "gmp"])
def test_vertex_id_stability_on_truncated_icosahedron(backend):
    k = 3
    _pts_true, A_ub_true, b_ub_true, _hull_true = build_truncated_icosahedron()

    method = CddMethod(k, norm="l2", backend=backend)

    id_to_vertex: dict[int, np.ndarray] = {}
    mismatches = []

    real_refresh = method._refresh

    def snapshotting_refresh():
        real_refresh()
        if method._vertex_ids is None:
            return  # id tracking unavailable for this refresh (e.g. a fallback full rebuild) -- nothing to check
        for vid, vertex in zip(method._vertex_ids, method._outer_vertices):
            seen = id_to_vertex.get(vid)
            if seen is None:
                id_to_vertex[vid] = vertex.copy()
            elif not np.allclose(seen, vertex, atol=1e-6):
                mismatches.append((vid, seen.copy(), vertex.copy()))

    method._refresh = snapshotting_refresh  # instance attribute shadows the class method; find_trial_point/max_gap still call self._refresh()

    def furthest_point_fn(directions):
        points = np.array([furthest_point(d, A_ub_true, b_ub_true) for d in directions])
        return points, np.asarray(directions, dtype=float)

    def closest_point(target):
        return closest_point_in_region(target, A_ub_true, b_ub_true, norm=method.norm)

    def optimal_fn():
        point, _normal = closest_point_in_region(np.zeros(k), A_ub_true, b_ub_true, norm=method.norm)
        return point

    drive(method, furthest_point_fn, closest_point, optimal_fn, tol=1e-6, max_iter=None, verbose=False)

    assert len(id_to_vertex) > 5, "test observed too few tracked vertex ids to prove anything"
    assert not mismatches, (
        f"{len(mismatches)} vertex id(s) pointed to different coordinates at different times, e.g. "
        f"id={mismatches[0][0]}: {mismatches[0][1]} vs {mismatches[0][2]}"
    )
