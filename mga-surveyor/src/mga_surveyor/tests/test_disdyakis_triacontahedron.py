"""pytest suite: `VertexEnumMethod` and `CddMethod` converge to the
disdyakis triacontahedron test region (see `disdyakis_triacontahedron.py`
for its construction) under the norms each one handles cleanly on this
particular shape.

`BilevelMethod` is deliberately not exercised here -- its MILP
reformulation is slow enough on a 120-facet region that it isn't a good
fit for a routine test run; use `disdyakis_triacontahedron.py`'s CLI
(`python ... bilevel linf`) directly if you want to check it by hand.

`CddMethod` + `l2` is marked `xfail`, not omitted: it reproducibly hits
cddlib's own documented float-mode precision limit on this shape's
accumulated halfspaces (see `disdyakis_triacontahedron.py`'s module
docstring) -- a real, disclosed limitation of the float-mode Double
Description solver, not a bug in this test.

Run with: pytest mga_surveyor/tests/test_disdyakis_triacontahedron.py -v

Marked `slow` (whole module -- ~1 minute across 7 method/norm
combinations): deselect with `pytest -m "not slow"`, or select just these
with `pytest -m slow`.
"""
import numpy as np
import pytest

from mga_surveyor import drive
from mga_surveyor.tests.lp_region import furthest_point, closest_point_in_region
from mga_surveyor.tests.disdyakis_triacontahedron import (
    build_disdyakis_triacontahedron, build_method, n_unique_facets, GAP_TOL,
)

pytestmark = pytest.mark.slow

K = 3
MAX_ITER = 500  # generous relative to the ~180-220 iterations actually observed at GAP_TOL


@pytest.fixture(scope="module")
def true_region():
    """`(points, A_ub, b_ub, hull)` for the disdyakis triacontahedron,
    built once and shared read-only across every test in this module."""
    return build_disdyakis_triacontahedron()


def test_disdyakis_triacontahedron_geometry(true_region):
    """Sanity-check the construction itself: 62 vertices, 120 facets,
    matching the known disdyakis triacontahedron."""
    _points, _A_ub, _b_ub, hull = true_region
    assert len(hull.vertices) == 62
    assert n_unique_facets(hull) == 120


@pytest.mark.parametrize("method_name,norm", [
    ("vertex_enum", "l1"),
    ("vertex_enum", "l2"),
    ("vertex_enum", "linf"),
    ("cdd", "l1"),
    pytest.param("cdd", "l2", marks=pytest.mark.xfail(
        reason="cddlib float-mode Double Description hits its own documented "
               "numerical-inconsistency limit on this shape's accumulated "
               "halfspaces; see disdyakis_triacontahedron.py's module docstring.",
        strict=False,
    )),
    ("cdd", "linf"),
])
def test_converges_to_disdyakis_triacontahedron(true_region, method_name, norm):
    _points, A_ub_true, b_ub_true, _hull = true_region
    method = build_method(method_name, K, norm)

    def furthest_point_fn(directions):
        # furthest_point is single-direction only -- loop it into the
        # batched FurthestPointFn contract drive() expects. normal is just
        # the query direction itself, echoed back.
        points = np.array([furthest_point(d, A_ub_true, b_ub_true) for d in directions])
        return points, np.asarray(directions, dtype=float)

    def closest_point(target):
        return closest_point_in_region(target, A_ub_true, b_ub_true, norm=method.norm)

    def optimal_fn():
        # No real "cost-optimal" point for this shape -- the origin,
        # projected onto the region, stands in as a known feasible point.
        point, _normal = closest_point_in_region(np.zeros(K), A_ub_true, b_ub_true, norm=method.norm)
        return point

    drive(method, furthest_point_fn, closest_point, optimal_fn, tol=GAP_TOL, max_iter=MAX_ITER, verbose=False)

    assert method.max_gap() < GAP_TOL
