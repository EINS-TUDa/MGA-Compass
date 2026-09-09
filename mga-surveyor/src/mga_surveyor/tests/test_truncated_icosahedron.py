"""pytest suite: both Methods (`CddMethod`, `BilevelMethod`) converge to
the truncated icosahedron test region (see
`truncated_icosahedron.py` for its construction) under every norm each one
supports.

Run with: pytest mga_surveyor/tests/test_truncated_icosahedron.py -v

Marked `slow` (whole module -- ~1 minute, dominated by BilevelMethod's MILP
solve): deselect with `pytest -m "not slow"`, or select just these with
`pytest -m slow`.
"""
import numpy as np
import pytest

from mga_surveyor import drive
from mga_surveyor.tests.lp_region import furthest_point, closest_point_in_region
from mga_surveyor.tests.truncated_icosahedron import (
    build_truncated_icosahedron, build_method, n_unique_facets,
)

pytestmark = pytest.mark.slow

K = 3
GAP_TOL = 1e-4  # loose relative to the ~1e-7..1e-15 final gaps actually observed
MAX_ITER = 300  # generous relative to the ~78-86 iterations actually observed


@pytest.fixture(scope="module")
def true_region():
    """`(points, A_ub, b_ub, hull)` for the truncated icosahedron, built
    once and shared read-only across every test in this module -- none of
    them mutate it."""
    return build_truncated_icosahedron()


def test_truncated_icosahedron_geometry(true_region):
    """Sanity-check the construction itself before trusting any
    convergence result against it: 60 vertices, 32 facets (12 pentagons +
    20 hexagons), matching the known truncated icosahedron."""
    _points, _A_ub, _b_ub, hull = true_region
    assert len(hull.vertices) == 60
    assert n_unique_facets(hull) == 32


@pytest.mark.parametrize("method_name,norm", [
    ("cdd", "l1"),
    ("cdd", "l2"),
    ("cdd", "linf"),
    ("bilevel", "linf"),  # the only norm BilevelMethod supports
])
def test_converges_to_truncated_icosahedron(true_region, method_name, norm):
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

    drive(method, furthest_point_fn, closest_point, optimal_fn, tol=1e-6, max_iter=MAX_ITER, verbose=False)

    assert method.max_gap() < GAP_TOL
