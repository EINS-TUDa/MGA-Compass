"""Example/test: run both Methods (`BilevelMethod`, `CddMethod`) against a
well-known, fixed 3D test region -- the truncated
icosahedron (60 vertices, 32 facets: 12 pentagons + 20 hexagons) -- instead
of `synthetic_ball.py`'s random perturbed-ball region, so convergence can
be checked against an exact, independently-known vertex/facet count rather
than just a shrinking gap.

Construction: truncate a regular icosahedron (12 vertices, all cyclic
permutations of `(0, +-1, +-phi)`) at exactly 1/3 of each of its 30 edges
-- the standard ratio for this particular Archimedean solid -- giving 60
points; `scipy.spatial.ConvexHull` derives the halfspace `(A_ub, b_ub)`
representation from those points exactly the same way `synthetic_ball.py`
does for its own region. Scaled to fit inside `[-1, 1]^3` (a "unit cube"
centered at the origin) -- the raw construction is already symmetric about
the origin with bounding box `[-phi, phi]^3`, so a single uniform scale
suffices.

Run with: python mga_surveyor/tests/truncated_icosahedron.py [method] [norm] [n_iters]
  method in {bilevel, cdd}, default cdd
  norm in {l1, l2, linf}, default l2 (bilevel forces linf regardless)
  n_iters default None (run to convergence, see `explore.drive`)
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `import mga_surveyor`

logging.basicConfig(level=logging.INFO, format="%(message)s")  # explore.drive's per-iteration logging

from mga_surveyor import (
    BilevelMethod, CddMethod,
    drive,
)
from mga_surveyor.tests.lp_region import furthest_point, closest_point_in_region

PHI = (1 + np.sqrt(5)) / 2


def build_truncated_icosahedron():
    """The 60 vertices of a truncated icosahedron, scaled to fit exactly
    inside `[-1, 1]^3`, plus its `(A_ub, b_ub)` halfspace representation
    and the `ConvexHull` they were derived from (for reporting the true
    vertex/facet counts)."""
    base = [(0, 1, PHI), (0, 1, -PHI), (0, -1, PHI), (0, -1, -PHI)]
    ico = []
    for x, y, z in base:
        ico.append((x, y, z))
        ico.append((z, x, y))
        ico.append((y, z, x))
    ico = np.array(ico)

    dists = np.linalg.norm(ico[:, None, :] - ico[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    edge_len = dists.min()
    edges = [(i, j) for i in range(12) for j in range(i + 1, 12) if abs(dists[i, j] - edge_len) < 1e-6]
    assert len(edges) == 30, f"expected 30 icosahedron edges, got {len(edges)}"

    t = 1 / 3
    pts = []
    for i, j in edges:
        pts.append(ico[i] + t * (ico[j] - ico[i]))
        pts.append(ico[j] + t * (ico[i] - ico[j]))
    pts = np.array(pts)
    assert pts.shape == (60, 3), f"expected 60 truncated vertices, got {pts.shape}"

    pts = pts / np.abs(pts).max()  # fit inside [-1, 1]^3

    hull = ConvexHull(pts)
    A_ub_true = hull.equations[:, :-1]
    b_ub_true = -hull.equations[:, -1]
    return pts, A_ub_true, b_ub_true, hull


def n_unique_facets(hull, decimals=6):
    """Number of actual facets (planes), deduping the triangulated
    simplices `scipy.spatial.ConvexHull` always splits coplanar facets
    into."""
    return len(np.unique(np.round(hull.equations, decimals), axis=0))


def build_method(name, k, norm, executor=None):
    if name == "bilevel":
        return BilevelMethod(k, norm="linf")  # only norm this method supports
    if name == "cdd":
        return CddMethod(k, norm=norm, executor=executor)
    raise ValueError(f"unknown method {name!r}")


def main():
    method_name = sys.argv[1] if len(sys.argv) > 1 else "cdd"
    norm = sys.argv[2] if len(sys.argv) > 2 else "l2"
    n_iters = int(sys.argv[3]) if len(sys.argv) > 3 else None

    k = 3
    pts_true, A_ub_true, b_ub_true, hull_true = build_truncated_icosahedron()
    n_true_facets = n_unique_facets(hull_true)
    print(f"true region: truncated icosahedron, {len(hull_true.vertices)} vertices, "
          f"{n_true_facets} facets (expect 60, 32)")

    method = build_method(method_name, k, norm)
    print(f"method={method_name} norm={method.norm}")

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
        point, _normal = closest_point_in_region(np.zeros(k), A_ub_true, b_ub_true, norm=method.norm)
        return point

    t0 = time.time()
    history = drive(method, furthest_point_fn, closest_point, optimal_fn, tol=1e-6, max_iter=n_iters, verbose=True)
    elapsed = time.time() - t0

    final_gap = method.max_gap()
    n_inner = len(method.inner_vertices)
    print(f"\nfinished {len(history)} iterations in {elapsed:.2f}s, final max_gap={final_gap:.3e}")
    print(f"discovered {n_inner} inner (feasible) points (true region has {len(hull_true.vertices)} vertices)")


if __name__ == "__main__":
    main()
