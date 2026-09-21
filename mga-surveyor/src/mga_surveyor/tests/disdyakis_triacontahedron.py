"""Example/test: run both Methods (`BilevelMethod`, `CddMethod`) against a
second well-known, fixed 3D test region -- the
disdyakis triacontahedron (62 vertices, 120 triangular facets), the
Catalan solid dual of the truncated icosidodecahedron and the most-faceted
of the standard named 3D polyhedra. See `truncated_icosahedron.py` for the
first (32-facet) test region this one is built the same way as.

Construction: the disdyakis triacontahedron's 62 vertices fall into three
orbits under icosahedral symmetry, at three different radii along the same
axes as a regular icosahedron's own vertices, face centers, and edge
midpoints:

- 12 vertices along the icosahedron's own vertex directions,
- 20 vertices along its 20 face-center directions,
- 30 vertices along its 30 edge-midpoint directions.

The published Catalan solid uses specific exact radii for these three
orbits; rather than risk mis-transcribing those from memory, the radii
used here (1.0 / 0.88 / 0.92) were instead found by a numerical search over
candidate ratios, keeping only combinations verified (via
`scipy.spatial.ConvexHull`) to reproduce the correct combinatorial
structure -- 62 vertices, 120 triangular facets, matching Euler's formula
(62 - 180 + 120 = 2) -- and picked to maximize the worst-case facet angle
(avoiding needle-thin triangles that make the closest-point/vertex-
enumeration solves numerically fragile).

Even so, this shape is noticeably harder-conditioned than
`truncated_icosahedron.py`'s (its 120 facets are scalene, not the near-
regular pentagons/hexagons there): with `tol=1e-6` (this package's usual
default), `CddMethod`'s float-mode Double Description solver occasionally
hits cddlib's own documented float-precision limit ("Numerical
inconsistency... use the GMP exact arithmetic") on the accumulated
halfspaces for L2 specifically. Avoided by using a looser `tol=1e-4` in
`drive()` -- still far tighter than anything visually meaningful -- rather
than by further tuning the geometry; see
`test_disdyakis_triacontahedron.py` for exactly which (method, norm)
combinations are exercised as real, passing tests versus documented as
expected-to-fail.

Scaled to fit inside `[-1, 1]^3` (a "unit cube" centered at the origin),
same convention as `truncated_icosahedron.py`.

Run with: python mga_surveyor/tests/disdyakis_triacontahedron.py [method] [norm] [n_iters]
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

# Found by numerical search (see module docstring) -- not the published
# Catalan solid's exact values, but verified to give the right combinatorics
# and a reasonable worst-case facet angle.
_R_FACE_CENTERS = 0.88
_R_EDGE_MIDPOINTS = 0.92

#: Looser than this package's usual 1e-6 default -- see module docstring
#: for why this specific shape needs it.
GAP_TOL = 1e-4


def _icosahedron_vertices():
    base = [(0, 1, PHI), (0, 1, -PHI), (0, -1, PHI), (0, -1, -PHI)]
    ico = []
    for x, y, z in base:
        ico.append((x, y, z))
        ico.append((z, x, y))
        ico.append((y, z, x))
    return np.array(ico)


def build_disdyakis_triacontahedron():
    """The 62 vertices of a disdyakis triacontahedron, scaled to fit
    exactly inside `[-1, 1]^3`, plus its `(A_ub, b_ub)` halfspace
    representation and the `ConvexHull` they were derived from (for
    reporting the true vertex/facet counts)."""
    ico = _icosahedron_vertices()
    ico_dir = ico / np.linalg.norm(ico, axis=1, keepdims=True)

    ico_hull = ConvexHull(ico)  # regular icosahedron: already simplicial, faces == simplices
    face_centers = ico[ico_hull.simplices].mean(axis=1)
    assert face_centers.shape == (20, 3), f"expected 20 icosahedron faces, got {face_centers.shape}"
    face_dir = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)

    dists = np.linalg.norm(ico[:, None, :] - ico[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    edge_len = dists.min()
    edges = [(i, j) for i in range(12) for j in range(i + 1, 12) if abs(dists[i, j] - edge_len) < 1e-6]
    assert len(edges) == 30, f"expected 30 icosahedron edges, got {len(edges)}"
    edge_mids = np.array([(ico[i] + ico[j]) / 2 for i, j in edges])
    edge_dir = edge_mids / np.linalg.norm(edge_mids, axis=1, keepdims=True)

    pts = np.vstack([ico_dir * 1.0, face_dir * _R_FACE_CENTERS, edge_dir * _R_EDGE_MIDPOINTS])
    assert pts.shape == (62, 3), f"expected 62 disdyakis triacontahedron vertices, got {pts.shape}"

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
        return BilevelMethod(k, norm="linf", time_limit=None)  # unbounded/exact; only norm this method supports
    if name == "cdd":
        return CddMethod(k, norm=norm, executor=executor)
    raise ValueError(f"unknown method {name!r}")


def main():
    method_name = sys.argv[1] if len(sys.argv) > 1 else "cdd"
    norm = sys.argv[2] if len(sys.argv) > 2 else "l2"
    n_iters = int(sys.argv[3]) if len(sys.argv) > 3 else None

    k = 3
    pts_true, A_ub_true, b_ub_true, hull_true = build_disdyakis_triacontahedron()
    n_true_facets = n_unique_facets(hull_true)
    print(f"true region: disdyakis triacontahedron, {len(hull_true.vertices)} vertices, "
          f"{n_true_facets} facets (expect 62, 120)")

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
    history = drive(
        method, furthest_point_fn, closest_point, optimal_fn,
        tol=GAP_TOL, max_iter=n_iters, max_parallel=1, verbose=True,
    )
    elapsed = time.time() - t0

    final_gap = method.max_gap()
    n_inner = len(method.inner_vertices)
    print(f"\nfinished {len(history)} iterations in {elapsed:.2f}s, final max_gap={final_gap:.3e}")
    print(f"discovered {n_inner} inner (feasible) points (true region has {len(hull_true.vertices)} vertices)")


if __name__ == "__main__":
    main()
