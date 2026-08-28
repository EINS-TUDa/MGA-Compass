"""Example: run any of the three Methods against a synthetic test
region (points sampled from a perturbed ball, as used throughout this
project's validation), via mga_surveyor's generic runner.

This is the only place in this example that knows anything about the test
region -- the library itself (mga_surveyor/) never does. Swap the
`closest_point` closure in `main()` for a callback into a real
system model to use the same methods/runner on a real problem.

Run with: python mga_surveyor/tests/synthetic_ball.py [method] [norm] [n_iters]
  method in {vertex_enum, bilevel, cdd}, default vertex_enum
  norm in {l1, l2, linf}, default l2 (bilevel forces linf regardless)
  n_iters default 50
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `import mga_surveyor`

from mga_surveyor import (
    VertexEnumMethod, BilevelMethod, CddMethod,
    run,
)
from mga_surveyor.tests.lp_region import furthest_point, closest_point_in_region


def sample_ball_with_perturbation(n, k, radius=1.0, perturbation=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    directions = rng.normal(size=(n, k))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * rng.random(n) ** (1.0 / k)
    points = directions * radii[:, None]
    if perturbation > 0:
        points += rng.normal(scale=perturbation, size=(n, k))
    return points


def build_test_region(n=400, k=6, seed=0):
    rng = np.random.default_rng(seed)
    points = sample_ball_with_perturbation(n, k, radius=1.0, perturbation=0.05, rng=rng)
    hull = ConvexHull(points)
    A_ub_true = hull.equations[:, :-1]
    b_ub_true = -hull.equations[:, -1]
    return A_ub_true, b_ub_true, hull


def build_method(name, k, norm, executor=None):
    if name == "vertex_enum":
        return VertexEnumMethod(k, norm=norm, executor=executor)
    if name == "bilevel":
        return BilevelMethod(k, norm="linf")  # only norm this method supports
    if name == "cdd":
        return CddMethod(k, norm=norm, executor=executor)
    raise ValueError(f"unknown method {name!r}")


def main():
    method_name = sys.argv[1] if len(sys.argv) > 1 else "vertex_enum"
    norm = sys.argv[2] if len(sys.argv) > 2 else "l2"
    n_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    n, k = 400, 6
    A_ub_true, b_ub_true, hull = build_test_region(n=n, k=k, seed=0)
    print(f"test region: n={n} points, k={k} dims, {len(hull.vertices)} true hull vertices")

    method = build_method(method_name, k, norm)
    print(f"method={method_name} norm={method.norm}")

    def furthest_point_fn(directions):
        # furthest_point is single-direction only -- loop it into the
        # batched FurthestPointFn contract run() expects. normal is just
        # the query direction itself, echoed back.
        points = np.array([furthest_point(d, A_ub_true, b_ub_true) for d in directions])
        return points, np.asarray(directions, dtype=float)

    def closest_point(target):
        return closest_point_in_region(target, A_ub_true, b_ub_true, norm=method.norm)

    def optimal_fn():
        # No real "cost-optimal" point for this synthetic region -- the
        # origin, projected onto the region, stands in as a known feasible
        # point to seed the inner approximation with.
        point, _normal = closest_point_in_region(np.zeros(k), A_ub_true, b_ub_true, norm=method.norm)
        return point

    t0 = time.time()
    history = run(method, furthest_point_fn, closest_point, optimal_fn, tol=1e-3, max_iter=n_iters, verbose=True)
    elapsed = time.time() - t0

    print(f"\nfinished {len(history)} iterations in {elapsed:.2f}s, "
          f"final gap={history[-1]['gap'] if history else float('nan'):.4e}, "
          f"final max_gap={method.max_gap():.4e}")


if __name__ == "__main__":
    main()
