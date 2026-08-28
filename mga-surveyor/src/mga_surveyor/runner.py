"""A generic driving loop (`run`): bootstraps a `Method` from the 2k axis
directions plus a known optimum, then runs the main loop until
convergence. Parameterized by which `Method` to use and by external
callbacks that query the true region -- the only place in the package
tying a method to where points/cuts come from; drive/bootstrap a method
directly with your own code if this doesn't fit your use case.
"""
import itertools
from typing import Protocol

import numpy as np

from .methods.base import Method


class FurthestPointFn(Protocol):
    """Batched support-function query: given `(n, k)` directions, the
    points in the true region maximizing each `direction @ x`, together
    with each point's own separating-hyperplane normal (for a
    furthest-point query, always just the query direction itself)."""
    def __call__(self, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class ClosestPointFn(Protocol):
    """The closest point in the true region to `target` under the method's
    norm, together with a valid separating-hyperplane normal there."""
    def __call__(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class OptimalPointFn(Protocol):
    """Zero-argument query: the true region's own known optimum -- a free
    extra point to seed the inner approximation with."""
    def __call__(self) -> np.ndarray: ...


def run(
    method: Method, furthest_point: FurthestPointFn, closest_point: ClosestPointFn, optimal_fn: OptimalPointFn,
    tol: float = 1e-6, max_iter: int | None = None, verbose: bool = True,
) -> list[dict]:
    """Bootstrap `method` from the 2k axis directions +-e_i plus
    `optimal_fn`'s point, then drive it through the main loop until the gap
    closes or `max_iter` is hit. `max_iter=None` (the default) means no cap.

    - `furthest_point(directions) -> (points, normals)`: called once, with
      all `2k` box directions, for bootstrapping only.
    - `closest_point(target) -> (point, normal)`: called each iteration
      with the worst-point search's trial point.
    - `optimal_fn() -> point`: called once, before bootstrapping.

    See `mga_surveyor.tests.lp_region`/`mga_surveyor.tests.synthetic_ball`
    for example implementations, or `pypsa_ext` for PyPSA-backed ones.

    Uses the closest-point-plus-dual-normal construction for the main
    loop: correct for any of the three norms, with no separate argument
    needed per norm.

    Returns a history list of dicts, one per iteration, with keys "iter",
    "gap", "trial_point", "new_point", "cut_added".
    """
    box_directions = np.vstack([np.eye(method.k), -np.eye(method.k)])
    box_points, box_normals = furthest_point(box_directions)
    box_points = np.asarray(box_points, dtype=float)
    initial_points = [*box_points, np.asarray(optimal_fn(), dtype=float)]
    initial_halfspaces = list(zip(box_normals, box_points))
    method.bootstrap(initial_points, initial_halfspaces)

    history = []
    iters = itertools.count() if max_iter is None else range(max_iter)
    for it in iters:
        trial_point, gap = method.find_trial_point()
        if trial_point is None or gap < tol:
            if verbose:
                print(f"[run] converged at iter {it}: gap={gap:.3e}")
            break

        new_point, normal = closest_point(trial_point)
        new_point = np.asarray(new_point, dtype=float)

        method.add_point(new_point)
        # Cut only needed if the closest point differs from the trial
        # point, i.e. the trial point itself was infeasible.
        cut_added = np.linalg.norm(new_point - np.asarray(trial_point)) > tol
        if cut_added:
            method.add_cut(normal, new_point)

        history.append({
            "iter": it + 1, "gap": gap, "trial_point": trial_point,
            "new_point": new_point, "cut_added": cut_added,
        })
        if verbose:
            print(f"[run] iter {it+1:4d}: gap={gap:.6f} cut_added={cut_added}")

    return history
