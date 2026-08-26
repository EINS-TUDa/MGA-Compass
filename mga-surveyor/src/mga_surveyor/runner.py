"""A generic driving loop (`run`) plus the standard bootstrap procedure
(`bootstrap_box`) that seeds a `Method` before that loop starts, each
parameterized by which `Method` to use and by an external callback that
queries the true region. This is the only place in the package that ties a
method to "where points and cuts come from" -- none of the `Method`
implementations (`methods/base.py`, `methods/vertex_enum.py`,
`methods/bilevel.py`, `methods/cdd.py`) depend on it, so you can also
drive/bootstrap a method directly with your own code if these don't fit
your use case.
"""
import itertools
from typing import Protocol

import numpy as np

from .methods.base import Method


class FurthestPointFn(Protocol):
    """Contract `bootstrap_box` needs: a support-function query returning
    the point in the true region maximizing `direction @ x`."""
    def __call__(self, direction: np.ndarray) -> np.ndarray: ...


class ClosestFeasiblePointFn(Protocol):
    """Contract `run` needs: the closest point in the true region to
    `target` under the method's norm, together with a valid
    separating-hyperplane normal there."""
    def __call__(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def bootstrap_box(method: Method, furthest_point: FurthestPointFn, k: int) -> None:
    """Bootstrap `method`'s inner/outer approximation from the 2k axis
    directions +-e_i -- the box-direction seeding every driver script in
    this repo used to duplicate individually, factored out here since it's
    identical regardless of which `Method` or region is used.

    `furthest_point(direction) -> point` must return the point in the TRUE
    region maximizing `direction @ x` (a support-function query) --
    `mga_surveyor.tests.lp_region.furthest_point` against an explicit LP
    region is the obvious implementation for testing; a real system model would
    answer this with its own solver instead. Note this is a DIFFERENT
    oracle contract than `run`'s `closest_feasible_point`: bootstrapping
    only needs a furthest-point-in-a-given-direction query (the box
    directions are already known upfront), while the main loop needs a
    closest-point-to-a-given-target query instead (see `run`'s docstring
    for why the direction-based query isn't used there too).

    Mutates `method` in place via `method.bootstrap(...)`; returns `None`.
    """
    box_directions = np.vstack([np.eye(k), -np.eye(k)])
    initial_points, initial_halfspaces = [], []
    for direction in box_directions:
        point = np.asarray(furthest_point(direction), dtype=float)
        initial_points.append(point)
        initial_halfspaces.append((direction, point))
    method.bootstrap(initial_points, initial_halfspaces)


def run(
    method: Method, closest_feasible_point: ClosestFeasiblePointFn,
    tol: float = 1e-6, max_iter: int | None = None, verbose: bool = True,
) -> list[dict]:
    """Drive `method` through the main loop: find the worst point, query
    the true region for the closest feasible point to it, and update the
    method's state accordingly, until the gap closes or `max_iter` is hit.
    `max_iter=None` (the default) means no cap -- loop until convergence.

    This uses the exact closest-point-plus-dual-normal construction: any
    of the three worst-point methods provably returns the same trial point
    as the original bilevel-MILP formulation, and querying the true
    region's closest point (with the cut normal read off the dual of that
    projection) is provably correct for any of the three norms, with no
    separate argument needed per norm.

    An earlier, direction-based variant (querying the true region for the
    furthest point in a cut direction, rather than the closest point to
    the trial point) used to live here too. It's provably equivalent to
    this one for L2 (L2's self-duality), but only empirically, not
    provably, equivalent for L1/Linf -- removed rather than kept as an
    unproven alternative; see project notes.

    `method` must already be bootstrapped (see `Method.bootstrap`).
    `closest_feasible_point(target) -> (point, normal)` is called with the
    trial point from the worst-point search and must return the closest
    point in the TRUE region to `target` (under `method.norm`) together
    with a valid separating-hyperplane normal there --
    `mga_surveyor.tests.lp_region.closest_point_in_region` against an
    explicit LP region is the obvious implementation for testing; a real
    system model would answer this with its own solver instead. This is
    the only place this function touches anything specific to a
    particular system model or test example.

    Returns a history list of dicts, one per iteration, with keys "iter",
    "gap", "trial_point", "new_point", "cut_added".
    """
    history = []
    iters = itertools.count() if max_iter is None else range(max_iter)
    for it in iters:
        trial_point, gap = method.find_trial_point()
        if trial_point is None or gap < tol:
            if verbose:
                print(f"[run] converged at iter {it}: gap={gap:.3e}")
            break

        new_point, normal = closest_feasible_point(trial_point)
        new_point = np.asarray(new_point, dtype=float)

        method.add_point(new_point)
        # A cut is only needed if the closest feasible point differs from
        # the trial point -- i.e. the trial point itself was infeasible. If
        # they match, the trial point was confirmed feasible and O is left
        # as-is.
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
