"""A generic driving loop (`drive`): bootstraps a `Method` from the 2k axis
directions, a known optimum, and any other already-known points/cuts, all
in one `method.bootstrap` call, then drives the main loop until convergence.
Parameterized by which `Method` to use and by external callbacks that
query the true region -- the only place in the package tying a method to
where points/cuts come from; drive/bootstrap a method directly with your
own code if this doesn't fit your use case.
"""
import itertools
import logging
import time
from typing import Protocol

import numpy as np

from .common import Halfspace, gaps
from .methods.base import Method

logger = logging.getLogger(__name__)


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


def box_directions(k: int) -> np.ndarray:
    """The `2k` axis directions `+-e_i` `drive`'s bootstrap queries
    `furthest_point` with -- factored out so anything that needs to
    presolve/intercept that same batch before `drive` gets to it (e.g.
    `mga_surveyor.normalize.presolve_box_points`) builds it identically."""
    return np.vstack([np.eye(k), -np.eye(k)])


def drive(
    method: Method, furthest_point: FurthestPointFn, closest_point: ClosestPointFn, optimal_fn: OptimalPointFn,
    tol: float = 1e-6, max_iter: int | None = None, verbose: bool = True,
    extra_points: list[np.ndarray] | None = None, extra_halfspaces: list[Halfspace] | None = None,
    gap_scale: float = 1.0, gap_unit: str = "",
) -> list[dict]:
    """Bootstrap `method` from the 2k axis directions +-e_i, `optimal_fn`'s
    point, and `extra_points`/`extra_halfspaces` if given -- all together
    in one `method.bootstrap` call, since `Method.bootstrap` must be called
    exactly once, before any `add_point`/`add_cut`/`find_trial_point` call
    -- then drive it through the main loop until the gap closes or
    `max_iter` is hit. `max_iter=None` (the default) means no cap.

    - `furthest_point(directions) -> (points, normals)`: called once, with
      all `2k` box directions, for bootstrapping only.
    - `closest_point(target) -> (point, normal)`: called each iteration
      with the worst-point search's trial point.
    - `optimal_fn() -> point`: called once, before bootstrapping.
    - `extra_points`/`extra_halfspaces`: points/cuts already known from
      elsewhere (e.g. an earlier, interrupted run), folded into the same
      bootstrap call as the box directions and `optimal_fn`'s point.

    See `mga_surveyor.tests.lp_region`/`mga_surveyor.tests.synthetic_ball`
    for example implementations, or `pypsa_ext` for PyPSA-backed ones.

    Uses the closest-point-plus-dual-normal construction for the main
    loop: correct for any of the three norms, with no separate argument
    needed per norm.

    `gap_scale`/`gap_unit` (default `1.0`/`""`, i.e. no change): every
    logged gap is multiplied by `gap_scale` and suffixed with `gap_unit`
    before printing -- purely cosmetic, doesn't touch `tol`/`history`'s own
    values. Lets a caller display the gap however it originally expressed
    `tol` (e.g. as a percentage) instead of `drive`'s own internal units.

    Returns a history list of dicts, one per iteration, with keys "iter",
    "gap", "trial_point", "new_point", "cut_added", "trial_point_time" (
    seconds spent inside this iteration's own `method.find_trial_point()`
    call -- the worst-point search that decides the next point to query,
    logged since it's the one part of each iteration whose cost scales
    with the method's own accumulated state, e.g. VertexEnumMethod's
    growing outer_vertices/outer_halfspaces, unlike closest_point's cost
    which is external to `drive` itself). If `max_iter` is exhausted before
    `gap < tol`, logs a warning (the return value itself doesn't flag this
    -- check `history[-1]["gap"]` against `tol`, or the caller's own
    `method.max_gap()`, to detect it programmatically).
    """
    if verbose:
        logger.info("[drive] furthest-point queries: bootstrapping %d box direction(s)", 2 * method.k)
    box_points, box_normals = furthest_point(box_directions(method.k))
    box_points = np.asarray(box_points, dtype=float)
    initial_points = [*box_points, np.asarray(optimal_fn(), dtype=float), *(extra_points or [])]
    initial_halfspaces = [*zip(box_normals, box_points), *(extra_halfspaces or [])]
    method.bootstrap(initial_points, initial_halfspaces)

    history = []
    last_gap = float("nan")
    if verbose:
        logger.info("[drive] closest-point queries: main loop starting")
    iters = itertools.count() if max_iter is None else range(max_iter)
    for it in iters:
        trial_point_start = time.perf_counter()
        trial_point, gap = method.find_trial_point()
        trial_point_time = time.perf_counter() - trial_point_start
        last_gap = gap
        if trial_point is None or gap < tol:
            if verbose:
                logger.info(
                    "[drive] converged at iter %d: gap=%.4f%s (find_trial_point: %.3fs)",
                    it, gap * gap_scale, gap_unit, trial_point_time,
                )
            break

        new_point, normal = closest_point(trial_point)
        new_point = np.asarray(new_point, dtype=float)

        method.add_point(new_point)
        # Cut only needed if the closest point differs from the trial
        # point, i.e. the trial point itself was infeasible. Measured
        # under method.norm -- the same metric find_trial_point's own gap
        # (and hence this same tol, in the convergence check above) is
        # measured under -- not a hardcoded L2 distance, which would
        # otherwise compare a different-norm quantity against tol than
        # the one the convergence check itself uses (up to sqrt(k)x off
        # for L2 vs Linf in k dimensions).
        cut_added = float(gaps(new_point[None, :], np.asarray(trial_point)[None, :], norm=method.norm)[0]) > tol
        if cut_added:
            method.add_cut(normal, new_point)

        history.append({
            "iter": it + 1, "gap": gap, "trial_point": trial_point,
            "new_point": new_point, "cut_added": cut_added, "trial_point_time": trial_point_time,
        })
        if verbose:
            logger.info(
                "[drive] iter %4d: gap=%.4f%s cut_added=%s (find_trial_point: %.3fs)",
                it + 1, gap * gap_scale, gap_unit, cut_added, trial_point_time,
            )
    else:
        # for...else: only reached if the loop ran out of iterations
        # without ever hitting the `break` above -- i.e. max_iter was
        # exhausted while still above tol. Silent otherwise (no log line,
        # no exception), so a caller that doesn't inspect the returned
        # history could easily mistake this for genuine convergence.
        if max_iter is not None:
            logger.warning(
                "[drive] did NOT converge within max_iter=%d iterations: final gap=%.4f%s (tol=%.4f%s)",
                max_iter, last_gap * gap_scale, gap_unit, tol * gap_scale, gap_unit,
            )

    return history
