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
    max_iter: int | None, tol: float, max_parallel: int, verbose: bool = True,
    extra_points: list[np.ndarray] | None = None, extra_halfspaces: list[Halfspace] | None = None,
    gap_scale: float = 1.0, gap_unit: str = "", incremental_bootstrap: bool = False,
) -> list[dict]:
    """Bootstrap `method` from the 2k axis directions +-e_i, `optimal_fn`'s
    point, and `extra_points`/`extra_halfspaces` if given -- all together
    in one `method.bootstrap` call, since `Method.bootstrap` must be called
    exactly once, before any `add_point`/`add_cut`/`find_trial_point` call
    -- then drive it through the main loop until the gap closes or
    `max_iter` is hit. No default -- every caller must decide; pass `None`
    explicitly for no cap (silent non-convergence is exactly what this
    forces a caller to consciously opt into, rather than getting it for
    free by omitting the argument).

    - `furthest_point(directions) -> (points, normals)`: called once, with
      all `2k` box directions, for bootstrapping only.
    - `closest_point(target) -> (point, normal)`: called each iteration
      with the worst-point search's trial point.
    - `optimal_fn() -> point`: called once, before bootstrapping.
    - `extra_points`/`extra_halfspaces`: points/cuts already known from
      elsewhere (e.g. an earlier, interrupted run), folded into the same
      bootstrap call as the box directions and `optimal_fn`'s point --
      unless `incremental_bootstrap` is set, see below.

    `incremental_bootstrap` (default `False`, preserving the single-batch
    behavior above for every existing caller): if `True`, `method.bootstrap`
    is instead called with ONLY the box directions + `optimal_fn`'s point,
    and `extra_points`/`extra_halfspaces` are folded in afterward one at a
    time (`add_point`/`add_cut`, each halfspace immediately followed by a
    forced `method.max_gap()` refresh) instead of batched into that same
    call. Exists because `CddMethod`'s single-batch from-scratch build
    (`cdd.polyhedron_from_matrix` over the WHOLE initial set at once) is
    far more prone to float mode's "Numerical inconsistency" failure than
    the same halfspaces appended one at a time -- confirmed against this
    project's real network: an alphabetically-ordered ~650-halfspace
    restart-reload crashed from-scratch but never once crashed when
    replayed incrementally (see `pypsa_ext.generate_mga_files`, the only
    caller that currently opts into this). Costs one extra `max_gap()` per
    extra halfspace, so only worth it for a method where that's cheap
    relative to a single huge from-scratch build -- true for `CddMethod`,
    almost certainly not for e.g. `BilevelMethod`, whose `max_gap()` solves
    a full MILP each time, so this is opt-in rather than the default.

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

    Returns a history list of dicts, one per POINT processed (see
    `max_parallel` below for when that can be more than one per loop
    iteration), with keys "iter", "gap", "trial_point", "new_point",
    "cut_added", "trial_point_time" (seconds spent inside this iteration's
    own worst-point search -- the one part of each iteration whose cost
    scales with the method's own accumulated state, e.g. CddMethod's
    growing outer_halfspaces, unlike closest_point's cost which is
    external to `drive` itself; attributed to the first point of a
    multi-point iteration, `0.0` for the rest of that same iteration, so
    summing it across `history` still gives the true total search-time
    cost). If `max_iter` is exhausted before `gap < tol`, logs a warning
    (the return value itself doesn't flag this -- check
    `history[-1]["gap"]` against `tol`, or the caller's own
    `method.max_gap()`, to detect it programmatically).

    `max_parallel`: no default -- every caller must decide, same reasoning
    as `max_iter`/`tol` above. When `max_parallel > 1` AND `method`
    actually supports it (checked via `hasattr` -- currently only
    `CddMethod.find_pool_trial_points`; anything else, `BilevelMethod`
    included, silently keeps the original single-target path), each
    iteration finds up to `max_parallel` of the worst, still-unconverged
    candidate trial points at once instead of just the single worst one,
    and solves all of them in one batched `closest_point` call --
    exploiting `closest_point`'s own existing batch/parallel support (see
    `pypsa_ext.make_closest_point_fn`, whose own worker-pool size this
    same `max_parallel` also controls) instead of the strictly
    one-at-a-time solves the sequential path is limited to. Pass `1` for
    today's original single-target behavior.

    No extra reconciliation is needed when several candidates land in one
    round: each one's own `(point, normal)` is an independently valid
    supporting halfspace of the TRUE region (it comes straight from that
    candidate's own `closest_point` query against the real system, never
    from the outer polytope's own current state) -- intersecting
    halfspaces is commutative/associative, so folding in several at once
    from one round (all chosen against the same, necessarily slightly
    stale, pre-round snapshot) is exactly as correct as adding them one at
    a time would have been. It just isn't necessarily as efficient: a
    later candidate in the round might not still have been the worst
    point had the earlier ones already been folded in first -- an
    exploration-strategy tradeoff, not a correctness one.
    """
    if verbose:
        logger.info("[drive] furthest-point queries: bootstrapping %d box direction(s)", 2 * method.k)
    box_points, box_normals = furthest_point(box_directions(method.k))
    box_points = np.asarray(box_points, dtype=float)
    if incremental_bootstrap:
        method.bootstrap([*box_points, np.asarray(optimal_fn(), dtype=float)], list(zip(box_normals, box_points)))
        for point in extra_points or []:
            method.add_point(point)
        for direction, point in extra_halfspaces or []:
            method.add_cut(direction, point)
            method.max_gap()  # forces this cut's own incremental append now, one at a time
    else:
        initial_points = [*box_points, np.asarray(optimal_fn(), dtype=float), *(extra_points or [])]
        initial_halfspaces = [*zip(box_normals, box_points), *(extra_halfspaces or [])]
        method.bootstrap(initial_points, initial_halfspaces)

    use_pool = max_parallel > 1 and hasattr(method, "find_pool_trial_points")
    if max_parallel > 1 and not use_pool and verbose:
        logger.info(
            "[drive] max_parallel=%d requested, but %s has no find_pool_trial_points -- "
            "falling back to the single-target path", max_parallel, type(method).__name__,
        )

    history = []
    last_gap = float("nan")
    if verbose:
        logger.info("[drive] closest-point queries: main loop starting")
    iters = itertools.count() if max_iter is None else range(max_iter)
    for it in iters:
        trial_point_start = time.perf_counter()
        if use_pool:
            trial_points, trial_gaps = method.find_pool_trial_points(max_parallel, tol)
            trial_point_time = time.perf_counter() - trial_point_start
            if len(trial_points) == 0:
                # No gap to report here -- find_pool_trial_points only
                # returns candidates still above tol, so an empty result
                # doesn't tell us the actual (necessarily below-tol) gap,
                # unlike find_trial_point's own always-computed one below.
                if verbose:
                    logger.info(
                        "[drive] converged at iter %d (find_pool_trial_points: %.3fs)",
                        it, trial_point_time,
                    )
                break

            last_gap = float(trial_gaps[0])
            if verbose:
                logger.info(
                    "[drive] iter %4d: gap=%.4f%s pool_size=%d -- solving (find_pool_trial_points: %.3fs)",
                    it + 1, last_gap * gap_scale, gap_unit, len(trial_points), trial_point_time,
                )

            new_points, normals = closest_point(trial_points)
            new_points = np.atleast_2d(np.asarray(new_points, dtype=float))
            normals = np.atleast_2d(np.asarray(normals, dtype=float))

            # No extra reconciliation needed across this round's own
            # candidates -- see this function's own docstring for why
            # each one's (point, normal) is independently valid regardless
            # of the others.
            for i in range(len(trial_points)):
                trial_point, new_point, normal = trial_points[i], new_points[i], normals[i]
                method.add_point(new_point)
                cut_added = float(gaps(new_point[None, :], trial_point[None, :], norm=method.norm)[0]) > tol
                if cut_added:
                    method.add_cut(normal, new_point)
                history.append({
                    "iter": it + 1, "gap": float(trial_gaps[i]), "trial_point": trial_point,
                    "new_point": new_point, "cut_added": cut_added,
                    "trial_point_time": trial_point_time if i == 0 else 0.0,
                })
            if verbose:
                n_cuts = sum(h["cut_added"] for h in history[-len(trial_points):])
                logger.info(
                    "[drive] iter %4d: done -- cuts_added=%d/%d", it + 1, n_cuts, len(trial_points),
                )
            continue

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
