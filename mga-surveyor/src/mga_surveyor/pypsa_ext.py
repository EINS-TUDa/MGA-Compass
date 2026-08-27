"""Run PyPSA's Modelling-to-Generate-Alternatives (MGA) and persist one solved
network per direction as NetCDF.

optimize_mga_in_direction and optimize_mga_in_multiple_directions wrap PyPSA's own
n.optimize methods of the same name (OptimizationAbstractMGAMixin), reusing
n.optimize.build_linexpr_from_weights/_add_near_opt_constraint/project_solved
as-is, with two behavioral differences:

- optimize_mga_in_direction re-solves the original objective with the projected MGA
  coordinates pinned via equality constraints, so the network ends up at a feasible
  solution that matches the returned coordinates exactly, and returns the shadow
  prices of those pinned constraints.
- optimize_mga_in_multiple_directions exports each direction's solved network to
  ``output_dir/mga_result_{i}.nc`` and appends the objective value (capex + opex,
  labeled ``obj_label``) to the returned coordinates, so downstream code can extract
  per-point timeseries/capacity data from the saved networks.

make_furthest_point_fn builds a batched support-function query against a
real, already-solved PyPSA network -- the PyPSA-side half of plugging a real
system model into `mga_surveyor`'s inner/outer approximation engine instead
of only `mga_surveyor.tests.lp_region`'s explicit-LP test region. Unlike
`mga_surveyor.runner.FurthestPointFn` (one direction in, one point out, e.g.
`runner.bootstrap_box`'s per-direction loop), it takes several directions at
once and solves them in parallel via `optimize_mga_in_multiple_directions`'s
multiprocessing.Pool -- e.g. call it once with all `2k` box directions
instead of looping a single-direction query `2k` times.

make_closest_point_fn builds the other oracle half, a
`mga_surveyor.runner.ClosestFeasiblePointFn` for `runner.run`'s main loop --
PyPSA has no equivalent query built in (only direction-based ones), so it's
written from scratch here, structurally mirroring
`mga_surveyor.tests.lp_region.closest_point_in_region`'s residual-variables-
plus-norm-specific-objective construction but against `n`'s own real
constraints instead of an explicit `(A_ub, b_ub)`. Unlike the box
directions, `run`'s loop can't be batched: each iteration's query depends on
the previous iteration's result. It also can't be rebuilt from scratch every
call the way `make_furthest_point_fn` is (that's fine there -- only `2k`
calls; `run`'s loop can take hundreds): the near-optimal region itself never
changes across calls, only `target` does, so the model is built once and
each call just updates the residual constraints' RHS in place and re-solves.
Only `l1`/`linf` are supported for now (LP-representable, HiGHS-compatible);
`l2` would need a quadratic objective, Gurobi-only like everywhere else L2
shows up in this package.
"""
from __future__ import annotations

import logging
import tempfile
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pypsa._options import options
from pypsa.optimization.mga import _convert_to_dict, _worker_init

from .runner import ClosestFeasiblePointFn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pypsa import Network

logger = logging.getLogger(__name__)
# Explicit level so per-point progress logging (below) is visible even though
# worker processes' `_worker_init` sets the root logger level to WARNING.
logger.setLevel(logging.INFO)

__all__ = [
    "make_furthest_point_fn",
    "make_closest_point_fn",
    "optimize_mga_in_direction",
    "optimize_mga_in_multiple_directions",
]


def make_furthest_point_fn(
    n: Network,
    dimensions: dict,
    output_dir: str | Path | None = None,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    slack: float = 0.05,
    model_kwargs: dict | None = None,
    max_parallel: int = 4,
    **kwargs: Any,
) -> Callable[..., np.ndarray]:
    """Build a batched support-function query against a real,
    already-solved PyPSA network `n`, backed by
    `optimize_mga_in_multiple_directions`'s `multiprocessing.Pool` so that
    passing several directions at once solves them in parallel.

    The returned callable takes `directions` -- an `(n_directions, k)` array
    (a single 1D `(k,)` direction is promoted to a length-1 batch) -- and
    returns the corresponding `(n_directions, k)` array of points in `n`'s
    own near-optimal feasible region, each maximizing its direction. E.g.
    call it once with all `2k` box directions
    (`np.vstack([np.eye(k), -np.eye(k)])`) to bootstrap a `Method`, instead
    of looping a single-direction query `2k` times.

    `dimensions`' key order fixes the axis order of both `directions` and
    the returned points -- the same fixed `k`-dimensional order
    `mga_surveyor` assumes throughout (see `mga_surveyor.common`).

    Every call passes `resolve_main_objective=False` through to
    `optimize_mga_in_multiple_directions`: only the projected points are
    needed here, not the pinned re-solve/duals that flag skips (see
    `optimize_mga_in_direction`'s docstring).

    Caching: pass a real, persistent `output_dir` (one that already exists,
    or should be created and kept) to cache each direction's result there
    across separate calls -- a later call whose `point_ids` line up with an
    already-solved `output_dir/mga_result_{point_id}.nc` reads that
    direction's point straight off disk (via `n.meta["coordinates"]`,
    stashed there for exactly this by `optimize_mga_in_direction`) instead
    of re-solving it, on top of `optimize_mga_in_multiple_directions`'s own
    `skip_existing` already skipping the solve itself. This is an exact,
    ID-keyed cache, the same tradeoff `optimize_mga_in_multiple_directions`'s
    `skip_existing` already documents: it's the CALLER's job to use a fresh
    `output_dir` (or `point_ids`) whenever `dimensions`/`slack`/`n`'s own
    state changes in a way that would make an old cached result wrong --
    nothing here detects that automatically. `point_ids` defaults to
    `range(len(directions))`, i.e. each direction's row position -- already
    a stable, natural key for the common case of repeatedly calling this
    with the same `2k` box directions built the same way every time.

    Parameters
    ----------
    output_dir : str | Path | None
        Where each direction's solved network gets exported, and (see
        "Caching" above) where a later call reads already-solved directions
        back from instead of re-solving. If None (the default), a fresh
        temporary directory is used per call and deleted afterward -- no
        caching is possible across separate calls in that case, only
        (trivially) within a single call's own batch.

    :raises RuntimeError: If any direction doesn't come back solved or
        cached. `optimize_mga_in_multiple_directions` itself already raises
        on a genuine solve failure (see its docstring for why a well-formed
        direction should never actually fail); the only other way to come
        back short is a worker being interrupted (e.g. Ctrl+C), treated the
        same as a failure here since a partial batch isn't useful to the
        caller.
    """
    keys = list(dimensions.keys())

    def furthest_points(directions: np.ndarray, point_ids: Sequence[Any] | None = None) -> np.ndarray:
        directions = np.atleast_2d(np.asarray(directions, dtype=float))
        directions_df = pd.DataFrame(directions, columns=keys)
        if point_ids is None:
            point_ids = list(range(len(directions)))

        # Caching only makes sense against a persistent directory -- a fresh
        # temp dir deleted at the end of this call could never produce a hit
        # on a later call anyway, so skip_existing would just be dead weight.
        caching = output_dir is not None
        tmp_dir = None if caching else tempfile.TemporaryDirectory(prefix="mga_surveyor_furthest_points_")
        resolved_output_dir = Path(output_dir) if caching else Path(tmp_dir.name)
        try:
            _dirs_df, coordinates_df, _duals_df = optimize_mga_in_multiple_directions(
                n,
                directions=directions_df,
                dimensions=dimensions,
                output_dir=resolved_output_dir,
                point_ids=point_ids,
                skip_existing=caching,
                snapshots=snapshots,
                multi_investment_periods=multi_investment_periods,
                slack=slack,
                model_kwargs=model_kwargs,
                resolve_main_objective=False,
                max_parallel=max_parallel,
                **kwargs,
            )

            points_by_id = {
                pid: coordinates_df.loc[pid, keys].to_numpy(dtype=float) for pid in coordinates_df.index
            }
            # Anything not just solved: either a cache hit skip_existing made
            # (caching=True) or -- if it's still missing after that -- a
            # genuinely failed/interrupted point, which is an error either way.
            for pid in point_ids:
                if pid in points_by_id:
                    continue
                cached_fn = resolved_output_dir / f"mga_result_{pid}.nc"
                if not cached_fn.exists():
                    msg = f"Direction {pid!r} was neither solved nor found cached at {cached_fn}."
                    raise RuntimeError(msg)
                from pypsa.networks import Network as _Network  # noqa: PLC0415

                cached_coordinates = pd.Series(_Network(cached_fn).meta["coordinates"])
                points_by_id[pid] = cached_coordinates.loc[keys].to_numpy(dtype=float)
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()

        return np.array([points_by_id[pid] for pid in point_ids])

    return furthest_points


_CLOSEST_POINT_SUPPORTED_NORMS = ("l1", "linf")


def make_closest_point_fn(
    n: Network,
    dimensions: dict,
    norm: str,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    slack: float = 0.05,
    model_kwargs: dict | None = None,
    **kwargs: Any,
) -> ClosestFeasiblePointFn:
    """Build a `mga_surveyor.runner.ClosestFeasiblePointFn` against a real,
    already-solved PyPSA network `n` -- the query `runner.run`'s main loop
    needs: the closest point in `n`'s own near-optimal feasible region to a
    given `target`, together with a valid separating-hyperplane normal there.

    Structurally this mirrors `mga_surveyor.tests.lp_region.closest_point_in_region`
    (same residual-variables-plus-norm-specific-objective construction, same
    `r_j == target_j - dim_j(x)` sign convention for the residual and for
    reading the normal off its dual, unmodified), but against `n`'s own real
    constraints (`create_model` + `_add_near_opt_constraint`) instead of an
    explicit `(A_ub, b_ub)`.

    Only `norm in ('l1', 'linf')` is supported for now -- both are
    LP-representable via the usual auxiliary-variable linearization, so this
    works with any solver `n.optimize.solve_model` supports (HiGHS
    included). `l2` would need a quadratic objective -- Gurobi-only, like
    everywhere else L2 shows up in this package (`gurobi_projection`,
    `BilevelMethod`) -- and isn't implemented here yet.

    Unlike `make_furthest_point_fn` (rebuilt from scratch every call --
    fine there, `bootstrap_box` only calls it `2k` times), the model here is
    built ONCE, before the returned callable is ever invoked: `run`'s main
    loop can call this hundreds of times, but the near-optimal region itself
    (the budget constraint plus everything `create_model` built) never
    changes across those calls, only `target` does -- so each call just
    updates the residual constraints' RHS in place
    (`linopy.Constraint.update(rhs=...)`) and re-solves, rather than paying
    for a full model rebuild every time.

    :raises NotImplementedError: If `norm` is not `'l1'` or `'linf'`.
    :raises RuntimeError: If a call's solve does not reach `status="ok"`.
    """
    if norm not in _CLOSEST_POINT_SUPPORTED_NORMS:
        raise NotImplementedError(
            f"make_closest_point_fn only supports norm in {_CLOSEST_POINT_SUPPORTED_NORMS} for now, "
            f"got {norm!r} -- l2 needs a quadratic objective (Gurobi-only, like gurobi_projection."
            f"closest_point_on_hull)."
        )
    keys = list(dimensions.keys())
    if model_kwargs is None:
        model_kwargs = options.params.optimize.model_kwargs.copy()
    if snapshots is None:
        snapshots = n.snapshots
    if not n.is_solved:
        msg = "Network needs to be solved with `n.optimize()` before running MGA."
        raise ValueError(msg)

    m = n.optimize.create_model(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
        **model_kwargs,
    )
    n.optimize._add_near_opt_constraint(multi_investment_periods, slack)

    # r_key == target_key - dim_key(x), i.e. r_key + dim_key(x) == target_key
    # -- RHS starts at 0 (target=0) and is updated to the real target on
    # every call to `closest_point` below.
    resid_constrs = {}
    for key in keys:
        r = m.add_variables(lower=-np.inf, name=f"mga_surveyor_resid_{key}")
        dim_expr = n.optimize.build_linexpr_from_weights(dimensions[key], model=m).sum()
        resid_constrs[key] = m.add_constraints(r + dim_expr == 0.0, name=f"mga_surveyor_resid_constr_{key}")

    if norm == "linf":
        t = m.add_variables(lower=0.0, name="mga_surveyor_closest_point_t")
        for key in keys:
            r = m.variables[f"mga_surveyor_resid_{key}"]
            m.add_constraints(t - r >= 0.0, name=f"mga_surveyor_tpos_{key}")
            m.add_constraints(t + r >= 0.0, name=f"mga_surveyor_tneg_{key}")
        m.objective = 1 * t  # bare Variable isn't a valid objective -- coerce to LinearExpression
    else:  # l1
        s_vars = []
        for key in keys:
            r = m.variables[f"mga_surveyor_resid_{key}"]
            s = m.add_variables(lower=0.0, name=f"mga_surveyor_s_{key}")
            m.add_constraints(s - r >= 0.0, name=f"mga_surveyor_spos_{key}")
            m.add_constraints(s + r >= 0.0, name=f"mga_surveyor_sneg_{key}")
            s_vars.append(s)
        m.objective = sum(s_vars)

    def closest_point(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(target, dtype=float)
        for key, target_value in zip(keys, target):
            resid_constrs[key].update(rhs=float(target_value))

        status, condition = n.optimize.solve_model(**kwargs)
        if status != "ok":
            msg = f"Closest-point solve for target {dict(zip(keys, target.tolist()))} failed: status={status}, condition={condition}"
            raise RuntimeError(msg)

        point = n.optimize.project_solved(dimensions).loc[keys].to_numpy(dtype=float)
        normal = np.array([float(resid_constrs[key].dual) for key in keys])
        return point, normal

    return closest_point


def optimize_mga_in_direction(
    n: Network,
    direction: dict | pd.Series,
    dimensions: dict,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    slack: float = 0.05,
    model_kwargs: dict | None = None,
    resolve_main_objective: bool = True,
    **kwargs: Any,
) -> tuple[str, str, pd.Series, pd.Series]:
    """Run MGA in a given direction, optionally re-solving pinned to the resulting coordinates.

    Same as pypsa's `n.optimize.optimize_mga_in_direction`, except (when
    `resolve_main_objective` is True, the default) after finding the MGA
    point's projected coordinates, the network is re-solved with the original
    objective under added equality constraints fixing those coordinates exactly.

    Both solves are guaranteed feasible for well-formed inputs: the budget constraint
    always admits the network's own (already-solved) optimum, and the final pinned
    re-solve always admits the point found by the direction-objective solve. So a
    non-"ok" status here means either a bug (e.g. `direction`/`dimensions` mismatch)
    or a genuine solver/numerical problem — either way this raises rather than
    returning a sentinel, so callers don't silently lose data points.

    Parameters
    ----------
    resolve_main_objective : bool
        If True (default), re-solve the original cost objective pinned to the
        MGA point's coordinates, so the network ends up at a feasible solution
        matching them exactly and `duals` holds the shadow price of each
        pinned-coordinate constraint. If False, skip that second solve
        entirely and return right after the direction-objective solve --
        `duals` is then all-zero (no pinning constraints were ever added, so
        there's nothing to price). Set this to False when only the projected
        coordinates are needed and the extra solve isn't worth paying for,
        e.g. a support-function query like `make_furthest_point_fn`'s.

    Returns
    -------
    status, condition : str
        As returned by the final solve. Always "ok"/"optimal" — raises otherwise.
    coordinates : pd.Series
        Projected coordinates of the solution.
    duals : pd.Series
        Shadow price of each dimension's pinned-coordinate constraint, indexed by
        the same keys as `dimensions`. Dimensions with `direction[key] == 0` (no
        constraint was added for them) get 0.0 -- as does every dimension when
        `resolve_main_objective` is False (no constraints were added at all).

    """
    if model_kwargs is None:
        model_kwargs = options.params.optimize.model_kwargs.copy()

    if set(direction.keys()) != set(dimensions.keys()):
        msg = (
            "Keys of `direction` and `dimensions` arguments must match. "
            f"Got {set(direction.keys())} and {set(dimensions.keys())}."
        )
        raise ValueError(msg)

    if snapshots is None:
        snapshots = n.snapshots

    if not n.is_solved:
        msg = "Network needs to be solved with `n.optimize()` before running MGA."
        raise ValueError(msg)

    m = n.optimize.create_model(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
        **model_kwargs,
    )

    if resolve_main_objective:
        # m.objective is already populated by create_model (the network's normal cost
        # objective) — no need to solve first just to capture it. n's component tables
        # (read by _add_near_opt_constraint below) still hold the values from the
        # network's prior n.optimize() call, since m hasn't been solved yet.
        main_obj = m.objective

    n.optimize._add_near_opt_constraint(multi_investment_periods, slack)

    m.objective = -sum(
        direction[key] * n.optimize.build_linexpr_from_weights(dimensions[key], model=m)
        for key in direction.keys()
    )

    status, condition = n.optimize.solve_model(**kwargs)
    if status != "ok":
        msg = f"MGA solve in direction {direction} failed: status={status}, condition={condition}"
        raise RuntimeError(msg)
    coordinates = n.optimize.project_solved(dimensions)

    n.meta["slack"] = slack
    n.meta["dimensions"] = _convert_to_dict(dimensions)
    n.meta["direction"] = direction
    # Stashed so a network reloaded from a plain NetCDF export (no live
    # linopy model attached -- project_solved would fail on it) can still
    # recover its own coordinates, e.g. to read a cached result back without
    # re-solving; see make_furthest_point_fn.
    n.meta["coordinates"] = {key: float(value) for key, value in coordinates.items()}

    if not resolve_main_objective:
        duals = pd.Series(0.0, index=list(dimensions.keys()))
        return status, condition, coordinates, duals

    pinned_keys = [key for key in dimensions.keys() if direction[key] != 0]
    for key in pinned_keys:
        # .sum() is required here: build_linexpr_from_weights returns one term
        # per component (e.g. per generator) rather than a pre-aggregated scalar,
        # and add_constraints treats each coordinate slice as a separate
        # constraint row. Without summing first, this would add one (mostly
        # malformed, e.g. "0 == coordinates[key]") constraint per component
        # instead of a single constraint pinning the weighted sum.
        m.add_constraints(
            n.optimize.build_linexpr_from_weights(dimensions[key], model=m).sum() == coordinates[key].item(),
            name=f"coord_{key}",
        )
    # The budget constraint can't make this re-solve infeasible (the point found
    # above already satisfies it), and minimizing cost at fixed coordinates can
    # only match or undercut that bound, so it stays non-binding. Drop it so the
    # duals below reflect only the pinned-coordinate constraints.
    m.remove_constraints("budget")
    m.objective = main_obj
    status, condition = n.optimize.solve_model(**kwargs)
    if status != "ok":
        msg = f"Re-solve at pinned coordinates for direction {direction} failed: status={status}, condition={condition}"
        raise RuntimeError(msg)
    duals = pd.Series(
        {key: float(m.constraints[f"coord_{key}"].dual) for key in pinned_keys}
    ).reindex(dimensions.keys(), fill_value=0.0)

    return status, condition, coordinates, duals


def _solve_single_direction(
    fn: str,
    direction: dict,
    dimensions: dict,
    snapshots: Sequence | None,
    multi_investment_periods: bool,
    slack: float,
    model_kwargs: dict,
    resolve_main_objective: bool,
    kwargs: dict,
    output_fn: str,
) -> tuple[dict, None, None, None] | tuple[dict, pd.Series, pd.Series, float]:
    """Solve a single direction for parallel execution (helper for `optimize_mga_in_multiple_directions`).

    The network is read from `fn`, solved, and exported to `output_fn` as a NetCDF
    file so downstream code can extract per-point data from it later.

    Real failures (a bug, or a genuine solver/numerical problem — see
    `optimize_mga_in_direction`) are deliberately not caught here: they propagate
    through `pool.starmap` and fail the whole batch loudly, rather than silently
    dropping a point. The only case handled here is a worker being interrupted
    (e.g. Ctrl+C), which is not a failure to report, just a graceful shutdown.
    """
    from pypsa.networks import Network  # noqa: PLC0415

    try:
        n = Network(fn)
        _, _, coordinates, duals = optimize_mga_in_direction(
            n,
            direction=direction,
            dimensions=dimensions,
            snapshots=snapshots,
            multi_investment_periods=multi_investment_periods,
            slack=slack,
            model_kwargs=model_kwargs,
            resolve_main_objective=resolve_main_objective,
            **kwargs,
        )
        n.export_to_netcdf(output_fn)
        obj = n.statistics.capex().sum() + n.statistics.opex().sum()
    except KeyboardInterrupt:
        logger.info("Worker process interrupted")
        return (direction, None, None, None)
    else:
        logger.info("slack=%s: solved %s", slack, Path(output_fn).stem)
        return (direction, coordinates, duals, obj)


def optimize_mga_in_multiple_directions(
    n: Network,
    directions: list[dict] | pd.DataFrame,
    dimensions: dict,
    output_dir: str | Path,
    obj_label: str = "TOTEX",
    point_ids: Sequence[Any] | None = None,
    skip_existing: bool = False,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    slack: float = 0.05,
    model_kwargs: dict | None = None,
    resolve_main_objective: bool = True,
    max_parallel: int = 4,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run MGA in multiple directions in parallel, saving each solved network as NetCDF.

    Same as pypsa's `n.optimize.optimize_mga_in_multiple_directions`, except each
    direction's solved network is exported to `output_dir/mga_result_{point_id}.nc`
    (kept permanently), the returned coordinates include the objective value
    (capex + opex) as an extra column labeled `obj_label`, and a third DataFrame of
    duals is returned (shadow price of each dimension's pinned-coordinate constraint,
    one row per point).

    Unlike pypsa's version, a direction that fails to solve does NOT get silently
    dropped — it raises and aborts the whole batch (see `optimize_mga_in_direction`
    for why a well-formed direction should never actually fail). The only thing
    that can shrink the result below `len(directions)` rows is a worker being
    interrupted (e.g. Ctrl+C) or `skip_existing` skipping already-solved points.

    Parameters
    ----------
    output_dir : str | Path
        Directory where `mga_result_{point_id}.nc` files are written, one per
        solved direction.
    obj_label : str
        Column name used for the objective value (capex + opex) in the returned
        coordinates DataFrame. Defaults to "TOTEX".
    point_ids : Sequence | None
        Labels identifying each row of `directions`, used both for naming the
        NetCDF files (`mga_result_{point_id}.nc`) and as the index of the returned
        DataFrames. Must match `directions` in length and order. Defaults to
        `range(len(directions))`, matching pypsa's own (purely positional) behavior.
        Pass your own stable IDs (e.g. ones that survive across separate runs) to
        make `skip_existing` meaningful.
    skip_existing : bool
        If True, directions whose `output_dir/mga_result_{point_id}.nc` already
        exists are not re-solved; only the still-missing directions are run. Useful
        for resuming an interrupted or incrementally-extended dataset generation
        run. Requires `point_ids` to be stable across runs.
    resolve_main_objective : bool
        Forwarded to `optimize_mga_in_direction` in every worker (see its
        docstring). Defaults to True for this function's usual dataset-
        generation use case, which needs `duals_df`; pass False for a
        cheaper, duals-free batch (e.g. `make_furthest_point_fn`, which
        only needs `coordinates_df`).

    Returns
    -------
    directions_df, coordinates_df, duals_df : pd.DataFrame
        One row per solved direction each (directions skipped via `skip_existing`,
        or whose worker was interrupted, are not included), indexed by `point_id`.
        `duals_df` has the same columns as `dimensions` (see
        `optimize_mga_in_direction`'s `duals` return value).

    See Also
    --------
    pypsa.optimization.mga.OptimizationAbstractMGAMixin.optimize_mga_in_multiple_directions

    """
    if model_kwargs is None:
        model_kwargs = options.params.optimize.model_kwargs.copy()

    if isinstance(directions, pd.DataFrame):
        directions = list(directions.T.to_dict().values())

    if point_ids is None:
        point_ids = list(range(len(directions)))
    elif len(point_ids) != len(directions):
        msg = f"point_ids must have the same length as directions ({len(point_ids)} != {len(directions)})"
        raise ValueError(msg)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_fns = {pid: output_dir / f"mga_result_{pid}.nc" for pid in point_ids}

    if skip_existing:
        todo = [(pid, d) for pid, d in zip(point_ids, directions) if not output_fns[pid].exists()]
        skipped = len(directions) - len(todo)
        if skipped:
            logger.info("Skipping %s already-solved direction(s)", skipped)
        if not todo:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        point_ids, directions = (list(x) for x in zip(*todo, strict=True))

    # Export the network to a temporary file. Note: cannot pass the network as an
    # argument directly since it is not picklable.
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False, dir=output_dir) as f:
        fn = f.name

    try:
        n.export_to_netcdf(fn)

        with (
            get_context("spawn").Pool(
                processes=max_parallel,
                initializer=_worker_init,
                # No maxtasksperchild: each worker re-imports the whole scientific
                # stack (pypsa, linopy, pandas, xarray) on every cold start, which
                # dwarfs the actual solve time. Reusing workers across directions
                # pays that import cost ~max_parallel times instead of once per
                # direction. Revisit if per-worker memory growth becomes an issue
                # over very large direction counts.
            ) as pool
        ):
            try:
                results = pool.starmap(
                    _solve_single_direction,
                    [
                        (
                            fn,
                            direction,
                            dimensions,
                            snapshots,
                            multi_investment_periods,
                            slack,
                            model_kwargs,
                            resolve_main_objective,
                            kwargs,
                            str(output_fns[pid]),
                        )
                        for pid, direction in zip(point_ids, directions)
                    ],
                )
            except Exception:
                pool.terminate()
                pool.join()
                raise

        # coords is only None here if that worker was interrupted (e.g. Ctrl+C):
        # real solve failures raise inside the worker and abort the whole batch
        # via the `except Exception: pool.terminate(); ...; raise` above.
        successful = [
            (pid, direction, pd.concat([coords, pd.Series({obj_label: obj})]), duals)
            for pid, (direction, coords, duals, obj) in zip(point_ids, results)
            if coords is not None
        ]
        interrupted_count = len(results) - len(successful)
        if interrupted_count > 0:
            logger.warning(
                "%s out of %s directions were interrupted before solving", interrupted_count, len(results)
            )

        if not successful:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        successful_ids, successful_directions, successful_coordinates, successful_duals = zip(*successful, strict=True)
        index = pd.Index(successful_ids)
        return (
            pd.DataFrame(successful_directions, index=index),
            pd.DataFrame(successful_coordinates, index=index),
            pd.DataFrame(successful_duals, index=index),
        )
    finally:
        if Path(fn).exists():
            Path(fn).unlink()
