"""Run PyPSA's Modelling-to-Generate-Alternatives (MGA) against a real,
already-solved network, optionally caching each solved point as NetCDF.

make_optimality_fn builds a callable that solves and caches n's own
cost-minimizing optimum at output_dir/optimal.nc, returning its point in
dimensions space -- the precondition every other function here assumes.
Also pins that point's coordinates and re-solves for real per-dimension
duals, cached in n.meta["duals"] -- same pinned-resolve idea as
make_furthest_point_fn/make_closest_point_fn's own resolve_main_objective.

make_furthest_point_fn solves MGA in one or more directions: a batched
support-function query against n's near-optimal feasible region. Always
returns (points, normals) -- normal is just the query direction, free.
resolve_main_objective=True (default) additionally does the
pinned-coordinate re-solve for real duals/obj, cached in n.meta but not
part of the return value (pass False to skip that extra solve). Has its
own dedicated solver/worker/cache
(_build_furthest_point_solver/_init_furthest_point_worker/
_solve_furthest_point_direction), reused across a whole batch instead of
rebuilt per direction.

make_closest_point_fn finds the closest point in n's near-optimal region to
a target, plus a separating-hyperplane normal -- PyPSA has no built-in
equivalent. resolve_main_objective=True (default) also pins that point's
coordinates and re-solves the original objective for real per-dimension
duals, cached in n.meta but not part of the return value (still just
(point, normal), matching runner.ClosestPointFn); pass False to skip that
re-solve. A single-target call reuses one persistent in-process model; a
batch spreads targets across a multiprocessing.Pool, each worker building
its own solver once. Only l1/linf supported (l2 needs a quadratic
objective, Gurobi-only).

Both batch paths pass the PyPSA Network directly to worker processes
(picklable as of PyPSA 1.3.0, github.com/PyPSA/PyPSA/pull/1889) instead of
exporting to NetCDF and reloading. make_closest_point_fn's batch path also
snapshots n via pickle before any solve can mutate it, since
Network.copy() refuses on a network with an attached solver model.

make_oracle_fns builds all three of the above in one call, sharing
n/dimensions/output_dir -- a convenience for feeding runner.run's three
callback parameters directly.
"""
from __future__ import annotations

import logging
import pickle
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from linopy import LinearExpression, QuadraticExpression
from pypsa._options import options
from pypsa.optimization.mga import _worker_init

from .runner import ClosestPointFn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pypsa import Network

logger = logging.getLogger(__name__)
# Explicit level so per-point progress logging (below) stays visible even
# though worker processes' _worker_init sets the root logger to WARNING.
logger.setLevel(logging.INFO)

__all__ = [
    "make_optimality_fn",
    "make_furthest_point_fn",
    "make_closest_point_fn",
    "make_oracle_fns",
    "encode_cache_key",
    "decode_cache_key",
]


def make_optimality_fn(n: Network, dimensions: dict, output_dir: str | Path, **kwargs: Any) -> Callable[[], np.ndarray]:
    """Build a callable that solves `n`'s own cost-minimizing optimum,
    caches it at `output_dir/optimal.nc`, and returns its point in
    `dimensions` space -- structurally mirrors `make_furthest_point_fn`/
    `make_closest_point_fn`.

    After solving, the optimal point's coordinates are pinned and the same
    cost objective is re-solved to read real per-dimension duals off those
    pinned constraints -- the marginal cost of each dimension at the
    optimum -- cached alongside the point in `n.meta["duals"]`, same idea
    as `make_furthest_point_fn`/`make_closest_point_fn`'s own
    `resolve_main_objective`.

    If `output_dir/optimal.nc` already exists, its point is read back from
    `n.meta["coordinates"]` (stashed there on save, same as everywhere else
    in this module) instead of re-solving.

    :raises RuntimeError: If a solve does not reach `status="ok"`.
    """
    keys = list(dimensions.keys())
    optimal_fn = Path(output_dir) / "optimal.nc"

    def solve_optimal() -> np.ndarray:
        if optimal_fn.exists():
            from pypsa import Network as _Network  # noqa: PLC0415

            cached_coordinates = _Network(optimal_fn).meta["coordinates"]
            return np.array([cached_coordinates[k] for k in keys])

        m = n.optimize.create_model()
        status, condition = n.optimize.solve_model(**kwargs)
        if status != "ok":
            msg = f"Solve to optimality failed: status={status}, condition={condition}"
            raise RuntimeError(msg)
        point = n.optimize.project_solved(dimensions).loc[keys].to_numpy(dtype=float)

        for key, value in zip(keys, point):
            m.add_constraints(
                n.optimize.build_linexpr_from_weights(dimensions[key], model=m).sum() == float(value),
                name=f"coord_{key}",
            )
        status, condition = n.optimize.solve_model(**kwargs)
        if status != "ok":
            msg = f"Re-solve at pinned optimal coordinates failed: status={status}, condition={condition}"
            raise RuntimeError(msg)
        duals = np.array([float(m.constraints[f"coord_{key}"].dual) for key in keys])

        n.meta["coordinates"] = {k: float(v) for k, v in zip(keys, point)}
        n.meta["duals"] = {k: float(v) for k, v in zip(keys, duals)}

        optimal_fn.parent.mkdir(parents=True, exist_ok=True)
        n.export_to_netcdf(str(optimal_fn))
        return point

    return solve_optimal


def encode_cache_key(vec: np.ndarray, decimals: int) -> str:
    """Reversible, filename-safe encoding of a numeric vector: round each
    component to `decimals` places (default 8, so vectors differing only
    beyond that collide onto the same key by design), then hex-encode the
    rounded array's float64 bytes.
    """
    rounded = np.round(np.asarray(vec, dtype=np.float64), decimals) + 0.0  # +0.0 normalizes -0.0 to 0.0
    return rounded.tobytes().hex()


def decode_cache_key(key: str) -> np.ndarray:
    """Inverse of `encode_cache_key`."""
    return np.frombuffer(bytes.fromhex(key), dtype=np.float64)


def _build_furthest_point_solver(
    n: Network,
    dimensions: dict,
    slack: float,
    snapshots: Sequence | None,
    multi_investment_periods: bool,
    model_kwargs: dict | None,
    kwargs: dict,
    resolve_main_objective: bool,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, Network, str]]:
    """Build the furthest-point model against `n` once, and return a
    single-direction `(point, duals, n, condition)` solver reusing it.

    Each call reassigns `m.objective` to that direction's support function
    and re-solves against a fixed `budget` near-optimal-cost constraint.

    If `resolve_main_objective`: also pins the solved coordinates, drops
    `budget`, re-solves the original cost objective for duals, then undoes
    the pin and re-adds `budget` -- using `optimal_cost`/`fixed_cost`
    captured once up front, NOT by calling
    `n.optimize._add_near_opt_constraint()` again, which would recompute
    them from `n`'s by-then-mutated state.

    :raises RuntimeError: If a solve does not reach `status="ok"`.
    """
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
    main_obj = m.objective

    # Captured once, before n is mutated by any solve -- see docstring.
    if not multi_investment_periods:
        optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
        fixed_cost = n.statistics.installed_capex().sum()
    else:
        w = n.investment_period_weightings.objective
        optimal_cost = (n.statistics.capex().sum() * w + n.statistics.opex().sum() * w).sum()
        fixed_cost = (n.statistics.installed_capex().sum() * w).sum()

    def add_budget_constraint() -> None:
        # Only ever called with m.objective == main_obj.
        objective = m.objective
        if not isinstance(objective, (LinearExpression, QuadraticExpression)):
            objective = objective.expression
        m.add_constraints(objective + fixed_cost <= (1 + slack) * optimal_cost, name="budget")

    add_budget_constraint()

    def solve_one(direction_row: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, Network, str]:
        m.objective = -sum(
            direction_row[i] * n.optimize.build_linexpr_from_weights(dimensions[key], model=m)
            for i, key in enumerate(keys)
        )
        status, condition = n.optimize.solve_model(**kwargs)
        if status != "ok":
            msg = (
                f"Furthest-point solve for direction {dict(zip(keys, direction_row.tolist()))} "
                f"failed: status={status}, condition={condition}"
            )
            raise RuntimeError(msg)
        coordinates = n.optimize.project_solved(dimensions)
        point = coordinates.loc[keys].to_numpy(dtype=float)

        if not resolve_main_objective:
            return point, None, n, condition

        pinned_keys = [key for i, key in enumerate(keys) if direction_row[i] != 0]
        for key in pinned_keys:
            m.add_constraints(
                n.optimize.build_linexpr_from_weights(dimensions[key], model=m).sum() == coordinates[key].item(),
                name=f"coord_{key}",
            )
        m.remove_constraints("budget")
        m.objective = main_obj
        status, condition = n.optimize.solve_model(**kwargs)
        if status != "ok":
            msg = (
                f"Re-solve at pinned coordinates for direction {dict(zip(keys, direction_row.tolist()))} "
                f"failed: status={status}, condition={condition}"
            )
            raise RuntimeError(msg)
        duals_by_key = {key: float(m.constraints[f"coord_{key}"].dual) for key in pinned_keys}
        duals = np.array([duals_by_key.get(key, 0.0) for key in keys])

        # Undo this call's own additions so the model is ready for the next call.
        if pinned_keys:
            m.remove_constraints([f"coord_{key}" for key in pinned_keys])
        add_budget_constraint()

        return point, duals, n, condition

    return solve_one


#: This worker's own furthest-point solver, built once by
#: _init_furthest_point_worker and reused by _solve_furthest_point_direction
#: for every task this worker process handles.
_worker_furthest_point_solver: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None, Network, str]] | None = None


def _init_furthest_point_worker(
    n: Network,
    dimensions: dict,
    slack: float,
    snapshots: Sequence | None,
    multi_investment_periods: bool,
    model_kwargs: dict | None,
    kwargs: dict,
    resolve_main_objective: bool,
) -> None:
    """Pool initializer for `make_furthest_point_fn`'s batch path -- builds
    this worker's solver once via `_build_furthest_point_solver`, before
    any tasks are dispatched.
    """
    _worker_init()
    global _worker_furthest_point_solver
    _worker_furthest_point_solver = _build_furthest_point_solver(
        n, dimensions, slack, snapshots, multi_investment_periods, model_kwargs, kwargs, resolve_main_objective,
    )


def _solve_furthest_point_direction(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, Network, str]:
    """One `pool.starmap` task: solve a single direction against this
    worker's already-built solver (see `_init_furthest_point_worker`).
    """
    return _worker_furthest_point_solver(direction)


def _cache_furthest_point_result(
    n: Network, output_dir: str | Path, key: str, keys: list,
    point: np.ndarray, duals: np.ndarray | None, obj: float, direction: np.ndarray,
) -> None:
    """Stash `coordinates`/`obj`/`normal` into `n.meta` and export to
    `output_dir/mga_furthest_result_{key}.nc`. `normal` is the query `direction`
    itself -- already a valid separating-hyperplane normal at the point it
    maximizes, same concept `_cache_closest_point_result`'s own `normal`
    is, just sourced differently (the query itself here, a residual
    constraint's dual there). `duals` is only stashed (as `n.meta["duals"]`)
    when not `None` -- i.e. when `resolve_main_objective` actually computed
    it; otherwise `n.meta` has no `"duals"` key at all, rather than a
    placeholder.
    """
    n.meta["coordinates"] = {k: float(v) for k, v in zip(keys, point)}
    if duals is not None:
        n.meta["duals"] = {k: float(v) for k, v in zip(keys, duals)}
    n.meta["obj"] = float(obj)
    n.meta["normal"] = {k: float(v) for k, v in zip(keys, direction)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(str(output_dir / f"mga_furthest_result_{key}.nc"))


def make_furthest_point_fn(
    n: Network,
    dimensions: dict,
    slack: float,
    output_dir: str | Path | None = None,
    decimals: int = 8,
    resolve_main_objective: bool = True,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    model_kwargs: dict | None = None,
    max_parallel: int = 4,
    **kwargs: Any,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build a batched MGA-direction query against a real, already-solved
    PyPSA network `n`.

    The returned callable takes `directions` -- an `(n_directions, k)` array
    (a single `(k,)` direction is promoted to a length-1 batch) -- and
    always returns `(points, normals)`: each direction's point in `n`'s
    near-optimal feasible region (`(n_directions, k)`, not shape-preserving
    -- a single direction still comes back as a length-1 array) and its
    separating-hyperplane normal, which for a furthest-point query is just
    the query direction itself, echoed back (free -- no extra solve).

    `resolve_main_objective=True` (default) additionally pins each found
    point's coordinates and re-solves the original cost objective for real
    per-dimension duals and that point's own capex+opex cost -- cached
    alongside the point in `n.meta["duals"]`/`n.meta["obj"]`, but NOT part
    of the return value (kept fixed at `(points, normals)` either way, so
    callers -- e.g. `runner.run` -- don't need to know which was used).
    Pass `False` to skip that extra solve.

    `dimensions`' key order fixes the axis order of `directions`/`points`/
    `normals`.

    Caching: pass a persistent `output_dir` to cache each direction's
    result at `output_dir/mga_furthest_result_{key}.nc`, content-addressed by the
    direction itself (see `encode_cache_key`). `None` (default) disables
    caching. A cache hit on the single-direction path skips solving
    entirely, so `n` is NOT mutated by a call that only reads a cache hit.

    :raises RuntimeError: If a solve does not reach `status="ok"` (see
        `_build_furthest_point_solver`).
    """
    keys = list(dimensions.keys())

    # Snapshot n before any solve can mutate it -- the batch path's workers
    # need n's true original state for the budget constraint.
    pristine_n = pickle.loads(pickle.dumps(n))

    solve_one = _build_furthest_point_solver(
        n, dimensions, slack, snapshots, multi_investment_periods, model_kwargs, kwargs,
        resolve_main_objective=resolve_main_objective,
    )

    def furthest_points(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Always returns `(points, normals)` -- see `make_furthest_point_fn`'s
        docstring. `duals`/`obj` are computed (if `resolve_main_objective`)
        and cached to `output_dir`, but never part of the return value.

        A length-1 batch is solved directly against the persistent
        in-process model built above (or read from `output_dir`'s cache);
        a longer batch has its already-cached rows read from disk and every
        other row dispatched as its own `pool.starmap` task, spread across
        `min(max_parallel, n_todo)` worker processes each building its own
        solver once.
        """
        directions = np.atleast_2d(np.asarray(directions, dtype=float))

        if len(directions) == 1:
            d = directions[0]
            cached = None
            if output_dir is not None:
                key = encode_cache_key(d, decimals)
                cached_fn = Path(output_dir) / f"mga_furthest_result_{key}.nc"
                if cached_fn.exists():
                    from pypsa.networks import Network as _Network  # noqa: PLC0415

                    cached_meta = _Network(cached_fn).meta
                    cached = (
                        np.array([cached_meta["coordinates"][k] for k in keys]),
                        np.array([cached_meta["normal"][k] for k in keys]),
                    )
            if cached is not None:
                results = [cached]
            else:
                point, duals, n_result, _condition = solve_one(d)
                obj = n_result.statistics.capex().sum() + n_result.statistics.opex().sum()
                if output_dir is not None:
                    _cache_furthest_point_result(n_result, output_dir, key, keys, point, duals, obj, d)
                results = [(point, d)]
        else:
            # Passes pristine_n, not n's current state -- n may already be
            # mutated by an earlier single-direction call above.
            results = [None] * len(directions)
            todo_idx = list(range(len(directions)))
            cache_keys = None
            if output_dir is not None:
                cache_keys = [encode_cache_key(row, decimals) for row in directions]
                todo_idx = []
                for i, key in enumerate(cache_keys):
                    cached_fn = Path(output_dir) / f"mga_furthest_result_{key}.nc"
                    if not cached_fn.exists():
                        todo_idx.append(i)
                        continue
                    from pypsa.networks import Network as _Network  # noqa: PLC0415

                    cached_meta = _Network(cached_fn).meta
                    results[i] = (
                        np.array([cached_meta["coordinates"][k] for k in keys]),
                        np.array([cached_meta["normal"][k] for k in keys]),
                    )
                skipped = len(directions) - len(todo_idx)
                if skipped:
                    logger.info("Skipping %s already-cached direction(s)", skipped)

            if todo_idx:
                todo_directions = directions[todo_idx]
                n_workers = min(max_parallel, len(todo_directions))
                with get_context("spawn").Pool(
                    processes=n_workers,
                    initializer=_init_furthest_point_worker,
                    initargs=(
                        pristine_n, dimensions, slack, snapshots,
                        multi_investment_periods, model_kwargs, kwargs, resolve_main_objective,
                    ),
                ) as pool:
                    try:
                        todo_results = pool.starmap(
                            _solve_furthest_point_direction, [(row,) for row in todo_directions],
                        )
                    except Exception:
                        pool.terminate()
                        pool.join()
                        raise
                for i, (point, duals, n_result, _condition) in zip(todo_idx, todo_results):
                    obj = n_result.statistics.capex().sum() + n_result.statistics.opex().sum()
                    if output_dir is not None:
                        _cache_furthest_point_result(
                            n_result, output_dir, cache_keys[i], keys, point, duals, obj, directions[i],
                        )
                    results[i] = (point, directions[i])

        points = np.array([r[0] for r in results])
        normals = np.array([r[1] for r in results])
        return points, normals

    return furthest_points


_CLOSEST_POINT_SUPPORTED_NORMS = ("l1", "linf")


def _build_closest_point_solver(
    n: Network,
    dimensions: dict,
    norm: str,
    slack: float,
    snapshots: Sequence | None,
    multi_investment_periods: bool,
    model_kwargs: dict | None,
    kwargs: dict,
    resolve_main_objective: bool,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray | None, Network]]:
    """Build the closest-point model against `n` once, and return a
    single-target `(point, normal, duals, n)` solver reusing it.

    One residual variable per dimension (`r_key + dim_key(x) == target_key`,
    RHS updated per call) plus a norm-specific objective: `t >= |r_key|`
    for `linf`, `sum(s_key)` with `s_key >= |r_key|` for `l1`. `normal` is
    the dual of these residual constraints -- the query `target`'s own
    separating-hyperplane normal, always computed.

    If `resolve_main_objective`: also pins the found point's coordinates,
    drops `budget`, re-solves the network's original cost objective for
    `duals` (the shadow price of each pinned-coordinate constraint -- the
    same pinned-resolve `_build_furthest_point_solver` does), then undoes
    the pin, re-adds `budget`, and restores the residual-LP objective, all
    so this same reused model is ready for the next call -- same
    budget-restore care as `_build_furthest_point_solver` (`optimal_cost`/
    `fixed_cost` captured once, not recomputed via
    `n.optimize._add_near_opt_constraint()` again). `duals` is `None`
    when `resolve_main_objective` is False.

    :raises NotImplementedError: If `norm` is not `'l1'` or `'linf'`.
    :raises RuntimeError: If a solve does not reach `status="ok"`.
    """
    if norm not in _CLOSEST_POINT_SUPPORTED_NORMS:
        raise NotImplementedError(
            f"closest-point solving only supports norm in {_CLOSEST_POINT_SUPPORTED_NORMS} for now, "
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
    main_obj = m.objective

    # Captured once, before n is mutated by any solve -- see
    # _build_furthest_point_solver's docstring for why.
    if not multi_investment_periods:
        optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
        fixed_cost = n.statistics.installed_capex().sum()
    else:
        w = n.investment_period_weightings.objective
        optimal_cost = (n.statistics.capex().sum() * w + n.statistics.opex().sum() * w).sum()
        fixed_cost = (n.statistics.installed_capex().sum() * w).sum()

    def add_budget_constraint() -> None:
        # Only ever called with m.objective == main_obj.
        objective = m.objective
        if not isinstance(objective, (LinearExpression, QuadraticExpression)):
            objective = objective.expression
        m.add_constraints(objective + fixed_cost <= (1 + slack) * optimal_cost, name="budget")

    add_budget_constraint()

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
    residual_obj = m.objective

    def solve_one(target_row: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Network]:
        for key, target_value in zip(keys, target_row):
            resid_constrs[key].update(rhs=float(target_value))

        status, condition = n.optimize.solve_model(**kwargs)
        if status != "ok":
            msg = f"Closest-point solve for target {dict(zip(keys, target_row.tolist()))} failed: status={status}, condition={condition}"
            raise RuntimeError(msg)

        point = n.optimize.project_solved(dimensions).loc[keys].to_numpy(dtype=float)
        normal = np.array([float(resid_constrs[key].dual) for key in keys])

        if not resolve_main_objective:
            return point, normal, None, n

        for key, value in zip(keys, point):
            m.add_constraints(
                n.optimize.build_linexpr_from_weights(dimensions[key], model=m).sum() == float(value),
                name=f"coord_{key}",
            )
        m.remove_constraints("budget")
        m.objective = main_obj
        status, condition = n.optimize.solve_model(**kwargs)
        if status != "ok":
            msg = f"Re-solve at pinned coordinates for target {dict(zip(keys, target_row.tolist()))} failed: status={status}, condition={condition}"
            raise RuntimeError(msg)
        duals = np.array([float(m.constraints[f"coord_{key}"].dual) for key in keys])

        # Undo this call's own additions so the model is ready for the next call.
        m.remove_constraints([f"coord_{key}" for key in keys])
        add_budget_constraint()
        m.objective = residual_obj

        return point, normal, duals, n

    return solve_one


#: This worker's own closest-point solver, built once by
#: _init_closest_point_worker and reused by _solve_closest_point_target for
#: every task this worker process handles. Also how n gets back to the
#: parent for caching -- solve_one returns (point, normal, duals, n).
_worker_closest_point_solver: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray | None, Network]] | None = None


def _init_closest_point_worker(
    n: Network,
    dimensions: dict,
    norm: str,
    slack: float,
    snapshots: Sequence | None,
    multi_investment_periods: bool,
    model_kwargs: dict | None,
    kwargs: dict,
    resolve_main_objective: bool,
) -> None:
    """Pool initializer for `make_closest_point_fn`'s batch path -- builds
    this worker's solver once via `_build_closest_point_solver`, before any
    tasks are dispatched.
    """
    _worker_init()
    global _worker_closest_point_solver
    _worker_closest_point_solver = _build_closest_point_solver(
        n, dimensions, norm, slack, snapshots, multi_investment_periods, model_kwargs, kwargs, resolve_main_objective,
    )


def _solve_closest_point_target(target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Network]:
    """One `pool.starmap` task: solve a single target against this
    worker's already-built solver (see `_init_closest_point_worker`).
    """
    return _worker_closest_point_solver(target)


def _cache_closest_point_result(
    n: Network, output_dir: str | Path, key: str, keys: list,
    point: np.ndarray, normal: np.ndarray, duals: np.ndarray | None, obj: float,
) -> None:
    """Stash `point`/`normal`/`duals`/`obj` into `n.meta` and export to
    `output_dir/mga_closest_result_{key}.nc` (a distinct prefix from
    `_cache_furthest_point_result`'s `mga_furthest_result_`, so both caches can
    safely share one `output_dir`). `duals` is only stashed (as
    `n.meta["duals"]`) when not `None` -- i.e. when `resolve_main_objective`
    actually computed it; otherwise `n.meta` has no `"duals"` key at all,
    rather than a placeholder.
    """
    n.meta["point"] = {k: float(v) for k, v in zip(keys, point)}
    n.meta["normal"] = {k: float(v) for k, v in zip(keys, normal)}
    if duals is not None:
        n.meta["duals"] = {k: float(v) for k, v in zip(keys, duals)}
    n.meta["obj"] = float(obj)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(str(output_dir / f"mga_closest_result_{key}.nc"))


def make_closest_point_fn(
    n: Network,
    dimensions: dict,
    norm: str,
    slack: float,
    output_dir: str | Path | None = None,
    decimals: int = 8,
    resolve_main_objective: bool = True,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    model_kwargs: dict | None = None,
    max_parallel: int = 4,
    **kwargs: Any,
) -> ClosestPointFn:
    """Build a `mga_surveyor.runner.ClosestPointFn` against a real,
    already-solved PyPSA network `n`: the closest point in `n`'s
    near-optimal feasible region to a given `target`, plus a valid
    separating-hyperplane normal there.

    Caching: pass a persistent `output_dir` to cache each query's result at
    `output_dir/mga_closest_result_{key}.nc`, content-addressed by the
    target itself (see `encode_cache_key`). `None` (default) disables
    caching. A cache hit on the single-target path skips solving entirely,
    so `n` is NOT mutated by a call that only reads a cache hit. Each cached
    `.nc` also stashes the found point's own capex+opex cost in
    `n.meta["obj"]`, same as `make_furthest_point_fn`'s cache.

    If `resolve_main_objective`: after finding the closest point, its
    coordinates are pinned and the network's original cost objective is
    re-solved to get real per-dimension duals, cached alongside
    `point`/`normal` in `n.meta["duals"]` -- same idea as
    `make_furthest_point_fn`'s own `resolve_main_objective`. This is extra
    work cached as a side effect only: `closest_point`'s own return value
    is unaffected, always `(point, normal)`, so it still matches
    `runner.ClosestPointFn`.

    :raises NotImplementedError: If `norm` is not `'l1'` or `'linf'`.
    :raises RuntimeError: If a call's solve does not reach `status="ok"`.
    """
    keys = list(dimensions.keys())
    # Snapshot n before any solve can mutate it -- the batch path's workers
    # need n's true original state for the budget constraint. A pickle
    # round-trip, not n.copy(), since Network.copy() refuses on a network
    # with an attached solver model.
    pristine_n = pickle.loads(pickle.dumps(n))

    solve_one = _build_closest_point_solver(
        n, dimensions, norm, slack, snapshots, multi_investment_periods, model_kwargs, kwargs, resolve_main_objective,
    )

    def closest_point(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shape-preserving: a `(k,)` target returns one `(point, normal)`
        pair; an `(n_targets, k)` batch returns `(n_targets, k)` arrays.

        Single target: read from `output_dir`'s cache if a hit, else
        solved in-process against the persistent model built above. Batch:
        cached rows read from disk, the rest dispatched one
        `pool.starmap` task per target across
        `min(max_parallel, n_todo)` worker processes, each building its own
        solver once.
        """
        target = np.asarray(target, dtype=float)
        single = target.ndim == 1
        target_2d = np.atleast_2d(target)

        if len(target_2d) == 1:
            t = target_2d[0]
            cached = None
            if output_dir is not None:
                key = encode_cache_key(t, decimals)
                cached_fn = Path(output_dir) / f"mga_closest_result_{key}.nc"
                if cached_fn.exists():
                    from pypsa.networks import Network as _Network  # noqa: PLC0415

                    cached_meta = _Network(cached_fn).meta
                    cached = (
                        np.array([cached_meta["point"][k] for k in keys]),
                        np.array([cached_meta["normal"][k] for k in keys]),
                    )
            if cached is not None:
                results = [cached]
            else:
                point, normal, duals, n_result = solve_one(t)
                if output_dir is not None:
                    obj = n_result.statistics.capex().sum() + n_result.statistics.opex().sum()
                    _cache_closest_point_result(n_result, output_dir, key, keys, point, normal, duals, obj)
                results = [(point, normal)]
        else:
            # Passes pristine_n, not n's current state -- n may already be
            # mutated by an earlier single-target call above.
            results = [None] * len(target_2d)
            todo_idx = list(range(len(target_2d)))
            cache_keys = None
            if output_dir is not None:
                cache_keys = [encode_cache_key(row, decimals) for row in target_2d]
                todo_idx = []
                for i, key in enumerate(cache_keys):
                    cached_fn = Path(output_dir) / f"mga_closest_result_{key}.nc"
                    if not cached_fn.exists():
                        todo_idx.append(i)
                        continue
                    from pypsa.networks import Network as _Network  # noqa: PLC0415

                    cached_meta = _Network(cached_fn).meta
                    results[i] = (
                        np.array([cached_meta["point"][k] for k in keys]),
                        np.array([cached_meta["normal"][k] for k in keys]),
                    )
                skipped = len(target_2d) - len(todo_idx)
                if skipped:
                    logger.info("Skipping %s already-cached target(s)", skipped)

            if todo_idx:
                todo_targets = target_2d[todo_idx]
                n_workers = min(max_parallel, len(todo_targets))
                with get_context("spawn").Pool(
                    processes=n_workers,
                    initializer=_init_closest_point_worker,
                    initargs=(
                        pristine_n, dimensions, norm, slack, snapshots,
                        multi_investment_periods, model_kwargs, kwargs, resolve_main_objective,
                    ),
                ) as pool:
                    try:
                        todo_results = pool.starmap(_solve_closest_point_target, [(row,) for row in todo_targets])
                    except Exception:
                        pool.terminate()
                        pool.join()
                        raise
                for i, (point, normal, duals, n_result) in zip(todo_idx, todo_results):
                    if output_dir is not None:
                        obj = n_result.statistics.capex().sum() + n_result.statistics.opex().sum()
                        _cache_closest_point_result(
                            n_result, output_dir, cache_keys[i], keys, point, normal, duals, obj,
                        )
                    results[i] = (point, normal)

        points = np.array([r[0] for r in results])
        normals = np.array([r[1] for r in results])

        if single:
            return points[0], normals[0]
        return points, normals

    return closest_point


def make_oracle_fns(
    n: Network,
    dimensions: dict,
    slack: float,
    norm: str,
    output_dir: str | Path,
    decimals: int = 8,
    resolve_main_objective: bool = True,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    model_kwargs: dict | None = None,
    max_parallel: int = 4,
    **kwargs: Any,
) -> tuple[Callable, ClosestPointFn, Callable[[], np.ndarray]]:
    """Build all three oracle callables `runner.run` needs in one call
    against a real, already-solved PyPSA network `n`, sharing
    `n`/`dimensions`/`output_dir` -- just calls `make_furthest_point_fn`,
    `make_closest_point_fn`, and `make_optimality_fn` with their common
    arguments and returns the three results.

    Each gets its own independent `pickle` copy of `n`, not `n` itself:
    `create_model()` stores its model at `n.model`, a single slot, so three
    factories built directly against one shared `n` would silently
    overwrite each other's model there, leaving at most one of the three
    actually usable.

    Returns `(furthest_point, closest_point, solve_optimal)`, in that
    order -- `runner.run`'s own parameter order.
    """
    furthest_point = make_furthest_point_fn(
        pickle.loads(pickle.dumps(n)), dimensions, slack,
        output_dir=output_dir, decimals=decimals, resolve_main_objective=resolve_main_objective,
        snapshots=snapshots, multi_investment_periods=multi_investment_periods,
        model_kwargs=model_kwargs, max_parallel=max_parallel, **kwargs,
    )
    closest_point = make_closest_point_fn(
        pickle.loads(pickle.dumps(n)), dimensions, norm, slack,
        output_dir=output_dir, decimals=decimals, resolve_main_objective=resolve_main_objective,
        snapshots=snapshots, multi_investment_periods=multi_investment_periods,
        model_kwargs=model_kwargs, max_parallel=max_parallel, **kwargs,
    )
    solve_optimal = make_optimality_fn(n, dimensions, output_dir, **kwargs)

    return furthest_point, closest_point, solve_optimal
