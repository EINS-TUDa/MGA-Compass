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
equivalent. Always solves fresh -- output_dir, if given, is export-only,
never read back to skip a solve. resolve_main_objective=True (default)
also pins that point's coordinates and re-solves the original objective
for real per-dimension duals, exported in n.meta but not part of the
return value (still just (point, normal), matching explore.ClosestPointFn);
pass False to skip that re-solve. A single-target call reuses one
persistent in-process model; a batch spreads targets across a
multiprocessing.Pool, each worker building its own solver once. Only
l1/linf supported (l2 needs a quadratic objective, Gurobi-only).

Both batch paths pass the PyPSA Network directly to worker processes
(picklable as of PyPSA 1.3.0, github.com/PyPSA/PyPSA/pull/1889) instead of
exporting to NetCDF and reloading. make_closest_point_fn's batch path also
snapshots n via pickle before any solve can mutate it, since
Network.copy() refuses on a network with an attached solver model.

make_oracle_fns builds all three of the above in one call, sharing
n/dimensions/output_dir -- a convenience for feeding explore.drive's three
callback parameters directly.

generate_mga_files runs a CddMethod per slack level (via
explore.drive) against a real network, writing every result to .nc files
under output_dir -- resuming from whatever's already there (cached
optimal/furthest-point solves are read back, and existing closest-point
results are folded into each slack level's own explore.drive call as its
extra_points/extra_halfspaces). consolidate_mga_results is the standalone
other half: given just an output_dir, it scans for every result .nc file
there and builds the consolidated points/duals DataFrames, independent of
a live run. Call both in sequence for the common case of running and then
reading the result back.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
import weakref
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from linopy import LinearExpression, QuadraticExpression
from pypsa._options import options
from pypsa.optimization.mga import _worker_init

from .common import Halfspace, Norm, gaps, linearized_objective_cut
from .explore import ClosestPointFn, drive
from .methods import CddMethod
from .normalize import AffineMap, normalize_oracle_fns

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pypsa import Network

logger = logging.getLogger(__name__)
# Explicit level so per-point progress logging (below) stays visible even
# though worker processes' _worker_init sets the root logger to WARNING.
logger.setLevel(logging.INFO)


def _log_solve(m: Any, status: str, condition: str, what: str) -> None:
    """One condensed line per successful solve -- status/condition are
    already returned by n.optimize.solve_model itself, and objective is
    read directly off the model (m.objective.value, a plain linopy
    Objective property set right after solving), not scraped from
    linopy.constants's own multi-line log block, which this replaces.
    Runtime/dual bound/solution size are skipped for now (not currently
    needed, and not available as a plain attribute the way objective is --
    they live on the solver's own report object -- so plumbing them
    through can wait until something actually needs them).
    """
    logger.info("%s solved: status=%s condition=%s objective=%.6e", what, status, condition, m.objective.value)


def _solver_kwargs_with_log(kwargs: dict, log_path: Path | None) -> dict:
    """`kwargs` unchanged if `log_path` is None or the active solver isn't
    Gurobi (HiGHS's own log-file option name isn't handled here, to avoid
    guessing wrong and silently breaking that fallback path) -- else a
    shallow copy with `solver_options["LogFile"]` set to `log_path`, so one
    solve writes its own log file (e.g. alongside its cached .nc result)
    without mutating the shared `kwargs` dict every other call also uses.
    Safe to call with the same `log_path` across multiple solves in a row
    (Gurobi keeps appending to an already-open log file when `LogFile` is
    set to the same path again) -- e.g. a furthest-point direction's
    primary solve and its own pinned-duals resolve share one file.
    """
    if log_path is None or kwargs.get("solver_name") != "gurobi":
        return kwargs
    # Gurobi's own LogFile param doesn't create missing directories --
    # output_dir/slack_X/ may not exist yet (its own mkdir is otherwise
    # deferred until a result is cached, after the solve), so this would
    # otherwise fail with GurobiError("Unable to write file") the first
    # time a slack level's own results directory hasn't been created yet.
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    solver_options = {**kwargs.get("solver_options", {}), "LogFile": str(log_path)}
    return {**kwargs, "solver_options": solver_options}


def _append_coordinates_to_log(log_path: Path | None, keys: list, point: np.ndarray) -> None:
    """Appends the primary solve's own coordinates to `log_path` (the same
    per-direction/target file Gurobi's own LogFile writes to, see
    `_solver_kwargs_with_log`) -- so, once the point that gets pinned in
    the pinned-duals resolve right after is known, it's sitting right next
    to that resolve's own solver log for easy correlation (e.g. whether a
    "Numerical trouble"/"may be infeasible" pinned-resolve failure lines up
    with a suspiciously large or otherwise odd coordinate value). No-op if
    `log_path` is None (output_dir disabled, so there's no file to write
    into -- matches _solver_kwargs_with_log's own no-op in that case).
    """
    if log_path is None:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    coordinates = {k: float(v) for k, v in zip(keys, point)}
    with open(log_path, "a") as f:
        f.write(f"\nCoordinates after primary solve: {coordinates}\n")


def _pin_coord_scale(value: float) -> float:
    """Scale factor for one pinned-coordinate equality constraint (a
    furthest/closest-point query's own found point, pinned exactly so its
    duals can be read off afterward) -- sqrt(|value|), same spirit as
    add_budget_constraint's own budget_scale. A pinned coordinate can be
    ~1e8 in magnitude (e.g. this network's own H2 storage dimension),
    dwarfing every other variable's natural coefficient range and
    breaking the pinned-duals resolve's own barrier solve with
    "Numerical trouble" -- confirmed on the real network, and confirmed
    NOT fixed by giving that solve an exact (crossover) vertex instead of
    an interior-point approximation, so the pinned value's sheer
    magnitude is the actual problem, not its precision. 1.0 for a
    (near-)zero value, since sqrt(0) would make the constraint degenerate
    (dividing by zero) and there's no large-magnitude problem to fix
    there anyway.
    """
    return abs(value) ** 0.5 if value else 1.0


#: Relative half-width of the band a pinned coordinate is allowed to sit in,
#: instead of an exact equality. An exact pin reliably broke the H2
#: dimension's pinned-duals resolve with Gurobi's "Numerical trouble
#: encountered" (status=warning, condition=other) -- confirmed NOT fixed by
#: BarConvTol tightening (1e-5/1e-6/1e-8), crossover on the primary solve,
#: or algebraic row-scaling of the same exact equality (_pin_coord_scale
#: alone): none of those change the underlying equation, so the same
#: degenerate feasible-set geometry remained. Relaxing to a band is the one
#: change that alters the feasible set itself, giving the barrier method
#: room to land somewhere solvable.
#: This value found by bisection on the real H2 case: 5e-4 and 1e-5 both
#: converged cleanly (status=ok, condition=optimal); 1e-6 (Gurobi's own
#: default FeasibilityTol) still completed but got demoted to
#: condition=suboptimal ("big uncrush violation"); 5e-6 converged cleanly
#: again. The true boundary sits between 1e-6 and 5e-6 -- 5e-6 kept as a
#: safety margin above it rather than cutting it exactly.
_PIN_TOLERANCE = 5e-6


def _add_pinned_coord_band(m: Any, n: Network, dimensions: dict, key: str, value: float, scale: float) -> None:
    """Pin dimension `key` to within `_PIN_TOLERANCE` of `value` (an
    inequality band: `coord_{key}_lb`/`coord_{key}_ub`) instead of an exact
    equality -- see `_PIN_TOLERANCE`'s own docstring for why."""
    expr = n.optimize.build_linexpr_from_weights(dimensions[key], model=m).sum()
    lo = min(value * (1 - _PIN_TOLERANCE), value * (1 + _PIN_TOLERANCE))
    hi = max(value * (1 - _PIN_TOLERANCE), value * (1 + _PIN_TOLERANCE))
    m.add_constraints(expr / scale >= lo / scale, name=f"coord_{key}_lb")
    m.add_constraints(expr / scale <= hi / scale, name=f"coord_{key}_ub")


def _read_pinned_coord_dual(m: Any, key: str, scale: float) -> float:
    """Dual of a `_add_pinned_coord_band` pin -- only one side of the band
    is ever binding at the optimum (the other's dual is 0), so summing both
    recovers exactly that one side's sensitivity. Divided back out by
    `scale` for the same reason `_pin_coord_scale`'s callers do."""
    dual_lb = float(m.constraints[f"coord_{key}_lb"].dual)
    dual_ub = float(m.constraints[f"coord_{key}_ub"].dual)
    return (dual_lb + dual_ub) / scale


__all__ = [
    "make_optimality_fn",
    "make_furthest_point_fn",
    "make_closest_point_fn",
    "make_oracle_fns",
    "generate_mga_files",
    "consolidate_mga_results",
    "encode_cache_key",
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
    `resolve_main_objective`. The optimum's own cost is cached too, in
    `n.meta["obj"]`, same as every other result file this module writes --
    so `consolidate_mga_results` never needs to recompute it. `dimensions`
    itself is cached in `n.meta["dimensions"]` too -- the one piece of
    per-run metadata every other `.nc` file in `output_dir` needs to be
    interpreted, so `consolidate_mga_results` can read it back from here
    instead of needing it passed in again.

    If `output_dir/optimal.nc` already exists, its point is read back from
    `n.meta["coordinates"]` (stashed there on save, same as everywhere else
    in this module) instead of re-solving.

    :raises RuntimeError: If a solve does not reach `status="ok"`.
    """
    keys = list(dimensions.keys())
    optimal_fn = Path(output_dir) / "optimal.nc"
    # Same idea as mga_furthest_result_{key}.log/mga_closest_result_{key}.log
    # (see _solver_kwargs_with_log) -- optimal.nc's own solve gets its own
    # log file too, right next to it.
    log_path = Path(output_dir) / "optimal.log"

    def solve_optimal() -> np.ndarray:
        if optimal_fn.exists():
            cached_coordinates = _read_meta(optimal_fn)["coordinates"]
            return np.array([cached_coordinates[k] for k in keys])

        call_kwargs = _solver_kwargs_with_log(kwargs, log_path)
        m = n.optimize.create_model(include_objective_constant=True)
        status, condition = n.optimize.solve_model(**call_kwargs)
        if status != "ok":
            msg = f"Solve to optimality failed: status={status}, condition={condition}"
            raise RuntimeError(msg)
        _log_solve(m, status, condition, "optimal")
        point = n.optimize.project_solved(dimensions).loc[keys].to_numpy(dtype=float)
        obj = n.statistics.capex().sum() + n.statistics.opex().sum()
        _append_coordinates_to_log(log_path, keys, point)

        for key, value in zip(keys, point):
            m.add_constraints(
                n.optimize.build_linexpr_from_weights(dimensions[key], model=m).sum() == float(value),
                name=f"coord_{key}",
            )
        status, condition = n.optimize.solve_model(**call_kwargs)
        if status != "ok":
            msg = f"Re-solve at pinned optimal coordinates failed: status={status}, condition={condition}"
            raise RuntimeError(msg)
        _log_solve(m, status, condition, "optimal (pinned duals)")
        duals = np.array([float(m.constraints[f"coord_{key}"].dual) for key in keys])

        n.meta["coordinates"] = {k: float(v) for k, v in zip(keys, point)}
        n.meta["duals"] = {k: float(v) for k, v in zip(keys, duals)}
        n.meta["obj"] = float(obj)
        n.meta["dimensions"] = dimensions

        optimal_fn.parent.mkdir(parents=True, exist_ok=True)
        n.export_to_netcdf(str(optimal_fn))
        logger.info("saved to file: %s", optimal_fn.name)
        return point

    return solve_optimal


def encode_cache_key(vec: np.ndarray) -> str:
    """Filename-safe cache key for a numeric vector: SHA-256 of its raw
    float64 bytes, truncated to 10 hex characters (40 bits -- collision
    risk is negligible at the scale of a project's cached files). Not
    reversible -- unlike a rounding-based encoding, this hashes the exact
    bytes, so it's only meant for vectors that are exactly reproducible
    bit-for-bit across calls (e.g. `np.eye`-derived directions), not ones
    that might differ by float noise.

    `-0.0` is normalized to `0.0` first, since IEEE 754 gives it a
    different bit pattern despite being numerically equal to `0.0` (e.g.
    `-np.eye(k)`'s off-diagonal entries).
    """
    normalized = np.asarray(vec, dtype=np.float64) + 0.0
    return hashlib.sha256(normalized.tobytes()).hexdigest()[:10]


def _read_meta(nc_path: str | Path) -> dict:
    """Read just a result `.nc` file's `meta` dict, via `netCDF4` directly
    on the file's own JSON-encoded global attribute -- not by instantiating
    a full `pypsa.Network`, which parses every component/snapshot into
    pandas DataFrames just to reach a handful of scalars PyPSA itself only
    ever stashed as a JSON string in the first place (~1000x slower in
    practice: full-network loads dominate the runtime of anything reading
    back many cached files, e.g. `consolidate_mga_results` or a
    `generate_mga_files` restart's `_read_existing_closest_points`)."""
    from netCDF4 import Dataset  # noqa: PLC0415

    with Dataset(nc_path) as ds:
        return json.loads(ds.meta)


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
    # Explicit, not left to PyPSA's own default -- silences its own
    # FutureWarning about that default changing in 2.0, and keeps this
    # model's objective value meaning stable regardless of the PyPSA
    # version installed (add_budget_constraint below compares m.objective
    # directly against optimal_cost, so a silent default flip would change
    # what "slack" means without anything here noticing).
    model_kwargs.setdefault("include_objective_constant", True)
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

    # A real network's optimal_cost (capex+opex, annual) can be ~1e11.
    # scaled_objective ties to main_obj ALONE (no fixed_cost, no slack/
    # optimal_cost added in) via one equality, so that equality never
    # embeds any large constant -- unlike a prior attempt that added
    # fixed_cost inside the equality itself (`objective + fixed_cost ==
    # scaled_obj * budget_scale`), which just relocated the same
    # ~1e11-magnitude constant into a different constraint. budget_rhs
    # here is plain Python arithmetic -- computed once, in full
    # double-precision float, before it ever becomes a linopy expression
    # -- so the only number handed to the solver for this constraint is
    # the final, already-small result, never a raw ~1e11 constant.
    budget_scale = optimal_cost**0.5
    main_obj_expr = main_obj if isinstance(main_obj, (LinearExpression, QuadraticExpression)) else main_obj.expression
    scaled_objective = m.add_variables(lower=-np.inf, name="mga_surveyor_scaled_objective")
    m.add_constraints(main_obj_expr == scaled_objective * budget_scale, name="scaled_objective_def")
    budget_rhs = ((1 + slack) * optimal_cost - fixed_cost) / budget_scale

    def add_budget_constraint() -> None:
        m.add_constraints(scaled_objective <= budget_rhs, name="budget")

    add_budget_constraint()

    def solve_one(
        direction_row: np.ndarray, log_path: Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, Network, str]:
        call_kwargs = _solver_kwargs_with_log(kwargs, log_path)
        m.objective = -sum(
            direction_row[i] * n.optimize.build_linexpr_from_weights(dimensions[key], model=m)
            for i, key in enumerate(keys)
        )
        status, condition = n.optimize.solve_model(**call_kwargs)
        if status != "ok":
            msg = (
                f"Furthest-point solve for direction {dict(zip(keys, direction_row.tolist()))} "
                f"failed: status={status}, condition={condition}"
            )
            raise RuntimeError(msg)
        _log_solve(m, status, condition, "furthest-point")
        coordinates = n.optimize.project_solved(dimensions)
        point = coordinates.loc[keys].to_numpy(dtype=float)
        _append_coordinates_to_log(log_path, keys, point)

        if not resolve_main_objective:
            return point, None, n, condition

        pinned_keys = [key for i, key in enumerate(keys) if direction_row[i] != 0]
        pin_scales = {}
        for key in pinned_keys:
            value = coordinates[key].item()
            scale = _pin_coord_scale(value)
            pin_scales[key] = scale
            _add_pinned_coord_band(m, n, dimensions, key, value, scale)
        m.remove_constraints("budget")
        m.objective = main_obj
        status, condition = n.optimize.solve_model(**call_kwargs)
        if status != "ok":
            msg = (
                f"Re-solve at pinned coordinates for direction {dict(zip(keys, direction_row.tolist()))} "
                f"failed: status={status}, condition={condition}"
            )
            raise RuntimeError(msg)
        _log_solve(m, status, condition, "furthest-point (pinned duals)")
        duals_by_key = {key: _read_pinned_coord_dual(m, key, pin_scales[key]) for key in pinned_keys}
        duals = np.array([duals_by_key.get(key, 0.0) for key in keys])

        # Undo this call's own additions so the model is ready for the next call.
        if pinned_keys:
            m.remove_constraints([f"coord_{key}_lb" for key in pinned_keys])
            m.remove_constraints([f"coord_{key}_ub" for key in pinned_keys])
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


def _solve_furthest_point_direction(
    item: tuple[int, np.ndarray, Path | None],
) -> tuple[int, np.ndarray, np.ndarray | None, Network, str]:
    """One `pool.imap_unordered` task: solve a single direction against
    this worker's already-built solver (see `_init_furthest_point_worker`).
    Carries `i` (its position in this batch's own todo-list) through so
    the caller can cache/report each direction's result as soon as it
    finishes, in whatever order workers actually complete them in --
    not only after every direction in the batch is done. `log_path`, if
    given, is where this direction's own solver log gets written (see
    `_solver_kwargs_with_log`).
    """
    i, direction, log_path = item
    point, duals, n, condition = _worker_furthest_point_solver(direction, log_path)
    return i, point, duals, n, condition


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
    out_fn = output_dir / f"mga_furthest_result_{key}.nc"
    n.export_to_netcdf(str(out_fn))
    logger.info("saved to file: %s", out_fn.name)


def make_furthest_point_fn(
    n: Network,
    dimensions: dict,
    slack: float,
    output_dir: str | Path | None = None,
    resolve_main_objective: bool = True,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    model_kwargs: dict | None = None,
    max_parallel: int = 4,
    on_solve: Callable[[np.ndarray, np.ndarray, float, np.ndarray | None], None] | None = None,
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
    callers -- e.g. `explore.drive` -- don't need to know which was used).
    Pass `False` to skip that extra solve.

    `dimensions`' key order fixes the axis order of `directions`/`points`/
    `normals`.

    Caching: pass a persistent `output_dir` to cache each direction's
    result at `output_dir/mga_furthest_result_{key}.nc`, content-addressed
    by the direction itself (see `encode_cache_key`). `None` (default)
    disables caching. A cache hit on the single-direction path skips
    solving entirely, so `n` is NOT mutated by a call that only reads a
    cache hit.

    `on_solve(point, normal, obj, duals)` (default `None`): called once per
    direction actually reported here -- fresh solve or cache hit alike --
    with data that's otherwise discarded from the return value (`obj`,
    `duals`; `duals` is `None` when that direction's own
    `resolve_main_objective` was `False`, cached or not). Lets a caller
    (e.g. `generate_mga_files`'s `use_objective_cuts`) tap into the same
    solve without changing `furthest_points`' own `(points, normals)`
    contract. Always called back in this process, never inside a worker --
    the batch path's Pool results are collected here first.

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
                key = encode_cache_key(d)
                cached_fn = Path(output_dir) / f"mga_furthest_result_{key}.nc"
                if cached_fn.exists():
                    cached_meta = _read_meta(cached_fn)
                    cached = (
                        np.array([cached_meta["coordinates"][k] for k in keys]),
                        np.array([cached_meta["normal"][k] for k in keys]),
                    )
                    if on_solve is not None:
                        cached_duals = np.array([cached_meta["duals"][k] for k in keys]) if "duals" in cached_meta else None
                        on_solve(cached[0], cached[1], cached_meta["obj"], cached_duals)
            if cached is not None:
                results = [cached]
            else:
                log_path = Path(output_dir) / f"mga_furthest_result_{key}.log" if output_dir is not None else None
                point, duals, n_result, _condition = solve_one(d, log_path)
                obj = n_result.statistics.capex().sum() + n_result.statistics.opex().sum()
                if output_dir is not None:
                    _cache_furthest_point_result(n_result, output_dir, key, keys, point, duals, obj, d)
                if on_solve is not None:
                    on_solve(point, d, obj, duals)
                results = [(point, d)]
        else:
            # Passes pristine_n, not n's current state -- n may already be
            # mutated by an earlier single-direction call above.
            results = [None] * len(directions)
            todo_idx = list(range(len(directions)))
            cache_keys = None
            if output_dir is not None:
                cache_keys = [encode_cache_key(row) for row in directions]
                todo_idx = []
                for i, key in enumerate(cache_keys):
                    cached_fn = Path(output_dir) / f"mga_furthest_result_{key}.nc"
                    if not cached_fn.exists():
                        todo_idx.append(i)
                        continue
                    cached_meta = _read_meta(cached_fn)
                    results[i] = (
                        np.array([cached_meta["coordinates"][k] for k in keys]),
                        np.array([cached_meta["normal"][k] for k in keys]),
                    )
                    if on_solve is not None:
                        cached_duals = np.array([cached_meta["duals"][k] for k in keys]) if "duals" in cached_meta else None
                        on_solve(results[i][0], results[i][1], cached_meta["obj"], cached_duals)
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
                    tasks = [
                        (
                            pos, row,
                            Path(output_dir) / f"mga_furthest_result_{cache_keys[todo_idx[pos]]}.log"
                            if output_dir is not None else None,
                        )
                        for pos, row in enumerate(todo_directions)
                    ]
                    try:
                        # imap_unordered, not starmap -- yields each direction's
                        # result as soon as ITS worker finishes (in completion
                        # order, not submission order), so the cache/on_solve
                        # below runs per-direction instead of only after every
                        # direction in this batch is done (which could hide a
                        # long-running run's progress for the entire batch's
                        # duration otherwise).
                        for pos, point, duals, n_result, _condition in pool.imap_unordered(
                            _solve_furthest_point_direction, tasks,
                        ):
                            i = todo_idx[pos]
                            obj = n_result.statistics.capex().sum() + n_result.statistics.opex().sum()
                            if output_dir is not None:
                                _cache_furthest_point_result(
                                    n_result, output_dir, cache_keys[i], keys, point, duals, obj, directions[i],
                                )
                            if on_solve is not None:
                                on_solve(point, directions[i], obj, duals)
                            results[i] = (point, directions[i])
                    except Exception:
                        pool.terminate()
                        pool.join()
                        raise

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
    weights: dict[str, float] | None = None,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray | None, Network]]:
    """Build the closest-point model against `n` once, and return a
    single-target `(point, normal, duals, n)` solver reusing it.

    One residual variable per dimension (`r_key + dim_key(x) == target_key`,
    RHS updated per call) plus a norm-specific objective: `t >= |w_key *
    r_key|` for `linf`, `sum(s_key)` with `s_key >= |w_key * r_key|` for
    `l1` -- `weights` (default all-ones, i.e. the plain unweighted norm)
    lets a caller make "closest" mean closest under a per-dimension-scaled
    metric instead of raw real units (see `mga_surveyor.normalize` for
    why: solving with `weights = 1 / half_range` makes this equivalent to
    solving directly in a `[-1, 1]^k`-normalized space). `normal` is the
    dual of the residual constraints themselves -- still a valid
    real-unit separating-hyperplane normal whatever `weights` is (its
    value reflects whichever objective -- weighted or not -- the model
    actually has when it's read).

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
    if weights is None:
        weights = dict.fromkeys(keys, 1.0)
    # Solve with weights uniformly rescaled by a positive constant --
    # mathematically a no-op (minimizing C*t(x) has the same argmin as
    # minimizing t(x) for any C>0, so `point` is unchanged), but it moves
    # resid_constrs[key].dual's absolute magnitude by that same factor C
    # -- which matters because that dual is otherwise proportional to
    # weights[key] itself (see solve_one's own normal derivation), and
    # normalize's weights (1/half_range) can be tiny (e.g. ~1e-7) for a
    # wide-range dimension. At a degenerate optimum (multiple dimensions
    # simultaneously tied at the L-inf/L1 max -- confirmed common in
    # practice, not rare), the solver's reported dual there satisfies
    # complementary slackness only to within its own optimality
    # tolerance, not exactly; when that dual's own magnitude is only
    # ~1-2 orders of magnitude above the solver's tolerance, this
    # otherwise-negligible imprecision is enough to make the resulting
    # geometric cut genuinely invalid (confirmed against the real
    # network: a cut derived this way excluded a provably feasible
    # point). Choosing the scale so the *smallest* weight lands at
    # exactly 1 -- i.e. C = 1 / min(weights) -- self-adapts to whatever
    # real-world units this network happens to use, giving every
    # dimension's weight comfortable headroom above the solver's
    # tolerance floor regardless of scale, without touching solver
    # tolerance settings (confirmed empirically: fixes the real failing
    # case, stable across C from 1/min(weights) up through 10000x
    # larger, so this isn't a narrow tuned value). `normal` is divided
    # back down by this same C in solve_one before being returned, so
    # this is purely an internal precision improvement -- the point/
    # normal contract callers see is completely unchanged.
    _dual_scale = 1.0 / min(weights.values())
    solve_weights = {key: value * _dual_scale for key, value in weights.items()}
    if model_kwargs is None:
        model_kwargs = options.params.optimize.model_kwargs.copy()
    # Explicit, not left to PyPSA's own default -- silences its own
    # FutureWarning about that default changing in 2.0, and keeps this
    # model's objective value meaning stable regardless of the PyPSA
    # version installed (add_budget_constraint below compares m.objective
    # directly against optimal_cost, so a silent default flip would change
    # what "slack" means without anything here noticing).
    model_kwargs.setdefault("include_objective_constant", True)
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

    # A real network's optimal_cost (capex+opex, annual) can be ~1e11.
    # scaled_objective ties to main_obj ALONE (no fixed_cost, no slack/
    # optimal_cost added in) via one equality, so that equality never
    # embeds any large constant -- unlike a prior attempt that added
    # fixed_cost inside the equality itself (`objective + fixed_cost ==
    # scaled_obj * budget_scale`), which just relocated the same
    # ~1e11-magnitude constant into a different constraint. budget_rhs
    # here is plain Python arithmetic -- computed once, in full
    # double-precision float, before it ever becomes a linopy expression
    # -- so the only number handed to the solver for this constraint is
    # the final, already-small result, never a raw ~1e11 constant.
    budget_scale = optimal_cost**0.5
    main_obj_expr = main_obj if isinstance(main_obj, (LinearExpression, QuadraticExpression)) else main_obj.expression
    scaled_objective = m.add_variables(lower=-np.inf, name="mga_surveyor_scaled_objective")
    m.add_constraints(main_obj_expr == scaled_objective * budget_scale, name="scaled_objective_def")
    budget_rhs = ((1 + slack) * optimal_cost - fixed_cost) / budget_scale

    def add_budget_constraint() -> None:
        m.add_constraints(scaled_objective <= budget_rhs, name="budget")

    add_budget_constraint()

    resid_constrs = {}
    for key in keys:
        r = m.add_variables(lower=-np.inf, name=f"mga_surveyor_resid_{key}")
        dim_expr = n.optimize.build_linexpr_from_weights(dimensions[key], model=m).sum()
        resid_constrs[key] = m.add_constraints(r + dim_expr == 0.0, name=f"mga_surveyor_resid_constr_{key}")

    if norm == "linf":
        t = m.add_variables(lower=0.0, name="mga_surveyor_closest_point_t")
        for key in keys:
            r = solve_weights[key] * m.variables[f"mga_surveyor_resid_{key}"]
            m.add_constraints(t - r >= 0.0, name=f"mga_surveyor_tpos_{key}")
            m.add_constraints(t + r >= 0.0, name=f"mga_surveyor_tneg_{key}")
        m.objective = 1 * t  # bare Variable isn't a valid objective -- coerce to LinearExpression
    else:  # l1
        s_vars = []
        for key in keys:
            r = solve_weights[key] * m.variables[f"mga_surveyor_resid_{key}"]
            s = m.add_variables(lower=0.0, name=f"mga_surveyor_s_{key}")
            m.add_constraints(s - r >= 0.0, name=f"mga_surveyor_spos_{key}")
            m.add_constraints(s + r >= 0.0, name=f"mga_surveyor_sneg_{key}")
            s_vars.append(s)
        m.objective = sum(s_vars)
    residual_obj = m.objective

    def solve_one(
        target_row: np.ndarray, log_path: Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Network]:
        solve_start = time.perf_counter()
        call_kwargs = _solver_kwargs_with_log(kwargs, log_path)
        for key, target_value in zip(keys, target_row):
            resid_constrs[key].update(rhs=float(target_value))

        status, condition = n.optimize.solve_model(**call_kwargs)
        if status != "ok":
            msg = f"Closest-point solve for target {dict(zip(keys, target_row.tolist()))} failed: status={status}, condition={condition}"
            raise RuntimeError(msg)
        _log_solve(m, status, condition, "closest-point")

        point = n.optimize.project_solved(dimensions).loc[keys].to_numpy(dtype=float)
        _append_coordinates_to_log(log_path, keys, point)
        # Divide back out the internal solve-time rescaling (see
        # _dual_scale's own comment above) -- resid_constrs[key].dual was
        # computed against solve_weights (C times the real weights), and
        # a dual scales exactly linearly with its objective's own overall
        # scale, so dividing by that same C recovers the real-weights
        # dual exactly (mathematically), just computed with better
        # numerical conditioning than solving at the tiny raw scale
        # would have given.
        normal = np.array([float(resid_constrs[key].dual) for key in keys]) / _dual_scale

        if not resolve_main_objective:
            logger.info("closest-point solve time: %.3fs", time.perf_counter() - solve_start)
            return point, normal, None, n

        pin_scales = {}
        for key, value in zip(keys, point):
            value = float(value)
            scale = _pin_coord_scale(value)
            pin_scales[key] = scale
            _add_pinned_coord_band(m, n, dimensions, key, value, scale)
        m.remove_constraints("budget")
        m.objective = main_obj
        status, condition = n.optimize.solve_model(**call_kwargs)
        if status != "ok":
            msg = f"Re-solve at pinned coordinates for target {dict(zip(keys, target_row.tolist()))} failed: status={status}, condition={condition}"
            raise RuntimeError(msg)
        _log_solve(m, status, condition, "closest-point (pinned duals)")
        duals = np.array([_read_pinned_coord_dual(m, key, pin_scales[key]) for key in keys])

        # Undo this call's own additions so the model is ready for the next call.
        m.remove_constraints([f"coord_{key}_lb" for key in keys])
        m.remove_constraints([f"coord_{key}_ub" for key in keys])
        add_budget_constraint()
        m.objective = residual_obj

        logger.info("closest-point solve time: %.3fs", time.perf_counter() - solve_start)
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
    weights: dict[str, float] | None = None,
) -> None:
    """Pool initializer for `make_closest_point_fn`'s batch path -- builds
    this worker's solver once via `_build_closest_point_solver`, before any
    tasks are dispatched.
    """
    _worker_init()
    global _worker_closest_point_solver
    _worker_closest_point_solver = _build_closest_point_solver(
        n, dimensions, norm, slack, snapshots, multi_investment_periods, model_kwargs, kwargs, resolve_main_objective,
        weights,
    )


def _solve_closest_point_target(
    item: tuple[int, np.ndarray, Path | None],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray | None, Network]:
    """One `pool.imap_unordered` task: solve a single target against this
    worker's already-built solver (see `_init_closest_point_worker`).
    Carries `i` (its position in this batch) through -- see
    `_solve_furthest_point_direction`'s own docstring for why. `log_path`,
    if given, is where this target's own solver log gets written (see
    `_solver_kwargs_with_log`).
    """
    i, target, log_path = item
    point, normal, duals, n = _worker_closest_point_solver(target, log_path)
    return i, point, normal, duals, n


def _cache_closest_point_result(
    n: Network, output_dir: str | Path, key: str, keys: list,
    point: np.ndarray, normal: np.ndarray, duals: np.ndarray | None, obj: float, target: np.ndarray,
) -> None:
    """Stash `point`/`normal`/`duals`/`obj`/`target` into `n.meta` and export
    to `output_dir/mga_closest_result_{key}.nc` (a distinct prefix from
    `_cache_furthest_point_result`'s `mga_furthest_result_`, so both caches can
    safely share one `output_dir`). `duals` is only stashed (as
    `n.meta["duals"]`) when not `None` -- i.e. when `resolve_main_objective`
    actually computed it; otherwise `n.meta` has no `"duals"` key at all,
    rather than a placeholder.

    `target` (the point this solve was actually asked to get closest to,
    i.e. `explore.drive`'s own trial point) is stashed alongside `point`
    so `_read_existing_closest_points` can, on a later restart, replicate
    `explore.drive`'s own `cut_added = norm(new_point - trial_point) >
    tol` check before trusting this solve's `normal` as a halfspace --
    without it, a solve whose closest point came back (near-)equal to its
    own target (nothing separated -- typically a highly degenerate LP,
    every dimension tied near a zero residual at once) would otherwise be
    replayed as a supporting halfspace it never actually was, silently
    corrupting the outer approximation on restart. See that function's
    own docstring for the real-network case this was found from.
    """
    n.meta["point"] = {k: float(v) for k, v in zip(keys, point)}
    n.meta["normal"] = {k: float(v) for k, v in zip(keys, normal)}
    if duals is not None:
        n.meta["duals"] = {k: float(v) for k, v in zip(keys, duals)}
    n.meta["obj"] = float(obj)
    n.meta["target"] = {k: float(v) for k, v in zip(keys, target)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_fn = output_dir / f"mga_closest_result_{key}.nc"
    n.export_to_netcdf(str(out_fn))
    logger.info("saved to file: %s", out_fn.name)


def _terminate_pool_box(pool_box: dict) -> None:
    """weakref.finalize callback for make_closest_point_fn's own batch-path
    pool (see its own docstring for why it's kept alive across calls
    instead of recreated every time) -- takes the mutable box, not the
    pool itself or anything referencing the closest_point closure being
    finalized, since a finalize callback holding a strong reference back
    to its own target would keep it alive forever, defeating the point.
    """
    pool = pool_box.get("pool")
    if pool is not None:
        pool.terminate()
        pool_box["pool"] = None


def make_closest_point_fn(
    n: Network,
    dimensions: dict,
    norm: str,
    slack: float,
    output_dir: str | Path | None = None,
    resolve_main_objective: bool = True,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    model_kwargs: dict | None = None,
    max_parallel: int = 4,
    weights: dict[str, float] | None = None,
    on_solve: Callable[[np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray], None] | None = None,
    **kwargs: Any,
) -> ClosestPointFn:
    """Build a `mga_surveyor.explore.ClosestPointFn` against a real,
    already-solved PyPSA network `n`: the closest point in `n`'s
    near-optimal feasible region to a given `target`, plus a valid
    separating-hyperplane normal there.

    Export only, not a cache: pass a persistent `output_dir` and every call
    solves fresh and then writes its result to
    `output_dir/mga_closest_result_{key}.nc`, content-addressed by the
    target itself (see `encode_cache_key`) -- but an existing file there is
    never read back or used to skip solving; re-running with the same
    target just re-solves and overwrites it. `None` (default) disables
    exporting entirely. Each exported `.nc` also stashes the found point's
    own capex+opex cost in `n.meta["obj"]`, same as `make_furthest_point_fn`'s
    own export.

    `weights` (default `None`, i.e. every dimension weighted `1.0` --
    the plain unweighted norm): scales each dimension's residual in the
    closest-point objective before it enters the L1/Linf norm, so
    "closest" means closest under a per-dimension-scaled metric instead of
    raw real units -- see `mga_surveyor.normalize.normalize_oracle_fns`,
    which is what actually needs this (pass its `AffineMap.weights` here
    to make the point found consistent with the cube-normalized geometry
    `explore.drive` is exploring). Points/normals returned are still
    always in real units regardless of `weights` -- only the objective
    used to find them changes.

    If `resolve_main_objective`: after finding the closest point, its
    coordinates are pinned and the network's original cost objective is
    re-solved to get real per-dimension duals, cached alongside
    `point`/`normal` in `n.meta["duals"]` -- same idea as
    `make_furthest_point_fn`'s own `resolve_main_objective`. This is extra
    work cached as a side effect only: `closest_point`'s own return value
    is unaffected, always `(point, normal)`, so it still matches
    `explore.ClosestPointFn`.

    `on_solve(point, normal, obj, duals, target)` (default `None`): called
    once per target actually solved -- single or each one of a batch --
    with data that's otherwise discarded from the return value (`obj`,
    `duals`; `duals` is `None` when `resolve_main_objective` is `False`),
    plus the `target` that was queried (unlike `make_furthest_point_fn`'s
    own `on_solve`, which has no equivalent -- a support-function query
    has no "target" a found point could be compared against). Always
    called back in this process, never inside a worker.

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
        weights,
    )

    # Lazily built on the first BATCH call (never for single-target calls,
    # which reuse solve_one above and never need a Pool at all) and reused
    # for every later batch call from this same closest_point, instead of
    # recreated per call -- a fresh Pool means a fresh
    # _init_closest_point_worker per worker, i.e. a fresh
    # _build_closest_point_solver/create_model, which can dominate an
    # actual solve's own cost. Since explore.drive's max_parallel-driven
    # pooling feature calls closest_point with a batch once per iteration,
    # recreating the Pool every time would pay that full model-build cost
    # on every single round instead of once for the whole run. Grown (old
    # pool terminated, new one started) if a later call needs more
    # workers than it was originally sized for -- e.g. a smaller final
    # round doesn't shrink it, but max_parallel changing between calls
    # could grow it. Terminated by
    # _terminate_pool_box via weakref.finalize below, tied to
    # closest_point's own lifetime, since ClosestPointFn has no explicit
    # "done with this" hook of its own for an ordinary caller to use.
    pool_box: dict[str, Any] = {"pool": None, "n_workers": 0}

    def get_batch_pool(n_workers: int):
        pool = pool_box["pool"]
        if pool is not None and pool_box["n_workers"] < n_workers:
            pool.terminate()
            pool = None
        if pool is None:
            pool = get_context("spawn").Pool(
                processes=n_workers,
                initializer=_init_closest_point_worker,
                initargs=(
                    pristine_n, dimensions, norm, slack, snapshots,
                    multi_investment_periods, model_kwargs, kwargs, resolve_main_objective, weights,
                ),
            )
            pool_box["pool"] = pool
            pool_box["n_workers"] = n_workers
        return pool

    def closest_point(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shape-preserving: a `(k,)` target returns one `(point, normal)`
        pair; an `(n_targets, k)` batch returns `(n_targets, k)` arrays.
        Always solves -- no existing `output_dir` file is ever read back.

        Single target: solved in-process against the persistent model
        built above. Batch: one task per target, spread across
        `min(max_parallel, n_targets)` worker processes -- a persistent
        Pool (see `get_batch_pool` above), each worker building its own
        solver once for as long as this `closest_point` itself lives, not
        once per batch call.
        """
        target = np.asarray(target, dtype=float)
        single = target.ndim == 1
        target_2d = np.atleast_2d(target)

        if len(target_2d) == 1:
            t = target_2d[0]
            key = encode_cache_key(t) if output_dir is not None else None
            log_path = Path(output_dir) / f"mga_closest_result_{key}.log" if output_dir is not None else None
            if log_path is not None:
                logger.info("closest-point target given: solving, log at %s", log_path.name)
            point, normal, duals, n_result = solve_one(t, log_path)
            if output_dir is not None or on_solve is not None:
                obj = n_result.statistics.capex().sum() + n_result.statistics.opex().sum()
            if output_dir is not None:
                _cache_closest_point_result(n_result, output_dir, key, keys, point, normal, duals, obj, t)
            if on_solve is not None:
                on_solve(point, normal, obj, duals, t)
            results = [(point, normal)]
        else:
            # Passes pristine_n, not n's current state -- n may already be
            # mutated by an earlier single-target call above.
            n_workers = min(max_parallel, len(target_2d))
            pool = get_batch_pool(n_workers)
            results = [None] * len(target_2d)
            tasks = [
                (
                    i, row,
                    Path(output_dir) / f"mga_closest_result_{encode_cache_key(row)}.log"
                    if output_dir is not None else None,
                )
                for i, row in enumerate(target_2d)
            ]
            try:
                # imap_unordered, not starmap -- see
                # _solve_furthest_point_direction's own docstring for why:
                # caches/reports each target's result as soon as ITS
                # worker finishes, not only after the whole batch is done.
                for i, point, normal, duals, n_result in pool.imap_unordered(
                    _solve_closest_point_target, tasks,
                ):
                    row = target_2d[i]
                    if output_dir is not None or on_solve is not None:
                        obj = n_result.statistics.capex().sum() + n_result.statistics.opex().sum()
                    if output_dir is not None:
                        key = encode_cache_key(row)
                        _cache_closest_point_result(n_result, output_dir, key, keys, point, normal, duals, obj, row)
                    if on_solve is not None:
                        on_solve(point, normal, obj, duals, row)
                    results[i] = (point, normal)
            except Exception:
                # Unlike the old per-call Pool, this one is meant to
                # survive past this call -- but not in a possibly-broken
                # state after an unhandled worker exception, so it's
                # torn down here rather than kept for reuse. get_batch_pool
                # builds a fresh one on the next call, same as if this had
                # been the first call.
                pool.terminate()
                pool.join()
                pool_box["pool"] = None
                raise

        points = np.array([r[0] for r in results])
        normals = np.array([r[1] for r in results])

        if single:
            return points[0], normals[0]
        return points, normals

    weakref.finalize(closest_point, _terminate_pool_box, pool_box)
    return closest_point


def make_oracle_fns(
    n: Network,
    dimensions: dict,
    slack: float,
    norm: str,
    output_dir: str | Path,
    resolve_main_objective: bool = True,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    model_kwargs: dict | None = None,
    max_parallel: int = 4,
    **kwargs: Any,
) -> tuple[Callable, ClosestPointFn, Callable[[], np.ndarray]]:
    """Build all three oracle callables `explore.drive` needs in one call
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
    order -- `explore.drive`'s own parameter order.
    """
    furthest_point = make_furthest_point_fn(
        pickle.loads(pickle.dumps(n)), dimensions, slack,
        output_dir=output_dir, resolve_main_objective=resolve_main_objective,
        snapshots=snapshots, multi_investment_periods=multi_investment_periods,
        model_kwargs=model_kwargs, max_parallel=max_parallel, **kwargs,
    )
    closest_point = make_closest_point_fn(
        pickle.loads(pickle.dumps(n)), dimensions, norm, slack,
        output_dir=output_dir, resolve_main_objective=resolve_main_objective,
        snapshots=snapshots, multi_investment_periods=multi_investment_periods,
        model_kwargs=model_kwargs, max_parallel=max_parallel, **kwargs,
    )
    solve_optimal = make_optimality_fn(n, dimensions, output_dir, **kwargs)

    return furthest_point, closest_point, solve_optimal


def _broadcast_per_slack(value: Any, n_levels: int, name: str) -> list:
    """`value` as a per-slack-level list: a scalar (anything but a
    `list`/`tuple`) is repeated `n_levels` times; a `list`/`tuple` is used
    as-is, and must already have exactly `n_levels` entries.

    :raises ValueError: If `value` is a `list`/`tuple` of the wrong length.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != n_levels:
            msg = f"{name} has {len(value)} entries but there are {n_levels} slack_levels"
            raise ValueError(msg)
        return list(value)
    return [value] * n_levels


def _read_existing_closest_points(
    slack_dir: Path, keys: list, tol: float, norm: Norm, affine_map: AffineMap | None = None,
    obj_optimal: float | None = None, slack: float | None = None,
) -> tuple[list[np.ndarray], list[Halfspace]]:
    """Read every already-exported `mga_closest_result_*.nc` file under
    `slack_dir` (e.g. from an earlier, interrupted run) back into
    `(point, (normal, point))` pairs, ready to pass as `explore.drive`'s
    `extra_points`/`extra_halfspaces` -- so a restart doesn't have to
    rediscover them via `closest_point`. Empty (not an error) if
    `slack_dir` doesn't exist yet.

    `tol`/`norm`/`affine_map`: replicates `explore.drive`'s own
    `cut_added = gaps(...) > tol` check (measured under `norm` -- the
    same metric `Method`'s own convergence gap is measured under, not a
    hardcoded L2 distance, which would otherwise compare a
    different-norm quantity against `tol` than the one the convergence
    check itself uses; in whichever space `tol` is itself expressed --
    normalized cube if `affine_map` is given, real units if not) before
    trusting a file's `normal` as a halfspace at all. A solve whose
    closest point came back (near-)equal to its own
    `target` (cached alongside `point`/`normal` -- see
    `_cache_closest_point_result`) separated nothing, and at that kind of
    near-zero residual the L1/Linf closest-point LP is maximally
    degenerate (every dimension ties near the max at once); the dual it
    reports can be an arbitrary split across dimensions rather than a
    genuine supporting-hyperplane normal. Confirmed in practice against
    the real network: one such halfspace, read back and folded into a
    single `Method.bootstrap` call alongside everything else, excluded
    the network's own true optimum -- corrupting the entire outer
    approximation (and, downstream, either crashing outright or silently
    shrinking the explored region below what's actually near-optimal).
    `point` itself is still kept in the returned `points` list either way
    (it's a genuinely feasible point regardless of whether it separated
    anything) -- only the two derived halfspaces below are skipped for a
    non-separating solve. A file cached before this check existed (no
    `"target"` in its meta) is always treated as separating, matching the
    old unconditional behavior, since there's nothing on disk to check it
    against.

    `obj_optimal`/`slack` (both `None` by default): if given (and this
    solve was separating), also derive the file's own
    `linearized_objective_cut` from its cached `duals`/`obj` (present
    whenever that solve's `resolve_main_objective` was `True`) and fold
    it into `halfspaces` too -- these were already paid for by an earlier
    run, same as the geometric point/normal above, never used until now.
    Added unconditionally otherwise (no parallel-to-own-normal check) --
    see `generate_mga_files`'s own docstring on why that check was
    dropped for closest-point-derived cuts specifically.
    """
    points = []
    halfspaces = []
    for nc_path in sorted(slack_dir.glob("mga_closest_result_*.nc")):
        meta = _read_meta(nc_path)
        point = np.array([meta["point"][k] for k in keys])
        normal = np.array([meta["normal"][k] for k in keys])
        points.append(point)

        separating = True
        if "target" in meta:
            target = np.array([meta["target"][k] for k in keys])
            check_point = affine_map.to_normalized(point) if affine_map is not None else point
            check_target = affine_map.to_normalized(target) if affine_map is not None else target
            separating = float(gaps(check_point[None, :], check_target[None, :], norm=norm)[0]) > tol
        if not separating:
            continue

        halfspaces.append((normal, point))
        if obj_optimal is not None and "duals" in meta:
            duals = np.array([meta["duals"][k] for k in keys])
            cut = linearized_objective_cut(point, meta["obj"], duals, obj_optimal, slack)
            if cut is not None:
                halfspaces.append(cut)
    return points, halfspaces


def generate_mga_files(
    n: Network,
    dimensions: dict,
    output_dir: str | Path,
    norm: str,
    slack_levels: Sequence[float],
    solver_name: str,
    max_iter: int | None | Sequence[int | None],
    tol: float | Sequence[float],
    max_parallel: int,
    verbose: bool = True,
    normalize: bool = True,
    use_objective_cuts: bool = True,
    **kwargs: Any,
) -> None:
    """Run a `CddMethod` (exact `'gmp'` arithmetic) per slack level (`explore.drive`, up to
    `max_iter` iterations each) against a real, already-solved PyPSA
    network `n`, writing every result to `.nc` files under `output_dir`.
    Doesn't consolidate them into DataFrames -- call `consolidate_mga_results`
    afterward for that.

    `max_parallel`: no default -- must be chosen explicitly for every
    run, same as `max_iter`/`tol`. Dual-purpose: spreads
    `furthest_point`/`closest_point`'s own solves across this many worker
    processes (see `make_furthest_point_fn`/`make_closest_point_fn`), AND
    is passed straight through to each slack level's own `explore.drive`
    call as its own `max_parallel` -- when greater than `1`, every
    iteration there solves up to that many of the worst candidate
    closest-points at once instead of just the single worst one, spread
    across the very same worker pool -- see `explore.drive`'s own
    docstring for the full picture, including why no extra reconciliation is needed
    across a round's own results.

    `tol`/`max_iter` are each either a single value (used for every slack
    level) or a sequence with one entry per `slack_levels` (raises if the
    length doesn't match).

    `normalize` (default `True`): if `True`, each slack level's
    `CddMethod` explores a `[-1, 1]^k`-normalized cube instead of
    raw dimension units, via `mga_surveyor.normalize.normalize_oracle_fns`
    -- see that module for why (mixing e.g. MW and MWh dimensions in one
    unweighted polytope means the ones with the larger natural range
    dominate; on by default since that's true of most real networks, not
    a narrow edge case). Computed fresh per slack level (each one's own
    achievable box can differ), from that level's own `furthest_point`'s
    box-point solves -- raises if any dimension turns out fixed (no
    achievable range) under that slack budget, see
    `normalize.compute_affine_map`. Everything on disk (`.nc` files,
    `consolidate_mga_results`'s output) is completely unaffected -- still
    real units throughout, since the oracle functions themselves are
    unchanged; only what `explore.drive`/`CddMethod` sees is
    normalized. Pass `False` to explore raw units instead.

    **`tol` is interpreted differently depending on `normalize`, on
    purpose -- there is no unit that's natural in both modes at once:**
    - `normalize=False`: `tol` is a plain real-unit distance, exactly as
      it always was.
    - `normalize=True`: `tol` is instead a **percentage** (`tol=1` means
      `1%`), converted internally (`tol_cube = tol / 100 * 2`, since each
      dimension's real range maps onto a normalized width of `2`) before
      being used as the actual `[-1, 1]^k`-cube stopping distance. What
      that percentage bounds depends on `norm` -- each is the same norm,
      just applied to the vector of per-dimension percentage errors
      instead of raw values: `norm="linf"` bounds every dimension's own
      error individually (each is at most `tol`%, not just their
      average); `norm="l1"` bounds their *sum* (`tol`% split across
      dimensions however the worst case demands); `norm="l2"` bounds
      their root-sum-square. All three are exact, provable bounds, not
      heuristics -- just different ways of combining per-dimension
      percentages into one stopping number.

    Same per-slack-level broadcasting as `max_iter`.

    `use_objective_cuts` (default `True`; measured ~5x fewer iterations to
    converge at `tol=2%` on `model_energy`, and the degenerate-dual/
    scale-inconsistency bugs that once made this risky are now fixed --
    pass `False` to fall back to plain geometric cuts only): also add a
    `mga_surveyor.common.linearized_objective_cut` for every point the
    main loop finds (`resolve_main_objective=True`, the default) -- both
    live (`method.add_cut`, same as its geometric cut) and for any
    existing `mga_closest_result_*.nc` file read back on a restart
    (folded into the same `extra_halfspaces` bootstrap call as everything
    else there, since `Method.bootstrap` isn't called yet at that point).
    Reuses `duals`/`obj` that are already computed for caching and
    otherwise never used again. These are provably valid cuts (see that
    function's docstring: convexity of the achievable-cost-at-x value
    function makes each one a global underestimator, not a heuristic), so
    this is free extra tightening, no new solves. Composes with
    `normalize=True`: every cut is derived in real units first, then
    transformed the same way any other halfspace is.

    Also adds one free extra cut per slack level from `n`'s own optimum
    (`optimal.nc`'s own cached `duals`/`obj`) -- unconditionally, since it
    has no same-run geometric normal to check it against (the optimum
    isn't a support-function/closest-point query) and, unlike a
    box-direction point (see below), there's no structural reason for its
    achievable-cost gradient to align with any single axis.

    Deliberately NOT derived from any box-direction point, even though
    those are also solved with `resolve_main_objective=True` and have
    `duals` sitting right there unused -- a box-direction point's own
    objective cut is, in practice, *always* exactly parallel to its own
    box halfspace, a direct consequence of the LP's structure at that
    point (maximizing dimension `i` under the slack budget usually makes
    the budget itself what limits `i`, and the achievable-cost gradient
    there works out to be proportional to `e_i`), and nearly-coincident
    hyperplanes are a real degeneracy risk regardless of which method
    processes them. Confirmed in practice against the incremental
    edge-walking construction this project used before `CddMethod` became
    the production method: unfiltered, this reliably exploded into
    hundreds of thousands of spurious crossing vertices and an
    out-of-memory crash -- kept excluded here as a precaution, not
    re-validated against `CddMethod`'s own construction. Closest-point
    and optimum cuts don't share that structural alignment (confirmed
    empirically: their angles to any same-run geometric normal are
    scattered across the whole range, not clustered near 0/180 the way
    box cuts are), so unlike box cuts, they're never filtered for
    near-parallelism -- an earlier version of this function did check
    each closest-point cut against its own same-solve geometric normal
    (cosine similarity, threshold 0.95) before adding it, but that check
    turned out to be net-negative: on the real test network it discarded
    cuts that still carried real tightening value, costing extra
    main-loop iterations (7 iterations with the
    check vs. 4 without, same network/slack/tolerance) for no compensating
    safety benefit -- cddlib's own Double Description construction already
    discards any genuinely duplicate/redundant cut for free, regardless.

    One more thing every candidate cut from the main loop (or a restart's
    existing files) is checked against before being added, skip-only
    (never turns a bad cut into a good one, just avoids adding one that
    isn't useful) -- the optimum's own cut above is exempt, since it's
    added unconditionally:
    - **Already feasible**: if the point this cut came from was itself
      already feasible (nothing was separated -- same test `explore.drive`
      itself uses for its own geometric cut, its `cut_added` check), the
      objective cut is skipped too, not just the geometric one.

    `n`'s own optimum is slack-independent, so it's solved/cached once
    (`output_dir/optimal.nc`), not once per slack level, even though each
    slack level's own `drive()` call still calls it (as its required
    `optimal_fn`) to seed that slack level's inner approximation.

    Resumable: `optimal_fn`'s and `furthest_point`'s `.nc` files (under
    `output_dir/optimal.nc` and `output_dir/slack_{slack}/`) are read back
    on a restart instead of re-solved (see `make_optimality_fn`/
    `make_furthest_point_fn`). `closest_point` itself has no such cache --
    it always solves fresh (see `make_closest_point_fn`) -- but every
    `mga_closest_result_*.nc` already under a slack level's own directory
    (from an earlier, interrupted run) is read back and passed as that
    slack level's own `drive()` call's `extra_points`/`extra_halfspaces`,
    folded into its one `Method.bootstrap` call, so those points/cuts
    don't need to be rediscovered. Assumes those earlier results are
    genuine near-optimal points for `n`/`dimensions`/this slack level --
    a wildly incompatible stray file (e.g. from a different network or
    corrupted by hand) could otherwise seed `CddMethod.bootstrap` with
    points/halfspaces that don't describe a coherent region.

    `solver_name` also becomes each slack level's `CddMethod`
    backend. `**kwargs` (e.g. `solver_options`) is forwarded to every
    oracle factory.

    :raises ValueError: If `tol` or `max_iter` is a sequence whose length
        doesn't match `slack_levels`; or if `normalize=True` and any
        dimension has no achievable range under some slack level's own
        budget (see `normalize.compute_affine_map`).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = list(dimensions.keys())
    k = len(keys)
    tols = _broadcast_per_slack(tol, len(slack_levels), "tol")
    if normalize:
        # tol is a percentage in this mode (see docstring) -- converted to
        # the actual [-1, 1]^k cube stopping distance here, once, so
        # nothing downstream needs to know the difference.
        tols = [t / 100 * 2 for t in tols]
    max_iters = _broadcast_per_slack(max_iter, len(slack_levels), "max_iter")
    # Log every gap/tol the same way tol was originally passed in (a
    # percentage under normalize=True, since that's what a cube-space
    # number like "0.08" means nothing against otherwise; a plain real-unit
    # number when normalize=False) -- see gap_scale = tol_cube -> tol
    # percent's own inverse, `/ 100 * 2` above.
    gap_scale = 50.0 if normalize else 1.0
    gap_unit = "%" if normalize else ""

    # Each factory below gets its own independent pickled copy of n, not n
    # itself: n.optimize.create_model() stores its model at n.model, a
    # single slot, so factories sharing one n would silently overwrite
    # each other's model there -- the same bug make_oracle_fns avoids.
    solve_optimal = make_optimality_fn(
        pickle.loads(pickle.dumps(n)), dimensions, output_dir, solver_name=solver_name, **kwargs,
    )

    # optimal.nc's own coordinates/duals/obj -- read back once, up front,
    # if use_objective_cuts needs them (obj_optimal itself, needed by
    # every slack level's own linearized_objective_cut; coordinates/duals
    # for the optimum's own cut, added once per slack level below).
    # Forces optimal.nc to exist now rather than lazily inside the first
    # slack level's drive() bootstrap -- a pure cache hit if it already
    # does.
    optimal_meta = None
    obj_optimal = None
    if use_objective_cuts:
        solve_optimal()
        optimal_meta = _read_meta(output_dir / "optimal.nc")
        obj_optimal = optimal_meta["obj"]

    for slack, slack_tol, slack_max_iter in zip(slack_levels, tols, max_iters):
        slack_dir = output_dir / f"slack_{slack}"
        method = None  # assigned below -- add_objective_cut_from_closest_point's closure reads it late

        if verbose:
            logger.info("=" * 60)
            logger.info("slack=%s", slack)
            logger.info("=" * 60)

        furthest_point = make_furthest_point_fn(
            pickle.loads(pickle.dumps(n)), dimensions, slack=slack, output_dir=slack_dir,
            resolve_main_objective=True, max_parallel=max_parallel, solver_name=solver_name, **kwargs,
        )

        def add_objective_cut_from_closest_point(
            point: np.ndarray, normal: np.ndarray, obj: float, duals: np.ndarray | None, target: np.ndarray,
        ) -> None:
            # Legal here: the main loop only ever calls closest_point
            # after Method.bootstrap has already run, so a live add_cut is
            # fine (unlike a box- or optimum-derived cut would be).
            if duals is None:
                return
            # Same feasibility test drive() itself uses for its own
            # geometric cut (explore.py's cut_added) -- if the trial point
            # was already feasible, nothing was separated/learned, so skip
            # the objective cut too, not just the geometric one. Measured
            # under norm (not a hardcoded L2 distance -- see explore.py's
            # own cut_added for why that matters), in whichever space
            # slack_tol is itself in (normalized, if that's the mode),
            # since point/target here are always real units regardless of
            # normalize.
            check_point = affine_map.to_normalized(point) if affine_map is not None else point
            check_target = affine_map.to_normalized(target) if affine_map is not None else target
            if float(gaps(check_point[None, :], check_target[None, :], norm=norm)[0]) <= slack_tol:
                return
            cut = linearized_objective_cut(point, obj, duals, obj_optimal, slack)
            if cut is None:
                return
            direction, cut_point = cut
            if affine_map is not None:
                direction = affine_map.normal_to_normalized(direction)
                cut_point = affine_map.to_normalized(cut_point)
            method.add_cut(direction, cut_point)

        def build_closest_point(weights_arr: np.ndarray | None, _slack=slack, _slack_dir=slack_dir) -> ClosestPointFn:
            weights = dict(zip(keys, weights_arr)) if weights_arr is not None else None
            return make_closest_point_fn(
                pickle.loads(pickle.dumps(n)), dimensions, norm=norm, slack=_slack, output_dir=_slack_dir,
                max_parallel=max_parallel, solver_name=solver_name, weights=weights,
                on_solve=add_objective_cut_from_closest_point if use_objective_cuts else None, **kwargs,
            )

        affine_map = None
        optimal_fn = solve_optimal
        if normalize:
            # normalize_oracle_fns solves the 2k box directions itself
            # (presolve_box_points, to compute the affine map), earlier
            # than drive()'s own bootstrap ever would -- so that solving
            # (and its "saved to file" lines) needs its own banner here,
            # not just drive()'s own furthest-point one, which by the time
            # it fires has nothing left to do (box points are already
            # cached in the wrapped furthest_point closure by then).
            if verbose:
                logger.info("[normalize] furthest-point queries: bootstrapping %d box direction(s)", 2 * k)
            furthest_point, closest_point, optimal_fn, affine_map = normalize_oracle_fns(
                furthest_point, build_closest_point, solve_optimal, k, keys,
            )
        else:
            closest_point = build_closest_point(None)

        # Points/cuts already known from an earlier, interrupted run --
        # folded into drive()'s own bootstrap call below (Method.bootstrap
        # must be called exactly once, before any add_point/add_cut/
        # find_trial_point), so the main loop doesn't rediscover them.
        # Always real units on disk; normalized here too when this slack
        # level is. tol/affine_map let it replicate explore.drive's own
        # cut_added check before trusting a file's geometric normal at
        # all (see its own docstring for the real-network bug this
        # closes). use_objective_cuts also derives each (separating)
        # existing file's own linearized_objective_cut from its cached
        # duals (no box-derived cuts at all -- see generate_mga_files's
        # own docstring for why those are excluded).
        extra_points, extra_halfspaces = _read_existing_closest_points(
            slack_dir, keys, tol=slack_tol, norm=norm, affine_map=affine_map,
            obj_optimal=obj_optimal, slack=slack if use_objective_cuts else None,
        )

        if use_objective_cuts:
            # The optimum's own cut -- one more free extra per slack level
            # (K depends on slack, so it differs per level even though the
            # underlying solve is the same one, cached once). Unlike a
            # box-direction point, the true cost-optimum's own gradient
            # has no structural reason to align with any single axis, so
            # there's no known degeneracy risk here -- added
            # unconditionally, no same-run geometric counterpart exists to
            # check it against (the optimum isn't a support-function/
            # closest-point query).
            opt_point = np.array([optimal_meta["coordinates"][key] for key in keys])
            opt_duals = np.array([optimal_meta["duals"][key] for key in keys])
            opt_cut = linearized_objective_cut(opt_point, optimal_meta["obj"], opt_duals, obj_optimal, slack)
            if opt_cut is not None:
                extra_halfspaces.append(opt_cut)

        if affine_map is not None:
            extra_points = [affine_map.to_normalized(p) for p in extra_points]
            extra_halfspaces = [
                (affine_map.normal_to_normalized(normal), affine_map.to_normalized(point))
                for normal, point in extra_halfspaces
            ]
        if verbose and extra_points:
            logger.info("slack=%s: seeding with %d existing closest-point result(s)", slack, len(extra_points))

        method = CddMethod(k=k, norm=norm, backend="gmp", projection_backend=solver_name)
        if verbose:
            logger.info(
                "slack=%s: running CddMethod (tol=%.4f%s, max_iter=%s)",
                slack, slack_tol * gap_scale, gap_unit, slack_max_iter,
            )
        drive(
            method, furthest_point, closest_point, optimal_fn,
            tol=slack_tol, max_iter=slack_max_iter, verbose=verbose,
            extra_points=extra_points, extra_halfspaces=extra_halfspaces,
            gap_scale=gap_scale, gap_unit=gap_unit,
            max_parallel=max_parallel,
        )
        final_gap = method.max_gap()
        if final_gap > slack_tol:
            # drive() already logged a warning for this (it's the one that
            # knows whether max_iter was actually hit), but that's easy to
            # miss buried in per-iteration logging -- and this is the
            # summary line most callers actually look at, so make the
            # under-convergence visible here too rather than reusing the
            # same neutral info line convergence would also produce.
            logger.warning(
                "slack=%s: did NOT converge (max_iter=%s exhausted): %d inner point(s), final max_gap=%.4f%s > tol=%.4f%s",
                slack, slack_max_iter, len(method.inner_vertices),
                final_gap * gap_scale, gap_unit, slack_tol * gap_scale, gap_unit,
            )
        elif verbose:
            logger.info(
                "slack=%s: %d inner point(s), final max_gap=%.4f%s",
                slack, len(method.inner_vertices), final_gap * gap_scale, gap_unit,
            )


def consolidate_mga_results(
    output_dir: str | Path, obj_label: str = "obj",
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Scan `output_dir` for every MGA result `.nc` file this module's
    factories can produce there (`optimal.nc`, and each slack level's own
    `mga_furthest_result_*.nc`/`mga_closest_result_*.nc` under
    `output_dir/slack_*/`) and consolidate their `meta` into
    `points_df`/`duals_df`, one row per file, 0..N-1 indexed. Standalone --
    can be called independently of `generate_mga_files`, e.g. to rebuild
    the dataset from an `output_dir` an earlier (possibly interrupted) run
    already populated, without re-running anything -- `output_dir` is the
    only thing this needs: dimension names/order come from `optimal.nc`'s
    own `meta["dimensions"]` (stashed there by `make_optimality_fn`), not a
    parameter here, so there's no way to pass mismatched ones by accident.

    Each file is either `make_furthest_point_fn`'s/`make_optimality_fn`'s
    own export (`meta["coordinates"]`) or `make_closest_point_fn`'s own
    (`meta["point"]`) -- `duals` absent (all-zero) when that call's own
    `resolve_main_objective` was `False`. `obj` is read straight out of
    `meta` too (every writer in this module stashes it there already) --
    never recomputed via `n.statistics`, which would mean a full
    `pypsa.Network` load per file for no reason (see `_read_meta`).

    Returns `points_df`, `duals_df`, and the sorted list of NetCDF paths
    (matching the DataFrames' row order).

    :raises FileNotFoundError: If `output_dir/optimal.nc` doesn't exist --
        it's the precondition every other function in this module assumes,
        so there's nothing meaningful to consolidate without it.
    """
    output_dir = Path(output_dir)
    optimal_nc = output_dir / "optimal.nc"
    if not optimal_nc.exists():
        msg = f"{optimal_nc} not found -- run generate_mga_files (or at least make_optimality_fn) first."
        raise FileNotFoundError(msg)
    keys = list(_read_meta(optimal_nc)["dimensions"].keys())

    nc_paths: set[Path] = {optimal_nc}
    nc_paths.update(output_dir.glob("slack_*/mga_furthest_result_*.nc"))
    nc_paths.update(output_dir.glob("slack_*/mga_closest_result_*.nc"))

    all_coords = []
    all_duals = []
    all_nc_paths = sorted(nc_paths, key=str)
    for nc_path in all_nc_paths:
        meta = _read_meta(nc_path)
        if "coordinates" in meta:
            coords = pd.Series(meta["coordinates"])[keys]
        else:
            coords = pd.Series(meta["point"])[keys]
        duals = pd.Series(meta["duals"])[keys] if "duals" in meta else pd.Series(0.0, index=keys)
        coords = coords.copy()
        coords[obj_label] = meta["obj"]
        all_coords.append(coords)
        all_duals.append(duals)

    points_df = pd.DataFrame(all_coords).reset_index(drop=True)
    duals_df = pd.DataFrame(all_duals).reset_index(drop=True)
    points_df.index.name = "index"
    duals_df.index.name = "index"

    return points_df, duals_df, all_nc_paths
