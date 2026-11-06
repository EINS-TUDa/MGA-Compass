
import linopy
import logging
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from .manifest import Plot
from .schemes import Constraints, Breakpoint, Point, Points, Alpha, ConstraintChange, LinearBound, LowerBoundPoint


def configure_solver_logging(log_path: Path) -> None:
    """Route linopy/solver logs to a file instead of the console. Call this once at startup."""
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    for logger_name in ("linopy.model", "linopy.io", "linopy.constants", "google.cloud.storage"):
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.propagate = False  # don't echo to console


class SolverError(RuntimeError):
    """An LP did not solve to optimality. linopy reports a solver that gives up
    (e.g. HiGHS' "unknown") as status "ok" and still fills in a solution -- all
    zeros -- so every solve's termination condition has to be checked."""


def _solve(mdl: linopy.Model, what: str, solver_name: str, solver_options: dict | None) -> None:
    mdl.solve(solver_name=solver_name, **(solver_options or {}))
    if mdl.status != "ok" or mdl.termination_condition != "optimal":
        raise SolverError(f"{what}: status={mdl.status}, condition={mdl.termination_condition}")


# ---------------------------------------------------------------- column scaling
#
# navigate, interpolate and point_2_alpha solve small LPs over the points'
# values. Columns of very different magnitude (a cost of ~1e11 next to
# capacities of ~1e4) make those LPs numerically ill-conditioned, and solvers
# then report feasible problems as infeasible. So each public function divides
# every column by its own scale before building its LPs and converts the results
# back. Dividing column j by s_j > 0 maps sum_i alpha_i z_ij to the same sum
# divided by s_j, so every constraint and objective keeps the same feasible and
# optimal weights alpha: the results are unchanged, only better conditioned.
# A division only, no shift, so interpolate's threshold, relative to the cost's
# magnitude, keeps its meaning.


def _column_scales(points: Points) -> pd.Series:
    """Each column's largest absolute value, or 1 for an all-zero column."""
    scales = points.abs().max()
    return scales.where(scales > 0, 1.0)


def _scale_constraints(constraints: Constraints, scales: pd.Series) -> Constraints:
    scaled = constraints.copy(deep=True)
    factors = scales.reindex(scaled.index)
    scaled["value"] = scaled["value"] / factors
    scaled["delta"] = scaled["delta"] / factors
    return scaled


def navigate(
    points: Points,
    constraints: Constraints,
    obj_label: str,
    solver_name: str = "highs",
    solver_options: dict | None = None,
) -> tuple[Constraints, Point]:
    constraints.validate(points)
    scales = _column_scales(points)
    scaled_input = _scale_constraints(constraints, scales)
    scaled_output, alpha = _navigate_scaled(points / scales, scaled_input, obj_label, solver_name, solver_options)
    output_df = constraints.copy(deep=True)
    output_df["direction"] = scaled_output["direction"]
    # Unchanged deltas are copied, not scaled back, so that rounding cannot make
    # describe_changes report a change that navigation never made.
    changed = scaled_output["delta"] != scaled_input["delta"]
    output_df.loc[changed, "delta"] = scaled_output.loc[changed, "delta"] * scales.reindex(output_df.index)[changed]
    return output_df, alpha_2_point(alpha, points)


def _navigate_scaled(
    points: Points,
    constraints: Constraints,
    obj_label: str,
    solver_name: str,
    solver_options: dict | None,
) -> tuple[Constraints, Alpha]:
    output_df = constraints.copy(deep=True)
    mdl = linopy.Model()
    alpha = mdl.add_variables(coords=[points.index], lower=0, name="alpha")
    f = mdl.add_variables(coords=[points.columns], name="f")
    abs_delta = mdl.add_variables(coords=[points.columns], lower=0, name="abs_delta")
    mdl.add_constraints(alpha.sum() == 1)
    for func in points.columns:
        mdl.add_constraints(f.loc[func] == (points[func] * alpha).sum())

    # First Loop - Find feasible solution
    for func in constraints.index:
        if constraints.loc[func, "direction"] == ">=":
            mdl.add_objective(f.loc[func], sense="max", overwrite=True)
            _solve(mdl, f"navigate: {func!r}", solver_name, solver_options)
            delta = (f.loc[func].solution - constraints.loc[func, "value"]).values
            if delta >= 0:
                if delta >= constraints.loc[func, "delta"]:
                    mdl.add_constraints(f.loc[func] >= constraints.loc[func, "value"] + constraints.loc[func, "delta"])
                else:
                    mdl.add_constraints(f.loc[func] >= constraints.loc[func, "value"] + delta)
                    output_df.loc[func, "delta"] = delta

            else:
                output_df.loc[func, "delta"] = -delta
                output_df.loc[func, "direction"] = "=="
                mdl.add_constraints(f.loc[func] >= constraints.loc[func, "value"] - np.abs(delta))
                mdl.add_constraints(f.loc[func] <= constraints.loc[func, "value"] + np.abs(delta))
        elif constraints.loc[func, "direction"] == "<=":
            mdl.add_objective(f.loc[func], sense="min", overwrite=True)
            _solve(mdl, f"navigate: {func!r}", solver_name, solver_options)
            delta = (constraints.loc[func, "value"] - f.loc[func].solution).values
            if delta >= 0:
                if delta >= constraints.loc[func, "delta"]:
                    mdl.add_constraints(f.loc[func] <= constraints.loc[func, "value"] - constraints.loc[func, "delta"])
                else:
                    mdl.add_constraints(f.loc[func] <= constraints.loc[func, "value"] - delta)
                    output_df.loc[func, "delta"] = delta
            else:
                output_df.loc[func, "delta"] = -delta
                output_df.loc[func, "direction"] = "=="
                mdl.add_constraints(f.loc[func] >= constraints.loc[func, "value"] - np.abs(delta))
                mdl.add_constraints(f.loc[func] <= constraints.loc[func, "value"] + np.abs(delta))
        elif constraints.loc[func, "direction"] == "==":
            mdl.add_constraints(abs_delta.loc[func] >= f.loc[func] - constraints.loc[func, "value"])
            mdl.add_constraints(abs_delta.loc[func] >= constraints.loc[func, "value"] - f.loc[func])
            mdl.add_objective(abs_delta.loc[func], sense="min", overwrite=True)
            _solve(mdl, f"navigate: '==' on {func!r}", solver_name, solver_options)
            delta = np.abs((f.loc[func].solution - constraints.loc[func, "value"]).values)
            if delta <= constraints.loc[func, "delta"]:
                mdl.add_constraints(abs_delta.loc[func] <= constraints.loc[func, "delta"])
            else:
                output_df.loc[func, "delta"] = delta
                mdl.add_constraints(abs_delta.loc[func] <= delta)
        elif constraints.loc[func, "direction"] == "":
            pass
        else:
            raise ValueError(f"Invalid constraint direction: {constraints.loc[func, 'direction']}")

    # Second Loop - push solutions further in the desired direction
    for func in constraints.index:
        if constraints.loc[func, "direction"] == "":
            continue
        elif constraints.loc[func, "direction"] == ">=":
            mdl.add_objective(f.loc[func], sense="max", overwrite=True)
            _solve(mdl, f"navigate: {func!r}", solver_name, solver_options)
        elif constraints.loc[func, "direction"] == "<=":
            mdl.add_objective(f.loc[func], sense="min", overwrite=True)
            _solve(mdl, f"navigate: {func!r}", solver_name, solver_options)
        elif constraints.loc[func, "direction"] == "==":
            mdl.add_objective(abs_delta.loc[func], sense="min", overwrite=True)
            _solve(mdl, f"navigate: {func!r}", solver_name, solver_options)
        mdl.add_constraints(f.loc[func] == f.loc[func].solution)

    # Find the solution with the minimum cost
    mdl.add_objective(f.loc[obj_label], sense="min", overwrite=True)
    _solve(mdl, "navigate: minimum cost", solver_name, solver_options)

    return output_df, _normalized_alpha(alpha.solution.values, points.index)


_interpolate_logger = logging.getLogger(__name__ + ".interpolate")


def interpolate(
    points: Points,
    point_start: Point,
    point_end: Point,
    obj_label: str,
    threshold: float = 1e-3,
    solver_name: str = "highs",
    solver_options: dict | None = None,
) -> list[Breakpoint]:
    scales = _column_scales(points)
    breakpoints = _interpolate_scaled(
        points / scales, point_start / scales, point_end / scales, obj_label, threshold, solver_name, solver_options,
    )
    return [Breakpoint(bp.beta, bp.alpha, alpha_2_point(bp.alpha, points)) for bp in breakpoints]


def _interpolate_scaled(
    points: Points,
    point_start: Point,
    point_end: Point,
    obj_label: str,
    threshold: float,
    solver_name: str,
    solver_options: dict | None,
) -> list[Breakpoint]:
    output = []

    def _period_interpolate(beta_start, p_start, beta_end, p_end, depth=0):
        beta_mid = (beta_start + beta_end) / 2
        p_mid = (p_start + p_end) / 2
        alpha_mid_interpolate = _point_2_alpha_scaled(p_mid, points, obj_label, solver_name, solver_options)
        p_mid_interpolate = alpha_2_point(alpha_mid_interpolate, points)
        gap = p_mid[obj_label] - p_mid_interpolate[obj_label]
        abs_threshold = threshold * abs(p_mid[obj_label])
        _interpolate_logger.debug("depth=%d beta=[%.4f,%.4f] gap=%.4f abs_threshold=%.4f", depth, beta_start, beta_end, gap, abs_threshold)
        if p_mid_interpolate[obj_label] < p_mid[obj_label] - abs_threshold:
            output.append(Breakpoint(beta_mid, alpha_mid_interpolate, p_mid_interpolate))
            _period_interpolate(beta_mid, p_mid_interpolate, beta_end, p_end, depth + 1)
            _period_interpolate(beta_start, p_start, beta_mid, p_mid_interpolate, depth + 1)

    _interpolate_logger.info("point_start:\n%s", point_start.to_string())
    _interpolate_logger.info("point_end:\n%s", point_end.to_string())
    alpha_start = _point_2_alpha_scaled(point_start, points, obj_label, solver_name, solver_options)
    alpha_end = _point_2_alpha_scaled(point_end, points, obj_label, solver_name, solver_options)
    p_start = alpha_2_point(alpha_start, points)
    p_end = alpha_2_point(alpha_end, points)
    output.append(Breakpoint(0, alpha_start, p_start))
    output.append(Breakpoint(1, alpha_end, p_end))
    _interpolate_logger.info("start: recursive interpolation")
    _period_interpolate(0, p_start, 1, p_end)
    _interpolate_logger.info("done: %d breakpoints", len(output))
    return sorted(output, key=lambda x: x.beta)


def alpha_2_point(alpha: Alpha, points: Points) -> Point:
    return pd.Series({func: sum(points.loc[i, func] * alpha[i] for i in alpha.index) for func in points.columns})


def aggregate_by_alpha(da: xr.DataArray, alpha: Alpha) -> xr.DataArray:
    weights = xr.DataArray(alpha, dims=["index"])
    # xr.dot aligns on "index" like (da * weights).sum("index") does, but contracts
    # without a da-sized temporary -- on a large output dataset that product was
    # several times the dataset's own memory. Weights in da's dtype so a float32
    # dataset isn't upcast (and copied) to float64 first.
    return xr.dot(da, weights.astype(da.dtype), dim="index")


def xr_to_series(da: xr.DataArray, plot: Plot) -> dict | list:
    # da: xarray aggregated over alpha (no "index" dimension).
    # Returns a dict {category -> values over x_dim} if the plot has a categories_dim,
    # or a plain list over x_dim otherwise.
    dims_to_sum = [d for d in da.dims if d != plot.x_dim and d != plot.categories_dim]
    aggregated = da.sum(dims_to_sum) if dims_to_sum else da

    if plot.categories_dim:
        return {
            str(cat): aggregated.sel({plot.categories_dim: cat}).values.tolist()
            for cat in aggregated.coords[plot.categories_dim].values
        }
    return aggregated.values.tolist()


def point_2_alpha(
    point: Point,
    points: Points,
    obj_label: str,
    solver_name: str = "highs",
    solver_options: dict | None = None,
) -> Alpha:
    scales = _column_scales(points)
    return _point_2_alpha_scaled(point / scales, points / scales, obj_label, solver_name, solver_options)


def _point_2_alpha_scaled(
    point: Point,
    points: Points,
    obj_label: str,
    solver_name: str,
    solver_options: dict | None,
) -> Alpha:
    mdl = linopy.Model()
    alpha = mdl.add_variables(coords=[points.index], lower=0, name="alpha")
    mdl.add_constraints(alpha.sum() == 1)
    for func in points.columns:
        if func != obj_label:
            mdl.add_constraints((points[func] * alpha).sum() == point[func])
    mdl.add_objective((points[obj_label] * alpha).sum(), sense="min")
    _solve(mdl, "point_2_alpha", solver_name, solver_options)
    return _normalized_alpha(alpha.solution.values, points.index)


def _normalized_alpha(values: np.ndarray, index: pd.Index) -> Alpha:
    """Solver weights as an Alpha. The solver meets sum(alpha) == 1 only to its
    feasibility tolerance, and a point built from weights summing to 1 + 1e-7 is
    the points' hull stretched by 1e-7 -- outside it, so the next LP that has to
    match that point exactly is infeasible. Divided by their sum, they are not."""
    weights = pd.Series(values, index=index).clip(lower=0)
    return Alpha(weights / weights.sum())


def get_plot_data(alphas: list[Alpha], plot: Plot, da: xr.DataArray) -> dict:
    return {
        "x_dim": da.coords[plot.x_dim].values.tolist(),
        "data": [xr_to_series(aggregate_by_alpha(da, alpha), plot) for alpha in alphas],
    }


def describe_changes(input_constraints: Constraints, output_constraints: Constraints) -> list[ConstraintChange]:
    """Compare input and output constraints and return (code, message) tuples for each change."""
    messages = []
    for dim in input_constraints.index:
        in_dir = input_constraints.loc[dim, "direction"]
        out_dir = output_constraints.loc[dim, "direction"]
        in_val = input_constraints.loc[dim, "value"]
        in_delta = input_constraints.loc[dim, "delta"]
        out_delta = output_constraints.loc[dim, "delta"]

        if in_dir != out_dir:
            if out_dir != "==":
                raise ValueError(
                    f"{dim}: unexpected direction change from '{in_dir}' to '{out_dir}'; "
                    f"output direction must be '=='"
                )
            if in_dir == ">=":
                messages.append(ConstraintChange(
                    "infeasible_lb",
                    f"targeted minimum {{}} + {{}} = {{}} is infeasible; "
                    f"constrained to equality within a \u00b1{{}} window around {{}};"
                    f"the maximum feasible is {{}} - {{}} = {{}}",
                    [in_val, in_delta, in_val + in_delta, out_delta, in_val, in_val, out_delta, in_val - out_delta],
                    dim,
                ))
            elif in_dir == "<=":
                messages.append(ConstraintChange(
                    "infeasible_ub",
                    f"targeted maximum {{}} - {{}} = {{}} is infeasible; "
                    f"constrained to equality within a \u00b1{{}} window around {{}};"
                    f"the minimum feasible is {{}} + {{}} = {{}}",
                    [in_val, in_delta, in_val - in_delta, out_delta, in_val, in_val, out_delta, in_val + out_delta],
                    dim,
                ))
        else:
            if out_delta > in_delta:
                # equality constraint was relaxed
                if in_dir != "==":
                    raise ValueError(
                        f"{dim}: delta widened for a '{in_dir}' constraint; "
                        f"only equality constraints can be relaxed"
                    )
                messages.append(ConstraintChange(
                    "delta_widened",
                    f"equality tolerance around {{}} widened from {{}} to {{}}",
                    [in_val, in_delta, out_delta],
                    dim,
                ))
            elif out_delta < in_delta:
                # the direction is the same, either >= or <=
                if in_dir not in (">=", "<="):
                    raise ValueError(
                        f"{dim}: delta reduced for a '{in_dir}' constraint; "
                        f"only '>=' and '<=' constraints can have their margin reduced"
                    )
                messages.append(ConstraintChange(
                    "delta_reduced",
                    f"delta reduced from {{}} to {{}}",
                    [in_delta, out_delta],
                    dim,
                ))

    return messages


def linear_lower_bounds(
    points: Points,
    breakpoints: list[Breakpoint],
    duals: Points,
    obj_label: str,
) -> list[LinearBound]:
    nonzero_index = set()
    for bp in breakpoints:
        nonzero_index.update(bp.alpha[bp.alpha != 0].index)
    p_start = breakpoints[0].point
    p_target = breakpoints[-1].point
    result = []
    for i in nonzero_index:
        slope = float(((p_target - p_start).drop(obj_label) * duals.loc[i]).sum())
        intercept = float(((p_start.drop(obj_label) - points.iloc[i].drop(obj_label)) * duals.loc[i]).sum() + points.iloc[i][obj_label])
        result.append(LinearBound(slope=slope, intercept=intercept))
    return result


def build_outer_approximation(
    outer_approximation: pd.DataFrame,
) -> linopy.Model:
    coef_cols = [c for c in outer_approximation.columns if c not in ("direction", "RHS")]
    mdl = linopy.Model()
    f = mdl.add_variables(coords=[coef_cols], name="f")

    invalid = ~outer_approximation["direction"].isin([">=", "<=", "=="])
    if invalid.any():
        raise ValueError(f"Invalid directions: {outer_approximation.loc[invalid, 'direction'].unique().tolist()}")

    for direction, group in outer_approximation.groupby("direction", sort=False):
        coef_da = xr.DataArray(
            group[coef_cols].values,
            dims=["con", f.dims[0]],
            coords={"con": range(len(group)), f.dims[0]: coef_cols},
        )
        lhs = (coef_da * f).sum(f.dims[0])
        rhs = group["RHS"].values
        if direction == ">=":
            mdl.add_constraints(lhs >= rhs)
        elif direction == "<=":
            mdl.add_constraints(lhs <= rhs)
        else:
            mdl.add_constraints(lhs == rhs)

    return mdl


def navigate_outer_approximation(
    outer_approximation: linopy.Model,
    input_constraints: Constraints,
    obj_label: str,
    solver_name: str = "highs",
    solver_options: dict | None = None,
) -> tuple[Constraints, Point]:
    output_df = input_constraints.copy(deep=True)
    mdl = outer_approximation
    f = mdl.variables["f"]
    abs_delta = mdl.add_variables(coords=f.coords, lower=0, name="abs_delta")

    # First Loop - Find feasible solution
    for func in input_constraints.index:
        if input_constraints.loc[func, "direction"] == ">=":
            mdl.add_objective(f.loc[func], sense="max", overwrite=True)
            _solve(mdl, f"navigate_outer_approximation: {func!r}", solver_name, solver_options)
            delta = (f.loc[func].solution - input_constraints.loc[func, "value"]).values
            if delta >= 0:
                if delta >= input_constraints.loc[func, "delta"]:
                    mdl.add_constraints(f.loc[func] >= input_constraints.loc[func, "value"] + input_constraints.loc[func, "delta"])
                else:
                    mdl.add_constraints(f.loc[func] >= input_constraints.loc[func, "value"] + delta)
                    output_df.loc[func, "delta"] = delta
            else:
                output_df.loc[func, "delta"] = -delta
                output_df.loc[func, "direction"] = "=="
                mdl.add_constraints(f.loc[func] >= input_constraints.loc[func, "value"] - np.abs(delta))
                mdl.add_constraints(f.loc[func] <= input_constraints.loc[func, "value"] + np.abs(delta))
        elif input_constraints.loc[func, "direction"] == "<=":
            mdl.add_objective(f.loc[func], sense="min", overwrite=True)
            _solve(mdl, f"navigate_outer_approximation: {func!r}", solver_name, solver_options)
            delta = (input_constraints.loc[func, "value"] - f.loc[func].solution).values
            if delta >= 0:
                if delta >= input_constraints.loc[func, "delta"]:
                    mdl.add_constraints(f.loc[func] <= input_constraints.loc[func, "value"] - input_constraints.loc[func, "delta"])
                else:
                    mdl.add_constraints(f.loc[func] <= input_constraints.loc[func, "value"] - delta)
                    output_df.loc[func, "delta"] = delta
            else:
                output_df.loc[func, "delta"] = -delta
                output_df.loc[func, "direction"] = "=="
                mdl.add_constraints(f.loc[func] >= input_constraints.loc[func, "value"] - np.abs(delta))
                mdl.add_constraints(f.loc[func] <= input_constraints.loc[func, "value"] + np.abs(delta))
        elif input_constraints.loc[func, "direction"] == "==":
            mdl.add_constraints(abs_delta.loc[func] >= f.loc[func] - input_constraints.loc[func, "value"])
            mdl.add_constraints(abs_delta.loc[func] >= input_constraints.loc[func, "value"] - f.loc[func])
            mdl.add_objective(abs_delta.loc[func], sense="min", overwrite=True)
            _solve(mdl, f"navigate_outer_approximation: {func!r}", solver_name, solver_options)
            delta = np.abs((f.loc[func].solution - input_constraints.loc[func, "value"]).values)
            if delta <= input_constraints.loc[func, "delta"]:
                mdl.add_constraints(abs_delta.loc[func] <= input_constraints.loc[func, "delta"])
            else:
                output_df.loc[func, "delta"] = delta
                mdl.add_constraints(abs_delta.loc[func] <= delta)
        elif input_constraints.loc[func, "direction"] == "":
            pass
        else:
            raise ValueError(f"Invalid constraint direction: {input_constraints.loc[func, 'direction']}")

    # Second Loop - push solutions further in the desired direction
    for func in input_constraints.index:
        if input_constraints.loc[func, "direction"] == ">=":
            mdl.add_objective(f.loc[func], sense="max", overwrite=True)
            _solve(mdl, f"navigate_outer_approximation: {func!r}", solver_name, solver_options)
            mdl.add_constraints(f.loc[func] == f.loc[func].solution)
        elif input_constraints.loc[func, "direction"] == "<=":
            mdl.add_objective(f.loc[func], sense="min", overwrite=True)
            _solve(mdl, f"navigate_outer_approximation: {func!r}", solver_name, solver_options)
        elif input_constraints.loc[func, "direction"] == "==":
            mdl.add_objective(abs_delta.loc[func], sense="min", overwrite=True)
            _solve(mdl, f"navigate_outer_approximation: {func!r}", solver_name, solver_options)
        mdl.add_constraints(f.loc[func] == f.loc[func].solution)

    # Find the solution with the minimum cost
    mdl.add_objective(f.loc[obj_label], sense="min", overwrite=True)
    _solve(mdl, "navigate_outer_approximation: minimum cost", solver_name, solver_options)

    return output_df, f.solution.to_series()


def upper_envelope(
    bounds: list[LinearBound],
) -> list[LowerBoundPoint]:
    if not bounds:
        return []

    def _intersect(a: LinearBound, b: LinearBound) -> float:
        """beta where line a and line b cross (a.slope != b.slope)."""
        return (a.intercept - b.intercept) / (b.slope - a.slope)

    # Sort by slope; for equal slopes keep only the highest intercept
    sorted_bounds = sorted(bounds, key=lambda b: (b.slope, -b.intercept))
    deduped: list[LinearBound] = []
    for b in sorted_bounds:
        if deduped and deduped[-1].slope == b.slope:
            continue  # already have the dominant line for this slope
        deduped.append(b)

    # Convex hull trick: build stack of lines that contribute to upper envelope
    stack: list[LinearBound] = []
    for b in deduped:
        while len(stack) >= 2:
            # line stack[-1] is redundant if b takes over before stack[-1] does
            if _intersect(stack[-2], b) <= _intersect(stack[-2], stack[-1]):
                stack.pop()
            else:
                break
        stack.append(b)

    # Collect LowerBoundPoints at beta=0, interior crossings within (0,1), beta=1
    # Intersection points between consecutive surviving lines
    crossings = [_intersect(stack[i], stack[i + 1]) for i in range(len(stack) - 1)]

    # Find which line is active at beta=0 (last crossing <= 0)
    start = sum(1 for x in crossings if x <= 0.0)

    result: list[LowerBoundPoint] = []
    active = stack[start]
    result.append(LowerBoundPoint(beta=0.0, value=active.intercept))

    for i in range(start, len(crossings)):
        x = crossings[i]
        if x <= 0.0:
            continue
        if x >= 1.0:
            break
        active = stack[i + 1]
        result.append(LowerBoundPoint(beta=x, value=active.slope * x + active.intercept))

    last = stack[min(start + len([x for x in crossings[start:] if 0.0 < x < 1.0]), len(stack) - 1)]
    result.append(LowerBoundPoint(beta=1.0, value=last.slope + last.intercept))

    return result

