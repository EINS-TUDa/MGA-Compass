
import linopy
import logging
import numpy as np
import pandas as pd
import xarray as xr

from .config import settings, Plot
from .schemes import Constraints, Breakpoint, Point, Points, Alpha, ConstraintChange, LinearBound, LowerBoundPoint

_solver_handler = logging.FileHandler(settings.solver_log_path)
_solver_handler.setLevel(logging.DEBUG)
_solver_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
for _logger_name in ("linopy.model", "linopy.io", "linopy.constants", "google.cloud.storage"):
    _log = logging.getLogger(_logger_name)
    _log.addHandler(_solver_handler)
    _log.propagate = False  # don't echo to console


def navigate(
    points: Points,
    constraints: Constraints,
    obj_label: str,
) -> tuple[Constraints, Point]:
    constraints.validate(points)
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
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
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
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
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
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
            if mdl.status != "ok" or mdl.termination_condition != "optimal":
                raise ValueError(f"Solver failed for '==' constraint on '{func}': status={mdl.status}, condition={mdl.termination_condition}")
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
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
        elif constraints.loc[func, "direction"] == "<=":
            mdl.add_objective(f.loc[func], sense="min", overwrite=True)
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
        elif constraints.loc[func, "direction"] == "==":
            mdl.add_objective(abs_delta.loc[func], sense="min", overwrite=True)
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
        mdl.add_constraints(f.loc[func] == f.loc[func].solution)

    # Find the solution with the minimum cost
    mdl.add_objective(f.loc[obj_label], sense="min", overwrite=True)
    mdl.solve(solver_name=settings.solver_name, **settings.solver_options)

    point = f.solution.to_series()
    return output_df, point


_interpolate_logger = logging.getLogger(__name__ + ".interpolate")


def interpolate(
    points: Points,
    point_start: Point,
    point_end: Point,
    obj_label: str,
    threshold: float = 1e-3,
) -> list[Breakpoint]:
    output = []

    def _period_interpolate(beta_start, p_start, beta_end, p_end, depth=0):
        beta_mid = (beta_start + beta_end) / 2
        p_mid = (p_start + p_end) / 2
        alpha_mid_interpolate = point_2_alpha(p_mid, points, obj_label)
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
    alpha_start = point_2_alpha(point_start, points, obj_label)
    alpha_end = point_2_alpha(point_end, points, obj_label)
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
    return (da * weights).sum("index")


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


def point_2_alpha(point: Point, points: Points, obj_label: str) -> Alpha:
    mdl = linopy.Model()
    alpha = mdl.add_variables(coords=[points.index], lower=0, name="alpha")
    mdl.add_constraints(alpha.sum() == 1)
    for func in points.columns:
        if func != obj_label:
            mdl.add_constraints((points[func] * alpha).sum() == point[func])
    mdl.add_objective((points[obj_label] * alpha).sum(), sense="min")
    mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
    return Alpha(alpha.solution.clip(min=0))


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
                    f"{dim}: targeted minimum {in_val:.4g} + {in_delta:.4g} = {(in_val+in_delta):.4g} is infeasible; "
                    f"constrained to equality within a \u00b1{out_delta:.4g} window around {in_val:.4g};"
                    f"the maximum feasible is {in_val:.4g} - {out_delta:.4g} = {(in_val-out_delta):.4g}"
                ))
            elif in_dir == "<=":
                messages.append(ConstraintChange(
                    "infeasible_ub",
                    f"{dim}: targeted maximum {in_val:.4g} - {in_delta:.4g} = {(in_val-in_delta):.4g} is infeasible; "
                    f"constrained to equality within a \u00b1{out_delta:.4g} window around {in_val:.4g};"
                    f"the minimum feasible is {in_val:.4g} + {out_delta:.4g} = {(in_val+out_delta):.4g}"
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
                    f"{dim}: equality tolerance around {in_val:.4g} widened from {in_delta:.4g} to {out_delta:.4g}"
                ))
            elif out_delta < in_delta:
                # the direction is the same, either >= or <=
                if in_dir not in (">=", "<="):
                    raise ValueError(
                        f"{dim}: delta reduced for a '{in_dir}' constraint; "
                        f"only '>=' and '<=' constraints can have their margin reduced"
                    )
                direction_word = "above" if in_dir == ">=" else "below"
                messages.append(ConstraintChange(
                    "delta_reduced",
                    f"{dim}: margin {direction_word} {in_val:.4g} reduced from {in_delta:.4g} to {out_delta:.4g}"
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
) -> tuple[Constraints, Point]:
    output_df = input_constraints.copy(deep=True)
    mdl = outer_approximation
    f = mdl.variables["f"]
    abs_delta = mdl.add_variables(coords=f.coords, lower=0, name="abs_delta")

    # First Loop - Find feasible solution
    for func in input_constraints.index:
        if input_constraints.loc[func, "direction"] == ">=":
            mdl.add_objective(f.loc[func], sense="max", overwrite=True)
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
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
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
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
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
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
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
            mdl.add_constraints(f.loc[func] == f.loc[func].solution)
        elif input_constraints.loc[func, "direction"] == "<=":
            mdl.add_objective(f.loc[func], sense="min", overwrite=True)
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
        elif input_constraints.loc[func, "direction"] == "==":
            mdl.add_objective(abs_delta.loc[func], sense="min", overwrite=True)
            mdl.solve(solver_name=settings.solver_name, **settings.solver_options)
        mdl.add_constraints(f.loc[func] == f.loc[func].solution)

    # Find the solution with the minimum cost
    mdl.add_objective(f.loc[obj_label], sense="min", overwrite=True)
    mdl.solve(solver_name=settings.solver_name, **settings.solver_options)

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

