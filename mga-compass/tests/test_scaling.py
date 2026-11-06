"""navigate, interpolate and point_2_alpha scale every column internally, so
columns of very different magnitude no longer break their LPs."""
from pathlib import Path

import pandas as pd
import pytest

from mga_compass import Constraints, default_solver_options, describe_changes, interpolate, navigate, point_2_alpha

#: 38 near-optimal points of PyPSA's model_energy example in model units: the cost
#: in EUR (~4e9) next to capacities in MW (~1e4). Navigation's last LP used to be
#: reported infeasible on these.
RAW_POINTS = pd.read_csv(Path(__file__).parent / "data" / "raw_units_points.csv", index_col=0)
OBJ = "TOTEX"
OPTIONS = default_solver_options("highs")


def _request(points: pd.DataFrame) -> tuple[pd.Series, Constraints]:
    start = points.loc[points[OBJ].idxmin()]
    constraints = Constraints(pd.DataFrame({
        "direction": [">=", "==", "", "", ""],
        "value": [start["wind"], start[OBJ], start["solar"], start["battery"], start["hydrogen"]],
        "delta": [0.2 * start["wind"], 0.03 * start[OBJ], 0.0, 0.0, 0.0],
    }, index=["wind", OBJ, "solar", "battery", "hydrogen"]))
    return start, constraints


def test_navigation_works_in_raw_units():
    start, constraints = _request(RAW_POINTS)
    _, target = navigate(RAW_POINTS, constraints, OBJ, solver_options=OPTIONS)
    assert target["wind"] >= 1.2 * start["wind"] * (1 - 1e-6)
    assert target[OBJ] <= 1.03 * start[OBJ] * (1 + 1e-6)
    breakpoints = interpolate(RAW_POINTS, start, target, OBJ, solver_options=OPTIONS)
    assert breakpoints[0].point[OBJ] == pytest.approx(start[OBJ])
    assert breakpoints[-1].point[OBJ] == pytest.approx(target[OBJ])


def test_results_do_not_depend_on_units():
    """The same points in other units (GW, million EUR) give the same result."""
    factors = pd.Series({"wind": 1e-3, "solar": 1e-3, "battery": 1e-3, "hydrogen": 1e-6, OBJ: 1e-6})
    _, raw_constraints = _request(RAW_POINTS)
    _, raw_target = navigate(RAW_POINTS, raw_constraints, OBJ, solver_options=OPTIONS)

    converted = RAW_POINTS * factors
    _, converted_constraints = _request(converted)
    _, converted_target = navigate(converted, converted_constraints, OBJ, solver_options=OPTIONS)

    pd.testing.assert_series_equal(converted_target / factors, raw_target, rtol=1e-6, check_names=False)


def test_unchanged_preferences_are_returned_exactly():
    """Scaling back must not introduce rounding that looks like a change."""
    start, constraints = _request(RAW_POINTS)
    constraints.loc["wind", "delta"] = 0.01 * start["wind"]  # easily achievable: kept as asked
    output, _ = navigate(RAW_POINTS, constraints, OBJ, solver_options=OPTIONS)
    assert output.loc["wind", "delta"] == constraints.loc["wind", "delta"]
    assert (output["value"] == constraints["value"]).all()
    assert "wind" not in {change.dim for change in describe_changes(constraints, output)}


def test_point_2_alpha_in_raw_units():
    point = RAW_POINTS.iloc[:2].mean()
    alpha = point_2_alpha(point, RAW_POINTS, OBJ, solver_options=OPTIONS)
    mixed = (RAW_POINTS.loc[alpha.index].mul(alpha, axis=0)).sum()
    for column in ["wind", "solar", "battery", "hydrogen"]:
        assert mixed[column] == pytest.approx(point[column], rel=1e-6, abs=1e-6)
    assert mixed[OBJ] <= point[OBJ] * (1 + 1e-9)
