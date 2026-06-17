import linopy
import pytest
import pandas as pd
import xarray as xr
from mga_compass.core import navigate, interpolate, alpha_2_point, point_2_alpha, describe_changes, linear_lower_bounds, upper_envelope, build_outer_approximation, navigate_outer_approximation, aggregate_by_alpha
from mga_compass.schemes import Alpha, Breakpoint, Constraints, ConstraintChange, LinearBound, LowerBoundPoint


def test_alpha_2_point(points_df, dimensions):
    alpha = Alpha([0.3, 0.7], index=[0, 2])
    point = alpha_2_point(alpha, points_df)
    expected = pd.Series({
        "wind": 0.3 * 10 + 0.7 * 18,
        "solar": 0.3 * 20 + 0.7 * 30,
        "hydrogen": 0.3 * 45 + 0.7 * 30,
        "battery": 0.3 * 25 + 0.7 * 20,
        "TOTEX": 0.3 * 35 + 0.7 * 40
    })
    pd.testing.assert_series_equal(point, expected)


def test_point_2_alpha(points_df, dimensions):
    point = pd.Series({
        "wind": 0.3 * 10 + 0.7 * 18,
        "solar": 0.3 * 20 + 0.7 * 30,
        "hydrogen": 0.3 * 45 + 0.7 * 30,
        "battery": 0.3 * 25 + 0.7 * 20,
        "TOTEX": 0.3 * 35 + 0.7 * 40
    })
    alpha = point_2_alpha(point, points_df, obj_label="TOTEX")
    expected = Alpha([0.3, 0.7], index=[0, 2])
    pd.testing.assert_series_equal(alpha, expected)


def test_interpolate():
    points = pd.DataFrame({
        "wind": [1, 2, 3, 4, 5],
        "TOTEX": [5, 3, 1, 2, 5]
    })
    point_start = points.iloc[0]
    point_end = points.iloc[4]
    breakpoints = interpolate(points, point_start, point_end, obj_label="TOTEX")

    expected = [
        Breakpoint(beta=0,    alpha=Alpha([1.0], index=[0]), point=pd.Series({"wind": 1.0, "TOTEX": 5.0})),
        Breakpoint(beta=0.5,  alpha=Alpha([1.0], index=[2]), point=pd.Series({"wind": 3.0, "TOTEX": 1.0})),
        Breakpoint(beta=0.75, alpha=Alpha([1.0], index=[3]), point=pd.Series({"wind": 4.0, "TOTEX": 2.0})),
        Breakpoint(beta=1,    alpha=Alpha([1.0], index=[4]), point=pd.Series({"wind": 5.0, "TOTEX": 5.0})),
    ]
    assert len(breakpoints) == len(expected)
    for bp, exp in zip(breakpoints, expected):
        assert bp.beta == exp.beta
        pd.testing.assert_series_equal(bp.alpha.astype(float), exp.alpha.astype(float), check_names=False)
        pd.testing.assert_series_equal(bp.point.astype(float), exp.point.astype(float), check_names=False)


# def test_navigate(points_df, constraints_df):
#     navigate(points_df, constraints_df)
#     pass



class TestAggregateByAlpha:
    def test_returns_xarray_dataarray(self):
        da = xr.DataArray([10.0, 20.0], dims=["index"], coords={"index": [0, 1]})
        alpha = Alpha([0.5, 0.5], index=[0, 1])
        assert isinstance(aggregate_by_alpha(da, alpha), xr.DataArray)

    def test_index_dimension_removed(self):
        da = xr.DataArray([10.0, 20.0], dims=["index"], coords={"index": [0, 1]})
        alpha = Alpha([0.5, 0.5], index=[0, 1])
        assert "index" not in aggregate_by_alpha(da, alpha).dims

    def test_weighted_sum_1d(self):
        da = xr.DataArray([10.0, 30.0], dims=["index"], coords={"index": [0, 1]})
        alpha = Alpha([0.3, 0.7], index=[0, 1])
        assert float(aggregate_by_alpha(da, alpha)) == pytest.approx(0.3 * 10.0 + 0.7 * 30.0)

    def test_weighted_sum_2d(self):
        da = xr.DataArray(
            [[10.0, 20.0], [30.0, 40.0]],
            dims=["index", "technology"],
            coords={"index": [0, 1], "technology": ["gas", "coal"]},
        )
        alpha = Alpha([0.3, 0.7], index=[0, 1])
        result = aggregate_by_alpha(da, alpha)
        assert float(result.sel(technology="gas")) == pytest.approx(0.3 * 10.0 + 0.7 * 30.0)
        assert float(result.sel(technology="coal")) == pytest.approx(0.3 * 20.0 + 0.7 * 40.0)

    def test_single_point_alpha_returns_original_values(self):
        da = xr.DataArray(
            [[5.0, 15.0], [25.0, 35.0]],
            dims=["index", "technology"],
            coords={"index": [0, 1], "technology": ["gas", "coal"]},
        )
        alpha = Alpha([1.0], index=[0])
        result = aggregate_by_alpha(da, alpha)
        assert float(result.sel(technology="gas")) == pytest.approx(5.0)
        assert float(result.sel(technology="coal")) == pytest.approx(15.0)

    def test_sparse_alpha_ignores_missing_indices(self):
        da = xr.DataArray(
            [10.0, 20.0, 30.0, 40.0],
            dims=["index"],
            coords={"index": [0, 1, 2, 3]},
        )
        alpha = Alpha([0.4, 0.6], index=[0, 2])
        assert float(aggregate_by_alpha(da, alpha)) == pytest.approx(0.4 * 10.0 + 0.6 * 30.0)

    def test_other_dimensions_preserved(self):
        da = xr.DataArray(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            dims=["index", "snapshot", "technology"],
            coords={"index": [0, 1], "snapshot": ["t1", "t2"], "technology": ["gas", "coal"]},
        )
        alpha = Alpha([0.5, 0.5], index=[0, 1])
        result = aggregate_by_alpha(da, alpha)
        assert set(result.dims) == {"snapshot", "technology"}


class TestLinearLowerBounds:
    @pytest.fixture
    def simple_points(self):
        return pd.DataFrame({
            "x":     [1.0, 3.0],
            "y":     [2.0, 4.0],
            "TOTEX": [10.0, 20.0],
        })

    @pytest.fixture
    def simple_duals(self):
        return pd.DataFrame({
            "x": [0.5, 0.2],
            "y": [0.3, 0.1],
        })

    @pytest.fixture
    def simple_breakpoints(self, simple_points):
        bp_start = Breakpoint(beta=0.0, alpha=Alpha([1.0], index=[0]), point=simple_points.iloc[0])
        bp_end   = Breakpoint(beta=1.0, alpha=Alpha([1.0], index=[1]), point=simple_points.iloc[1])
        return [bp_start, bp_end]

    def test_returns_one_bound_per_nonzero_alpha(self, simple_points, simple_duals, simple_breakpoints):
        result = linear_lower_bounds(simple_points, simple_breakpoints, simple_duals, obj_label="TOTEX")
        # nonzero_index = {0, 1} → two bounds
        assert len(result) == 2

    def test_all_results_are_linear_bound(self, simple_points, simple_duals, simple_breakpoints):
        result = linear_lower_bounds(simple_points, simple_breakpoints, simple_duals, obj_label="TOTEX")
        assert all(isinstance(b, LinearBound) for b in result)

    def test_slope_and_intercept_for_point_0(self, simple_points, simple_duals, simple_breakpoints):
        # p_start = [x=1, y=2, TOTEX=10], p_target = [x=3, y=4, TOTEX=20]
        # duals[0] = [x=0.5, y=0.3]
        # slope     = (3-1)*0.5 + (4-2)*0.3 = 1.0 + 0.6 = 1.6
        # intercept = (1-1)*0.5 + (2-2)*0.3 + 10 = 10.0
        result = linear_lower_bounds(simple_points, simple_breakpoints, simple_duals, obj_label="TOTEX")
        bounds = {round(b.intercept, 6): b for b in result}
        b0 = bounds[10.0]
        assert b0.slope == pytest.approx(1.6)
        assert b0.intercept == pytest.approx(10.0)

    def test_slope_and_intercept_for_point_1(self, simple_points, simple_duals, simple_breakpoints):
        # duals[1] = [x=0.2, y=0.1]
        # slope     = (3-1)*0.2 + (4-2)*0.1 = 0.4 + 0.2 = 0.6
        # intercept = (1-3)*0.2 + (2-4)*0.1 + 20 = -0.4 - 0.2 + 20 = 19.4
        result = linear_lower_bounds(simple_points, simple_breakpoints, simple_duals, obj_label="TOTEX")
        bounds = {round(b.intercept, 6): b for b in result}
        b1 = bounds[round(19.4, 6)]
        assert b1.slope == pytest.approx(0.6)
        assert b1.intercept == pytest.approx(19.4)

    def test_duplicate_alpha_indices_counted_once(self, simple_points, simple_duals, simple_breakpoints):
        # Repeat bp_start — index 0 appears in two breakpoints, should still yield one bound for it
        result = linear_lower_bounds(simple_points, simple_breakpoints * 2, simple_duals, obj_label="TOTEX")
        assert len(result) == 2

    def test_single_breakpoint_pair_same_point(self, simple_points, simple_duals):
        # start == end: slope should be 0
        bp = Breakpoint(beta=0.0, alpha=Alpha([1.0], index=[0]), point=simple_points.iloc[0])
        result = linear_lower_bounds(simple_points, [bp, bp], simple_duals, obj_label="TOTEX")
        for b in result:
            assert b.slope == pytest.approx(0.0)


class TestUpperEnvelope:
    def test_single_line_returns_two_endpoints(self):
        # f(beta) = 2*beta + 1  →  (0, 1) and (1, 3)
        result = upper_envelope([LinearBound(slope=2.0, intercept=1.0)])
        assert len(result) == 2
        assert result[0] == pytest.approx(LowerBoundPoint(beta=0.0, value=1.0))
        assert result[1] == pytest.approx(LowerBoundPoint(beta=1.0, value=3.0))

    def test_all_results_are_lower_bound_points(self):
        bounds = [LinearBound(slope=1.0, intercept=0.0), LinearBound(slope=-1.0, intercept=2.0)]
        result = upper_envelope(bounds)
        assert all(isinstance(p, LowerBoundPoint) for p in result)

    def test_two_lines_crossing_inside_interval(self):
        # f1 = 1*beta + 0  (starts lower, ends higher)
        # f2 = -1*beta + 2 (starts higher, ends lower)
        # intersection: beta = 1, value = 1  → at beta=0.5, value=0.5 wait...
        # f1(0.5)=0.5, f2(0.5)=1.5 ... let's use:
        # f1 = 2*beta + 0,  f2 = -2*beta + 3
        # intersection: 2b = -2b+3 → b=0.75, val=1.5
        bounds = [LinearBound(slope=2.0, intercept=0.0), LinearBound(slope=-2.0, intercept=3.0)]
        result = upper_envelope(bounds)
        betas = [p.beta for p in result]
        assert 0.0 in betas
        assert 1.0 in betas
        assert 0.75 in [pytest.approx(b) for b in betas]

    def test_two_lines_crossing_outside_interval_only_endpoints(self):
        # f1 = 3*beta + 0, f2 = 1*beta + 0.5
        # intersection: 3b = b + 0.5 → b = 0.25  — inside [0,1]
        # at b=0: f1=0, f2=0.5  → f2 wins at start
        # at b=1: f1=3, f2=1.5  → f1 wins at end
        # envelope has breakpoint at b=0.25
        # Now try lines that only cross outside [0,1]:
        # f1 = 1*beta + 5, f2 = 2*beta + 5  → f2 always >= f1 on [0,1], cross at b=0
        bounds = [LinearBound(slope=1.0, intercept=5.0), LinearBound(slope=2.0, intercept=5.0)]
        result = upper_envelope(bounds)
        # f2 dominates everywhere, so only 2 points
        assert len(result) == 2
        assert result[0].value == pytest.approx(5.0)   # max(6, 5) at beta=0
        assert result[1].value == pytest.approx(7.0)   # max(6, 7) at beta=1

    def test_envelope_values_are_pointwise_max(self):
        bounds = [
            LinearBound(slope=3.0, intercept=0.0),
            LinearBound(slope=0.0, intercept=2.0),
            LinearBound(slope=-1.0, intercept=4.0),
        ]
        result = upper_envelope(bounds)
        for pt in result:
            expected = max(b.slope * pt.beta + b.intercept for b in bounds)
            assert pt.value == pytest.approx(expected)

    def test_output_is_sorted_by_beta(self):
        bounds = [LinearBound(slope=2.0, intercept=0.0), LinearBound(slope=-2.0, intercept=3.0)]
        result = upper_envelope(bounds)
        betas = [p.beta for p in result]
        assert betas == sorted(betas)

    def test_coincident_lines_no_duplicate_breakpoints(self):
        # Two identical lines should produce just 2 endpoints
        b = LinearBound(slope=1.0, intercept=1.0)
        result = upper_envelope([b, b])
        assert len(result) == 2


def _make_constraints(data: dict) -> Constraints:
    df = pd.DataFrame(data).T.astype({"value": float, "delta": float})
    return Constraints(df)


class TestDescribeChanges:
    def test_no_changes_returns_empty(self):
        c = _make_constraints({"wind": {"direction": ">=", "value": 10.0, "delta": 2.0},
                                "solar": {"direction": "<=", "value": 20.0, "delta": 3.0}})
        messages = describe_changes(c, c)
        assert messages == []

    def test_infeasible_minimum_direction_change(self):
        inp = _make_constraints({"wind": {"direction": ">=", "value": 10.0, "delta": 2.0},
                                  "solar": {"direction": "", "value": 0.0, "delta": 0.0}})
        out = _make_constraints({"wind": {"direction": "==", "value": 10.0, "delta": 1.5},
                                  "solar": {"direction": "", "value": 0.0, "delta": 0.0}})
        messages = describe_changes(inp, out)
        print(messages)
        assert len(messages) == 1
        assert messages[0].code == "infeasible_lb"

    def test_infeasible_maximum_direction_change(self):
        inp = _make_constraints({"wind": {"direction": "<=", "value": 20.0, "delta": 3.0},
                                  "solar": {"direction": "", "value": 0.0, "delta": 0.0}})
        out = _make_constraints({"wind": {"direction": "==", "value": 20.0, "delta": 2.0},
                                  "solar": {"direction": "", "value": 0.0, "delta": 0.0}})
        messages = describe_changes(inp, out)
        assert len(messages) == 1
        assert messages[0].code == "infeasible_ub"

    def test_delta_reduced_for_inequality(self):
        inp = _make_constraints({"wind": {"direction": ">=", "value": 10.0, "delta": 5.0},
                                  "solar": {"direction": "", "value": 0.0, "delta": 0.0}})
        out = _make_constraints({"wind": {"direction": ">=", "value": 10.0, "delta": 2.0},
                                  "solar": {"direction": "", "value": 0.0, "delta": 0.0}})
        messages = describe_changes(inp, out)
        assert len(messages) == 1
        assert messages[0].code == "delta_reduced"

    def test_delta_widened_for_equality(self):
        inp = _make_constraints({"wind": {"direction": "==", "value": 10.0, "delta": 1.0},
                                  "solar": {"direction": "", "value": 0.0, "delta": 0.0}})
        out = _make_constraints({"wind": {"direction": "==", "value": 10.0, "delta": 4.0},
                                  "solar": {"direction": "", "value": 0.0, "delta": 0.0}})
        messages = describe_changes(inp, out)
        assert len(messages) == 1
        assert messages[0].code == "delta_widened"

    def test_unconstrained_dimensions_are_skipped(self):
        inp = _make_constraints({"wind": {"direction": "", "value": 0.0, "delta": 0.0},
                                  "solar": {"direction": "", "value": 0.0, "delta": 0.0}})
        out = _make_constraints({"wind": {"direction": "==", "value": 10.0, "delta": 2.0},
                                  "solar": {"direction": "", "value": 0.0, "delta": 0.0}})
        messages = describe_changes(inp, out)
        assert messages == []

    def test_multiple_changes_reported(self):
        inp = _make_constraints({"wind": {"direction": ">=", "value": 10.0, "delta": 5.0},
                                  "solar": {"direction": "<=", "value": 20.0, "delta": 3.0}})
        out = _make_constraints({"wind": {"direction": "==", "value": 10.0, "delta": 2.0},
                                  "solar": {"direction": "==", "value": 20.0, "delta": 1.0}})
        messages = describe_changes(inp, out)
        assert len(messages) == 2
        codes = [c.code for c in messages]
        assert "infeasible_lb" in codes
        assert "infeasible_ub" in codes

    def test_unexpected_direction_change_raises(self):
        # >= -> <= is not a valid navigate() output
        inp = _make_constraints({"wind": {"direction": ">=", "value": 10.0, "delta": 2.0}})
        out = _make_constraints({"wind": {"direction": "<=", "value": 10.0, "delta": 2.0}})
        with pytest.raises(ValueError, match="output direction must be"):
            describe_changes(inp, out)

    def test_delta_widened_on_inequality_raises(self):
        # delta can only grow on "==" constraints
        inp = _make_constraints({"wind": {"direction": ">=", "value": 10.0, "delta": 1.0}})
        out = _make_constraints({"wind": {"direction": ">=", "value": 10.0, "delta": 5.0}})
        with pytest.raises(ValueError, match="only equality constraints can be relaxed"):
            describe_changes(inp, out)

    def test_delta_reduced_on_equality_raises(self):
        # margin reduction only applies to >= / <= constraints
        inp = _make_constraints({"wind": {"direction": "==", "value": 10.0, "delta": 5.0}})
        out = _make_constraints({"wind": {"direction": "==", "value": 10.0, "delta": 1.0}})
        with pytest.raises(ValueError, match="only '>=' and '<=' constraints"):
            describe_changes(inp, out)


class TestBuildOuterApproximation:
    @pytest.fixture
    def simple_oa(self):
        # 2 variables: x, y
        # row 0: x + 2y >= 5
        # row 1: 3x + y <= 10
        # row 2: x + y == 4
        return pd.DataFrame({
            "x": [1.0, 3.0, 1.0],
            "y": [2.0, 1.0, 1.0],
            "direction": [">=", "<=", "=="],
            "RHS": [5.0, 10.0, 4.0],
        })

    def test_returns_linopy_model(self, simple_oa):
        mdl = build_outer_approximation(simple_oa)
        assert isinstance(mdl, linopy.Model)

    def test_variables_match_coef_cols(self, simple_oa):
        mdl = build_outer_approximation(simple_oa)
        assert "f" in mdl.variables
        assert set(mdl.variables["f"].coords["dim_0"].values) == {"x", "y"}

    def test_constraint_count_equals_rows(self, simple_oa):
        mdl = build_outer_approximation(simple_oa)
        assert mdl.constraints.ncons == len(simple_oa)

    def test_invalid_direction_raises(self):
        bad = pd.DataFrame({
            "x": [1.0],
            "direction": ["!="],
            "RHS": [5.0],
        })
        with pytest.raises(ValueError, match="Invalid directions"):
            build_outer_approximation(bad)

    def test_invalid_direction_message_lists_bad_values(self):
        bad = pd.DataFrame({
            "x": [1.0, 2.0],
            "direction": ["!=", ">"],
            "RHS": [5.0, 3.0],
        })
        with pytest.raises(ValueError) as exc_info:
            build_outer_approximation(bad)
        assert "!=" in str(exc_info.value)
        assert ">" in str(exc_info.value)

    def test_only_ge_rows(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0],
            "direction": [">=", ">="],
            "RHS": [1.0, 2.0],
        })
        mdl = build_outer_approximation(df)
        assert mdl.constraints.ncons == len(df)

    def test_only_le_rows(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0],
            "direction": ["<=", "<="],
            "RHS": [1.0, 2.0],
        })
        mdl = build_outer_approximation(df)
        assert mdl.constraints.ncons == len(df)

    def test_only_eq_rows(self):
        df = pd.DataFrame({
            "x": [1.0],
            "direction": ["=="],
            "RHS": [3.0],
        })
        mdl = build_outer_approximation(df)
        assert mdl.constraints.ncons == len(df)

    def test_fixture_constraint_count(self, outer_approximation_df):
        mdl = build_outer_approximation(outer_approximation_df)
        assert mdl.constraints.ncons == len(outer_approximation_df)


class TestNavigateOuterApproximation:
    @pytest.fixture
    def simple_oa_df(self):
        # Polytope:  x in [0, 20],  TOTEX >= x  (x - TOTEX <= 0),  TOTEX <= 50
        return pd.DataFrame({
            "x":         [-1.0,  1.0,  1.0,  0.0],
            "TOTEX":     [ 0.0,  0.0, -1.0,  1.0],
            "direction": ["<=", "<=", "<=", "<="],
            "RHS":       [ 0.0, 20.0,  0.0, 50.0],
        })

    @pytest.fixture
    def simple_model(self, simple_oa_df):
        return build_outer_approximation(simple_oa_df)

    def test_return_types(self, simple_model):
        c = _make_constraints({
            "x":     {"direction": ">=", "value": 5.0, "delta": 2.0},
            "TOTEX": {"direction": ">=", "value": 0.0, "delta": 0.0},
        })
        output_c, point = navigate_outer_approximation(simple_model, c, obj_label="TOTEX")
        assert isinstance(output_c, Constraints)
        assert isinstance(point, pd.Series)

    def test_output_index_matches_input(self, simple_model):
        c = _make_constraints({
            "x":     {"direction": ">=", "value": 5.0, "delta": 2.0},
            "TOTEX": {"direction": ">=", "value": 0.0, "delta": 0.0},
        })
        output_c, _ = navigate_outer_approximation(simple_model, c, obj_label="TOTEX")
        assert list(output_c.index) == list(c.index)

    def test_feasible_constraint_direction_unchanged(self, simple_model):
        # x >= 5, delta 2  →  x = 7 is feasible (max x = 20)  →  direction stays ">="
        c = _make_constraints({
            "x":     {"direction": ">=", "value": 5.0, "delta": 2.0},
            "TOTEX": {"direction": ">=", "value": 0.0, "delta": 0.0},
        })
        output_c, _ = navigate_outer_approximation(simple_model, c, obj_label="TOTEX")
        assert output_c.loc["x", "direction"] == ">="

    def test_infeasible_constraint_becomes_equality(self, simple_model):
        # x >= 25 is infeasible (x <= 20)  →  direction becomes "=="
        c = _make_constraints({
            "x":     {"direction": ">=", "value": 25.0, "delta": 2.0},
            "TOTEX": {"direction": ">=", "value":  0.0, "delta": 0.0},
        })
        output_c, _ = navigate_outer_approximation(simple_model, c, obj_label="TOTEX")
        assert output_c.loc["x", "direction"] == "=="

    def test_point_dimensions_match_oa_variables(self, simple_oa_df, simple_model):
        coef_cols = [c for c in simple_oa_df.columns if c not in ("direction", "RHS")]
        c = _make_constraints({col: {"direction": ">=", "value": 0.0, "delta": 0.0} for col in coef_cols})
        _, point = navigate_outer_approximation(simple_model, c, obj_label="TOTEX")
        assert set(point.index) == set(coef_cols)

    def test_point_satisfies_outer_approximation(self, simple_model):
        # returned point must lie inside the polytope: x in [0,20], TOTEX >= x, TOTEX <= 50
        c = _make_constraints({
            "x":     {"direction": ">=", "value": 5.0, "delta": 2.0},
            "TOTEX": {"direction": ">=", "value": 0.0, "delta": 0.0},
        })
        _, point = navigate_outer_approximation(simple_model, c, obj_label="TOTEX")
        assert float(point["x"]) >= -1e-5
        assert float(point["x"]) <= 20.0 + 1e-5
        assert float(point["TOTEX"]) >= float(point["x"]) - 1e-5
        assert float(point["TOTEX"]) <= 50.0 + 1e-5
