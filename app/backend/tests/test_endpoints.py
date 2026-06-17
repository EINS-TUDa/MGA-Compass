from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from mga_compass import Alpha, Breakpoint

from mgaserver.main import app


@pytest.fixture(scope="module")
def client(points_df, duals_df, outer_approximation_df):
    from mgaserver.config import settings
    _fake_da = xr.DataArray(
        np.ones((4, 2)),
        dims=["index", "technology"],
        coords={"index": [0, 1, 2, 3], "technology": ["gas", "coal"]},
    )
    _pickle_map = {
        Path(settings.data_path): points_df,
        Path(settings.duals_path): duals_df,
        Path(settings.outer_approximation_path): outer_approximation_df,
    }
    def _read_pickle(path):
        return _pickle_map[Path(path)]
    with patch("mgaserver.main.pd.read_pickle", side_effect=_read_pickle):
        with patch("mgaserver.main.pickle.loads", return_value=_fake_da):
            with patch("pathlib.Path.read_bytes", return_value=b""):
                with TestClient(app) as c:
                    yield c


class TestInitEndpoint:
    def test_returns_200(self, client):
        response = client.get("/init")
        assert response.status_code == 200

    def test_response_contains_obj_label(self, client):
        response = client.get("/init")
        data = response.json()
        assert "obj_label" in data

    def test_response_contains_dimensions(self, client):
        response = client.get("/init")
        data = response.json()
        assert "dimensions" in data

    def test_dimensions_match_dataframe_columns(self, client, points_df):
        response = client.get("/init")
        data = response.json()
        assert set(data["dimensions"].keys()) == set(points_df.columns)

    def test_dimension_ranges_have_min_and_max(self, client, points_df):
        response = client.get("/init")
        data = response.json()
        for col in points_df.columns:
            r = data["dimensions"][col]["range"]
            assert "min" in r
            assert "max" in r

    def test_dimension_min_max_values(self, client, points_df):
        response = client.get("/init")
        data = response.json()
        for col in points_df.columns:
            r = data["dimensions"][col]["range"]
            assert r["min"] == pytest.approx(float(points_df[col].min()))
            assert r["max"] == pytest.approx(float(points_df[col].max()))

    def test_response_contains_optimal_point(self, client):
        response = client.get("/init")
        data = response.json()
        assert "point" in data

    def test_optimal_point_has_all_dimensions(self, client, points_df):
        response = client.get("/init")
        data = response.json()
        assert set(data["point"].keys()) == set(points_df.columns)

    def test_optimal_point_minimizes_obj_label(self, client, points_df, manifest):
        response = client.get("/init")
        data = response.json()
        assert data["point"][manifest.obj_label] == pytest.approx(float(points_df[manifest.obj_label].min()))

    def test_response_does_not_contain_datasets(self, client):
        response = client.get("/init")
        data = response.json()
        assert "datasets" not in data


class TestInitPlotEndpoint:
    def test_returns_200(self, client):
        response = client.get("/init_plot")
        assert response.status_code == 200

    def test_response_contains_datasets(self, client):
        response = client.get("/init_plot")
        data = response.json()
        assert "datasets" in data
        assert isinstance(data["datasets"], list)

    def test_response_contains_obj_label(self, client, manifest):
        response = client.get("/init_plot")
        data = response.json()
        assert "obj_label" in data
        assert data["obj_label"] == manifest.obj_label

    def test_datasets_match_config(self, client, manifest):
        response = client.get("/init_plot")
        assert set(response.json()["datasets"]) == set(manifest.plots.keys())


@pytest.fixture
def navigate_payload(constraints_df):
    return {
        "constraints": {
            label: {"direction": row["direction"], "value": float(row["value"]), "delta": float(row["delta"])}
            for label, row in constraints_df.iterrows()
        }
    }


@pytest.fixture
def mock_breakpoints(points_df):
    alpha = Alpha([1.0], index=[0])
    point = points_df.iloc[0].astype(float)
    return [Breakpoint(beta=0.0, alpha=alpha, point=point)]


class TestNavigateEndpoint:
    @pytest.fixture(autouse=True)
    def mock_navigate_internals(self, constraints_df, mock_breakpoints, points_df):
        mock_point = points_df.iloc[0].astype(float)
        with (
            patch("mgaserver.main.navigate", return_value=(constraints_df, mock_point)),
            patch("mgaserver.main.interpolate", return_value=mock_breakpoints),
            patch("mgaserver.main.Constraints", return_value=constraints_df),
        ):
            yield

    def test_returns_200(self, client, navigate_payload):
        response = client.post("/navigate", json=navigate_payload)
        assert response.status_code == 200

    def test_response_has_constraints_and_breakpoints(self, client, navigate_payload):
        response = client.post("/navigate", json=navigate_payload)
        data = response.json()
        assert "constraints" in data
        assert "breakpoints" in data
        assert "changes" in data
        assert "point" in data

    def test_changes_is_list(self, client, navigate_payload):
        response = client.post("/navigate", json=navigate_payload)
        data = response.json()
        assert isinstance(data["changes"], list)

    def test_changes_entries_have_code_and_message(self, client, constraints_df, mock_breakpoints, navigate_payload, points_df):
        # return a modified output where wind direction changed to "==" → triggers infeasible_lb
        out_df = constraints_df.copy(deep=True)
        out_df.loc["wind", "direction"] = "=="
        out_df.loc["wind", "delta"] = 1.0
        mock_point = points_df.iloc[0].astype(float)
        with (
            patch("mgaserver.main.navigate", return_value=(out_df, mock_point)),
            patch("mgaserver.main.interpolate", return_value=mock_breakpoints),
        ):
            response = client.post("/navigate", json=navigate_payload)
        data = response.json()
        assert len(data["changes"]) >= 1
        for change in data["changes"]:
            assert "code" in change
            assert "message" in change

    def test_response_constraints_keys_match_payload(self, client, navigate_payload):
        response = client.post("/navigate", json=navigate_payload)
        data = response.json()
        assert set(data["constraints"].keys()) == set(navigate_payload["constraints"].keys())

    def test_response_constraint_has_required_fields(self, client, navigate_payload):
        response = client.post("/navigate", json=navigate_payload)
        data = response.json()
        for row in data["constraints"].values():
            assert "direction" in row
            assert "value" in row
            assert "delta" in row

    def test_response_breakpoints_have_required_fields(self, client, navigate_payload):
        response = client.post("/navigate", json=navigate_payload)
        data = response.json()
        for bp in data["breakpoints"]:
            assert "beta" in bp
            assert "alpha" in bp
            assert "point" in bp

    def test_invalid_direction_returns_422(self, client):
        payload = {
            "constraints": {
                "wind": {"direction": "INVALID", "value": 10.0, "delta": 2.0},
            }
        }
        response = client.post("/navigate", json=payload)
        assert response.status_code == 422

    def test_missing_constraints_field_returns_422(self, client):
        response = client.post("/navigate", json={})
        assert response.status_code == 422


class TestNavigateValidation:
    """Tests that exercise the real Constraints.validate() path."""

    def test_value_out_of_range_returns_422(self, client, navigate_payload):
        payload = {**navigate_payload}
        payload["constraints"] = {**navigate_payload["constraints"]}
        payload["constraints"]["wind"] = {**navigate_payload["constraints"]["wind"], "value": 9999.0}
        response = client.post("/navigate", json=payload)
        assert response.status_code == 422

    def test_unknown_dimension_returns_422(self, client, navigate_payload):
        payload = {**navigate_payload, "constraints": {"nonexistent": {"direction": "", "value": 1.0, "delta": 1.0}}}
        response = client.post("/navigate", json=payload)
        assert response.status_code == 422

    def test_negative_delta_returns_422(self, client, navigate_payload):
        payload = {**navigate_payload}
        payload["constraints"] = {**navigate_payload["constraints"]}
        payload["constraints"]["wind"] = {**navigate_payload["constraints"]["wind"], "delta": -1.0}
        response = client.post("/navigate", json=payload)
        assert response.status_code == 422


class TestPlotEndpoint:
    @pytest.fixture(autouse=True)
    def mock_get_plot_data(self):
        _result = {
            "x_dim": ["gas", "coal"],
            "data": [{"gas": [10.0], "coal": [30.0]}, {"gas": [20.0], "coal": [40.0]}],
        }
        with patch("mgaserver.main.get_plot_data", return_value=_result):
            yield _result

    def test_returns_200(self, client):
        payload = {"name": "capacity", "breakpoints": [{"beta": 0.0, "alpha": {"0": 1.0}}, {"beta": 1.0, "alpha": {"1": 1.0}}]}
        response = client.post("/plot_data", json=payload)
        assert response.status_code == 200

    def test_unknown_plot_returns_404(self, client):
        payload = {"name": "nonexistent", "breakpoints": [{"beta": 0.0, "alpha": {"0": 1.0}}]}
        response = client.post("/plot_data", json=payload)
        assert response.status_code == 404

    def test_response_has_betas_x_dim_data(self, client):
        payload = {"name": "capacity", "breakpoints": [{"beta": 0.5, "alpha": {"0": 1.0}}]}
        response = client.post("/plot_data", json=payload)
        data = response.json()
        assert "betas" in data
        assert "x_dim" in data
        assert "data" in data

    def test_betas_match_input(self, client):
        payload = {"name": "capacity", "breakpoints": [{"beta": 0.0, "alpha": {"0": 1.0}}, {"beta": 1.0, "alpha": {"1": 1.0}}]}
        response = client.post("/plot_data", json=payload)
        assert response.json()["betas"] == pytest.approx([0.0, 1.0])

    def test_x_dim_from_get_plot_data(self, client, mock_get_plot_data):
        payload = {"name": "capacity", "breakpoints": [{"beta": 0.0, "alpha": {"0": 1.0}}]}
        response = client.post("/plot_data", json=payload)
        assert response.json()["x_dim"] == mock_get_plot_data["x_dim"]

    def test_data_from_get_plot_data(self, client, mock_get_plot_data):
        payload = {"name": "capacity", "breakpoints": [{"beta": 0.0, "alpha": {"0": 1.0}}, {"beta": 1.0, "alpha": {"1": 1.0}}]}
        response = client.post("/plot_data", json=payload)
        assert response.json()["data"] == mock_get_plot_data["data"]


class TestLowerBoundEndpoint:
    @pytest.fixture
    def lower_bound_payload(self, points_df):
        return {
            "breakpoints": [
                {
                    "beta": 0.0,
                    "alpha": {"0": 1.0},
                    "point": points_df.iloc[0].astype(float).to_dict(),
                },
                {
                    "beta": 1.0,
                    "alpha": {"3": 1.0},
                    "point": points_df.iloc[3].astype(float).to_dict(),
                },
            ]
        }

    @pytest.fixture(autouse=True)
    def mock_lower_bound_internals(self):
        from mga_compass import LowerBoundPoint
        mock_envelope = [LowerBoundPoint(beta=0.0, value=35.0), LowerBoundPoint(beta=1.0, value=32.0)]
        with (
            patch("mgaserver.main.linear_lower_bounds", return_value=[]),
            patch("mgaserver.main.upper_envelope", return_value=mock_envelope),
        ):
            yield

    def test_returns_200(self, client, lower_bound_payload):
        response = client.post("/lower_bound", json=lower_bound_payload)
        assert response.status_code == 200

    def test_response_has_points(self, client, lower_bound_payload):
        response = client.post("/lower_bound", json=lower_bound_payload)
        assert "points" in response.json()

    def test_points_is_list(self, client, lower_bound_payload):
        response = client.post("/lower_bound", json=lower_bound_payload)
        assert isinstance(response.json()["points"], list)

    def test_each_point_has_beta_and_value(self, client, lower_bound_payload):
        response = client.post("/lower_bound", json=lower_bound_payload)
        for pt in response.json()["points"]:
            assert "beta" in pt
            assert "value" in pt

    def test_beta_and_value_are_floats(self, client, lower_bound_payload):
        response = client.post("/lower_bound", json=lower_bound_payload)
        for pt in response.json()["points"]:
            assert isinstance(pt["beta"], float)
            assert isinstance(pt["value"], float)

    def test_response_values_match_mock(self, client, lower_bound_payload):
        response = client.post("/lower_bound", json=lower_bound_payload)
        pts = response.json()["points"]
        assert pts[0] == {"beta": pytest.approx(0.0), "value": pytest.approx(35.0)}
        assert pts[1] == {"beta": pytest.approx(1.0), "value": pytest.approx(32.0)}

    def test_missing_breakpoints_returns_422(self, client):
        response = client.post("/lower_bound", json={})
        assert response.status_code == 422
