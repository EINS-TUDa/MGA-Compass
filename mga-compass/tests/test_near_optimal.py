import numpy as np
import pandas as pd
import pytest

import mga_format
from mga_compass.near_optimal import category_styles, mix_output, navigable_points, plot_data
from mga_compass.schemes import Alpha

SNAPSHOTS = pd.date_range("2030-01-01", periods=2, freq="h")
TECHS = ["gas", "wind"]


@pytest.fixture
def written(tmp_path):
    manifest = mga_format.Manifest.model_validate({
        "format": "mga-near-optimal-set", "version": 1, "created": "2026-11-06T12:00:00+00:00",
        "cost": {"name": "TOTEX", "unit": "EUR", "optimum": 10.0},
        "pois": [{"name": "wind", "unit": "MW", "definition": {}}],
        "slack_levels": [{"slack": 0.05, "norm": "linf", "normalize": True, "tol": 5, "tol_unit": "percent",
                          "vertex_method": "bb", "objective_cuts": True, "iterations": 1, "gap": 1, "converged": True}],
        "fingerprint": {"network": "x"},
        "sets": {"snapshot": {}, "technology": {}},
        "outputs": {"generation": {"index": ["point", "snapshot", "technology"]},
                    "capacity": {"index": ["point", "technology"]}},
        "plots": {"generation": {"type": "stacked_timeseries", "output": "generation", "x": "snapshot", "stack": "technology"},
                  "capacity": {"type": "bar", "output": "capacity", "x": "technology"},
                  "total": {"type": "timeseries", "output": "generation", "x": "snapshot"}},
    })
    index = pd.Index([0, 1, 2], name="point")
    points = pd.DataFrame({"slack": [0.0, 0.05, 0.05], "kind": ["optimum", "box", "nearest"],
                           "wind": [1.0, 2.0, 3.0], "TOTEX": [10.0, 10.4, 10.5]}, index=index)
    duals = pd.DataFrame({"wind": [0.1, 0.2, 0.3]}, index=index)
    sets = {"snapshot": pd.DataFrame({"snapshot": SNAPSHOTS}),
            "technology": pd.DataFrame({"technology": TECHS, "label": ["Gas Power", None], "color": ["#a85522", "#235ebc"]})}
    # generation[point, snapshot, tech] = 100 * point + 10 * snapshot + tech position
    generation = pd.MultiIndex.from_product([index, SNAPSHOTS, TECHS], names=["point", "snapshot", "technology"]).to_frame(index=False)
    generation["value"] = (100 * generation["point"] + 10 * generation["snapshot"].map({s: i for i, s in enumerate(SNAPSHOTS)})
                           + generation["technology"].map({t: i for i, t in enumerate(TECHS)})).astype(float)
    capacity = pd.DataFrame({"point": [0, 1, 2], "technology": ["wind"] * 3, "value": [1.0, 2.0, 3.0]})  # no gas rows at all
    directory = tmp_path / "set"
    mga_format.write(directory, mga_format.NearOptimalSet(manifest, points, duals, sets),
                     {"generation": generation, "capacity": capacity})
    return directory, mga_format.read(directory)


def test_navigable_points_drop_slack_and_kind(written):
    _, nos = written
    assert list(navigable_points(nos).columns) == ["wind", "TOTEX"]


def test_category_styles_skip_missing_values(written):
    _, nos = written
    labels, colors = category_styles(nos)
    assert labels == {"gas": "Gas Power"}
    assert colors == {"gas": "#a85522", "wind": "#235ebc"}


def test_mix_output_weights_points(written):
    directory, nos = written
    (mixed,) = mix_output(directory, nos, "generation", [Alpha(pd.Series({0: 0.25, 2: 0.75}))])
    # 0.25 * point 0 + 0.75 * point 2 -> 150 + 10 * snapshot + tech
    assert mixed.sel(snapshot=SNAPSHOTS[1], technology="wind").item() == pytest.approx(150 + 10 + 1)


def test_stacked_plot(written):
    directory, nos = written
    result = plot_data(directory, nos, "generation", [Alpha(pd.Series({1: 1.0}))])
    assert result["x_dim"] == SNAPSHOTS.as_unit("ns").asi8.tolist()
    assert result["data"] == [{"gas": [100.0, 110.0], "wind": [101.0, 111.0]}]


def test_unstacked_plot_sums_other_sets(written):
    directory, nos = written
    result = plot_data(directory, nos, "total", [Alpha(pd.Series({0: 1.0})), Alpha(pd.Series({1: 0.5, 2: 0.5}))])
    # point 0: (0 + 1), (20 + 1) summed over techs; points 1 and 2 mixed: 150 + 10 * snapshot per tech
    assert result["data"] == [[1.0, 21.0], [301.0, 321.0]]


def test_missing_combinations_count_as_zero(written):
    directory, nos = written
    result = plot_data(directory, nos, "capacity", [Alpha(pd.Series({0: 0.5, 2: 0.5}))])
    assert result["x_dim"] == TECHS
    assert result["data"] == [[0.0, 2.0]]
