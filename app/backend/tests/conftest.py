from pathlib import Path

import numpy as np
import pytest
import pandas as pd
import mga_format
from mga_compass import Constraints

from mgaserver.config import settings

UNITS = {"wind": "MW", "solar": "MW", "hydrogen": "MWh", "battery": "MW"}


@pytest.fixture(scope="session")
def points_df() -> pd.DataFrame:
    return pd.DataFrame({
        "wind": [10, 15, 18, 28],
        "solar": [20, 25, 30, 35],
        "hydrogen": [45, 38, 30, 25],
        "battery": [25, 22, 20, 15],
        "TOTEX": [35, 120, 40, 32]
    })


@pytest.fixture(scope="session")
def dimensions(points_df):
    return points_df.columns.tolist()


@pytest.fixture(scope="session", autouse=True)
def dataset_dir(tmp_path_factory, points_df) -> Path:
    """A small, real near-optimal set (see mga_format), loaded by the app through
    its normal startup path -- settings.data_dir points at it."""
    pois = [c for c in points_df.columns if c != "TOTEX"]
    manifest = mga_format.Manifest.model_validate({
        "format": "mga-near-optimal-set", "version": 1, "created": "2026-11-06T12:00:00+00:00",
        "cost": {"name": "TOTEX", "unit": "€", "optimum": float(points_df["TOTEX"].min())},
        "pois": [{"name": p, "unit": UNITS[p], "description": f"Installed {p} capacity.", "definition": {}}
                 for p in pois],
        "slack_levels": [{"slack": 0.05, "norm": "linf", "normalize": True, "tol": 5, "tol_unit": "percent",
                          "vertex_method": "bb", "objective_cuts": True, "iterations": 0, "gap": 1,
                          "converged": True}],
        "fingerprint": {"network": "test"},
        "sets": {"snapshot": {}, "technology": {}},
        "outputs": {"generation": {"index": ["point", "snapshot", "technology"], "unit": "MW"},
                    "capacity": {"index": ["point", "technology"], "unit": "MW"}},
        "plots": {
            "generation": {"type": "stacked_timeseries", "output": "generation", "x": "snapshot", "stack": "technology"},
            "capacity": {"type": "bar", "output": "capacity", "x": "technology"},
            "total generation": {"type": "timeseries", "output": "generation", "x": "snapshot"},
        },
        # Factor 1 for the cost keeps the points' values as in points_df, which most
        # tests compare against; the generation output is shown in GW.
        "display": {"TOTEX": {"unit": "k€", "factor": 1.0}, "generation": {"unit": "GW", "factor": 1e-3}},
    })
    index = pd.Index(points_df.index, name="point")
    optimum = points_df["TOTEX"].to_numpy().argmin()
    points = points_df.astype(float).set_axis(index)
    points.insert(0, "kind", ["optimum" if i == optimum else "box" for i in range(len(points))])
    points.insert(0, "slack", [0.0 if i == optimum else 0.05 for i in range(len(points))])
    duals = pd.DataFrame(0.0, index=index, columns=pois)
    techs = ["gas", "coal"]
    snapshots = pd.date_range("2030-01-01", periods=3, freq="h")
    sets = {
        "snapshot": pd.DataFrame({"snapshot": snapshots}),
        "technology": pd.DataFrame({"technology": techs, "label": ["Gas Power", None], "color": ["#a85522", None]}),
    }
    capacity = pd.MultiIndex.from_product([index, techs], names=["point", "technology"]).to_frame(index=False)
    capacity["value"] = 1.0
    generation = pd.MultiIndex.from_product([index, snapshots, techs], names=["point", "snapshot", "technology"]).to_frame(index=False)
    generation["value"] = 1.0
    directory = tmp_path_factory.mktemp("dataset") / "set"
    mga_format.write(directory, mga_format.NearOptimalSet(manifest, points, duals, sets),
                     {"generation": generation, "capacity": capacity})
    settings.data_dir = directory
    return directory


@pytest.fixture(scope="session")
def manifest(dataset_dir) -> mga_format.Manifest:
    return mga_format.read_manifest(dataset_dir)


@pytest.fixture(scope="session")
def duals_df(points_df, manifest) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = [c for c in points_df.columns if c != manifest.cost.name]
    return pd.DataFrame(
        rng.random((len(points_df), len(cols))),
        columns=cols,
    )


@pytest.fixture(scope="session")
def outer_approximation_df(points_df) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 4
    data = {col: rng.random(n) * (points_df[col].max() - points_df[col].min()) + points_df[col].min()
            for col in points_df.columns}
    data["direction"] = ["<=", ">=", "==", "<="]
    data["RHS"] = rng.random(n).tolist()
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def constraints_df():
    df = pd.DataFrame({
        "direction": [">=", "<=", "", "==", ""],
        "value": [10, 20, 45, 25, 100],
        "delta": [2, 3, 5, 7, 10]
    }, index=["wind", "solar", "hydrogen", "battery", "TOTEX"])
    df[["value", "delta"]] = df[["value", "delta"]].astype(float)
    return Constraints(df)
