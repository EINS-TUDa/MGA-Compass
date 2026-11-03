import shutil
from pathlib import Path

import numpy as np
import pytest
import pandas as pd
import xarray as xr
from mga_compass import Constraints, finalize_bundle, write_dataarray, write_frame

from mgaserver.config import load_manifest, settings, Manifest

MANIFEST_TEMPLATE = Path(__file__).parent / "manifest.json"


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
    """A small, real dataset bundle (see mga_compass.dataset), loaded by the app
    through its normal startup path -- settings.data_dir points at it."""
    bundle = tmp_path_factory.mktemp("dataset")
    shutil.copy(MANIFEST_TEMPLATE, bundle / "manifest.json")
    write_frame(points_df, bundle / "points.parquet")
    write_frame(points_df.drop(columns="TOTEX") * 0.0, bundle / "duals.parquet")
    index, techs = points_df.index.tolist(), ["gas", "coal"]
    write_dataarray(xr.DataArray(np.ones((len(index), 2)), dims=["index", "technology"],
                                 coords={"index": index, "technology": techs}),
                    bundle / "installed_capacity.parquet")
    snapshots = pd.date_range("2030-01-01", periods=3, freq="h")
    write_dataarray(xr.DataArray(np.ones((len(index), 3, 2)), dims=["index", "snapshot", "technology"],
                                 coords={"index": index, "snapshot": snapshots, "technology": techs}),
                    bundle / "generation_over_snapshots.parquet")
    finalize_bundle(bundle)
    settings.data_dir = bundle
    return bundle


@pytest.fixture(scope="session")
def manifest(dataset_dir) -> Manifest:
    return load_manifest(settings.manifest_path)


@pytest.fixture(scope="session")
def duals_df(points_df, manifest) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = [c for c in points_df.columns if c != manifest.obj_label]
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
