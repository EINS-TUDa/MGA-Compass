import numpy as np
import pytest
import pandas as pd
from mga_compass.schemes import Constraints

OBJ_LABEL = "TOTEX"


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


@pytest.fixture(scope="session")
def duals_df(points_df) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = [c for c in points_df.columns if c != OBJ_LABEL]
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
