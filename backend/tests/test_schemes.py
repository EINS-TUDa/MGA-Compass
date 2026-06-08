import pytest
import pandas as pd
from mgaserver.schemes import (
    Alpha, NegativeAlphaError, AlphaSumError,
    Constraints, ConstraintsIndexError, ConstraintsColumnsError,
    ConstraintsDirectionError, ConstraintsDeltaError, ConstraintsValueError,
)


def test_Alpha_type():
    alpha = Alpha([0.3, 0.7], index=[0, 2])
    assert isinstance(alpha, Alpha)
    assert alpha.sum() == 1.0

    with pytest.raises(AlphaSumError):
        Alpha([0.3, 0.5], index=[0, 2])
    with pytest.raises(NegativeAlphaError):
        Alpha([-0.1, 1.1], index=[0, 2])


def test_Constraints_validate(points_df, constraints_df):
    constraints_df.validate(points_df)  # valid — should not raise

    # wrong index
    bad_index = Constraints(constraints_df.rename(index={"wind": "gas"}))
    with pytest.raises(ConstraintsIndexError):
        bad_index.validate(points_df)

    # missing column
    bad_cols = Constraints(constraints_df.drop(columns=["delta"]))
    with pytest.raises(ConstraintsColumnsError):
        bad_cols.validate(points_df)

    # invalid direction
    bad_dir = constraints_df.copy()
    bad_dir.loc["wind", "direction"] = "!="
    with pytest.raises(ConstraintsDirectionError):
        bad_dir.validate(points_df)

    # negative delta
    bad_delta = constraints_df.copy()
    bad_delta.loc["solar", "delta"] = -1.0
    with pytest.raises(ConstraintsDeltaError):
        bad_delta.validate(points_df)

    # value out of range
    bad_value = constraints_df.copy()
    bad_value.loc["wind", "value"] = 999.0
    with pytest.raises(ConstraintsValueError):
        bad_value.validate(points_df)


def test_Constraints_changed(constraints_df):
    other = constraints_df.copy()

    # nothing changed
    result = constraints_df.changed(other)
    assert not result.any()

    # direction changed
    other_dir = constraints_df.copy()
    other_dir.loc["wind", "direction"] = "<="
    result = constraints_df.changed(other_dir)
    assert result["wind"]
    assert not result.drop("wind").any()

    # delta changed
    other_delta = constraints_df.copy()
    other_delta.loc["solar", "delta"] = 99.0
    result = constraints_df.changed(other_delta)
    assert result["solar"]
    assert not result.drop("solar").any()

    # both changed
    other_both = constraints_df.copy()
    other_both.loc["battery", "direction"] = ">="
    other_both.loc["battery", "delta"] = 1.0
    result = constraints_df.changed(other_both)
    assert result["battery"]
