import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mga_format import near_optimal_set as nos_format
from mga_format.near_optimal_set import (
    MANIFEST_NAME,
    Manifest,
    NearOptimalSet,
    NearOptimalSetError,
    read,
    read_manifest,
    read_output,
    validate_output,
    write,
)

POIS = ["wind", "solar"]
SNAPSHOTS = pd.date_range("2013-01-01", periods=3, freq="h")
TECHNOLOGIES = ["wind", "solar", "gas"]


def _manifest(**overrides) -> dict:
    data = {
        "format": "mga-near-optimal-set",
        "version": 1,
        "created": "2026-11-06T12:00:00+00:00",
        "software": {"mga-surveyor": "0.1.0"},
        "cost": {"name": "TOTEX", "unit": "EUR", "optimum": 100.0},
        "pois": [
            {"name": "wind", "unit": "MW", "definition": {"Generator": {"p_nom": {"wind": 1}}}},
            {"name": "solar", "unit": "MW", "definition": {"Generator": {"p_nom": {"solar": 1}}}},
        ],
        "slack_levels": [
            {"slack": s, "norm": "linf", "normalize": True, "tol": 5.0, "tol_unit": "percent",
             "vertex_method": "bb", "objective_cuts": True, "iterations": 3, "gap": 4.0, "converged": True}
            for s in (0.02, 0.05)
        ],
        "fingerprint": {"network": "abc123"},
        "sets": {"snapshot": {"description": "Time steps."}, "technology": {}},
        "outputs": {
            "generation": {"index": ["point", "snapshot", "technology"], "unit": "MW"},
            "capacity": {"index": ["point", "technology"], "unit": "MW"},
        },
        "plots": {
            "generation": {"type": "stacked_timeseries", "output": "generation", "x": "snapshot", "stack": "technology"},
            "capacity": {"type": "bar", "output": "capacity", "x": "technology"},
        },
    }
    data.update(overrides)
    return data


def _points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slack": [0.0, 0.02, 0.02, 0.05, 0.05],
            "kind": ["optimum", "box", "nearest", "box", "nearest"],
            "wind": [10.0, 14.0, 12.0, 20.0, 16.0],
            "solar": [5.0, 3.0, 6.0, 1.0, 8.0],
            "TOTEX": [100.0, 102.0, 101.0, 105.0, 104.0],
        },
        index=pd.Index(range(5), name="point"),
    )


def _duals() -> pd.DataFrame:
    return pd.DataFrame(
        {"wind": [0.1, 0.2, np.nan, 0.3, 0.4], "solar": [0.5, 0.6, np.nan, 0.7, 0.8]},
        index=pd.Index(range(5), name="point"),
    )


def _sets() -> dict[str, pd.DataFrame]:
    return {
        "snapshot": pd.DataFrame({"snapshot": SNAPSHOTS}),
        "technology": pd.DataFrame({
            "technology": TECHNOLOGIES,
            "label": ["Wind", "Solar", "Gas"],
            "color": ["#235ebc", "#f9d002", "#a85522"],
        }),
    }


def _outputs() -> dict[str, pd.DataFrame]:
    index = pd.MultiIndex.from_product([range(5), SNAPSHOTS, TECHNOLOGIES], names=["point", "snapshot", "technology"])
    generation = index.to_frame(index=False)
    generation["value"] = np.arange(len(generation), dtype=float)
    capacity = pd.MultiIndex.from_product([range(5), TECHNOLOGIES], names=["point", "technology"]).to_frame(index=False)
    capacity["value"] = 1.0
    # Unsorted on purpose: the writer sorts by point.
    return {"generation": generation.sample(frac=1, random_state=0), "capacity": capacity}


def _nos(**manifest_overrides) -> NearOptimalSet:
    return NearOptimalSet(
        manifest=Manifest.model_validate(_manifest(**manifest_overrides)),
        points=_points(), duals=_duals(), sets=_sets(),
    )


@pytest.fixture
def written(tmp_path):
    directory = tmp_path / "set"
    write(directory, _nos(), _outputs())
    return directory


# ---------------------------------------------------------------- round trip


def test_round_trip(written):
    nos = read(written)
    pd.testing.assert_frame_equal(nos.points, _points())
    pd.testing.assert_frame_equal(nos.duals, _duals())
    assert list(nos.sets["technology"]["label"]) == ["Wind", "Solar", "Gas"]
    assert nos.manifest.poi_names == POIS
    assert sorted(nos.manifest.files) == [
        "duals.parquet", "outputs/capacity.parquet", "outputs/generation.parquet",
        "points.parquet", "sets/snapshot.parquet", "sets/technology.parquet",
    ]


def test_missing_duals_stay_nan(written):
    assert read(written).duals.loc[2].isna().all()


def test_outputs_are_sorted_with_one_row_group_per_point(written):
    nos = read(written)
    validate_output(written, nos, "generation")
    assert pq.ParquetFile(written / "outputs/generation.parquet").metadata.num_row_groups == 5


def test_read_output_for_some_points(written):
    df = read_output(written, "generation", points=[1, 3])
    assert sorted(df["point"].unique()) == [1, 3]
    assert len(df) == 2 * len(SNAPSHOTS) * len(TECHNOLOGIES)


def test_write_refuses_existing_directory(written):
    with pytest.raises(FileExistsError):
        write(written, _nos(), _outputs())


# ---------------------------------------------------------------- manifest


def test_unknown_version_is_refused(written):
    data = json.loads((written / MANIFEST_NAME).read_text())
    data["version"] = 2
    (written / MANIFEST_NAME).write_text(json.dumps(data))
    with pytest.raises(NearOptimalSetError, match="version 2 is not supported"):
        read_manifest(written)


def test_other_format_is_refused(written):
    (written / MANIFEST_NAME).write_text(json.dumps({"obj_label": "TOTEX"}))
    with pytest.raises(NearOptimalSetError, match="not a near-optimal set"):
        read_manifest(written)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"created": "2026-11-06T12:00:00"}, "time zone"),
        ({"pois": [{"name": "wind", "unit": "MW", "definition": {}}] * 2}, "unique"),
        ({"pois": [{"name": "kind", "unit": "", "definition": {}}]}, "reserved"),
        ({"cost": {"name": "slack", "unit": "EUR", "optimum": 1.0}}, "cost column"),
        ({"sets": {"point": {}}}, "built-in"),
        ({"outputs": {"g": {"index": ["snapshot"]}}}, "start with 'point'"),
        ({"outputs": {"g": {"index": ["point", "node"]}}, "plots": {}}, "undeclared sets"),
        ({"plots": {"p": {"type": "bar", "output": "missing", "x": "technology"}}}, "unknown output"),
        ({"plots": {"p": {"type": "stacked_bar", "output": "capacity", "x": "technology"}}}, "needs 'stack'"),
        ({"plots": {"p": {"type": "bar", "output": "capacity", "x": "snapshot"}}}, "not a set of output"),
    ],
)
def test_invalid_manifest(overrides, message):
    with pytest.raises(ValueError, match=message):
        Manifest.model_validate(_manifest(**overrides))


# ---------------------------------------------------------------- tables


def _broken(points=None, duals=None, sets=None) -> NearOptimalSet:
    nos = _nos()
    return NearOptimalSet(
        nos.manifest,
        points if points is not None else nos.points,
        duals if duals is not None else nos.duals,
        sets if sets is not None else nos.sets,
    )


def _with(df, **changes):
    df = df.copy()
    for column, values in changes.items():
        df[column] = values
    return df


@pytest.mark.parametrize(
    ("nos", "message"),
    [
        (_broken(points=_points().set_axis(pd.Index([0, 1, 1, 3, 4], name="point"))), "unique"),
        (_broken(points=_points().drop(columns="solar")), "columns must be"),
        (_broken(points=_with(_points(), kind=["optimum", "box", "middle", "box", "nearest"])), "'kind' must be"),
        (_broken(points=_with(_points(), slack=[0.0, 0.0, 0.02, 0.05, 0.05])), "0 exactly for the optimum"),
        (_broken(points=_with(_points(), slack=[0.0, 0.03, 0.02, 0.05, 0.05])), "not in the manifest"),
        (_broken(points=_with(_points(), wind=[10.0, np.nan, 12.0, 20.0, 16.0])), "missing values"),
        (_broken(duals=_duals().iloc[:4]), "same as in points"),
        (_broken(sets={"snapshot": pd.DataFrame({"time": SNAPSHOTS}), "technology": _sets()["technology"]}), "first column"),
        (_broken(sets={"snapshot": pd.DataFrame({"snapshot": SNAPSHOTS[[0, 0, 1]]}), "technology": _sets()["technology"]}), "unique"),
        (_broken(sets={"snapshot": _sets()["snapshot"]}), "do not match"),
    ],
)
def test_invalid_tables(nos, message):
    with pytest.raises(NearOptimalSetError, match=message):
        nos.validate()


# ---------------------------------------------------------------- files and outputs


def test_corrupt_file_is_refused(written):
    pd.DataFrame({"x": [1]}).to_parquet(written / "sets/snapshot.parquet")
    with pytest.raises(NearOptimalSetError, match="missing or corrupt"):
        read(written)


def test_output_key_outside_its_set(tmp_path):
    outputs = _outputs()
    outputs["capacity"].loc[0, "technology"] = "coal"
    with pytest.raises(NearOptimalSetError, match="not members of set 'technology'"):
        write(tmp_path / "set", _nos(), outputs)


def test_output_point_outside_points(tmp_path):
    outputs = _outputs()
    outputs["capacity"].loc[0, "point"] = 99
    with pytest.raises(NearOptimalSetError, match="not members of set 'point'"):
        write(tmp_path / "set", _nos(), outputs)


def test_output_with_wrong_columns(tmp_path):
    outputs = _outputs()
    outputs["capacity"] = outputs["capacity"].rename(columns={"value": "MW"})
    with pytest.raises(NearOptimalSetError, match="columns must be"):
        write(tmp_path / "set", _nos(), outputs)


def test_output_without_row_group_per_point(written):
    path = written / "outputs/capacity.parquet"
    df = pd.read_parquet(path)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)  # a single row group
    with pytest.raises(NearOptimalSetError, match="one row group"):
        validate_output(written, read_without_checksums(written), "capacity")


def read_without_checksums(directory):
    """Load a set whose files were changed after writing, for testing validate_output."""
    manifest = read_manifest(directory)
    return NearOptimalSet(
        manifest,
        pd.read_parquet(directory / nos_format.POINTS_FILE).set_index("point"),
        pd.read_parquet(directory / nos_format.DUALS_FILE).set_index("point"),
        {name: pd.read_parquet(directory / f"sets/{name}.parquet") for name in manifest.sets},
    )


def test_manifest_created_is_preserved(written):
    assert read(written).manifest.created == datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
