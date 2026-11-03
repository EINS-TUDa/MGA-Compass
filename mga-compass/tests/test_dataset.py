import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mga_compass import finalize_bundle, load_bundle, read_dataarray, verify_bundle, write_dataarray, write_frame
from mga_compass.dataset import MANIFEST_NAME, fetch_bundle


def _bundle(tmp_path):
    points = pd.DataFrame({"wind": [1.0, 2.0], "TOTEX": [10.0, 11.0]})
    write_frame(points, tmp_path / "points.parquet")
    write_frame(points.drop(columns="TOTEX"), tmp_path / "duals.parquet")
    da = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=["index", "technology"],
                      coords={"index": [0, 1], "technology": ["wind", "gas", "coal"]})
    write_dataarray(da, tmp_path / "capacity.parquet")
    manifest = {"obj_label": "TOTEX", "output_datasets": {"capacity": {"path": "capacity.parquet",
                                                                       "dims": ["index", "technology"]}}}
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    finalize_bundle(tmp_path)
    return points, da


def test_bundle_round_trip_keeps_values_and_coordinate_order(tmp_path):
    points, da = _bundle(tmp_path)
    manifest, loaded_points, outputs = load_bundle(tmp_path)
    pd.testing.assert_frame_equal(loaded_points, points)
    xr.testing.assert_identical(outputs["capacity"].rename(None), da)
    assert list(outputs["capacity"]["technology"].values) == ["wind", "gas", "coal"]  # not sorted
    assert set(manifest.files) == {"points.parquet", "duals.parquet", "capacity.parquet"}


def test_read_dataarray_keeps_datetime_coordinates_and_float32(tmp_path):
    snapshots = pd.date_range("2030-01-01", periods=3, freq="h")
    da = xr.DataArray(np.arange(12.0).reshape(2, 3, 2), dims=["index", "snapshot", "technology"],
                      coords={"index": [5, 1], "snapshot": snapshots, "technology": ["solar", "coal"]})
    write_dataarray(da, tmp_path / "g.parquet", float32=True)
    back = read_dataarray(tmp_path / "g.parquet", ["index", "snapshot", "technology"])
    assert back["snapshot"].dtype.kind == "M"
    assert back.dtype == np.float32
    assert list(back["index"].values) == [5, 1]  # written order, not sorted
    np.testing.assert_array_equal(back.values, da.values)


def test_write_dataarray_needs_string_last_dim(tmp_path):
    da = xr.DataArray(np.ones((2, 3)), dims=["index", "snapshot"],
                      coords={"index": [0, 1], "snapshot": pd.date_range("2030-01-01", periods=3, freq="h")})
    with pytest.raises(ValueError, match="string coordinates"):
        write_dataarray(da, tmp_path / "g.parquet")


def test_verify_rejects_a_corrupted_or_missing_file(tmp_path):
    _bundle(tmp_path)
    manifest, _, _ = load_bundle(tmp_path)
    (tmp_path / "capacity.parquet").write_bytes(b"not parquet")
    with pytest.raises(ValueError, match="capacity.parquet"):
        verify_bundle(tmp_path, manifest)
    (tmp_path / "duals.parquet").unlink()
    with pytest.raises(ValueError, match="duals.parquet"):
        verify_bundle(tmp_path, manifest)


def test_finalize_requires_every_referenced_file(tmp_path):
    _bundle(tmp_path)
    (tmp_path / "points.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="points.parquet"):
        finalize_bundle(tmp_path)


def test_fetch_bundle_downloads_and_verifies(tmp_path):
    source = tmp_path / "release"
    source.mkdir()
    points, _ = _bundle(source)
    dest = tmp_path / "data"
    fetch_bundle(source.as_uri(), dest)
    _, loaded_points, _ = load_bundle(dest)
    pd.testing.assert_frame_equal(loaded_points, points)
