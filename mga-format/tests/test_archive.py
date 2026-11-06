import zipfile

import pandas as pd
import pytest
from test_near_optimal_set import _nos, _outputs, _points

from mga_format import NearOptimalSetError, fetch, pack, read, read_output, unpack, write


@pytest.fixture
def archive(tmp_path):
    write(tmp_path / "set", _nos(), _outputs())
    return pack(tmp_path / "set", tmp_path / "set.zip")


def test_pack_keeps_subdirectories(archive):
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert {"manifest.json", "points.parquet", "sets/technology.parquet", "outputs/generation.parquet"} <= names


def test_unpack_round_trips(archive, tmp_path):
    nos = unpack(archive, tmp_path / "unpacked")
    pd.testing.assert_frame_equal(nos.points, _points())
    assert sorted(read_output(tmp_path / "unpacked", "capacity", points=[1])["point"].unique()) == [1]


def test_unpack_replaces_existing_directory(archive, tmp_path):
    target = tmp_path / "data"
    target.mkdir()
    (target / "stale.txt").write_text("old dataset")
    unpack(archive, target)
    assert not (target / "stale.txt").exists()
    read(target)


def test_broken_archive_leaves_directory_untouched(tmp_path):
    target = tmp_path / "data"
    target.mkdir()
    (target / "keep.txt").write_text("current dataset")
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(broken, "w") as zf:
        zf.writestr("manifest.json", '{"format": "something-else"}')
    with pytest.raises(NearOptimalSetError):
        unpack(broken, target)
    assert (target / "keep.txt").read_text() == "current dataset"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".unpack-")] == []


@pytest.mark.parametrize("member", ["../evil.parquet", "/abs.parquet", "sets/../../evil.parquet"])
def test_unpack_refuses_paths_outside_the_set(tmp_path, member):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr(member, "x")
    with pytest.raises(NearOptimalSetError, match="outside the set"):
        unpack(bad, tmp_path / "data")


def test_fetch_downloads_and_unpacks(archive, tmp_path):
    nos = fetch(archive.as_uri(), tmp_path / "data")
    assert nos.manifest.poi_names == ["wind", "solar"]
    assert list(tmp_path.glob("*.zip")) == [archive]  # the download itself is removed
