import pandas as pd
import pytest

import mga_compass as mc
import mga_format

LEVEL = {"norm": "linf", "normalize": True, "tol": 5, "tol_unit": "percent",
         "vertex_method": "bb", "objective_cuts": True, "iterations": 1, "gap": 1, "converged": True}


@pytest.fixture
def set_dir(tmp_path):
    """Two PoIs, two slack levels: the optimum, two designs at 1% and two at 5%."""
    manifest = mga_format.Manifest.model_validate({
        "format": "mga-near-optimal-set", "version": 1, "created": "2026-11-07T12:00:00+00:00",
        "cost": {"name": "TOTEX", "unit": "EUR", "optimum": 1000.0},
        "pois": [{"name": "wind", "unit": "MW", "description": "wind capacity", "definition": {}},
                 {"name": "solar", "unit": "MW", "definition": {}}],
        "slack_levels": [{"slack": 0.01, **LEVEL}, {"slack": 0.05, **LEVEL}],
        "fingerprint": {"network": "x"},
        "display": {"wind": {"unit": "GW", "factor": 1e-3}, "TOTEX": {"unit": "kEUR", "factor": 1e-3}},
    })
    index = pd.Index([0, 1, 2, 3, 4], name="point")
    points = pd.DataFrame({
        "slack": [0.0, 0.01, 0.01, 0.05, 0.05],
        "kind": ["optimum", "box", "box", "box", "nearest"],
        "wind": [2000.0, 2500.0, 1500.0, 4000.0, 1000.0],
        "solar": [500.0, 400.0, 700.0, 100.0, 900.0],
        "TOTEX": [1000.0, 1010.0, 1008.0, 1050.0, 1040.0],
    }, index=index)
    duals = pd.DataFrame({"wind": [0.0] * 5, "solar": [0.0] * 5}, index=index)
    directory = tmp_path / "set"
    mga_format.write(directory, mga_format.NearOptimalSet(manifest, points, duals))
    return directory


def test_load_directory_in_display_units(set_dir):
    space = mc.load(set_dir)
    assert space.pois == ["wind", "solar"]
    assert space.cost == "TOTEX"
    assert space.units == {"wind": "GW", "solar": "MW", "TOTEX": "kEUR"}
    assert list(space.points.columns) == ["wind", "solar", "TOTEX"]
    assert list(space.points["wind"]) == pytest.approx([2.0, 2.5, 1.5, 4.0, 1.0])
    assert list(space.points["solar"]) == [500.0, 400.0, 700.0, 100.0, 900.0]  # no display entry
    assert space.slack == 0.05
    assert space.directory == set_dir


def test_model_units(set_dir):
    space = mc.load(set_dir, units="model")
    assert space.units == {"wind": "MW", "solar": "MW", "TOTEX": "EUR"}
    assert list(space.points["wind"]) == [2000.0, 2500.0, 1500.0, 4000.0, 1000.0]


def test_unknown_units(set_dir):
    with pytest.raises(ValueError, match="units"):
        mc.load(set_dir, units="SI")


def test_smaller_slack_keeps_optimum_and_designs_up_to_it(set_dir):
    space = mc.load(set_dir, slack=0.01)
    assert space.slack == 0.01
    assert list(space.points.index) == [0, 1, 2]


def test_slack_must_be_a_level_of_the_set(set_dir):
    with pytest.raises(ValueError, match="slack levels"):
        mc.load(set_dir, slack=0.03)


def test_optimum_is_the_cheapest_design(set_dir):
    optimum = mc.load(set_dir).optimum
    assert optimum.name == 0
    assert optimum["TOTEX"] == pytest.approx(1.0)


def test_summary(set_dir):
    summary = mc.load(set_dir).summary()
    assert list(summary.index) == ["wind", "solar", "TOTEX"]
    assert summary.loc["wind", "unit"] == "GW"
    assert summary.loc["wind", ["min", "max", "optimum"]].tolist() == pytest.approx([1.0, 4.0, 2.0])
    assert summary.loc["wind", "description"] == "wind capacity"
    assert summary.loc["TOTEX", "description"] == ""


def test_repr_shows_designs_slack_and_units(set_dir):
    text = repr(mc.load(set_dir, units="model"))
    assert "5 near-optimal designs in 2 PoIs, slack 0.05" in text
    assert "0.01, 0.05" in text
    assert "4,000" in text and "MW" in text and "wind capacity" in text
    assert "<table" in mc.load(set_dir)._repr_html_()


def test_load_archive_into_directory(set_dir, tmp_path):
    archive = mga_format.pack(set_dir, tmp_path / "set.zip")
    space = mc.load(archive, tmp_path / "unpacked")
    assert space.directory == tmp_path / "unpacked"
    assert len(space.points) == 5


def test_load_url(set_dir, tmp_path):
    archive = mga_format.pack(set_dir, tmp_path / "set.zip")
    space = mc.load(archive.as_uri(), tmp_path / "fetched")
    assert len(space.points) == 5


def test_archive_needs_directory(set_dir, tmp_path):
    archive = mga_format.pack(set_dir, tmp_path / "set.zip")
    with pytest.raises(ValueError, match="directory"):
        mc.load(archive)


def test_directory_takes_no_second_directory(set_dir, tmp_path):
    with pytest.raises(ValueError, match="only for archives"):
        mc.load(set_dir, tmp_path / "elsewhere")
