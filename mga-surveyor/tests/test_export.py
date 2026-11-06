import json
import shutil

import numpy as np
import pypsa
import pytest

import mga_format
from mga_surveyor import consolidate_mga_results, export_near_optimal_set, generate_mga_files, network_fingerprint

DIMENSIONS = {
    "wind": {"Generator": {"p_nom": {"wind": 1}}},
    "solar": {"Generator": {"p_nom": {"solar": 1}}},
}
UNITS = {"wind": "MW", "solar": "MW"}
SLACKS = [0.1, 0.2]


def _build_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[10, 15, 12])
    n.add("Generator", "wind", bus="bus", p_nom_extendable=True, capital_cost=1, marginal_cost=0,
          p_max_pu=[0.6, 0.3, 0.8])
    n.add("Generator", "solar", bus="bus", p_nom_extendable=True, capital_cost=1, marginal_cost=0,
          p_max_pu=[0.2, 0.9, 0.4])
    n.add("Generator", "gas", bus="bus", p_nom_extendable=True, capital_cost=0.5, marginal_cost=50)
    n.optimize(solver_name="highs", include_objective_constant=True)
    return n


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One generate_mga_files run, shared by the tests: (network, output_dir)."""
    n = _build_network()
    output_dir = tmp_path_factory.mktemp("mga")
    generate_mga_files(
        n, DIMENSIONS, output_dir, norm="linf", slack_levels=SLACKS, solver_name="highs",
        tol=10, max_iter=5, max_parallel=1, verbose=False, vertex_method="bb",
    )
    return n, output_dir


@pytest.fixture(scope="module")
def exported(run, tmp_path_factory):
    _, output_dir = run
    directory = tmp_path_factory.mktemp("export") / "set"
    export_near_optimal_set(output_dir, directory, units=UNITS, cost_unit="EUR",
                            descriptions={"wind": "Installed wind capacity."})
    return directory


# ---------------------------------------------------------------- run summaries


def test_run_summary_per_slack_level(run):
    _, output_dir = run
    for slack in SLACKS:
        slack_dir = output_dir / f"slack_{slack}"
        summary = json.loads((slack_dir / "run.json").read_text())
        assert summary["slack"] == slack
        assert summary["norm"] == "linf"
        assert summary["normalize"] is True
        assert summary["tol"] == pytest.approx(10)
        assert summary["tol_unit"] == "percent"
        assert summary["vertex_method"] == "bb"
        assert summary["objective_cuts"] is True
        assert summary["iterations"] == len(list(slack_dir.glob("mga_closest_result_*.nc")))
        assert summary["converged"] == (summary["gap"] <= summary["tol"])
        mga_format.SlackLevel.model_validate(summary)  # matches the format's slack_levels entry


# ---------------------------------------------------------------- export


def test_export_round_trips(run, exported):
    n, output_dir = run
    nos = mga_format.read(exported)
    points, _, files = consolidate_mga_results(output_dir, obj_label="cost")

    assert len(nos.points) == len(files)
    assert nos.points.loc[0, "kind"] == "optimum"
    assert nos.points.loc[0, "slack"] == 0
    assert (nos.points["kind"] == "optimum").sum() == 1
    for slack in SLACKS:
        at_slack = nos.points[nos.points["slack"] == slack]
        assert (at_slack["kind"] == "box").sum() == 2 * len(DIMENSIONS)
        assert (at_slack["kind"] == "nearest").sum() == len(list((output_dir / f"slack_{slack}").glob("mga_closest_*.nc")))

    # The same points and costs as consolidate_mga_results, just labelled.
    exported_rows = sorted(map(tuple, nos.points[["wind", "solar", "cost"]].round(6).to_numpy()))
    consolidated_rows = sorted(map(tuple, points[["wind", "solar", "cost"]].round(6).to_numpy()))
    assert exported_rows == consolidated_rows
    assert not nos.duals.isna().any().any()

    manifest = nos.manifest
    assert manifest.cost.name == "cost"
    assert manifest.cost.unit == "EUR"
    assert manifest.cost.optimum == pytest.approx(nos.points.loc[0, "cost"])
    assert manifest.poi_names == list(DIMENSIONS)
    assert [p.unit for p in manifest.pois] == ["MW", "MW"]
    assert manifest.pois[0].description == "Installed wind capacity."
    assert manifest.pois[0].definition == DIMENSIONS["wind"]
    assert manifest.slacks == SLACKS
    assert manifest.fingerprint.network == network_fingerprint(n)
    assert "mga-surveyor" in manifest.software


def test_export_needs_every_unit(run, tmp_path):
    _, output_dir = run
    with pytest.raises(ValueError, match="no unit given for PoIs"):
        export_near_optimal_set(output_dir, tmp_path / "set", units={"wind": "MW"}, cost_unit="EUR")


def test_export_needs_finished_slack_levels(run, tmp_path):
    _, output_dir = run
    copy = tmp_path / "mga"
    shutil.copytree(output_dir, copy)
    (copy / f"slack_{SLACKS[1]}" / "run.json").unlink()
    with pytest.raises(FileNotFoundError, match="did not finish"):
        export_near_optimal_set(copy, tmp_path / "set", units=UNITS, cost_unit="EUR")


def test_export_refuses_results_without_duals(run, tmp_path):
    _, output_dir = run
    copy = tmp_path / "mga"
    shutil.copytree(output_dir, copy)
    result = sorted((copy / f"slack_{SLACKS[0]}").glob("mga_furthest_result_*.nc"))[0]
    m = pypsa.Network(result)
    del m.meta["duals"]
    m.export_to_netcdf(result)
    with pytest.raises(ValueError, match="resolve_main_objective=False"):
        export_near_optimal_set(copy, tmp_path / "set", units=UNITS, cost_unit="EUR")
    assert not (tmp_path / "set").exists()


# ---------------------------------------------------------------- fingerprint


def test_fingerprint_ignores_results_but_not_inputs():
    n = _build_network()
    fingerprint = network_fingerprint(n)
    n.optimize(solver_name="highs", include_objective_constant=True)
    assert network_fingerprint(n) == fingerprint  # solving again changes only results

    n.generators.loc["gas", "capital_cost"] = 0.6
    assert network_fingerprint(n) != fingerprint
    n.generators.loc["gas", "capital_cost"] = 0.5
    assert network_fingerprint(n) == fingerprint

    n.generators_t.p_max_pu.loc[0, "wind"] = 0.5
    assert network_fingerprint(n) != fingerprint


def test_fingerprint_survives_netcdf(tmp_path):
    n = _build_network()
    n.export_to_netcdf(tmp_path / "n.nc")
    assert network_fingerprint(pypsa.Network(tmp_path / "n.nc")) == network_fingerprint(n)
