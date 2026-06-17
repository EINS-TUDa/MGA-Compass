import pandas as pd
import pypsa
import pytest

from mga_compass.pypsa_ext import optimize_mga_in_direction, optimize_mga_in_multiple_directions

DIMENSIONS = {
    "wind": {"Generator": {"p_nom": {"wind": 1}}},
    "solar": {"Generator": {"p_nom": {"solar": 1}}},
}


def _build_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "bus")
    n.add("Load", "load", bus="bus", p_set=[10, 15, 12])
    n.add(
        "Generator", "wind",
        bus="bus", p_nom_extendable=True, capital_cost=1, marginal_cost=0,
        p_max_pu=[0.6, 0.3, 0.8],
    )
    n.add(
        "Generator", "solar",
        bus="bus", p_nom_extendable=True, capital_cost=1, marginal_cost=0,
        p_max_pu=[0.2, 0.9, 0.4],
    )
    n.add(
        "Generator", "gas",
        bus="bus", p_nom_extendable=True, capital_cost=0.5, marginal_cost=50,
    )
    n.optimize(solver_name="highs")
    return n


def test_optimize_mga_in_direction_matches_projected_coordinates():
    n = _build_network()
    status, condition, coordinates, duals = optimize_mga_in_direction(
        n, direction={"wind": -1, "solar": 0}, dimensions=DIMENSIONS, slack=0.1, solver_name="highs",
    )
    assert status == "ok"
    assert condition == "optimal"
    assert coordinates is not None
    # network ends up re-solved exactly at the returned coordinates
    assert n.generators.at["wind", "p_nom_opt"] == pytest.approx(coordinates["wind"], abs=1e-4)
    assert n.generators.at["solar", "p_nom_opt"] == pytest.approx(coordinates["solar"], abs=1e-4)

    # duals: wind was pinned (direction != 0) so it has a shadow price; solar wasn't
    # pinned (direction == 0), so no constraint was added for it and its dual is 0.
    assert set(duals.index) == {"wind", "solar"}
    assert duals["solar"] == 0.0


def test_optimize_mga_in_multiple_directions_exports_nc_files(tmp_path):
    n = _build_network()
    directions = pd.DataFrame({"wind": [-1, 0], "solar": [0, -1]})
    output_dir = tmp_path / "mga_results"
    dirs_df, coords_df, duals_df = optimize_mga_in_multiple_directions(
        n, directions=directions, dimensions=DIMENSIONS, output_dir=output_dir,
        slack=0.1, solver_name="highs", max_parallel=2,
    )
    assert len(dirs_df) == 2
    assert "TOTEX" in coords_df.columns
    assert set(duals_df.columns) == {"wind", "solar"}
    assert len(duals_df) == 2

    nc_files = sorted(output_dir.glob("mga_result_*.nc"))
    assert len(nc_files) == 2
    # the temporary input network file should be cleaned up; only result files remain
    assert sorted(p.name for p in output_dir.iterdir()) == [f.name for f in nc_files]


def test_optimize_mga_in_multiple_directions_raises_on_failed_direction(tmp_path):
    n = _build_network()
    directions = [
        {"wind": -1, "solar": 0},
        {"wind": -1},  # missing "solar" key: mismatches DIMENSIONS, forces a failure
    ]
    output_dir = tmp_path / "mga_results"
    # A failed direction is a bug (or a genuine solver problem), not something to
    # silently drop and continue past — the whole batch should fail loudly.
    with pytest.raises(ValueError, match="must match"):
        optimize_mga_in_multiple_directions(
            n, directions=directions, dimensions=DIMENSIONS, output_dir=output_dir,
            slack=0.1, solver_name="highs", max_parallel=2,
        )


def test_optimize_mga_in_multiple_directions_skip_existing(tmp_path):
    n = _build_network()
    directions = pd.DataFrame({"wind": [-1, 0], "solar": [0, -1]})
    output_dir = tmp_path / "mga_results"

    dirs_df, coords_df, duals_df = optimize_mga_in_multiple_directions(
        n, directions=directions, dimensions=DIMENSIONS, output_dir=output_dir,
        point_ids=[10, 20], slack=0.1, solver_name="highs", max_parallel=2,
    )
    assert sorted(coords_df.index) == [10, 20]
    assert {p.name for p in output_dir.glob("mga_result_*.nc")} == {"mga_result_10.nc", "mga_result_20.nc"}

    # Re-running with the same point_ids and skip_existing=True should find both
    # already solved and return empty results without solving anything again.
    dirs_df2, coords_df2, duals_df2 = optimize_mga_in_multiple_directions(
        n, directions=directions, dimensions=DIMENSIONS, output_dir=output_dir,
        point_ids=[10, 20], skip_existing=True, slack=0.1, solver_name="highs", max_parallel=2,
    )
    assert dirs_df2.empty
    assert coords_df2.empty
    assert duals_df2.empty

    # A mix of one already-solved point and one new one only solves the new one.
    directions3 = pd.DataFrame({"wind": [-1, -1], "solar": [0, 0]})
    dirs_df3, coords_df3, duals_df3 = optimize_mga_in_multiple_directions(
        n, directions=directions3, dimensions=DIMENSIONS, output_dir=output_dir,
        point_ids=[10, 30], skip_existing=True, slack=0.1, solver_name="highs", max_parallel=2,
    )
    assert list(coords_df3.index) == [30]
    assert {p.name for p in output_dir.glob("mga_result_*.nc")} == {
        "mga_result_10.nc", "mga_result_20.nc", "mga_result_30.nc",
    }
