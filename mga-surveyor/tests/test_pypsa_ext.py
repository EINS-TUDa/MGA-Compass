import numpy as np
import pandas as pd
import pypsa
import pytest

from mga_surveyor import (
    VertexEnumMethod,
    make_furthest_point_fn,
    optimize_mga_in_direction,
    optimize_mga_in_multiple_directions,
)

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


def test_optimize_mga_in_direction_resolve_main_objective_false_skips_pin_resolve():
    n = _build_network()
    optimal_wind = n.generators.at["wind", "p_nom_opt"]

    status, condition, coordinates, duals = optimize_mga_in_direction(
        n, direction={"wind": -1, "solar": 0}, dimensions=DIMENSIONS, slack=0.1, solver_name="highs",
        resolve_main_objective=False,
    )
    assert status == "ok"
    assert condition == "optimal"
    # no pinning constraints were ever added, so every dual is 0 -- not just solar's
    assert (duals == 0.0).all()
    # No pin-resolve happened, so n is left wherever the one (direction-
    # objective) solve put it -- solve_model writes solution values back onto
    # n's component tables regardless of which objective was solved. That's
    # not necessarily the network's own cost-optimal point (that's exactly
    # what the skipped pin-resolve would otherwise restore/preserve): the
    # MGA direction pushes wind down within the slack budget, so it ends up
    # at or below the original optimum, not equal to it.
    assert n.generators.at["wind", "p_nom_opt"] == pytest.approx(coordinates["wind"], abs=1e-4)
    assert coordinates["wind"] <= optimal_wind + 1e-6


def test_make_furthest_point_fn_matches_optimize_mga_in_direction():
    n = _build_network()
    furthest_point = make_furthest_point_fn(n, DIMENSIONS, slack=0.1, solver_name="highs", max_parallel=2)

    # keys order fixes the array axes: DIMENSIONS = {"wind": ..., "solar": ...}.
    # Two directions in one batched call -- solved in parallel underneath via
    # optimize_mga_in_multiple_directions, not two serial single-direction calls.
    directions = np.array([[-1.0, 0.0], [0.0, -1.0]])
    points = furthest_point(directions)
    assert points.shape == (2, 2)

    # Cross-check each row against optimize_mga_in_direction's own
    # direction-objective solve (before its pinned re-solve, which shouldn't
    # move the coordinates at all -- that's the whole point of pinning).
    # Fresh, independently solved networks, so these reference calls don't
    # interfere with each other's (or the batch's) linopy model / n.model state.
    for row, direction in enumerate(directions):
        n2 = _build_network()
        direction_dict = dict(zip(DIMENSIONS.keys(), direction.tolist()))
        _, _, coordinates, _ = optimize_mga_in_direction(
            n2, direction=direction_dict, dimensions=DIMENSIONS, slack=0.1, solver_name="highs",
        )
        np.testing.assert_allclose(
            points[row], coordinates.loc[list(DIMENSIONS.keys())].to_numpy(), atol=1e-4,
        )


def test_make_furthest_point_fn_direction_pushes_to_bound():
    n = _build_network()
    furthest_point = make_furthest_point_fn(n, DIMENSIONS, slack=0.1, solver_name="highs", max_parallel=2)
    optimal_wind = n.generators.at["wind", "p_nom_opt"]

    # direction=[-1, 0] maximizes -wind (minimizes wind); direction=[1, 0]
    # maximizes wind -- both solved together in one batched call.
    points = furthest_point(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    assert points[0, 0] <= optimal_wind + 1e-6
    assert points[1, 0] >= optimal_wind - 1e-6


def test_make_furthest_point_fn_bootstraps_method_directly():
    """End-to-end: pass all 2k box directions to make_furthest_point_fn in
    one batched call (solved in parallel), then bootstrap a real Method
    directly from the resulting points/halfspaces -- the same state
    runner.bootstrap_box would seed for the single-direction FurthestPointFn
    case, but built from one parallel batch instead of a 2k-call loop."""
    n = _build_network()
    furthest_point = make_furthest_point_fn(n, DIMENSIONS, slack=0.1, solver_name="highs", max_parallel=4)

    k = len(DIMENSIONS)
    box_directions = np.vstack([np.eye(k), -np.eye(k)])
    points = furthest_point(box_directions)
    assert points.shape == (2 * k, k)

    method = VertexEnumMethod(k=k, norm="linf", backend="highs")
    method.bootstrap(list(points), list(zip(box_directions, points)))

    assert method.outer_vertices is not None
    assert len(method.outer_vertices) > 0
    assert method.max_gap() >= 0.0


def test_make_furthest_point_fn_reads_cache_instead_of_resolving(tmp_path):
    n = _build_network()
    cache_dir = tmp_path / "furthest_point_cache"
    furthest_point = make_furthest_point_fn(
        n, DIMENSIONS, output_dir=cache_dir, slack=0.1, solver_name="highs", max_parallel=2,
    )

    direction = np.array([[-1.0, 0.0]])
    first = furthest_point(direction, point_ids=[0])
    cached_fn = cache_dir / "mga_result_0.nc"
    assert cached_fn.exists()

    # Tamper with the cached result's stashed coordinates -- a value no
    # genuine solve of this direction could ever produce -- so getting it
    # back on the next call proves the cache was actually read, not
    # silently re-solved underneath.
    cached_n = pypsa.Network(cached_fn)
    cached_n.meta["coordinates"] = {"wind": 999.0, "solar": 999.0}
    cached_n.export_to_netcdf(cached_fn)

    second = furthest_point(direction, point_ids=[0])
    np.testing.assert_allclose(second, [[999.0, 999.0]])
    assert not np.allclose(second, first)


def test_make_furthest_point_fn_caches_alongside_new_directions(tmp_path):
    n = _build_network()
    cache_dir = tmp_path / "furthest_point_cache"
    furthest_point = make_furthest_point_fn(
        n, DIMENSIONS, output_dir=cache_dir, slack=0.1, solver_name="highs", max_parallel=2,
    )

    directions = np.array([[-1.0, 0.0], [0.0, -1.0]])
    furthest_point(directions, point_ids=[0, 1])
    assert {p.name for p in cache_dir.glob("mga_result_*.nc")} == {"mga_result_0.nc", "mga_result_1.nc"}

    # Corrupt point 0's cache so a real re-solve vs. a cache read are
    # unambiguously distinguishable, same trick as the test above.
    cached_n = pypsa.Network(cache_dir / "mga_result_0.nc")
    cached_n.meta["coordinates"] = {"wind": 999.0, "solar": 999.0}
    cached_n.export_to_netcdf(cache_dir / "mga_result_0.nc")

    # point 0 repeats (should read the tampered cache); point 2 is new
    # (should actually solve) -- one batched call mixing a cache hit and a
    # cache miss.
    directions2 = np.array([[-1.0, 0.0], [1.0, 0.0]])
    second = furthest_point(directions2, point_ids=[0, 2])
    assert {p.name for p in cache_dir.glob("mga_result_*.nc")} == {
        "mga_result_0.nc", "mga_result_1.nc", "mga_result_2.nc",
    }
    # point 0: served from the tampered cache, not re-solved.
    np.testing.assert_allclose(second[0], [999.0, 999.0])
    # point 2: a genuine new solve, nowhere near the tampered sentinel value.
    assert not np.allclose(second[1], [999.0, 999.0])


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
