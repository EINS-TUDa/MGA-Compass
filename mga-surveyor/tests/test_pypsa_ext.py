import numpy as np
import pandas as pd
import pypsa
import pytest

from mga_surveyor import (
    VertexEnumMethod,
    consolidate_mga_results,
    encode_cache_key,
    generate_mga_files,
    make_closest_point_fn,
    make_furthest_point_fn,
    make_optimality_fn,
    make_oracle_fns,
    drive as mga_surveyor_drive,
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
    n.optimize(solver_name="highs", include_objective_constant=True)
    return n


def test_encode_cache_key_is_deterministic_and_ten_hex_chars():
    key = encode_cache_key(np.array([1.0, 0.0]))
    assert len(key) == 10
    assert all(c in "0123456789abcdef" for c in key)
    # Same vector, independently constructed -- same key.
    assert key == encode_cache_key(np.eye(2)[0])
    # Different vector -- different key.
    assert key != encode_cache_key(np.array([0.0, 1.0]))


def test_encode_cache_key_normalizes_negative_zero():
    # -np.eye(k)'s off-diagonal entries are -0.0 (a different IEEE 754 bit
    # pattern from 0.0 despite being numerically equal) -- must still hash
    # the same as the +0.0 version.
    assert encode_cache_key(np.array([-1.0, 0.0])) == encode_cache_key(np.array([-1.0, -0.0]))


def test_make_optimality_fn_returns_point_and_caches(tmp_path):
    n = _build_network()
    solve_optimal = make_optimality_fn(n, DIMENSIONS, tmp_path, solver_name="highs")

    point = solve_optimal()
    expected = np.array([n.generators.at["wind", "p_nom_opt"], n.generators.at["solar", "p_nom_opt"]])
    np.testing.assert_allclose(point, expected, atol=1e-6)
    assert (tmp_path / "optimal.nc").exists()

    # second call, fresh network + a solver name that would raise if it
    # actually tried to solve: proves the cache was read, not re-solved.
    n2 = _build_network()
    solve_optimal2 = make_optimality_fn(n2, DIMENSIONS, tmp_path, solver_name="not_a_real_solver")
    point2 = solve_optimal2()
    np.testing.assert_allclose(point2, point)


def test_make_furthest_point_fn_resolve_main_objective_true_returns_duals(tmp_path):
    """resolve_main_objective=True is what optimize_mga_in_direction used to
    be for -- a single direction, pinned re-solve, duals cached to n.meta
    (not part of the return value, which is always (points, normals))."""
    n = _build_network()
    furthest_point = make_furthest_point_fn(
        n, DIMENSIONS, slack=0.1, output_dir=tmp_path, solver_name="highs", resolve_main_objective=True,
    )

    direction = np.array([-1.0, 0.0])  # wind pinned (!=0), solar not (==0)
    points, normals = furthest_point(direction)
    assert points.shape == (1, 2)
    assert normals.shape == (1, 2)
    np.testing.assert_allclose(normals[0], direction)  # normal is just the query direction

    # network ends up re-solved exactly at the returned coordinates (single-
    # direction path mutates n directly, same as the old optimize_mga_in_direction).
    assert n.generators.at["wind", "p_nom_opt"] == pytest.approx(points[0, 0], abs=1e-4)
    assert n.generators.at["solar", "p_nom_opt"] == pytest.approx(points[0, 1], abs=1e-4)

    # duals: cached in the exported .nc file's meta, not the return value.
    key = encode_cache_key(direction)
    cached_meta = pypsa.Network(tmp_path / f"mga_furthest_result_{key}.nc").meta
    keys = list(DIMENSIONS.keys())
    duals = np.array([cached_meta["duals"][k] for k in keys])
    # wind was pinned (direction != 0) so it has a shadow price; solar wasn't
    # pinned (direction == 0), so no constraint was added for it and its dual is 0.
    assert duals[keys.index("solar")] == 0.0


def test_make_furthest_point_fn_resolve_main_objective_false_skips_pin_resolve():
    n = _build_network()
    optimal_wind = n.generators.at["wind", "p_nom_opt"]

    furthest_point = make_furthest_point_fn(
        n, DIMENSIONS, slack=0.1, solver_name="highs", resolve_main_objective=False,
    )
    points, normals = furthest_point(np.array([-1.0, 0.0]))
    np.testing.assert_allclose(normals[0], [-1.0, 0.0])  # normal is always the query direction

    # No pin-resolve happened, so n is left wherever the one (direction-
    # objective) solve put it -- solve_model writes solution values back onto
    # n's component tables regardless of which objective was solved. That's
    # not necessarily the network's own cost-optimal point (that's exactly
    # what the skipped pin-resolve would otherwise restore/preserve): the
    # MGA direction pushes wind down within the slack budget, so it ends up
    # at or below the original optimum, not equal to it.
    assert n.generators.at["wind", "p_nom_opt"] == pytest.approx(points[0, 0], abs=1e-4)
    assert points[0, 0] <= optimal_wind + 1e-6


def test_make_furthest_point_fn_direction_pushes_to_bound():
    n = _build_network()
    furthest_point = make_furthest_point_fn(n, DIMENSIONS, slack=0.1, solver_name="highs", max_parallel=2)
    optimal_wind = n.generators.at["wind", "p_nom_opt"]

    # direction=[-1, 0] maximizes -wind (minimizes wind); direction=[1, 0]
    # maximizes wind -- both solved together in one batched call.
    points, _normals = furthest_point(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    assert points[0, 0] <= optimal_wind + 1e-6
    assert points[1, 0] >= optimal_wind - 1e-6


def test_make_furthest_point_fn_bootstraps_method_directly():
    """End-to-end: pass all 2k box directions to make_furthest_point_fn in
    one batched call (solved in parallel), then bootstrap a real Method
    directly from the resulting points/normals -- the same state
    explore.drive's own bootstrap step would seed via its own call, but built
    here directly instead."""
    n = _build_network()
    furthest_point = make_furthest_point_fn(n, DIMENSIONS, slack=0.1, solver_name="highs", max_parallel=4)

    k = len(DIMENSIONS)
    box_directions = np.vstack([np.eye(k), -np.eye(k)])
    points, normals = furthest_point(box_directions)
    assert points.shape == (2 * k, k)
    np.testing.assert_allclose(normals, box_directions)

    method = VertexEnumMethod(k=k, norm="linf", backend="highs")
    method.bootstrap(list(points), list(zip(normals, points)))

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
    key = encode_cache_key(direction[0])
    first, _first_normals = furthest_point(direction)
    cached_fn = cache_dir / f"mga_furthest_result_{key}.nc"
    assert cached_fn.exists()

    # Tamper with the cached result's stashed coordinates -- a value no
    # genuine solve of this direction could ever produce -- so getting it
    # back on the next call proves the cache was actually read, not
    # silently re-solved underneath.
    cached_n = pypsa.Network(cached_fn)
    cached_n.meta["coordinates"] = {"wind": 999.0, "solar": 999.0}
    cached_n.export_to_netcdf(cached_fn)

    second, _second_normals = furthest_point(direction)
    np.testing.assert_allclose(second, [[999.0, 999.0]])
    assert not np.allclose(second, first)


def test_make_furthest_point_fn_caches_alongside_new_directions(tmp_path):
    n = _build_network()
    cache_dir = tmp_path / "furthest_point_cache"
    furthest_point = make_furthest_point_fn(
        n, DIMENSIONS, output_dir=cache_dir, slack=0.1, solver_name="highs", max_parallel=2,
    )

    directions = np.array([[-1.0, 0.0], [0.0, -1.0]])
    keys = [encode_cache_key(d) for d in directions]
    furthest_point(directions)
    assert {p.name for p in cache_dir.glob("mga_furthest_result_*.nc")} == {f"mga_furthest_result_{k}.nc" for k in keys}

    # Corrupt direction 0's cache so a real re-solve vs. a cache read are
    # unambiguously distinguishable, same trick as the test above.
    cached_fn0 = cache_dir / f"mga_furthest_result_{keys[0]}.nc"
    cached_n = pypsa.Network(cached_fn0)
    cached_n.meta["coordinates"] = {"wind": 999.0, "solar": 999.0}
    cached_n.export_to_netcdf(cached_fn0)

    # direction 0 repeats (should read the tampered cache); [1.0, 0.0] is new
    # (should actually solve) -- one batched call mixing a cache hit and a
    # cache miss.
    directions2 = np.array([[-1.0, 0.0], [1.0, 0.0]])
    keys2 = [encode_cache_key(d) for d in directions2]
    second, _second_normals = furthest_point(directions2)
    assert {p.name for p in cache_dir.glob("mga_furthest_result_*.nc")} == {f"mga_furthest_result_{k}.nc" for k in {*keys, *keys2}}
    # direction 0: served from the tampered cache, not re-solved.
    np.testing.assert_allclose(second[0], [999.0, 999.0])
    # [1.0, 0.0]: a genuine new solve, nowhere near the tampered sentinel value.
    assert not np.allclose(second[1], [999.0, 999.0])


def test_make_closest_point_fn_rejects_l2():
    n = _build_network()
    with pytest.raises(NotImplementedError, match="l1.*linf|linf.*l1"):
        make_closest_point_fn(n, DIMENSIONS, norm="l2", slack=0.1, solver_name="highs")


def test_make_closest_point_fn_feasible_target_returns_target():
    n = _build_network()
    optimal = np.array([n.generators.at["wind", "p_nom_opt"], n.generators.at["solar", "p_nom_opt"]])
    closest_point = make_closest_point_fn(n, DIMENSIONS, norm="linf", slack=0.1, solver_name="highs")

    # The network's own optimum is itself a feasible point (trivially within
    # the near-optimal budget) -- projecting it onto the region should
    # return it unchanged, with a ~zero gap.
    point, _normal = closest_point(optimal)
    np.testing.assert_allclose(point, optimal, atol=1e-4)


def test_make_closest_point_fn_returns_valid_separating_hyperplane():
    """The whole point of `normal`: every feasible point must satisfy
    `normal @ (x - point) <= 0` -- this is what makes it usable as a cut in
    `Method.add_cut`. Checked against real extreme points of the network's
    own near-optimal region (from make_furthest_point_fn), not just trusted
    by construction."""
    n = _build_network()
    k = len(DIMENSIONS)
    furthest_point = make_furthest_point_fn(n, DIMENSIONS, slack=0.1, solver_name="highs", max_parallel=4)
    box_directions = np.vstack([np.eye(k), -np.eye(k)])
    feasible_points, _feasible_normals = furthest_point(box_directions)

    closest_point = make_closest_point_fn(n, DIMENSIONS, norm="linf", slack=0.1, solver_name="highs")
    target = np.array([-100.0, -100.0])  # far outside the near-optimal region
    point, normal = closest_point(target)

    # target itself was infeasible, so the projection must have moved
    assert not np.allclose(point, target)
    assert not np.allclose(normal, 0.0)

    violations = normal @ (feasible_points - point).T
    assert (violations <= 1e-6).all(), f"normal is not a valid separating hyperplane: {violations}"


def test_make_closest_point_fn_is_batch_capable():
    n = _build_network()
    closest_point = make_closest_point_fn(n, DIMENSIONS, norm="linf", slack=0.1, solver_name="highs")

    # single target: shape-preserving, matches ClosestPointFn's
    # contract exactly -- (k,) in, one (point, normal) pair out.
    target = np.array([-100.0, -100.0])
    point, normal = closest_point(target)
    assert point.shape == (2,)
    assert normal.shape == (2,)

    # batch of targets: (n_targets, k) in, (n_targets, k) arrays out.
    targets = np.array([[-100.0, -100.0], [100.0, 100.0], [-100.0, 100.0]])
    points, normals = closest_point(targets)
    assert points.shape == (3, 2)
    assert normals.shape == (3, 2)

    # row 0 of the batch is the exact same query as the single-target call
    # above, which ran first -- guards specifically against the batch path's
    # multiprocessing.Pool workers computing a different near-optimal budget
    # baseline than the single-target path, e.g. by re-exporting n's CURRENT
    # (by-then-mutated, since solve_model() writes solution values back onto
    # n regardless of objective) state instead of n's true original optimum.
    np.testing.assert_allclose(points[0], point)
    np.testing.assert_allclose(normals[0], normal)

    # the three targets pull in different directions -- not all identical.
    assert not np.allclose(points[0], points[1])
    assert not np.allclose(points[0], points[2])


def test_make_closest_point_fn_always_resolves_ignoring_existing_file(tmp_path):
    """No caching system: an existing output_dir file is export-only,
    never read back to skip a solve -- tampering with it must NOT affect
    the next call's result, and that call must overwrite it with the
    genuine re-solve."""
    n = _build_network()
    cache_dir = tmp_path / "closest_point_cache"
    closest_point = make_closest_point_fn(
        n, DIMENSIONS, norm="linf", slack=0.1, output_dir=cache_dir, solver_name="highs",
    )

    target = np.array([-100.0, -100.0])
    key = encode_cache_key(target)
    first_point, first_normal = closest_point(target)
    cached_fn = cache_dir / f"mga_closest_result_{key}.nc"
    assert cached_fn.exists()

    # Tamper with the exported result's stashed point/normal -- values no
    # genuine solve of this target could ever produce.
    cached_n = pypsa.Network(cached_fn)
    cached_n.meta["point"] = {"wind": 999.0, "solar": 999.0}
    cached_n.meta["normal"] = {"wind": 888.0, "solar": 888.0}
    cached_n.export_to_netcdf(cached_fn)

    second_point, second_normal = closest_point(target)
    np.testing.assert_allclose(second_point, first_point)
    np.testing.assert_allclose(second_normal, first_normal)
    assert not np.allclose(second_point, [999.0, 999.0])

    # The file itself is overwritten with the genuine result, not left tampered.
    overwritten_meta = pypsa.Network(cached_fn).meta
    keys = list(DIMENSIONS.keys())
    np.testing.assert_allclose([overwritten_meta["point"][k] for k in keys], second_point)


def test_make_closest_point_fn_exports_every_result_in_a_batch(tmp_path):
    n = _build_network()
    cache_dir = tmp_path / "closest_point_cache"
    closest_point = make_closest_point_fn(
        n, DIMENSIONS, norm="linf", slack=0.1, output_dir=cache_dir, solver_name="highs",
    )

    targets = np.array([[-100.0, -100.0], [100.0, 100.0]])
    keys = [encode_cache_key(row) for row in targets]
    points, _normals = closest_point(targets)
    assert {p.name for p in cache_dir.glob("mga_closest_result_*.nc")} == {
        f"mga_closest_result_{k}.nc" for k in keys
    }
    assert not np.allclose(points[0], points[1])


def test_make_closest_point_fn_end_to_end_with_explore_drive(tmp_path):
    """The strongest check: bootstrap a real VertexEnumMethod from a real
    network via make_oracle_fns (furthest_point, closest_point, and
    solve_optimal, wired straight into explore.drive's own parameter order)
    and drive it to convergence -- confirms all three PyPSA-side oracles
    actually work together, not just pass checks in isolation."""
    n = _build_network()
    k = len(DIMENSIONS)

    furthest_point, closest_point, solve_optimal = make_oracle_fns(
        n, DIMENSIONS, slack=0.1, norm="linf", output_dir=tmp_path, solver_name="highs", max_parallel=4,
    )

    method = VertexEnumMethod(k=k, norm="linf", backend="highs")
    history = mga_surveyor_drive(
        method, furthest_point, closest_point, solve_optimal, tol=1e-3, max_iter=30, verbose=False,
    )

    assert len(history) > 0
    initial_gap = history[0]["gap"]
    assert initial_gap > 0
    assert method.max_gap() < 1e-3
    assert method.max_gap() < initial_gap


def test_make_furthest_point_fn_resolve_main_objective_true_exports_nc_files(tmp_path):
    """resolve_main_objective=True's batch path is what
    optimize_mga_in_multiple_directions used to be for -- duals/obj land in
    each cached .nc file's meta, not in the return value."""
    n = _build_network()
    directions = np.array([[-1.0, 0.0], [0.0, -1.0]])
    output_dir = tmp_path / "mga_results"
    furthest_point = make_furthest_point_fn(
        n, DIMENSIONS, slack=0.1, output_dir=output_dir, resolve_main_objective=True,
        solver_name="highs", max_parallel=2,
    )
    points, normals = furthest_point(directions)
    assert points.shape == (2, 2)
    assert normals.shape == (2, 2)
    np.testing.assert_allclose(normals, directions)

    nc_files = sorted(output_dir.glob("mga_furthest_result_*.nc"))
    assert len(nc_files) == 2
    keys = list(DIMENSIONS.keys())
    for nc_file in nc_files:
        meta = pypsa.Network(nc_file).meta
        assert set(meta["duals"]) == set(keys)
        assert isinstance(meta["obj"], float)


def test_make_furthest_point_fn_caches_by_direction_content_with_resolve_main_objective(tmp_path):
    """No point_ids or skip_existing: passing output_dir alone makes caching
    automatic, and each direction's own (rounded) values -- not an arbitrary
    label -- determine its cache file, so identical directions always land
    on the same file across calls. Unlike the old
    optimize_mga_in_multiple_directions (which excluded already-cached rows
    from its return), a cache hit here is fully reconstructed into the
    returned arrays -- every call returns the FULL batch."""
    n = _build_network()
    directions = np.array([[-1.0, 0.0], [0.0, -1.0]])
    output_dir = tmp_path / "mga_results"
    keys = [encode_cache_key(row) for row in directions]
    furthest_point = make_furthest_point_fn(
        n, DIMENSIONS, slack=0.1, output_dir=output_dir, resolve_main_objective=True,
        solver_name="highs", max_parallel=2,
    )

    points, normals = furthest_point(directions)
    assert {p.name for p in output_dir.glob("mga_furthest_result_*.nc")} == {f"mga_furthest_result_{k}.nc" for k in keys}

    # Re-running with the exact same directions should find both already
    # cached and return the SAME full results, reconstructed from disk,
    # without solving anything again.
    points2, normals2 = furthest_point(directions)
    np.testing.assert_allclose(points2, points)
    np.testing.assert_allclose(normals2, normals)

    # A mix of one already-cached direction ([-1, 0]) and one new one ([-1, -1])
    # only solves the new one, but still returns the full (now 2-row) batch.
    directions3 = np.array([[-1.0, 0.0], [-1.0, -1.0]])
    keys3 = [encode_cache_key(row) for row in directions3]
    points3, normals3 = furthest_point(directions3)
    np.testing.assert_allclose(points3[0], points[0])  # cached row: unchanged
    assert {p.name for p in output_dir.glob("mga_furthest_result_*.nc")} == {
        f"mga_furthest_result_{k}.nc" for k in {*keys, keys3[1]}
    }


def test_make_oracle_fns_returns_three_independently_usable_callables(tmp_path):
    """Regression test for a real bug: n.optimize.create_model() stores its
    model at n.model, a single slot -- building furthest/closest/optimal
    directly against one shared n would silently overwrite each other's
    model there, so only whichever was built LAST would actually be live.
    Caught by interleaving all three after construction and checking each
    one's result is still correct, not just that calling it doesn't crash."""
    n = _build_network()
    k = len(DIMENSIONS)
    furthest_point, closest_point, solve_optimal = make_oracle_fns(
        n, DIMENSIONS, slack=0.1, norm="linf", output_dir=tmp_path, solver_name="highs", max_parallel=2,
    )

    optimal_point = solve_optimal()
    expected_optimal = np.array([n.generators.at["wind", "p_nom_opt"], n.generators.at["solar", "p_nom_opt"]])
    np.testing.assert_allclose(optimal_point, expected_optimal, atol=1e-6)

    box_directions = np.vstack([np.eye(k), -np.eye(k)])
    box_points, box_normals = furthest_point(box_directions)
    assert box_points.shape == (2 * k, k)
    np.testing.assert_allclose(box_normals, box_directions)

    target = np.array([-100.0, -100.0])
    point, normal = closest_point(target)
    assert not np.allclose(point, target)  # target was infeasible, so it must have moved

    # Call furthest_point again after closest_point ran -- if the two were
    # secretly sharing (and overwriting) one model, this would now solve
    # the wrong problem and disagree with the first call.
    box_points2, box_normals2 = furthest_point(box_directions)
    np.testing.assert_allclose(box_points2, box_points)
    np.testing.assert_allclose(box_normals2, box_normals)

    method = VertexEnumMethod(k=k, norm="linf", backend="highs")
    method.bootstrap([*box_points, optimal_point], list(zip(box_normals, box_points)))
    assert method.max_gap() > 0


def test_generate_mga_files_and_consolidate_end_to_end(tmp_path):
    """The full pipeline moved out of a caller script: builds all the
    oracles, drives a VertexEnumMethod per slack level, writes result .nc
    files, then consolidates every discovered point's
    coordinates/duals/obj into two DataFrames."""
    n = _build_network()
    keys = list(DIMENSIONS.keys())

    # tol/max_iter as single-element lists, not bare scalars -- exercises
    # the per-slack-level path, not just the scalar-broadcast path.
    generate_mga_files(
        n, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=[1e-2], max_iter=[5], max_parallel=2, verbose=False,
    )
    points_df, duals_df, nc_files = consolidate_mga_results(tmp_path, obj_label="cost")

    assert len(points_df) == len(nc_files) > 0
    assert len(duals_df) == len(points_df)
    assert set(points_df.columns) == {*keys, "cost"}
    assert set(duals_df.columns) == set(keys)
    assert all(f.exists() for f in nc_files)
    assert (tmp_path / "optimal.nc").exists()
    assert (tmp_path / "slack_0.1").is_dir()


def test_generate_mga_files_rejects_mismatched_tol_or_max_iter_length(tmp_path):
    """A list-form tol/max_iter must have one entry per slack level --
    checked before any solving starts."""
    n = _build_network()

    with pytest.raises(ValueError, match="tol"):
        generate_mga_files(
            n, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.02, 0.05], solver_name="highs",
            tol=[1e-2],  # only 1 entry for 2 slack levels
        )

    with pytest.raises(ValueError, match="max_iter"):
        generate_mga_files(
            n, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.02, 0.05], solver_name="highs",
            max_iter=[5, 5, 5],  # 3 entries for 2 slack levels
        )


def test_generate_mga_files_reuses_existing_closest_point_files_on_restart(tmp_path):
    """Points/cuts already found by a previous, interrupted run (existing
    mga_closest_result_*.nc files under a slack level's own directory) are
    read back and passed as explore.drive's extra_points/extra_halfspaces on
    a restart, instead of being rediscovered."""
    n = _build_network()

    # First (short) run: discover at least two closest-point results (so
    # they can be swapped with each other below).
    generate_mga_files(
        n, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=1e-2, max_iter=4, max_parallel=2, verbose=False,
    )
    closest_files = sorted((tmp_path / "slack_0.1").glob("mga_closest_result_*.nc"))
    assert len(closest_files) >= 2  # need two genuine results to swap below

    # Swap two files' stashed (point, normal) pairs with each other. Unlike
    # an arbitrary sentinel point (a plain point, paired with a made-up
    # normal, generally isn't a genuine supporting hyperplane of the
    # region at all -- e.g. pairing a real point with an unrelated normal
    # can carve out a "cut" that excludes part of the real box, which
    # broke this test's earlier 999/50/optimal-point attempts with a
    # scipy QhullError), a swap keeps both entries fully genuine: each
    # (point, normal) still came from a real solve, so it's guaranteed
    # geometrically consistent -- extra_points/extra_halfspaces are now
    # folded into the same scipy HalfspaceIntersection bootstrap call as
    # the box points, which requires a shared interior point strictly
    # inside every halfspace given, and only an actually-valid supporting
    # hyperplane is guaranteed safe there. Still unambiguously detectable
    # as "read back, not rediscovered", since the swapped-in point is a
    # different target's genuine result.
    file_a, file_b = closest_files[0], closest_files[1]
    n_a, n_b = pypsa.Network(file_a), pypsa.Network(file_b)
    point_a, normal_a = dict(n_a.meta["point"]), dict(n_a.meta["normal"])
    point_b, normal_b = dict(n_b.meta["point"]), dict(n_b.meta["normal"])
    assert point_a != point_b  # else this test proves nothing
    n_a.meta["point"], n_a.meta["normal"] = point_b, normal_b
    n_b.meta["point"], n_b.meta["normal"] = point_a, normal_a
    n_a.export_to_netcdf(file_a)
    n_b.export_to_netcdf(file_b)

    # Second run, max_iter=0: the main loop never runs (no new
    # closest-point solving at all) -- any point in the final result must
    # have come from seeding via the existing files.
    n2 = _build_network()
    generate_mga_files(
        n2, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=1e-2, max_iter=0, max_parallel=2, verbose=False,
    )
    points_df, _duals_df, _nc_files = consolidate_mga_results(tmp_path, obj_label="obj")
    assert ((points_df["wind"] == point_b["wind"]) & (points_df["solar"] == point_b["solar"])).any()
    assert ((points_df["wind"] == point_a["wind"]) & (points_df["solar"] == point_a["solar"])).any()


def test_consolidate_mga_results_is_a_repeatable_read(tmp_path):
    """consolidate_mga_results needs only an output_dir, and is a pure
    read: calling it again later on the same, untouched output_dir (no
    new solving in between) reproduces the exact same DataFrames/paths --
    it doesn't mutate anything it reads."""
    n = _build_network()

    generate_mga_files(
        n, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=1e-2, max_iter=5, max_parallel=2, verbose=False,
    )

    points_df, duals_df, nc_files = consolidate_mga_results(tmp_path, obj_label="cost")
    points_df2, duals_df2, nc_files2 = consolidate_mga_results(tmp_path, obj_label="cost")

    assert nc_files == nc_files2
    pd.testing.assert_frame_equal(points_df, points_df2)
    pd.testing.assert_frame_equal(duals_df, duals_df2)


def test_generate_mga_files_then_consolidate(tmp_path):
    """generate_mga_files (just runs, writes .nc files, returns None)
    followed by a separate consolidate_mga_results call is the whole
    pipeline, done in two explicit steps."""
    n = _build_network()
    keys = list(DIMENSIONS.keys())

    result = generate_mga_files(
        n, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=1e-2, max_iter=5, max_parallel=2, verbose=False,
    )
    assert result is None

    points_df, duals_df, nc_files = consolidate_mga_results(tmp_path, obj_label="cost")
    assert len(points_df) == len(nc_files) > 0
    assert len(duals_df) == len(points_df)
    assert set(points_df.columns) == {*keys, "cost"}
    assert (tmp_path / "optimal.nc").exists()
    assert (tmp_path / "slack_0.1").is_dir()


def test_make_optimality_fn_stashes_dimensions_in_optimal_nc_meta(tmp_path):
    """dimensions itself round-trips through optimal.nc's meta -- the piece
    of per-run metadata consolidate_mga_results reads back instead of
    needing dimensions/keys passed to it directly."""
    n = _build_network()
    solve_optimal = make_optimality_fn(n, DIMENSIONS, tmp_path, solver_name="highs")
    solve_optimal()

    cached_dimensions = pypsa.Network(tmp_path / "optimal.nc").meta["dimensions"]
    assert cached_dimensions == DIMENSIONS


def test_consolidate_mga_results_requires_optimal_nc(tmp_path):
    """No optimal.nc (e.g. generate_mga_files never ran, or it's the wrong
    directory) raises immediately -- there's no dimensions/keys to read
    back, so nothing meaningful could be consolidated anyway."""
    with pytest.raises(FileNotFoundError, match="optimal.nc"):
        consolidate_mga_results(tmp_path)


def test_make_closest_point_fn_weights_change_which_point_is_found():
    """Heavily downweighting one dimension's residual makes the solve
    indifferent to it, so it prefers moving along that dimension instead
    of the other one -- confirms `weights` actually changes what "closest"
    means (mga_surveyor.normalize's whole reason for needing it), not just
    accepted as a no-op parameter."""
    target = np.array([50.0, 50.0])  # well outside the feasible region in both dims

    n1 = _build_network()
    closest_ignore_solar = make_closest_point_fn(
        n1, DIMENSIONS, norm="linf", slack=0.1, solver_name="highs", weights={"wind": 1.0, "solar": 1e-6},
    )
    point_ignore_solar, _normal = closest_ignore_solar(target)

    n2 = _build_network()
    closest_ignore_wind = make_closest_point_fn(
        n2, DIMENSIONS, norm="linf", slack=0.1, solver_name="highs", weights={"wind": 1e-6, "solar": 1.0},
    )
    point_ignore_wind, _normal = closest_ignore_wind(target)

    # downweighting solar's residual lets the solve match target[0] (wind)
    # more closely than when it's solar that's weighted normally, and vice
    # versa -- the two solves must actually disagree on the point found.
    assert abs(point_ignore_solar[0] - target[0]) < abs(point_ignore_wind[0] - target[0])
    assert abs(point_ignore_wind[1] - target[1]) < abs(point_ignore_solar[1] - target[1])


def test_generate_mga_files_with_normalize_converges_and_output_is_unaffected(tmp_path):
    """normalize=True changes what VertexEnumMethod itself explores
    (a [-1, 1]^2 cube instead of raw MW), but every .nc file written and
    everything consolidate_mga_results reads back is still real units --
    the normalization is invisible outside generate_mga_files itself."""
    n = _build_network()

    generate_mga_files(
        n, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=1.0, max_iter=10, max_parallel=2, verbose=False, normalize=True,  # tol=1.0 -> 1%, normalize=True
    )
    points_df, duals_df, nc_files = consolidate_mga_results(tmp_path, obj_label="cost")

    assert len(points_df) == len(nc_files) > 0
    assert len(duals_df) == len(points_df)
    assert set(points_df.columns) == {"wind", "solar", "cost"}
    # real units, not squeezed into [-1, 1] -- e.g. capacities are >= 0 MW,
    # not clipped to the normalized cube's own range.
    assert (points_df["wind"] >= 0).all()


def test_make_furthest_point_fn_on_solve_fires_for_fresh_solve_and_cache_hit(tmp_path):
    """on_solve fires with real obj/duals on a fresh solve, and fires
    again (from the cached file's own meta) on a cache hit for the exact
    same direction -- a restart must not silently lose this data."""
    n = _build_network()
    calls = []

    def collect(point, normal, obj, duals):
        calls.append((point.copy(), normal.copy(), obj, None if duals is None else duals.copy()))

    furthest_point = make_furthest_point_fn(
        n, DIMENSIONS, slack=0.1, output_dir=tmp_path, solver_name="highs",
        resolve_main_objective=True, on_solve=collect,
    )
    direction = np.array([1.0, 0.0])
    furthest_point(direction)
    assert len(calls) == 1
    point1, _normal1, obj1, duals1 = calls[0]
    assert duals1 is not None
    assert np.isfinite(obj1)

    n2 = _build_network()
    furthest_point2 = make_furthest_point_fn(
        n2, DIMENSIONS, slack=0.1, output_dir=tmp_path, solver_name="not_a_real_solver",  # would raise if re-solved
        resolve_main_objective=True, on_solve=collect,
    )
    furthest_point2(direction)
    assert len(calls) == 2
    point2, _normal2, obj2, duals2 = calls[1]
    np.testing.assert_allclose(point1, point2)
    assert obj1 == obj2
    np.testing.assert_allclose(duals1, duals2)


def test_make_closest_point_fn_on_solve_fires_for_single_and_batch(tmp_path):
    """on_solve fires once per target actually solved, single or batch,
    with the same obj/duals that get cached (but discarded from the
    return value otherwise), plus the target itself."""
    calls = []

    def collect(point, normal, obj, duals, target):
        calls.append((
            point.copy(), normal.copy(), obj, None if duals is None else duals.copy(), target.copy(),
        ))

    n1 = _build_network()
    closest_point = make_closest_point_fn(
        n1, DIMENSIONS, norm="linf", slack=0.1, solver_name="highs", on_solve=collect,
    )
    closest_point(np.array([50.0, 50.0]))
    assert len(calls) == 1
    assert calls[0][3] is not None  # duals

    n2 = _build_network()
    closest_point_batch = make_closest_point_fn(
        n2, DIMENSIONS, norm="linf", slack=0.1, solver_name="highs", max_parallel=2, on_solve=collect,
    )
    closest_point_batch(np.array([[50.0, 50.0], [-50.0, -50.0]]))
    assert len(calls) == 3


def test_generate_mga_files_with_use_objective_cuts_converges_and_output_is_unaffected(tmp_path):
    """use_objective_cuts=True adds extra cuts to VertexEnumMethod's outer
    approximation, but every .nc file written and everything
    consolidate_mga_results reads back is still exactly what it would be
    without it -- the extra cuts are invisible outside generate_mga_files
    itself, same property normalize=True has."""
    n = _build_network()

    generate_mga_files(
        n, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=1e-2, max_iter=10, max_parallel=2, verbose=False, use_objective_cuts=True,
    )
    points_df, duals_df, nc_files = consolidate_mga_results(tmp_path, obj_label="cost")

    assert len(points_df) == len(nc_files) > 0
    assert len(duals_df) == len(points_df)
    assert set(points_df.columns) == {"wind", "solar", "cost"}
    assert (points_df["wind"] >= 0).all()
    assert (points_df["solar"] >= 0).all()


def test_generate_mga_files_with_normalize_and_use_objective_cuts_together(tmp_path):
    """Both opt-in features composed at once -- must not crash, and the
    objective cuts must still be expressed correctly once affine_map is
    known (see generate_mga_files's own docstring on how the two
    compose)."""
    n = _build_network()

    generate_mga_files(
        n, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=1.0, max_iter=10, max_parallel=2, verbose=False, normalize=True, use_objective_cuts=True,  # tol=1.0 -> 1%
    )
    points_df, duals_df, nc_files = consolidate_mga_results(tmp_path, obj_label="cost")

    assert len(points_df) == len(nc_files) > 0
    assert len(duals_df) == len(points_df)
    assert (points_df["wind"] >= 0).all()
    assert (points_df["solar"] >= 0).all()


def test_generate_mga_files_with_use_objective_cuts_uses_existing_closest_point_files_too(tmp_path):
    """_read_existing_closest_points' own use_objective_cuts extension
    only ever runs when there's an existing mga_closest_result_*.nc file
    to read back -- a fresh single run never exercises it. Mirrors
    test_generate_mga_files_reuses_existing_closest_point_files_on_restart's
    own restart structure, this time checking the restart doesn't crash
    once those existing files' own cached duals also feed
    linearized_objective_cut."""
    n = _build_network()

    generate_mga_files(
        n, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=1e-2, max_iter=4, max_parallel=2, verbose=False,
    )
    closest_files = sorted((tmp_path / "slack_0.1").glob("mga_closest_result_*.nc"))
    assert len(closest_files) > 0  # something for the restart below to read back
    assert all("duals" in pypsa.Network(f).meta for f in closest_files)  # resolve_main_objective=True by default

    n2 = _build_network()
    generate_mga_files(
        n2, DIMENSIONS, tmp_path, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=1e-2, max_iter=10, max_parallel=2, verbose=False, use_objective_cuts=True,
    )
    points_df, duals_df, nc_files = consolidate_mga_results(tmp_path, obj_label="cost")
    assert len(points_df) == len(nc_files) > 0
    assert len(duals_df) == len(points_df)


def test_generate_mga_files_tol_is_a_percentage_when_normalized(tmp_path):
    """tol under normalize=True is a percentage, not a raw cube distance
    -- a much looser tol (50%) should converge in far fewer iterations
    than a much tighter one (0.1%), observable (without needing to reach
    into drive()'s own internals) as: given the same small max_iter
    budget, the tight-tol run is still discovering new points when it
    runs out of budget, while the loose-tol run has already converged
    and stopped adding new ones well before that budget is used up."""
    tmp_path_loose = tmp_path / "loose"
    tmp_path_tight = tmp_path / "tight"

    n1 = _build_network()
    generate_mga_files(
        n1, DIMENSIONS, tmp_path_loose, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=50.0, max_iter=15, max_parallel=2, verbose=False, normalize=True,
    )
    points_loose, _duals_loose, _files_loose = consolidate_mga_results(tmp_path_loose, obj_label="cost")

    n2 = _build_network()
    generate_mga_files(
        n2, DIMENSIONS, tmp_path_tight, norm="linf", slack_levels=[0.1], solver_name="highs",
        tol=0.1, max_iter=15, max_parallel=2, verbose=False, normalize=True,
    )
    points_tight, _duals_tight, _files_tight = consolidate_mga_results(tmp_path_tight, obj_label="cost")

    assert len(points_tight) > len(points_loose)
