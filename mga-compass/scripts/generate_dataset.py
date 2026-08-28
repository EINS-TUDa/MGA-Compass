"""Generate the MGA dataset consumed by the MGA Compass server.

Solves a PyPSA example network, then for each sub-optimality (slack) level
drives a VertexEnumMethod (mga_surveyor's adaptive inner/outer polytope
approximation) to discover near-optimal points, and writes the
production-ready files consumed by the server (points.pkl, duals.pkl,
installed_capacity.pkl, generation_over_snapshots.pkl).

Requires the "scripts" extra (`uv sync --extra scripts`) for mga-surveyor/pypsa/netCDF4/gurobipy.

Run with: uv run python scripts/generate_dataset.py
"""
import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from netCDF4 import Dataset

from mga_surveyor import (
    VertexEnumMethod,
    encode_cache_key,
    make_closest_point_fn,
    make_furthest_point_fn,
    make_optimality_fn,
    run as mga_surveyor_run,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _detect_solver() -> tuple[str, dict]:
    HIGHS_FALLBACK = ("highs", {"output_flag": False})

    try:
        import gurobipy
    except ImportError:
        return HIGHS_FALLBACK

    try:
        # OutputFlag must be set on the env (created before the model) to
        # silence the license banner and "Read LP format..." messages, not
        # just the solve log — models inherit it from there, so no separate
        # top-level key is needed.
        with gurobipy.Env(params={"OutputFlag": 0}):
            return "gurobi", {"env": {"OutputFlag": 0}}
    except gurobipy.GurobiError:
        return HIGHS_FALLBACK


@dataclass
class RunConfig:
    """Bundles script-wide run configuration, so it can be overridden or
    swapped as a unit instead of editing module constants.
    """

    solver_name: str
    solver_options: dict
    norm: str
    obj_label: str
    slack_levels: list[float]
    tol: float = 1e-3
    max_iter: int = 120
    output_dir: Path = Path(__file__).parent / "output"
    max_parallel: int = 4


RUN_CONFIG = RunConfig(*_detect_solver(), norm="linf", obj_label="TOTEX", slack_levels=[0.02, 0.05, 0.08])


@dataclass
class NetworkConfig:
    """Bundles everything specific to one PyPSA network, so swapping networks
    means passing a different config instead of editing constants throughout
    this file.
    """

    build: Callable[[str, dict], pypsa.Network]
    dimensions: dict
    capacity_vars: list[tuple[str, str]]


def _build_model_energy(solver_name: str, solver_options: dict) -> pypsa.Network:
    n = pypsa.examples.model_energy()
    n.snapshots = n.snapshots[::3]  # crude reduction of time resolution for speed
    n.generators.at["load shedding", "marginal_cost"] = 20000
    n.optimize(solver_name=solver_name, solver_options=solver_options)
    return n


MODEL_ENERGY = NetworkConfig(
    build=_build_model_energy,
    dimensions={
        "wind": {"Generator": {"p_nom": {"wind": 1}}},
        "solar": {"Generator": {"p_nom": {"solar": 1}}},
        "battery": {"StorageUnit": {"p_nom": {"battery storage": 1}}},
        "hydrogen": {"Store": {"e_nom": {"hydrogen storage": 1}}},
    },
    capacity_vars=[
        ("generators", "generators_p_nom_opt"),
        ("links", "links_p_nom_opt"),
        ("storage_units", "storage_units_p_nom_opt"),
        ("stores", "stores_e_nom_opt"),
    ],
)


def run_mga(
    n: pypsa.Network,
    network: NetworkConfig,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Run a VertexEnumMethod per slack level (mga_surveyor's adaptive
    inner/outer polytope approximation, via runner.run, up to
    `config.max_iter` iterations each) and consolidate into one 0..N-1
    index.

    The network's own optimum is slack-independent, so it's solved/cached
    once (output_dir/optimal.nc), not once per slack level, even though
    each slack level's own run() still calls it (as its required
    optimal_fn) to seed that slack level's inner approximation.

    Resumable: every .nc file this function's own oracle calls produce is
    cached under output_dir/slack_{slack}/ (or output_dir/optimal.nc), so
    an interrupted run can simply be restarted with the same configuration
    -- already-cached queries are read back, not re-solved (see
    make_furthest_point_fn/make_closest_point_fn/make_optimality_fn).

    Returns points_df, duals_df, and the list of NetCDF paths (matching the
    DataFrames' row order) for the corresponding solved networks.
    """
    output_dir = config.output_dir
    output_dir.mkdir(exist_ok=True)
    keys = list(network.dimensions.keys())
    k = len(keys)

    solve_optimal = make_optimality_fn(
        n, network.dimensions, output_dir, solver_name=config.solver_name, solver_options=config.solver_options,
    )

    # Every .nc file any oracle call below actually touches, deduped by
    # path -- e.g. optimal.nc gets added once per slack level's own run()
    # call (each needs it for its own bootstrap) but is the same file every
    # time, so it naturally ends up counted once here.
    nc_paths: set[Path] = set()

    def logged_optimal_fn() -> np.ndarray:
        nc_paths.add(output_dir / "optimal.nc")
        return solve_optimal()

    for slack in config.slack_levels:
        slack_dir = output_dir / f"slack_{slack}"
        furthest_point = make_furthest_point_fn(
            n, network.dimensions, slack=slack, output_dir=slack_dir, resolve_main_objective=True,
            solver_name=config.solver_name, solver_options=config.solver_options, max_parallel=config.max_parallel,
        )
        closest_point = make_closest_point_fn(
            n, network.dimensions, norm=config.norm, slack=slack, output_dir=slack_dir,
            solver_name=config.solver_name, solver_options=config.solver_options, max_parallel=config.max_parallel,
        )

        def logged_furthest_point(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            for row in directions:
                nc_paths.add(slack_dir / f"mga_furthest_result_{encode_cache_key(row, decimals=8)}.nc")
            return furthest_point(directions)

        def logged_closest_point(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nc_paths.add(slack_dir / f"mga_closest_result_{encode_cache_key(target, decimals=8)}.nc")
            return closest_point(target)

        method = VertexEnumMethod(k=k, norm=config.norm, backend=config.solver_name)
        logging.info("slack=%s: running VertexEnumMethod (max_iter=%d)", slack, config.max_iter)
        mga_surveyor_run(
            method, logged_furthest_point, logged_closest_point, logged_optimal_fn,
            tol=config.tol, max_iter=config.max_iter, verbose=True,
        )
        logging.info(
            "slack=%s: %d inner point(s), final max_gap=%.3e",
            slack, len(method.inner_vertices), method.max_gap(),
        )

    all_coords = []
    all_duals = []
    all_nc_paths = sorted(nc_paths, key=str)
    for nc_path in all_nc_paths:
        n_result = pypsa.Network(nc_path)
        meta = n_result.meta
        obj = n_result.statistics.capex().sum() + n_result.statistics.opex().sum()
        if "coordinates" in meta:
            # make_furthest_point_fn's / make_optimality_fn's own cache.
            coords = pd.Series(meta["coordinates"])[keys]
        else:
            # make_closest_point_fn's own cache.
            coords = pd.Series(meta["point"])[keys]
        # Absent when that call's own resolve_main_objective was False.
        duals = pd.Series(meta["duals"])[keys] if "duals" in meta else pd.Series(0.0, index=keys)
        coords = coords.copy()
        coords[config.obj_label] = obj
        all_coords.append(coords)
        all_duals.append(duals)

    points_df = pd.DataFrame(all_coords).reset_index(drop=True)
    duals_df = pd.DataFrame(all_duals).reset_index(drop=True)
    points_df.index.name = "index"
    duals_df.index.name = "index"

    return points_df, duals_df, all_nc_paths


def extract_generation_and_capacity(
    nc_files: list[Path], index: pd.Index, snapshots: pd.DatetimeIndex, capacity_vars: list[tuple[str, str]]
):
    # netCDF4 directly is ~20x faster than xr.open_dataset for these files (the
    # xarray wrapper's overhead dominates when opening many small files in a loop).
    with Dataset(nc_files[0]) as ds0:
        # The static component list (generators_i), not the time-varying one
        # (generators_t_p_i): PyPSA only exports a generator's p time series if
        # it's ever dispatched, so generators_t_p_i's columns vary between
        # files (e.g. "load shedding" is absent whenever it's never used) and
        # can't be relied on for a consistent technology axis across files.
        gen_techs = [str(x) for x in ds0["generators_i"][:]]
        cap_techs = [str(x) for comp, _ in capacity_vars for x in ds0[f"{comp}_i"][:]]

    gen_data = np.zeros((len(nc_files), len(snapshots), len(gen_techs)))
    cap_data = np.empty((len(nc_files), len(cap_techs)))

    for pos, nc_file in enumerate(nc_files):
        with Dataset(nc_file) as ds:
            file_gen_techs = [str(x) for x in ds["generators_t_p_i"][:]]
            tech_positions = [gen_techs.index(tech) for tech in file_gen_techs]
            gen_data[pos, :, tech_positions] = np.asarray(ds["generators_t_p"][:]).T
            cap_data[pos] = np.concatenate([np.asarray(ds[var][:]) for _, var in capacity_vars])

    generation = xr.DataArray(
        gen_data,
        dims=["index", "snapshot", "technology"],
        coords={"index": index, "snapshot": snapshots, "technology": gen_techs},
        name="electricity_generation",
    )
    installed_capacity = xr.DataArray(
        cap_data,
        dims=["index", "technology"],
        coords={"index": index, "technology": cap_techs},
        name="installed_capacity",
    ).drop_sel(technology="load shedding")  # virtual slack generator, not a real tech

    return generation, installed_capacity


def main(network: NetworkConfig = MODEL_ENERGY, config: RunConfig = RUN_CONFIG) -> None:
    n = network.build(config.solver_name, config.solver_options)
    optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
    print(f"Optimal {config.obj_label}: {optimal_cost:,.0f}")

    points_df, duals_df, nc_files = run_mga(n, network, config)
    print(f"Total points: {len(points_df)}")

    points_df.to_pickle(config.output_dir / "points.pkl")
    duals_df.to_pickle(config.output_dir / "duals.pkl")

    generation, installed_capacity = extract_generation_and_capacity(
        nc_files, points_df.index, pd.DatetimeIndex(n.snapshots), network.capacity_vars
    )
    pd.to_pickle(generation, config.output_dir / "generation_over_snapshots.pkl")
    pd.to_pickle(installed_capacity, config.output_dir / "installed_capacity.pkl")

    print(f"Saved generation {generation.shape} (index x snapshot x technology)")
    print(f"Saved installed_capacity {installed_capacity.shape} (index x technology)")
    print(points_df)
    print(duals_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=RUN_CONFIG.output_dir, help="Directory to write outputs to."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = replace(RUN_CONFIG, output_dir=args.output_dir)
    main(config=config)
