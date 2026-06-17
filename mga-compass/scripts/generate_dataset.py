"""Generate the MGA dataset consumed by the MGA Compass server.

Solves a PyPSA example network, runs Modelling-to-Generate-Alternatives (MGA) across
several sub-optimality (slack) levels via mga_compass, and writes the production-ready
files consumed by the server (points.pkl, duals.pkl, installed_capacity.pkl,
generation_over_snapshots.pkl).

Run with: uv run python -c "from generate_dataset import main; main(slack_levels=[0.02, 0.05, 0.08])"
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from netCDF4 import Dataset
from pypsa.optimization.mga import generate_directions_halton

from mga_compass import optimize_mga_in_multiple_directions

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
    n_directions: int  # sampled directions per slack level
    obj_label: str
    output_dir: Path = Path(__file__).parent / "output"
    seed: int = 0
    max_parallel: int = 4


RUN_CONFIG = RunConfig(*_detect_solver(), n_directions=200, obj_label="TOTEX")


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
    optimal_cost: float,
    slack_levels: list[float],
    network: NetworkConfig,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Run MGA across all slack levels and consolidate into one 0..N-1 index.

    Resumable: each slack level's already-solved directions (tracked by their
    mga_result_{point_id}.nc file and a cached results.pkl) are skipped on rerun,
    so an interrupted run can simply be restarted with the same configuration.

    Returns points_df, duals_df, and the list of NetCDF paths (matching the
    DataFrames' row order) for the corresponding solved networks.
    """
    output_dir = config.output_dir
    output_dir.mkdir(exist_ok=True)
    directions = generate_directions_halton(network.dimensions.keys(), config.n_directions, seed=config.seed)
    point_ids = list(range(config.n_directions))

    all_coords = []
    all_duals = []
    all_nc_paths = []

    # Point 0: the network's own optimum, no slack. Cheap (no MGA solve needed).
    optimum_nc = output_dir / "optimum.nc"
    n.export_to_netcdf(optimum_nc)
    optimum_coords = n.optimize.project_solved(network.dimensions)
    optimum_coords[config.obj_label] = optimal_cost
    all_coords.append(optimum_coords)
    all_duals.append(pd.Series(0.0, index=list(network.dimensions.keys())))
    all_nc_paths.append(optimum_nc)

    for slack in slack_levels:
        slack_dir = output_dir / f"slack_{slack}"
        cache_path = slack_dir / "results.pkl"
        if cache_path.exists():
            cached_coords, cached_duals = pd.read_pickle(cache_path)
        else:
            cached_coords, cached_duals = pd.DataFrame(), pd.DataFrame()

        logging.info("slack=%s: starting (%d/%d already solved)", slack, len(cached_coords), config.n_directions)

        new_dirs_df, new_coords_df, new_duals_df = optimize_mga_in_multiple_directions(
            n,
            directions=directions,
            dimensions=network.dimensions,
            output_dir=slack_dir,
            obj_label=config.obj_label,
            point_ids=point_ids,
            skip_existing=True,
            slack=slack,
            solver_name=config.solver_name,
            solver_options=config.solver_options,
            max_parallel=config.max_parallel,
        )
        coords_df = pd.concat([cached_coords, new_coords_df])
        duals_df = pd.concat([cached_duals, new_duals_df])
        pd.to_pickle((coords_df, duals_df), cache_path)

        logging.info(
            "slack=%s: %d/%d total solved (%d new this run)",
            slack,
            len(coords_df),
            config.n_directions,
            len(new_coords_df),
        )

        for pid in coords_df.index:
            all_coords.append(coords_df.loc[pid])
            all_duals.append(duals_df.loc[pid])
            all_nc_paths.append(slack_dir / f"mga_result_{pid}.nc")

    points_df = pd.DataFrame(all_coords).reset_index(drop=True)
    duals_df_final = pd.DataFrame(all_duals).reset_index(drop=True)
    points_df.index.name = "index"
    duals_df_final.index.name = "index"

    return points_df, duals_df_final, all_nc_paths


def extract_generation_and_capacity(
    nc_files: list[Path], index: pd.Index, snapshots: pd.DatetimeIndex, capacity_vars: list[tuple[str, str]]
):
    # netCDF4 directly is ~20x faster than xr.open_dataset for these files (the
    # xarray wrapper's overhead dominates when opening many small files in a loop).
    with Dataset(nc_files[0]) as ds0:
        gen_techs = [str(x) for x in ds0["generators_t_p_i"][:]]
        cap_techs = [str(x) for comp, _ in capacity_vars for x in ds0[f"{comp}_i"][:]]

    gen_data = np.empty((len(nc_files), len(snapshots), len(gen_techs)))
    cap_data = np.empty((len(nc_files), len(cap_techs)))

    for pos, nc_file in enumerate(nc_files):
        with Dataset(nc_file) as ds:
            gen_data[pos] = np.asarray(ds["generators_t_p"][:])
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


def main(
    slack_levels: list[float], network: NetworkConfig = MODEL_ENERGY, config: RunConfig = RUN_CONFIG
) -> None:
    n = network.build(config.solver_name, config.solver_options)
    optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
    print(f"Optimal {config.obj_label}: {optimal_cost:,.0f}")

    points_df, duals_df, nc_files = run_mga(n, optimal_cost, slack_levels, network, config)
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


if __name__ == "__main__":
    main(slack_levels=[0.02, 0.05, 0.08])
