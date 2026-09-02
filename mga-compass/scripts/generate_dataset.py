"""Generate the MGA dataset consumed by the MGA Compass server.

Solves a PyPSA example network, then hands it to
mga_surveyor.generate_mga_files (which drives a VertexEnumMethod per
sub-optimality (slack) level to discover near-optimal points, writing
result .nc files), reads the result back via
mga_surveyor.consolidate_mga_results, and writes the production-ready
files consumed by the server (points.pkl, duals.pkl,
installed_capacity.pkl, generation_over_snapshots.pkl) from that.

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

from mga_surveyor import consolidate_mga_results, generate_mga_files

logging.basicConfig(level=logging.INFO, format="%(message)s")
# Silence PyPSA's own per-variable INFO log when assigning the solution back
# to the network: it fires for every mga_surveyor_* auxiliary variable we add
# to the model (residuals, the L-inf slack), since those don't follow
# PyPSA's "Component-attr" naming convention -- expected and harmless (we
# read them directly via linopy, not through the network), just noisy.
logging.getLogger("pypsa.optimization.optimize").setLevel(logging.WARNING)


def _add_file_logging(output_dir: Path) -> None:
    """Mirror everything logging.basicConfig above sends to the console into
    output_dir/run.log too -- same handler setup (level, format), just a
    second destination next to optimal.nc, so a run's full log survives
    after the terminal scrolls away."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(file_handler)


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
    normalize: bool = False
    use_objective_cuts: bool = False
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
    # include_objective_constant=True explicit, not left to PyPSA's own
    # default -- silences its own FutureWarning about that default changing
    # in 2.0, matching the same explicit choice mga_surveyor.pypsa_ext makes
    # for every model it builds against this network.
    n.optimize(solver_name=solver_name, solver_options=solver_options, include_objective_constant=True)
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
    _add_file_logging(config.output_dir)

    n = network.build(config.solver_name, config.solver_options)
    optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
    logging.info(f"Optimal {config.obj_label}: {optimal_cost:,.0f}")

    generate_mga_files(
        n, network.dimensions, config.output_dir, norm=config.norm, slack_levels=config.slack_levels,
        solver_name=config.solver_name, tol=config.tol, normalize=config.normalize, max_iter=config.max_iter,
        max_parallel=config.max_parallel, solver_options=config.solver_options,
        use_objective_cuts=config.use_objective_cuts,
    )
    points_df, duals_df, nc_files = consolidate_mga_results(config.output_dir, config.obj_label)
    logging.info(f"Total points: {len(points_df)}")

    points_df.to_pickle(config.output_dir / "points.pkl")
    duals_df.to_pickle(config.output_dir / "duals.pkl")

    generation, installed_capacity = extract_generation_and_capacity(
        nc_files, points_df.index, pd.DatetimeIndex(n.snapshots), network.capacity_vars
    )
    pd.to_pickle(generation, config.output_dir / "generation_over_snapshots.pkl")
    pd.to_pickle(installed_capacity, config.output_dir / "installed_capacity.pkl")

    logging.info(f"Saved generation {generation.shape} (index x snapshot x technology)")
    logging.info(f"Saved installed_capacity {installed_capacity.shape} (index x technology)")
    logging.info(points_df)
    logging.info(duals_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=RUN_CONFIG.output_dir, help="Directory to write outputs to."
    )
    parser.add_argument(
        "--tol", type=float, default=RUN_CONFIG.tol,
        help="Convergence tolerance: real units by default, or a percentage (e.g. 4 -> 4%%) if --normalize "
             "is also given -- see mga_surveyor.generate_mga_files's own docstring.",
    )
    parser.add_argument(
        "--normalize", action="store_true", default=RUN_CONFIG.normalize,
        help="Explore each slack level's near-optimal region in a [-1, 1]^k-normalized cube instead of raw "
             "units -- makes --tol a percentage instead of a real-unit distance.",
    )
    parser.add_argument(
        "--use-objective-cuts", action="store_true", default=RUN_CONFIG.use_objective_cuts,
        help="Also add each solved point's own linearized_objective_cut as free extra tightening -- see "
             "mga_surveyor.generate_mga_files's own docstring.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = replace(
        RUN_CONFIG, output_dir=args.output_dir, tol=args.tol, normalize=args.normalize,
        use_objective_cuts=args.use_objective_cuts,
    )
    main(config=config)
