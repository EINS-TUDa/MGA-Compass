"""Generate the near-optimal set the MGA Compass server serves.

Solves a PyPSA example network, hands it to mga_surveyor.generate_mga_files
(one BBMethod exploration, its default, per sub-optimality (slack) level,
writing a result .nc file per solve), collects the result as a near-optimal set (mga_surveyor.
collect_near_optimal_set), adds each point's installed capacity and generation
as outputs, and applies the dataset config (scripts/datasets/model_energy.json:
units, descriptions, plots, display units; see dataset_config.py). Writes the
set to <output-dir>/set and its archive to <output-dir>/near-optimal-set.zip,
which scripts/publish_dataset.py uploads for the server to download.

Requires the "scripts" extra (`uv sync --extra scripts`) for mga-surveyor/pypsa/netCDF4/gurobipy.

Run with: uv run python scripts/generate_dataset.py --tol 5 --max-iter 120 --max-parallel 4
"""
import argparse
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from netCDF4 import Dataset

from dataset_config import DatasetConfig, long_output, write_set
from mga_surveyor import collect_near_optimal_set, generate_mga_files

DATASET_CONFIG = Path(__file__).parent / "datasets" / "model_energy.json"

logging.basicConfig(level=logging.INFO, format="%(message)s")
# Silence PyPSA's own per-variable INFO log when assigning the solution back
# to the network: it fires for every mga_surveyor_* auxiliary variable we add
# to the model (residuals, the L-inf slack), since those don't follow
# PyPSA's "Component-attr" naming convention -- expected and harmless (we
# read them directly via linopy, not through the network), just noisy.
logging.getLogger("pypsa.optimization.optimize").setLevel(logging.WARNING)
# linopy.io logs "Writing time: Xs" and linopy.model logs "Solve problem
# using X solver" on every single solve -- the solver name is already
# logged once in main() below, and per-solve write timing isn't used for
# anything, so both are just noise at this log's per-solve granularity.
logging.getLogger("linopy.io").setLevel(logging.WARNING)
logging.getLogger("linopy.model").setLevel(logging.WARNING)
# linopy.constants logs one pre-formatted multi-line block per solve
# ("Status: ok\nTermination condition: optimal\nSolution: N primals, M
# duals\nObjective: X\nSolver: gurobi\nRuntime: Xs\nDual bound: X\nSolver
# model: available\nSolver message: 2"). pypsa_ext.py already logs its own
# condensed one-line summary per solve (status/condition/objective, read
# directly off the model), so this whole block would only duplicate that.
logging.getLogger("linopy.constants").setLevel(logging.WARNING)


# pypsa.network.io logs "Exported network 'X' saved to '<path>' contains:
# <every component>" on every export; pypsa_ext.py already logs its own
# short "saved to file: <name>" right after each one (same event, without
# the full path or component dump), so the pypsa.network.io version would
# only duplicate that. Its other message, "Imported network ...", fires
# once at the very start and is still useful, so a level change (which
# can't distinguish the two, same logger) would lose it too -- this drops
# only the one starting with "Exported network".
logging.getLogger("pypsa.network.io").addFilter(
    lambda record: not record.getMessage().startswith("Exported network")
)


def _add_file_logging(output_dir: Path) -> None:
    """Mirror everything logging.basicConfig above sends to the console into
    output_dir/run.log too -- same handler setup (level, format), just a
    second destination next to optimal.nc, so a run's full log survives
    after the terminal scrolls away. Appends (mode="a"), not overwrites --
    re-running against the same output_dir (e.g. a restart that resumes
    from cached .nc files) keeps every prior run's log instead of losing
    it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "run.log", mode="a")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    # gurobipy has its own Python-logging integration, entirely separate from
    # its native LogToConsole/OutputFlag parameters (see _detect_solver) --
    # once any logging.basicConfig() is active (as above), gurobipy mirrors
    # every console line through logging.getLogger("gurobipy").info(...) too,
    # which propagates to every handler on the root logger: both the console
    # (basicConfig's own StreamHandler, where it's the only live progress
    # signal during a solve -- LogFile is buffered and only flushes on that
    # solve's own completion) and, unwanted, this FileHandler. Deliberately
    # not silenced via logging.getLogger("gurobipy").setLevel(...) -- that
    # would suppress it on the console too. Filtering only this handler
    # keeps gurobipy's console visibility intact and run.log clean.
    file_handler.addFilter(lambda record: record.name != "gurobipy")
    # Written directly to the file (not through the logger), so it lands only
    # in run.log, not the console -- a separator marking where a new run
    # starts, since mode="a" above means this file can span several runs.
    file_handler.stream.write(f"\n\n{'=' * 80}\nRun started: {datetime.now().isoformat(timespec='seconds')}\n{'=' * 80}\n\n")
    file_handler.stream.flush()
    logging.getLogger().addHandler(file_handler)


def _detect_solver() -> tuple[str, dict]:
    HIGHS_FALLBACK = ("highs", {"output_flag": False})

    try:
        import gurobipy
    except ImportError:
        return HIGHS_FALLBACK

    try:
        # LogToConsole (not OutputFlag) must be set on the env (created before
        # the model) to silence per-solve console spam — models inherit it
        # from there, so no separate top-level key is needed. OutputFlag is a
        # master switch: 0 would suppress output everywhere, including each
        # solve's own per-call LogFile (see _solver_kwargs_with_log), leaving
        # those files with nothing but their one-time "logging started"
        # banner. LogToConsole=0 only silences the console, so LogFile still
        # captures the real trace. The env-creation license banner itself
        # isn't gated by either -- it still prints once per worker process,
        # regardless.
        with gurobipy.Env(params={"LogToConsole": 0}):
            return "gurobi", {"env": {"LogToConsole": 0}}
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
    max_iter: int
    tol: float
    max_parallel: int
    normalize: bool = True
    use_objective_cuts: bool = True
    output_dir: Path = Path(__file__).parent / "output"


RUN_CONFIG = RunConfig(
    *_detect_solver(), norm="linf", obj_label="TOTEX", slack_levels=[0.02, 0.05, 0.08], max_iter=120, tol=1e-3,
    max_parallel=4,
)


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
    logging.info(f"Solver: {config.solver_name}")

    n = network.build(config.solver_name, config.solver_options)
    optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
    logging.info(f"Optimal {config.obj_label}: {optimal_cost:,.0f}")

    generate_mga_files(
        n, network.dimensions, config.output_dir, norm=config.norm, slack_levels=config.slack_levels,
        solver_name=config.solver_name, tol=config.tol, normalize=config.normalize, max_iter=config.max_iter,
        max_parallel=config.max_parallel, solver_options=config.solver_options,
        use_objective_cuts=config.use_objective_cuts,
    )
    dataset = DatasetConfig.load(DATASET_CONFIG)
    nos, files = collect_near_optimal_set(config.output_dir, **dataset.collect_arguments())
    logging.info(f"Total points: {len(nos.points)}")

    snapshots = pd.DatetimeIndex(n.snapshots)
    generation, installed_capacity = extract_generation_and_capacity(
        files, nos.points.index.rename("index"), snapshots, network.capacity_vars
    )
    outputs = {"generation": long_output(generation), "capacity": long_output(installed_capacity)}
    members = {
        "snapshot": list(snapshots),
        "technology": list(dict.fromkeys([*generation.technology.values, *installed_capacity.technology.values])),
    }
    nos, outputs = dataset.apply(nos, outputs, members)
    archive = write_set(nos, outputs, config.output_dir / "set")

    logging.info(f"generation {generation.shape} (point x snapshot x technology)")
    logging.info(f"installed_capacity {installed_capacity.shape} (point x technology)")
    logging.info(f"Wrote {config.output_dir / 'set'} and {archive}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=RUN_CONFIG.output_dir, help="Directory to write outputs to."
    )
    parser.add_argument(
        "--tol", type=float, required=True,
        help="Convergence tolerance -- a percentage (e.g. 4 -> 4%%) since --normalize is on by default, "
             "or a real-unit distance if --no-normalize is passed -- see "
             "mga_surveyor.generate_mga_files's own docstring. No default -- must be chosen explicitly "
             "for every run.",
    )
    parser.add_argument(
        "--no-normalize", action="store_false", dest="normalize", default=RUN_CONFIG.normalize,
        help="Explore each slack level's near-optimal region in raw units instead of a [-1, 1]^k "
             "normalized cube (on by default) -- makes --tol a real-unit distance instead of a "
             "percentage.",
    )
    parser.add_argument(
        "--no-objective-cuts", action="store_false", dest="use_objective_cuts",
        default=RUN_CONFIG.use_objective_cuts,
        help="Disable adding each solved point's own linearized_objective_cut as free extra tightening "
             "(on by default) -- see mga_surveyor.generate_mga_files's own docstring.",
    )
    parser.add_argument(
        "--max-iter", type=int, required=True,
        help="Cap on drive() iterations per slack level -- see mga_surveyor.generate_mga_files's own docstring. "
             "No default -- must be chosen explicitly for every run.",
    )
    parser.add_argument(
        "--max-parallel", type=int, required=True,
        help="Worker processes to spread furthest_point/closest_point solves across. Also doubles as "
             "how many of the worst candidate closest-points drive() solves per iteration instead of "
             "just the single worst one (pass 1 for today's original single-target behavior) -- see "
             "mga_surveyor.explore.drive's own docstring. No default -- must be chosen explicitly for "
             "every run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = replace(
        RUN_CONFIG, output_dir=args.output_dir, tol=args.tol, normalize=args.normalize,
        use_objective_cuts=args.use_objective_cuts, max_iter=args.max_iter,
        max_parallel=args.max_parallel,
    )
    main(config=config)
