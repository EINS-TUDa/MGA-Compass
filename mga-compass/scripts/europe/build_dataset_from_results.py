"""Build the near-optimal set of a finished run_custom_mga.py run on PyPSA-Eur,
ready for scripts/publish_dataset.py.

Takes the run's optimum, its closest-point results, its run summary and a set of
box (furthest-point) results -- by default the run's own; pass --box-dir for ones
re-solved separately -- links them into one layout, collects them as a
near-optimal set (mga_surveyor.collect_near_optimal_set), and adds two outputs:

- generation: Europe-wide electricity generation per carrier and snapshot (MW)
  -- every generator plus hydro/PHS dispatch; battery and H2 discharge are
  storage, not generation;
- capacity: Europe-wide installed power capacity per carrier (MW) --
  generators, hydro/PHS, battery charger/discharger, H2 electrolysis and fuel
  cell; storage energy (MWh) is left out so the bars share one unit, and DC
  links are transmission.

Names, units, descriptions, labels, colours, plots and display units come from
scripts/datasets/pypsa_eur_34.json. Values stay in the model's units (MW, MWh,
€/a); the platform shows them in GW, TWh and bn €/a through the config's display
entries. Writes <output-dir>/set and <output-dir>/near-optimal-set.zip.

    uv run --extra scripts python scripts/europe/build_dataset_from_results.py \\
        --run-dir /path/to/mga_results_4 --box-dir /path/to/mga_results_4_box/slack_0.05 \\
        --network /path/to/base_s_34_elec_.nc --output-dir scripts/output_pypsa_eur
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/, for dataset_config
from dataset_config import DatasetConfig, long_output, write_set  # noqa: E402

from mga_surveyor import collect_near_optimal_set  # noqa: E402
from mga_surveyor.pypsa_ext import RUN_SUMMARY_NAME  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("build_dataset_from_results")

DATASET_CONFIG = Path(__file__).resolve().parent.parent / "datasets" / "pypsa_eur_34.json"
#: (component, nominal-capacity variable, carriers counted as power capacity)
CAPACITY_COMPONENTS = [
    ("generators", "generators_p_nom_opt", None),  # every carrier
    ("storage_units", "storage_units_p_nom_opt", None),
    ("links", "links_p_nom_opt", {"battery charger", "battery discharger", "H2 Electrolysis", "H2 Fuel Cell"}),
]


def link_layout(run_dir: Path, box_dir: Path, slack: str, layout_dir: Path) -> None:
    """Symlink the run's optimum, its closest-point results, its run summary and
    the chosen box results into one `output_dir` layout collect_near_optimal_set
    can read."""
    target = layout_dir / f"slack_{slack}"
    target.mkdir(parents=True, exist_ok=False)
    (layout_dir / "optimal.nc").symlink_to(run_dir / "optimal.nc")
    summary = run_dir / f"slack_{slack}" / RUN_SUMMARY_NAME
    if not summary.exists():
        raise SystemExit(f"{summary} not found: the run has no run summary (older runs do not write one)")
    (target / RUN_SUMMARY_NAME).symlink_to(summary)
    for f in sorted((run_dir / f"slack_{slack}").glob("mga_closest_result_*.nc")):
        (target / f.name).symlink_to(f)
    box = sorted(box_dir.glob("mga_furthest_result_*.nc"))
    if len(box) != 16:
        raise SystemExit(f"expected 16 box results in {box_dir}, found {len(box)}")
    for f in box:
        (target / f.name).symlink_to(f)


def _strings(ds: Dataset, var: str) -> list[str]:
    return [str(x) for x in ds[var][:]]


def _sum_by(values: np.ndarray, labels: list[str], order: list[str]) -> np.ndarray:
    """Sum the last axis of `values` by `labels`, in `order`."""
    out = np.zeros(values.shape[:-1] + (len(order),))
    pos = {c: i for i, c in enumerate(order)}
    for j, label in enumerate(labels):
        out[..., pos[label]] += values[..., j]
    return out


def extract(nc_files: list[Path], snapshots: pd.DatetimeIndex) -> tuple[xr.DataArray, xr.DataArray]:
    """Generation and capacity of each result file, indexed by its position,
    which is its point in the set."""
    with Dataset(nc_files[0]) as ds0:
        gen_carriers = list(dict.fromkeys(_strings(ds0, "generators_carrier") + _strings(ds0, "storage_units_carrier")))
        cap_carriers = []
        for comp, _, keep in CAPACITY_COMPONENTS:
            cap_carriers += [c for c in dict.fromkeys(_strings(ds0, f"{comp}_carrier")) if keep is None or c in keep]
        cap_carriers = list(dict.fromkeys(cap_carriers))

    generation = np.zeros((len(nc_files), len(snapshots), len(gen_carriers)))
    capacity = np.zeros((len(nc_files), len(cap_carriers)))
    for pos, f in enumerate(nc_files):
        with Dataset(f) as ds:
            for comp, var in (("generators", "generators_t_p"), ("storage_units", "storage_units_t_p_dispatch")):
                carrier_of = dict(zip(_strings(ds, f"{comp}_i"), _strings(ds, f"{comp}_carrier")))
                names = _strings(ds, f"{var}_i")  # only the elements that were ever dispatched
                generation[pos] += _sum_by(np.asarray(ds[var][:]), [carrier_of[x] for x in names], gen_carriers)
            for comp, var, keep in CAPACITY_COMPONENTS:
                carriers = _strings(ds, f"{comp}_carrier")
                values = np.asarray(ds[var][:])
                mask = [keep is None or c in keep for c in carriers]
                capacity[pos] += _sum_by(values[mask], [c for c, m in zip(carriers, mask) if m], cap_carriers)
        if pos % 25 == 0:
            logger.info("extracted %d/%d", pos + 1, len(nc_files))

    index = pd.RangeIndex(len(nc_files), name="index")
    gen = xr.DataArray(generation, dims=["index", "snapshot", "technology"],
                       coords={"index": index, "snapshot": snapshots, "technology": gen_carriers})
    cap = xr.DataArray(capacity, dims=["index", "technology"], coords={"index": index, "technology": cap_carriers})
    return gen, cap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--box-dir", type=Path, help="Box results to use (default: the run's own).")
    parser.add_argument("--network", type=Path, required=True, help="The network the run explored (for snapshots).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slack", default="0.05")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    box_dir = (args.box_dir or run_dir / f"slack_{args.slack}").resolve()
    layout = args.output_dir / "layout"
    link_layout(run_dir, box_dir, args.slack, layout)

    dataset = DatasetConfig.load(DATASET_CONFIG)
    nos, files = collect_near_optimal_set(layout, **dataset.collect_arguments())
    logger.info("points %s", nos.points.shape)
    snapshots = pd.DatetimeIndex(pypsa.Network(str(args.network)).snapshots)
    generation, capacity = extract(files, snapshots)

    outputs = {"generation": long_output(generation), "capacity": long_output(capacity)}
    members = {
        "snapshot": list(snapshots),
        "technology": list(dict.fromkeys([*generation.technology.values, *capacity.technology.values])),
    }
    nos, outputs = dataset.apply(nos, outputs, members)
    archive = write_set(nos, outputs, args.output_dir / "set")
    logger.info("generation %s %s, capacity %s %s", dict(generation.sizes), list(generation.technology.values),
                dict(capacity.sizes), list(capacity.technology.values))
    logger.info("wrote %s and %s", args.output_dir / "set", archive)


if __name__ == "__main__":
    main()
