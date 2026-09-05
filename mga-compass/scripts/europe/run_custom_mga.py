"""Run our own mga_surveyor package (not PyPSA's built-in MGA) against an
already-solved pypsa-eur electricity network.

Canonical copy lives here, in MGA-Dashboard (this is the project that owns
mga_surveyor and the pypsa version it needs); mga-compass/scripts/europe/
run_pipeline.py deploys it into a pypsa-eur checkout's own scripts/ folder
as part of provisioning, then runs it there with THIS project's venv --
never with pypsa-eur's own pixi env, which is deliberately pinned to an
older pypsa than mga_surveyor needs (>=1.3.0, for the Network-pickling fix
its multiprocessing batch paths rely on). Run standalone with:

    .venv\\Scripts\\python.exe scripts\\run_custom_mga.py \\
        results/<run>/networks/base_s_<clusters>_elec_<opts>.nc --output-dir mga_results/<run>

Dimensions are a fixed list of 8 (component, carrier) pairs (see
DIMENSIONS_SPEC) -- 6 generation, 2 storage, each confirmed extendable in
this network (34-country, elec-country-nodes) via a direct check of
n.generators/n.stores' own p_nom_extendable/e_nom_extendable flags.
StorageUnit (hydro/PHS) has zero extendable elements in this network, so
Store (battery/H2) is the only real storage option; OCGT was dropped for
having just 1 extendable element network-wide, too thin to be a meaningful
axis. Each dimension aggregates that carrier's capacity across every node
(so e.g. "onwind" sums onshore wind p_nom over all countries into one axis).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pypsa

from mga_surveyor import consolidate_mga_results, generate_mga_files

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("pypsa.optimization.optimize").setLevel(logging.WARNING)
logging.getLogger("linopy.io").setLevel(logging.WARNING)
logging.getLogger("linopy.model").setLevel(logging.WARNING)
logging.getLogger("linopy.constants").setLevel(logging.WARNING)
logger = logging.getLogger("run_custom_mga")

#: Fixed MGA axes: (key, component, nominal attribute, plural attribute name)
#: -- 6 generation + 2 storage, each confirmed to have extendable elements in
#: this network (see this module's own docstring for how/why these 8).
DIMENSIONS_SPEC = [
    ("onwind", "Generator", "p_nom", "generators"),
    ("offwind-ac", "Generator", "p_nom", "generators"),
    ("offwind-float", "Generator", "p_nom", "generators"),
    ("solar", "Generator", "p_nom", "generators"),
    ("solar-hsat", "Generator", "p_nom", "generators"),
    ("CCGT", "Generator", "p_nom", "generators"),
    ("battery", "Store", "e_nom", "stores"),
    ("H2", "Store", "e_nom", "stores"),
]


def _detect_solver() -> tuple[str, dict]:
    highs_fallback = ("highs", {"output_flag": False})
    try:
        import gurobipy
    except ImportError:
        return highs_fallback
    try:
        with gurobipy.Env(params={"OutputFlag": 0}):
            return "gurobi", {"env": {"OutputFlag": 0}}
    except gurobipy.GurobiError:
        return highs_fallback


def build_dimensions(n: pypsa.Network) -> dict:
    """One MGA dimension per entry in DIMENSIONS_SPEC, aggregating that
    carrier's nominal capacity across every bus/node in the network (weight
    1 per matching extendable element, 0 elsewhere -- see
    `mga_surveyor.pypsa_ext`/PyPSA's own `build_linexpr_from_weights`, which
    reindexes these weights against the component's own extendables). Raises
    if a spec'd carrier turns out to have no extendable elements in `n` --
    DIMENSIONS_SPEC is meant for this exact network; a silent empty
    dimension would otherwise break `VertexEnumMethod.bootstrap`'s Qhull call
    the same way a saturated one does (zero achievable range either way).
    """
    dimensions: dict = {}
    for key, comp, attr, plural in DIMENSIONS_SPEC:
        static = getattr(n, plural)
        extendable = static[(static["carrier"] == key) & static.get(f"{attr}_extendable", False)]
        if extendable.empty:
            msg = f"No extendable {comp!r} elements with carrier {key!r} found -- DIMENSIONS_SPEC doesn't match this network."
            raise SystemExit(msg)
        # A plain dict, not a pd.Series: n.meta["dimensions"] (this same
        # dict) gets JSON-serialized on every result .nc export (see
        # pypsa_ext.make_optimality_fn), and a Series isn't JSON
        # serializable -- build_linexpr_from_weights converts a dict to a
        # Series internally anyway, so this loses nothing.
        weights = {name: 1.0 for name in extendable.index}
        dimensions[key] = {comp: {attr: weights}}
    return dimensions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", type=Path, help="Path to a solved pypsa-eur network .nc file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write MGA results to")
    parser.add_argument("--slack-levels", type=str, default="0.05", help="Comma-separated slack levels, e.g. 0.02,0.05,0.08")
    parser.add_argument("--norm", choices=["l1", "l2", "linf"], default="linf")
    parser.add_argument(
        "--tol", type=float, required=True,
        help="Convergence tolerance -- a percentage (e.g. 2.0 -> 2%%) since --normalize is on by "
             "default, or a real-unit distance if --no-normalize is passed -- see "
             "mga_surveyor.generate_mga_files's own docstring. No default -- must be chosen "
             "explicitly for every run.",
    )
    parser.add_argument(
        "--no-normalize", action="store_false", dest="normalize", default=True,
        help="Explore each slack level's near-optimal region in raw units instead of a [-1, 1]^k "
             "normalized cube (on by default) -- makes --tol a real-unit distance instead of a "
             "percentage. See mga_surveyor.generate_mga_files's own docstring for why normalizing "
             "matters here: this network's dimensions span very different natural ranges (MW of "
             "wind/solar/gas capacity vs MWh of battery/H2 storage), so an unweighted, unnormalized "
             "tol would let the largest-range dimension dominate convergence.",
    )
    parser.add_argument(
        "--max-iter", type=int, required=True,
        help="Cap on drive() iterations per slack level -- no default, must be chosen explicitly.",
    )
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--obj-label", type=str, default="system_cost")
    args = parser.parse_args()

    slack_levels = [float(s) for s in args.slack_levels.split(",")]
    solver_name, solver_options = _detect_solver()
    logger.info("Solver: %s", solver_name)

    n = pypsa.Network(str(args.network))
    if not n.is_solved:
        msg = f"{args.network} is not a solved network -- run the pypsa-eur solve step first."
        raise SystemExit(msg)

    dimensions = build_dimensions(n)
    logger.info("Dimensions (%d): %s", len(dimensions), list(dimensions.keys()))

    generate_mga_files(
        n, dimensions, args.output_dir, norm=args.norm, slack_levels=slack_levels,
        solver_name=solver_name, tol=args.tol, normalize=args.normalize, max_iter=args.max_iter,
        max_parallel=args.max_parallel, solver_options=solver_options,
    )
    points_df, duals_df, _nc_files = consolidate_mga_results(args.output_dir, args.obj_label)

    points_df.to_pickle(args.output_dir / "points.pkl")
    duals_df.to_pickle(args.output_dir / "duals.pkl")
    points_df.to_csv(args.output_dir / "points.csv")

    logger.info("Total points: %d", len(points_df))
    logger.info("%s", points_df)


if __name__ == "__main__":
    main()
