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

from mga_surveyor import generate_mga_files

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("pypsa.optimization.optimize").setLevel(logging.WARNING)
logging.getLogger("linopy.io").setLevel(logging.WARNING)
logging.getLogger("linopy.model").setLevel(logging.WARNING)
logging.getLogger("linopy.constants").setLevel(logging.WARNING)
# gurobipy has its own Python-logging integration, entirely separate from
# its native OutputFlag/LogToConsole parameters (see _GUROBI_TUNING) --
# once *any* logging.basicConfig() is active (as above), gurobipy starts
# emitting every console line via logging.getLogger("gurobipy").info(...)
# too, which propagates to every handler on the root logger: both the
# console (basicConfig's own StreamHandler) and, unwanted, the run.log
# FileHandler added in main(). This is deliberately NOT silenced via
# logging.getLogger("gurobipy").setLevel(...) -- that was tried and
# reverted: it suppresses gurobipy's messages everywhere, including the
# console, which is the *only* live progress signal available during a
# solve (Gurobi's own LogFile is buffered and only flushes on that
# solve's own completion/interruption -- confirmed by direct testing).
# Losing that made an in-progress (if slow) solve look identical to a
# genuinely frozen one. Instead, main() attaches a filter to the run.log
# FileHandler specifically, excluding only records from the "gurobipy"
# logger -- so gurobipy's console visibility survives untouched, and only
# run.log stays clean.
class _ExcludeLoggerFilter(logging.Filter):
    def __init__(self, excluded_name: str) -> None:
        super().__init__()
        self.excluded_name = excluded_name

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != self.excluded_name


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


#: Same barrier-without-crossover tuning as pypsa-eur's own "gurobi-default"
#: preset (config.default.yaml) -- this script builds/solves the network's
#: optimum itself (make_optimality_fn's own two full solves) rather than
#: reusing pypsa-eur's already-solved result, so without these it falls back
#: to Gurobi's own defaults (crossover ON, no method pinned), which is
#: markedly slower on a network this size than the settings that solved it
#: in ~7 min inside pypsa-eur's pipeline.
#: BarConvTol tightened to 1e-8 (Gurobi's own out-of-the-box default,
#: confirmed via gurobipy -- pypsa-eur's own "gurobi-default" preset loosens
#: it to 1e-5): with crossover off, a furthest/closest-point solve's
#: returned coordinates are only an interior-point approximation, not an
#: exact vertex -- that point then gets pinned as a (small-tolerance-band,
#: not exact) constraint in the same solver's own pinned-duals resolve.
#: LogToConsole=0 unconditionally: Gurobi's console mirror is pure noise at
#: this scale (license banner, "Read LP format...", full barrier iteration
#: log) and, when this process's own stdout is captured into run.log (e.g.
#: a screen -L session or `> run.log 2>&1`), that noise ends up duplicated
#: into the app-level log too. This is deliberately NOT done via
#: OutputFlag=0 -- OutputFlag is Gurobi's master output switch, and setting
#: it to 0 also suppresses LogFile, which is the one place this solve's
#: full detail is supposed to end up (mga_surveyor.pypsa_ext's per-
#: direction/target LogFile, written next to that solve's own .nc result).
#: LogToConsole only silences the console mirror, leaving LogFile intact.
_GUROBI_TUNING = {
    "method": 2,
    "crossover": 0,
    "BarConvTol": 1.0e-08,
    "Seed": 123,
    "AggFill": 0,
    "PreDual": 0,
    "GURO_PAR_BARDENSETHRESH": 200,
    "LogToConsole": 0,
}


def _detect_solver(verbose: bool = True) -> tuple[str, dict]:
    """Gurobi's own console output is always off (see _GUROBI_TUNING's
    LogToConsole=0) -- full solver detail goes only to each direction/
    target's own LogFile, next to its .nc result; run.log stays limited to
    this app's own summary logging (mga_surveyor.pypsa_ext's `_log_solve`:
    status/condition per solve; explore.py's `drive`: gap per iteration).
    `verbose` (`--quiet-solver` inverted) only still matters for the HiGHS
    fallback, which has no per-direction LogFile of its own (
    mga_surveyor.pypsa_ext's `_solver_kwargs_with_log` only wires LogFile
    up for gurobi) -- there, its console output is the only detail
    available at all, so `verbose=False` there means no detail whatsoever,
    not "moved to a file" like it does for Gurobi."""
    highs_fallback = ("highs", {"output_flag": verbose})
    try:
        import gurobipy
    except ImportError:
        return highs_fallback
    try:
        with gurobipy.Env(params={}):
            return "gurobi", {"env": {}, **_GUROBI_TUNING}
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
    dimension gives no meaningful MGA axis at all -- same as a saturated one
    (zero achievable range either way).
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
    parser.add_argument("--norm", choices=["l1", "linf"], default="linf")  # the closest-point solver has no l2
    parser.add_argument(
        "--tol", type=float, required=True,
        help="Convergence tolerance -- a percentage (e.g. 2.0 -> 2%%) since --normalize is on by "
             "default, or a real-unit distance if --no-normalize is passed -- see "
             "mga_surveyor.generate_mga_files's own docstring. No default -- must be chosen "
             "explicitly for every run.",
    )
    parser.add_argument(
        "--sep-tol", type=float, default=None,
        help="Separation threshold, same units as --tol: a closest point more than this from its "
             "target adds its cuts, otherwise only the point is kept. Must be in [0, --tol). "
             "Default: --tol * 1e-3.",
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
        "--max-iter", type=int, default=None,
        help="Cap on drive() iterations per slack level. Omit for no cap (mga_surveyor.explore.drive's "
             "own default: runs until convergence).",
    )
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument(
        "--method", choices=["cdd", "bb"], default="cdd",
        help="Vertex enumeration of the outer polytope: cddlib (CddMethod, default) or Qhull's "
             "polar dual (BBMethod).",
    )
    parser.add_argument(
        "--quiet-solver", action="store_true", default=False,
        help="HiGHS fallback only -- suppresses its console progress log. No effect on Gurobi: its "
             "console output is always off (see _GUROBI_TUNING's LogToConsole=0); full solver detail "
             "goes only to each direction/target's own log file under --output-dir (see "
             "mga_surveyor.pypsa_ext's LogFile support), not the console, and not run.log.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(args.output_dir / "run.log")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.addFilter(_ExcludeLoggerFilter("gurobipy"))
    logging.getLogger().addHandler(file_handler)

    slack_levels = [float(s) for s in args.slack_levels.split(",")]
    solver_name, solver_options = _detect_solver(verbose=not args.quiet_solver)
    logger.info("Solver: %s", solver_name)

    n = pypsa.Network(str(args.network))
    if not n.is_solved:
        msg = f"{args.network} is not a solved network -- run the pypsa-eur solve step first."
        raise SystemExit(msg)

    dimensions = build_dimensions(n)
    logger.info("Dimensions (%d): %s", len(dimensions), list(dimensions.keys()))

    generate_mga_files(
        n, dimensions, args.output_dir, norm=args.norm, slack_levels=slack_levels,
        solver_name=solver_name, tol=args.tol, sep_tol=args.sep_tol, normalize=args.normalize, max_iter=args.max_iter,
        max_parallel=args.max_parallel, solver_options=solver_options, vertex_method=args.method,
    )
    logger.info("Done. Build the near-optimal set from %s with build_dataset_from_results.py.", args.output_dir)


if __name__ == "__main__":
    main()
