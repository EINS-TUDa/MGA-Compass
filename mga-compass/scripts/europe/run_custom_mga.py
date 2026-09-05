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

Dimensions are built automatically from the network: one per (component,
carrier) pair among the extendable Generator/StorageUnit/Store/Link elements,
aggregating that carrier's capacity across every node (so e.g. "onwind" sums
onshore wind p_nom over all nodes into a single MGA axis).
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

#: (component, nominal attribute) pairs to scan for extendable capacity.
#: Link deliberately excluded by default -- pypsa-eur's extendable links are
#: mostly derived conversion assets (H2 electrolysis/fuel cell, heat pumps,
#: ...) with no p_nom_max of their own, so a furthest-point query that tries
#: to maximize one in isolation is genuinely unbounded (confirmed: raises
#: infeasible_or_unbounded on the BE test network for "H2 Fuel Cell"). Pass
#: --include-links to add them back in anyway.
_COMPONENTS = [
    ("Generator", "p_nom", "generators"),
    ("StorageUnit", "p_nom", "storage_units"),
    ("Store", "e_nom", "stores"),
]
_LINK_COMPONENT = ("Link", "p_nom", "links")

#: Preferred MGA axes for an electricity-only, one-node-per-country network,
#: in priority order -- picked over pure capacity-ranking because the
#: highest-capacity carriers are often near-duplicates of each other (e.g.
#: solar-hsat is a minor variant of solar, OCGT of CCGT) and would otherwise
#: crowd out a technology that's small in installed MW but represents a
#: genuinely different system archetype. Covers: onshore + offshore wind,
#: solar, gas flexibility, short-duration storage (battery), and
#: seasonal/long-duration storage (H2) -- one representative of each
#: distinct technology category rather than several near-substitutes of the
#: same one. Any dimension not in this list still falls back to
#: capacity-ranking to fill remaining slots (see `build_dimensions`).
PREFERRED_DIMENSIONS = ["onwind", "solar", "offwind-ac", "CCGT", "battery", "H2"]


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


def _is_saturated(static, group, attr: str, tol: float = 1e-6) -> bool:
    """True if this dimension's aggregate capacity is already pinned at its
    own physical maximum at the network's true optimum -- i.e. its box
    "maximize" direction has zero headroom no matter how much cost slack is
    allowed, so every box vertex/interior point mga_surveyor's bootstrap
    would sample collapses onto the same value in this axis (confirmed: this
    is exactly what made scipy's HalfspaceIntersection fail on the BE test
    network's saturated "offwind-ac" dimension -- QhullError "feasible point
    is not clearly inside halfspace"). An uncapped element (p_nom_max/
    e_nom_max == inf) always has headroom, so the whole dimension is never
    saturated if any one element in it is uncapped."""
    max_attr = f"{attr}_max"
    opt_attr = f"{attr}_opt"
    sub = static.loc[group.index]
    if (sub[max_attr] == float("inf")).any():
        return False
    total_max = sub[max_attr].sum()
    total_opt = sub[opt_attr].sum()
    if total_max <= 0:
        return True
    return (total_max - total_opt) <= tol * total_max


def build_dimensions(
    n: pypsa.Network, include_links: bool = False, max_dimensions: int | None = 6,
) -> dict:
    """One MGA dimension per (component, carrier) among extendable elements,
    aggregating that carrier's nominal capacity across every bus/node in the
    network (weight 1 per matching element, 0 elsewhere -- see
    `mga_surveyor.pypsa_ext`/PyPSA's own `build_linexpr_from_weights`, which
    reindexes these weights against the component's own extendables).

    Dimensions already saturated at their own physical cap (see
    `_is_saturated`) are dropped -- they have zero achievable range and
    break `VertexEnumMethod.bootstrap`'s initial Qhull call. If more than
    `max_dimensions` remain, keeps `PREFERRED_DIMENSIONS` first (in that
    priority order, skipping any that were dropped as saturated or never
    found in this network), then fills any remaining slots with whatever's
    left, largest installed capacity at the optimum first -- `None` keeps
    all candidates (still preferred-first, just no cutoff).
    """
    candidates: list[tuple[str, dict, float]] = []  # (key, {comp: {attr: weights}}, total_opt)
    components = [*_COMPONENTS, _LINK_COMPONENT] if include_links else _COMPONENTS
    for comp, attr, plural in components:
        static = getattr(n, plural)
        if static.empty:
            continue
        extendable_col = f"{attr}_extendable"
        extendable = static[static.get(extendable_col, False)]
        if extendable.empty:
            continue
        for carrier, group in extendable.groupby("carrier"):
            key = carrier if carrier not in {k for k, _, _ in candidates} else f"{comp}_{carrier}"
            if _is_saturated(static, group, attr):
                logger.info("Dropping dimension %r: saturated at its own physical maximum (zero headroom)", key)
                continue
            # A plain dict, not a pd.Series: `n.meta["dimensions"]` (this
            # same dict) gets JSON-serialized on every result .nc export
            # (see pypsa_ext.make_optimality_fn), and a Series isn't JSON
            # serializable -- build_linexpr_from_weights converts a dict to
            # a Series internally anyway, so this loses nothing.
            weights = {name: 1.0 for name in group.index}
            total_opt = float(static.loc[group.index, f"{attr}_opt"].sum())
            candidates.append((key, {comp: {attr: weights}}, total_opt))

    by_key = {key: (weights, total_opt) for key, weights, total_opt in candidates}
    preferred = [key for key in PREFERRED_DIMENSIONS if key in by_key]
    rest = sorted((key for key in by_key if key not in preferred), key=lambda k: by_key[k][1], reverse=True)
    ordered = preferred + rest

    if max_dimensions is not None and len(ordered) > max_dimensions:
        logger.info(
            "Keeping %d dimension(s) (preferred list first, then largest-capacity), dropping: %s",
            max_dimensions, ordered[max_dimensions:],
        )
        ordered = ordered[:max_dimensions]

    return {key: by_key[key][0] for key in ordered}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", type=Path, help="Path to a solved pypsa-eur network .nc file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write MGA results to")
    parser.add_argument("--slack-levels", type=str, default="0.05", help="Comma-separated slack levels, e.g. 0.02,0.05,0.08")
    parser.add_argument("--norm", choices=["l1", "l2", "linf"], default="linf")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--obj-label", type=str, default="system_cost")
    parser.add_argument(
        "--include-links", action="store_true", default=False,
        help="Also add extendable Link carriers (e.g. H2 Electrolysis) as MGA dimensions. Off by "
             "default: these commonly have no p_nom_max, so maximizing one alone can be unbounded.",
    )
    parser.add_argument(
        "--max-dimensions", type=int, default=6,
        help="Cap on the number of MGA dimensions (axes) -- keeps the largest-capacity ones if more "
             "candidates are found. Pass 0 to keep all.",
    )
    args = parser.parse_args()

    slack_levels = [float(s) for s in args.slack_levels.split(",")]
    solver_name, solver_options = _detect_solver()
    logger.info("Solver: %s", solver_name)

    n = pypsa.Network(str(args.network))
    if not n.is_solved:
        msg = f"{args.network} is not a solved network -- run the pypsa-eur solve step first."
        raise SystemExit(msg)

    max_dimensions = args.max_dimensions if args.max_dimensions > 0 else None
    dimensions = build_dimensions(n, include_links=args.include_links, max_dimensions=max_dimensions)
    if not dimensions:
        msg = "No extendable capacity found to build MGA dimensions from."
        raise SystemExit(msg)
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
