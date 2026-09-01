"""mga_surveyor: reusable inner/outer polytope approximation for
near-optimal-space exploration, with three interchangeable worst-point /
convergence-metric methods:

- `VertexEnumMethod` ("my solution"): replaces the bilevel-MILP worst-point
  search with vertex enumeration of the outer approximation, exploiting the
  fact that a convex function's max over a polytope is attained at a
  vertex -- so no MILP is needed at all. Maintains the outer polytope's
  vertex set incrementally.
- `BilevelMethod` ("bilevel solution"): the original KKT + Big-M MILP
  reformulation. L_infinity only.
- `CddMethod` ("cdd solution"): recomputes the outer polytope's vertex set
  from scratch via cdd's Double Description method on every call. Simpler
  and more independently-verified than the incremental path, at the cost
  of redoing more work each time; also useful as a periodic ground-truth
  checkpoint on top of `VertexEnumMethod` (see `VertexEnumMethod.resync`).

All three share the same `Method` interface (`methods/base.py`) and
accept `norm` (`'l1'`, `'l2'`, or `'linf'`) as a constructor argument
(`BilevelMethod` is restricted to `'linf'`; see its docstring for why).

None of this package depends on any specific test example or system model:
`add_point`/`add_cut` are pure state updates on whatever point/direction
you hand them, and `find_trial_point` only ever looks at the state already
inside the method. `explore.drive` bootstraps a method from the 2k box
directions, a known optimum, and (optionally) any other already-known
points/cuts -- all folded into one `Method.bootstrap` call, since that
must be called exactly once, before any other state-changing call -- then
drives it to convergence, against external callbacks (a real system
model, or `mga_surveyor.tests.lp_region` against an explicit LP for
testing) -- or write your own bootstrap/loop entirely.

`pypsa_ext` is the PyPSA-side adapter supplying the oracle queries
`mga_surveyor.explore` needs against a real PyPSA network's own solve, so
the engine above can run against an actual system model instead of only a
test example. `make_optimality_fn` supplies `explore.OptimalPointFn`: a
callable that solves and caches a network's own optimum, returning its
point in dimensions space -- the one precondition everything else in
`pypsa_ext` assumes but never itself provides, and (optionally) a free
extra point for `drive`'s own bootstrap. `make_furthest_point_fn` supplies
`explore.FurthestPointFn` (also single-direction-capable beyond that batch
contract, for callers that want it) -- also the one place `pypsa_ext`
solves MGA in a direction at all, whether that's points-only (the
default) or a one-off/dataset-generation call with duals and each point's
own cost (`resolve_main_objective=True`; see its docstring).
`make_closest_point_fn` supplies a `explore.ClosestPointFn` for `drive`'s
main loop instead -- that one can't be batched the same way (each
iteration depends on the previous one's result), and currently only
supports `norm in ('l1', 'linf')`, not `'l2'` (would need a quadratic
objective, Gurobi-only).
"""
from .common import VALID_NORMS, gaps
from .gurobi_projection import closest_point_on_hull
# Both backend modules name their projection function `closest_point_on_hull`
# (module-qualified, e.g. `gurobi_projection.closest_point_on_hull` vs
# `highs_projection.closest_point_on_hull` -- they're never used side by side
# under one name, so there's no reason for either to carry a backend
# suffix). This flat top-level namespace re-exports both at once, though,
# so the highs one needs an alias here to avoid shadowing the gurobi one.
from .highs_projection import closest_point_on_hull as closest_point_on_hull_highs, VALID_NORMS_HIGHS
from .methods import (
    Method, VertexEnumMethod, BilevelMethod, solve_farthest_outer_point_milp,
    CddMethod, cdd_vertices_of,
)
from .explore import drive
from .pypsa_ext import (
    make_optimality_fn,
    make_furthest_point_fn,
    make_closest_point_fn,
    make_oracle_fns,
    generate_mga_files,
    consolidate_mga_results,
    encode_cache_key,
)

__all__ = [
    "VALID_NORMS", "closest_point_on_hull", "gaps",
    "closest_point_on_hull_highs", "VALID_NORMS_HIGHS",
    "Method", "VertexEnumMethod", "BilevelMethod", "solve_farthest_outer_point_milp",
    "CddMethod", "cdd_vertices_of", "drive",
    "make_optimality_fn", "make_furthest_point_fn", "make_closest_point_fn", "make_oracle_fns",
    "generate_mga_files", "consolidate_mga_results", "encode_cache_key",
]
