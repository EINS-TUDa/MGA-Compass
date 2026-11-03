"""mga_surveyor: reusable inner/outer polytope approximation for
near-optimal-space exploration, with three interchangeable worst-point /
convergence-metric methods:

- `CddMethod` ("cdd solution"): recomputes the outer polytope's vertex set
  via cdd's Double Description method -- from scratch on every call, or
  incrementally (folding in only the newest halfspaces) when the installed
  `cdd`/`cdd.gmp` has the local incremental-append patch (see
  `vendor/cdd/README.md`). Two arithmetic backends (`backend='float'`,
  the default, or `'gmp'` for exact rational arithmetic) and, separately, a
  choice of solver for its own inner-hull projection (`projection_backend`,
  `'gurobi'` or `'highs'`).
- `BBMethod` ("beneath-and-beyond"): maintains the outer polytope's
  vertex set incrementally through Qhull's polar dual (float arithmetic
  only), with the same choice of `projection_backend`.
- `BilevelMethod` ("bilevel solution"): the original KKT + Big-M MILP
  reformulation. L_infinity only.

All three share the same `Method` interface (`methods/base.py`) and
accept `norm` (`'l1'`, `'l2'`, or `'linf'`) as a constructor argument
(`BilevelMethod` is restricted to `'linf'`; see its docstring for why).
Only `CddMethod` also has `find_pool_trial_points`, which `drive` uses for
batched closest-point solves when `max_parallel > 1`.

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
main loop instead -- one target per iteration, or a batch of them when
`drive`'s `max_parallel > 1` and the method has `find_pool_trial_points`
(the next batch still depends on the previous one's results) -- and
currently only supports `norm in ('l1', 'linf')`, not `'l2'` (would need a quadratic
objective, Gurobi-only).

`normalize` wraps any three `drive` oracle functions so the polytope
`drive` explores lives in a `[-1, 1]^k`-normalized cube instead of raw
dimension units -- useful whenever those units differ wildly in scale
(e.g. MW vs MWh), since an unweighted polytope over such dimensions lets
the large-range ones dominate. `pypsa_ext.generate_mga_files`'s own
`normalize` (`True` by default) wires this in for a whole run.

`common.linearized_objective_cut` turns a single solved point's own
`duals` (already computed, otherwise unused, whenever
`resolve_main_objective=True`) into an extra supporting halfspace of the
true near-optimal region -- provably valid, not a heuristic (see its own
docstring: convexity of the achievable-cost value function makes the
tangent at any solved point a global underestimator). Free tightening, no
new solves. `pypsa_ext.generate_mga_files`'s own `use_objective_cuts`
(`True` by default) uses this for every point the main loop finds
(live, and for any existing result read back on a restart), plus one free
extra per slack level from the network's own optimum (added
unconditionally -- it has no same-run geometric normal to check it
against, and unlike a box-direction point there's no structural reason
for its gradient to align with any axis), composing with `normalize=True`
if that's also on. Deliberately NOT derived from any box-direction point,
even though those also have `duals` sitting unused -- a box-direction
point's own objective cut, in particular, comes out *exactly* parallel
to its own box halfspace -- not a coincidence: its pinned re-solve pins
only the queried dimension, so its duals are zero in every other one --
and nearly-coincident hyperplanes are
a real degeneracy risk regardless of which method processes them
(confirmed in practice against the incremental edge-walking construction
this project used before `CddMethod` became the production method:
unfiltered, this reliably triggered a combinatorial explosion in its
crossing-vertex computation; kept excluded here as a precaution, not
re-validated against `CddMethod`'s own construction). Closest-point and
optimum cuts don't share that structural alignment (their angles to any
same-run geometric normal come out scattered, not clustered near 0/180
the way box cuts are), so they're never filtered for near-parallelism --
an earlier version did check each closest-point cut against its own
same-solve geometric normal first, but on the real test network that
check discarded cuts that still carried real tightening value, costing
extra main-loop iterations for no compensating safety benefit, so it was
dropped; cddlib's own Double Description construction already discards
any genuinely duplicate/redundant cut for free, regardless. One remaining
check before any cut from the main loop
(or a restart's existing files) is added: skipped if the point it came
from was already feasible, i.e. within `sep_tol` of its target (the same
`common.separates` test `explore.drive` uses for its own geometric cut).
"""
from .common import SEP_TOL_FACTOR, VALID_NORMS, gaps, linearized_objective_cut, separates
from .gurobi_projection import closest_point_on_hull
# Both backend modules name their projection function `closest_point_on_hull`
# (module-qualified, e.g. `gurobi_projection.closest_point_on_hull` vs
# `highs_projection.closest_point_on_hull` -- they're never used side by side
# under one name, so there's no reason for either to carry a backend
# suffix). This flat top-level namespace re-exports both at once, though,
# so the highs one needs an alias here to avoid shadowing the gurobi one.
from .highs_projection import closest_point_on_hull as closest_point_on_hull_highs, VALID_NORMS_HIGHS
from .methods import (
    Method, BilevelMethod, solve_farthest_outer_point_milp, compute_safe_big_m,
    CddMethod, cdd_vertices_of, BBMethod,
)
from .explore import drive
from .normalize import AffineMap, compute_affine_map, normalize_from_box_points, normalize_oracle_fns
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
    "linearized_objective_cut", "separates", "SEP_TOL_FACTOR",
    "closest_point_on_hull_highs", "VALID_NORMS_HIGHS",
    "Method", "BilevelMethod", "solve_farthest_outer_point_milp", "compute_safe_big_m",
    "CddMethod", "cdd_vertices_of", "BBMethod", "drive",
    "AffineMap", "compute_affine_map", "normalize_oracle_fns", "normalize_from_box_points",
    "make_optimality_fn", "make_furthest_point_fn", "make_closest_point_fn", "make_oracle_fns",
    "generate_mga_files", "consolidate_mga_results", "encode_cache_key",
]
