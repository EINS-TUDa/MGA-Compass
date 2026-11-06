# Choosing a method

All trial-point methods share the {py:class}`~mga_surveyor.methods.base.Method` interface, so
{py:func}`~mga_surveyor.explore.drive` runs any of them unchanged. They differ only in how they
find the vertex of the outer approximation furthest from the inner one, which is Step 2 of
the [algorithm](g-oracle.md).

| Class | Approach | Paper name |
|---|---|---|
| {py:class}`~mga_surveyor.methods.bb.BBMethod` | G-ORACLE, beneath-and-beyond on the dual polytope (Qhull, floating point) | G-BB |
| {py:class}`~mga_surveyor.methods.cdd.CddMethod`, `backend="float"` | G-ORACLE, double description (patched cddlib, floating point) | G-DD-float |
| {py:class}`~mga_surveyor.methods.cdd.CddMethod`, `backend="gmp"` | G-ORACLE, double description in exact rational arithmetic | G-DD-rational |
| {py:class}`~mga_surveyor.methods.bilevel.BilevelMethod` | ORACLE's bilevel problem as a MILP (Gurobi, $\ell_\infty$ only) | O-MILP |

## What the experiments showed

The paper compared all four on the same inner and outer approximations of a PyPSA-Eur model
with eight PoIs:

- **G-BB** matched the exact reference (G-DD-rational) to within $3 \times 10^{-7}$
  percentage points at every iteration, and took at most 7 s per iteration.
- **G-DD-rational** is exact, but took up to 2.9 h per iteration (41.5 h in total).
- **G-DD-float** was about as fast as G-BB, but lost vertices as the run went on (only 12% of
  the exact vertex count by the end). From iteration 89 on it *underestimated* the distance
  in 66 iterations, which can stop a run before it has really converged.
- **O-MILP** grew by about 12% per iteration and exceeded 8 hours at iteration 94.

**Recommendation: use `BBMethod`.** It is as fast as the floating-point double description
and as accurate as the exact one. Use `CddMethod` with `backend="gmp"` when you need an exact
reference, and `BilevelMethod` only to compare against ORACLE.

{py:func}`~mga_surveyor.pypsa_ext.generate_mga_files` follows this recommendation: its
`vertex_method` defaults to `"bb"`. Pass `vertex_method="cdd"` to use the floating-point
double description instead.

## Norm

Every method takes `norm`, one of `"l1"`, `"l2"` or `"linf"`, which fixes how distances, and
so the tolerance, are measured. `BilevelMethod` supports only `"linf"`. The closest-point
solve against a PyPSA network ({py:func}`~mga_surveyor.pypsa_ext.make_closest_point_fn`)
currently supports `"l1"` and `"linf"`. With [normalization](normalization.md), `"linf"`
bounds the error of every PoI individually, which is usually what you want.

## Projection backend

The per-vertex projection $f(v)$ runs on Gurobi or HiGHS, chosen with `projection_backend`:

| Backend | Norms | Licence |
|---|---|---|
| `"gurobi"` ({py:mod}`~mga_surveyor.gurobi_projection`) | `l1`, `l2`, `linf` | Commercial (free for academic use) |
| `"highs"` ({py:mod}`~mga_surveyor.highs_projection`) | `l1`, `linf` | Open source |

HiGHS has no quadratic objectives, hence no $\ell_2$.

## Several trial points per iteration

Since G-ORACLE knows the distance of *every* vertex, it can solve the nearest-point problem
for several far vertices at once instead of only the furthest. With `max_parallel > 1`,
`drive` does this for methods that provide `find_pool_trial_points`, which is currently
`CddMethod` only. The paper notes the caveat: nearby vertices give largely redundant cuts, so
convergence does not speed up in proportion to the extra solves.
