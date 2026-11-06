# Run parameters

The parameters of {py:func}`~mga_surveyor.pypsa_ext.generate_mga_files`, which runs a full
exploration of a solved PyPSA network. For why they exist, see the
[Surveyor explanation pages](../surveyor/explanation.md).

| Parameter | Default | Meaning |
|---|---|---|
| `n` | required | An already solved `pypsa.Network`. |
| `dimensions` | required | The properties of interest; see [](#dimensions). |
| `output_dir` | required | Where all results are written; also where a run resumes from. |
| `norm` | required | `"l1"` or `"linf"`: how distances, and so `tol`, are measured. |
| `slack_levels` | required | Cost slacks $\epsilon$ to explore, e.g. `[0.02, 0.05]` for 2% and 5%. Each is explored separately. |
| `solver_name` | required | `"gurobi"` or `"highs"`. Used for the model solves and for the per-vertex projections. |
| `max_iter` | required | Iteration cap per slack level, or `None` for no cap. One value, or one per slack level. |
| `tol` | required | Convergence tolerance; see [](#tolerance). One value, or one per slack level. |
| `max_parallel` | required | Worker processes for the model solves. With `vertex_method="cdd"`, also the number of trial points solved per iteration. |
| `sep_tol` | `tol × 0.001` | A cut is added only if the nearest point is further than this from the trial point. Must satisfy `0 <= sep_tol < tol`. Same units as `tol`. |
| `verbose` | `True` | Log progress. |
| `normalize` | `True` | Explore in coordinates where every PoI spans $[-1, 1]$; see [normalization](../surveyor/normalization.md). |
| `use_objective_cuts` | `True` | Add a cost cut for every point found. |
| `vertex_method` | `"bb"` | `"bb"` (`BBMethod`) or `"cdd"` (`CddMethod`, float arithmetic); see [](../surveyor/methods.md). |
| `**kwargs` | | Passed to every model solve, e.g. `solver_options`. |

(dimensions)=
## Dimensions

`dimensions` maps each property-of-interest name to a linear combination of PyPSA
component attributes: name → component class → attribute → {component name: coefficient}.

```python
dimensions = {
    "wind": {"Generator": {"p_nom": {"wind": 1}}},
    "battery": {"StorageUnit": {"p_nom": {"battery storage": 1}}},
    "hydrogen": {"Store": {"e_nom": {"hydrogen storage": 1}}},
}
```

Here `wind` is the optimized `p_nom` of the generator named `wind`. Listing several
components, or several attributes, sums them with the given coefficients.

(tolerance)=
## Tolerance

With `normalize=True`, `tol` is a **percentage of each PoI's achievable range** at that slack.
Every normalized PoI spans a width of 2, so `tol=4` stops at a distance of $0.08$ in the
normalized coordinates. What the percentage bounds depends on `norm`:

| `norm` | `tol=4` guarantees |
|---|---|
| `"linf"` | every PoI individually is within 4% of its range |
| `"l1"` | the per-PoI percentage errors sum to at most 4 |

With `normalize=False`, `tol` is a plain distance in the PoIs' own units.

## Output directory

`output_dir` holds `optimal.nc`, the solved optimum, and one `slack_<ε>/` directory per slack
level with a NetCDF file per solve: `mga_furthest_result_*.nc` for the initial box directions
and `mga_closest_result_*.nc` for every nearest-point solve. Read them back with
{py:func}`~mga_surveyor.pypsa_ext.consolidate_mga_results`, which returns one table of points
and one of duals, with one row per file.
