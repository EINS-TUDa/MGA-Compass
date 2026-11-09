# Run without Gurobi

MGA Surveyor runs entirely on the open-source solver HiGHS. Pass `solver_name="highs"`:

```python
generate_mga_files(n, dimensions, "mga-output", norm="linf", solver_name="highs", ...)
```

`solver_name` is used both for the model solves and for the per-vertex projections, so
nothing else needs to change. `gurobipy` is still installed as a dependency, but without a
licence it is never used.

What needs Gurobi, and what to use instead:

| Needs Gurobi | Use instead |
|---|---|
| `norm="l2"` (HiGHS has no quadratic objectives) | `norm="linf"` or `norm="l1"` |
| `BilevelMethod` (ORACLE's MILP) | `BBMethod` or `CddMethod` |
| `mga_surveyor.tests.lp_region`, the test oracle | Nothing; it is only used by the test suite |

HiGHS is slower than Gurobi on large models, so expect each model solve to take longer.
