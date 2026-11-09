# Resume or extend a run

`generate_mga_files` writes every solve to `output_dir` as it goes, and reads those files back
when it starts. To resume an interrupted run, or to continue a finished one with a tighter
tolerance or a higher iteration cap, call it again with the **same network, dimensions and
`output_dir`**:

```python
generate_mga_files(n, dimensions, "mga-output", norm="linf", slack_levels=[0.05],
                   solver_name="highs", tol=2, max_iter=300, max_parallel=4)
```

On restart:

- The optimum and the initial box-direction solves are read back instead of solved again.
- Every earlier nearest-point result is replayed into the approximations, so neither its
  point nor its cuts have to be found again. Only new iterations solve the model.

Things to keep in mind:

- **Keep the network and dimensions unchanged.** The files are assumed to belong to the same
  problem. Results from a different network or different dimensions would describe a
  different set; use a new `output_dir` for those.
- **Adding a slack level** to `slack_levels` is fine: it gets its own `slack_<ε>/`
  directory and starts from scratch, while existing levels resume.
- `max_iter` counts the iterations of the current call, so `max_iter=300` adds up to 300
  iterations to what is already there.

(the-run-summary)=
## The run summary

When a slack level finishes, `generate_mga_files` writes `slack_<ε>/run.json`: the level's
settings and result.

```json
{
  "slack": 0.05, "norm": "linf", "normalize": true,
  "tol": 5.0, "tol_unit": "percent",
  "vertex_method": "bb", "objective_cuts": true,
  "iterations": 31, "gap": 4.45, "converged": true
}
```

`tol` and `gap` are in the unit `tol` was given in, a percentage when normalized.
`iterations` counts all nearest-point solves of the level, including those of earlier calls
that were resumed. A level that did not converge still gets a summary, with
`"converged": false`; a level without one was interrupted.

Exporting the results as a [near-optimal set](../reference/near-optimal-set-format.md) needs a
summary for every slack level: they become the set's `slack_levels`.

**A finished run without summaries**, from a version that did not write them yet, gets them
by resuming it with `max_iter=0` and the run's original settings (`norm`, `normalize`, `tol`,
`sep_tol`, `vertex_method`, `use_objective_cuts`). With `max_iter=0` no new nearest-point
problem is solved: the results are read back, the gap is measured on them, and the summary is
written. Check the log: it should report every box direction as already cached. If it starts
solving box directions instead, the files do not match these settings.
