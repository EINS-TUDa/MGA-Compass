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
