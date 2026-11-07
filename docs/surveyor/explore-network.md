# Explore the near-optimal set of your PyPSA network

This guide runs MGA Surveyor on your own PyPSA network and collects the near-optimal points.
It assumes you have [installed MGA Surveyor](install.md).

1. **Solve the network.** `generate_mga_files` starts from a solved network:

   ```python
   import pypsa

   n = pypsa.Network("my_network.nc")
   n.optimize(solver_name="highs", include_objective_constant=True)
   ```

2. **Choose the properties of interest.** Map each name to the component attributes it
   sums; see [](../reference/surveyor-parameters.md#dimensions). Keep the number small:
   the cost per iteration grows with it.

   ```python
   dimensions = {
       "wind": {"Generator": {"p_nom": {"wind": 1}}},
       "solar": {"Generator": {"p_nom": {"solar": 1}}},
   }
   ```

3. **Run the exploration.** Choose the cost slacks, a tolerance and an iteration cap:

   ```python
   from mga_surveyor import generate_mga_files

   if __name__ == "__main__":
       generate_mga_files(
           n,
           dimensions,
           "mga-output",
           norm="linf",
           slack_levels=[0.05],
           solver_name="highs",
           tol=5,          # stop once every PoI is within 5% of its range
           max_iter=100,
           max_parallel=4,
       )
   ```

   The `if __name__ == "__main__":` guard is required: `generate_mga_files` starts worker
   processes, which re-import your script on Windows and macOS.

   A run stops when it reaches `tol` or `max_iter`, whichever comes first, and logs which.
   Start with a coarse tolerance such as `tol=5`, check the result, then tighten it: the
   run resumes from the existing results (see [](resume.md)).

4. **Read the results:**

   ```python
   from mga_surveyor import consolidate_mga_results

   points, duals, files = consolidate_mga_results("mga-output", obj_label="cost")
   ```

   `points` has one row per solve and one column per property of interest, plus the cost
   column named by `obj_label`. `duals` holds the matching dual values, used for the
   [cost lower bound](../compass/traverse.md#a-lower-bound-on-the-cost).

All parameters are listed in [](../reference/surveyor-parameters.md).
