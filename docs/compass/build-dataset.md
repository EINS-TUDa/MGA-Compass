# Build a dataset for the platform

The web platform serves a **near-optimal set**: the near-optimal points with their costs and
duals, detailed outputs for plotting, and a manifest describing them, including how the
platform plots and labels them. This guide builds one from a PyPSA network. The format itself
is described in [](../reference/near-optimal-set-format.md).

You need the `scripts` extra, which adds MGA Surveyor and PyPSA:

```bash
uv sync --package mga-compass --extra scripts
```

## The example dataset

`mga-compass/scripts/generate_dataset.py` builds the set of PyPSA's `model_energy` example.
From the `mga-compass` directory:

```bash
uv run python scripts/generate_dataset.py --tol 5 --max-iter 120 --max-parallel 4
```

`--tol`, `--max-iter` and `--max-parallel` are required. The script uses Gurobi if a licence
is available and HiGHS otherwise, explores the 2%, 5% and 8% slack levels, and writes the set
to `scripts/output/set/` and its archive to `scripts/output/near-optimal-set.zip`. Names,
units, plots and display units come from the dataset config
`scripts/datasets/model_energy.json`; see [](../reference/scripts.md#dataset-configs).

Then check the set and publish it:

```bash
uv run python scripts/publish_dataset.py --set-dir scripts/output/set --tag dataset-<date>
```

This checks the set, checks that its archive unpacks to the same set, and uploads the archive
as a new GitHub release (needs the [GitHub CLI](https://cli.github.com/) and write access to
the repository). It prints the `DATASET_URL` to give the server. Without `--tag`, it only
checks.

## A dataset from your own network

1. **Explore the network** with MGA Surveyor; see [](../surveyor/explore-network.md).

2. **Collect the results** as a near-optimal set, with the unit of every property of
   interest and of the cost:

   ```python
   from mga_surveyor import collect_near_optimal_set

   nos, files = collect_near_optimal_set(
       "mga-output", units={"wind": "MW", "solar": "MW"}, cost_unit="EUR", cost_name="TOTEX",
   )
   ```

   Every slack level needs its [run summary](../surveyor/resume.md#the-run-summary), and
   every point its duals.

3. **Extract the outputs you want to plot**, as long tables: a `point` column, one column per
   index set, and a `value` column. `files[p]` is the solved network of point `p`. For
   example, installed capacity per technology:

   ```python
   import pandas as pd
   import pypsa

   rows = []
   for point, path in enumerate(files):
       n = pypsa.Network(path)
       capacity = n.generators.p_nom_opt
       rows += [(point, tech, value) for tech, value in capacity.items()]
   capacity = pd.DataFrame(rows, columns=["point", "technology", "value"])
   ```

   `generate_dataset.py`'s `extract_generation_and_capacity` reads generation and capacity
   for all points at once, without loading each network fully.

4. **Add the outputs, their index sets and the plots.** Write a dataset config, starting from
   `scripts/datasets/model_energy.json`, and apply it as the scripts do:

   ```python
   from pathlib import Path

   from dataset_config import DatasetConfig, write_set  # in mga-compass/scripts

   config = DatasetConfig.load("scripts/datasets/my_network.json")
   nos, outputs = config.apply(nos, {"capacity": capacity}, {"technology": list(capacity["technology"].unique())})
   write_set(nos, outputs, Path("my-output/set"))  # also writes my-output/near-optimal-set.zip
   ```

   Or set the manifest's `sets`, `outputs`, `plots` and `display` yourself and write the set
   with {py:func}`mga_format.write <mga_format.near_optimal_set.write>`.

5. **Publish it** with `publish_dataset.py`, as for the example above.

Keep the values in the model's units. To show them in other units, for example cost in
million euros and capacities in GW, give them a `display` entry in the dataset config: the
platform then shows the values times its factor.

To try the set, [run the platform locally](run-locally.md) with `DATA_DIR` pointing at its
directory.
