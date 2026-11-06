# Build a dataset for the platform

The web platform reads a **dataset bundle**: near-optimal points, their duals, detailed
outputs for plotting, and a manifest describing them. This guide builds one from a PyPSA
network. The format itself is described in [](../reference/dataset-bundle.md).

You need the `scripts` extra, which adds MGA Surveyor and PyPSA:

```bash
uv sync --package mga-compass --extra scripts
```

## The example dataset

`mga-compass/scripts/generate_dataset.py` builds the dataset of PyPSA's `model_energy`
example. From the `mga-compass` directory:

```bash
uv run python scripts/generate_dataset.py --tol 5 --max-iter 120 --max-parallel 4
```

`--tol`, `--max-iter` and `--max-parallel` are required. The script uses Gurobi if a licence
is available and HiGHS otherwise, explores the 2%, 5% and 8% slack levels, and writes the data
files to `scripts/output/`.

Then assemble the bundle:

```bash
uv run python scripts/publish_dataset.py --data-dir scripts/output \
    --manifest scripts/manifests/model_energy.json
```

This copies the files the manifest lists into `scripts/output/bundle/`, records their
checksums in the manifest, and checks that nothing is missing. To publish the bundle as a
GitHub release, add `--tag dataset-<date>` (needs the [GitHub CLI](https://cli.github.com/)
and write access to the repository); the script prints the `DATASET_URL` to give the server.

## A dataset from your own network

1. **Explore the network** with MGA Surveyor and read the results back; see
   [](../surveyor/explore-network.md).

2. **Write the points and duals** as Parquet:

   ```python
   from mga_compass import write_frame

   points, duals, nc_files = consolidate_mga_results("mga-output", obj_label="cost")
   write_frame(points, out / "points.parquet")
   write_frame(duals, out / "duals.parquet")
   ```

3. **Write the outputs you want to plot**, one `xarray.DataArray` per file, indexed by point
   first (`index`), with string coordinates on its last dimension:

   ```python
   from mga_compass import write_dataarray

   write_dataarray(installed_capacity, out / "installed_capacity.parquet")  # dims: index, technology
   ```

   `generate_dataset.py`'s `extract_generation_and_capacity` shows how to read generation
   and capacities out of the NetCDF files Surveyor wrote, one per point.

4. **Write a manifest** describing the files and the plots; start from
   `scripts/manifests/model_energy.json` and see [](../reference/dataset-bundle.md).

5. **Assemble the bundle** with `publish_dataset.py`, as for the example above.

```{tip}
Keep the columns of `points` at similar magnitudes, for example cost in million EUR and
capacities in GW. Navigation solves small linear programs over these numbers, and columns
that differ by many orders of magnitude can make them numerically infeasible.
```

To try the bundle, [run the platform locally](run-locally.md) with `DATA_DIR` pointing at it.
