# Dataset scripts

The scripts in `mga-compass/scripts` build and publish [dataset bundles](dataset-bundle.md).
Run them from the `mga-compass` directory with the `scripts` extra installed.

## generate_dataset.py

Solves PyPSA's `model_energy` example, explores it with MGA Surveyor at the 2%, 5% and 8%
slack levels (with properties of interest wind, solar, battery and hydrogen), and writes the
data files of a bundle. Uses Gurobi if a licence is available, HiGHS otherwise.

```bash
uv run python scripts/generate_dataset.py --tol 5 --max-iter 120 --max-parallel 4
```

| Option | Default | Meaning |
|---|---|---|
| `--tol` | required | Convergence tolerance, a percentage of each property's range (a plain distance with `--no-normalize`). |
| `--max-iter` | required | Iteration cap per slack level. |
| `--max-parallel` | required | Worker processes for the model solves. |
| `--output-dir` | `scripts/output` | Where to write the results and data files. |
| `--no-normalize` | normalize | Explore in raw units; `--tol` becomes a plain distance. |
| `--no-objective-cuts` | cuts on | Don't add a cost cut for every point found. |

Writes `points.parquet`, `duals.parquet`, `installed_capacity.parquet` and
`generation_over_snapshots.parquet`, plus Surveyor's NetCDF files and a `run.log`.

## publish_dataset.py

Assembles a bundle from data files and a manifest template, and optionally publishes it as a
GitHub release.

```bash
uv run python scripts/publish_dataset.py --data-dir scripts/output \
    --manifest scripts/manifests/model_energy.json --tag dataset-2026-11-06
```

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | required | Directory with the data files. |
| `--manifest` | required | Manifest template for this dataset. |
| `--bundle-dir` | `<data-dir>/bundle` | Where to build the bundle. Must not exist yet. |
| `--tag` | none | Publish the bundle as a GitHub release with this new tag. Needs the GitHub CLI. |
| `--repo` | `EINS-TUDa/MGA-Compass` | Repository for the release. |
| `--from-pickles` | off | Convert `.pkl` files written by older versions of `generate_dataset.py`. Only use on files you generated yourself. |

With `--tag`, it prints the `DATASET_URL` to give the server.
