# MGA Compass server

FastAPI backend serving one near-optimal set to the frontend.

## Data

The data is not part of this repository or the Docker image. It is a *near-optimal
set* (format: `docs/reference/near-optimal-set-format.md`, package `mga-format`): a
directory with a `manifest.json` that lists every file with its sha256, plus Parquet
files for the points, duals, index sets and outputs. It is published as one zip
archive.

At startup the server reads the set from `DATA_DIR` (default `data/`). If
`DATASET_URL` is set, it first downloads the archive from there and unpacks it into
`DATA_DIR`, and refuses to start if any file is missing or corrupt. Outputs stay on
disk: each plot request reads only the points it uses.

| setting | meaning |
|---|---|
| `DATASET_URL` | the set's archive, e.g. `https://github.com/EINS-TUDa/MGA-Compass/releases/download/<tag>/near-optimal-set.zip` |
| `DATA_DIR` | where the set is (unpacked to), default `data/` |

## Publishing a dataset

On a machine with the network and a solver (not on the server):

```sh
cd mga-compass
uv run --extra scripts python scripts/generate_dataset.py --output-dir scripts/output \
    --tol 5 --max-iter 120 --max-parallel 4
uv run python scripts/publish_dataset.py --set-dir scripts/output/set --tag dataset-<yyyy-mm-dd>
```

The dataset's names, units, plots and display units come from
`scripts/datasets/model_energy.json`. Then set `DATASET_URL` on the server to the
printed URL and restart it. Tags are never reused, so rolling back is pointing
`DATASET_URL` at an earlier tag.
