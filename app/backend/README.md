# MGA Compass server

FastAPI backend serving one MGA dataset to the frontend.

## Data

The data is not part of this repository or the Docker image. It is a *dataset
bundle* (see `mga_compass.dataset`): a `manifest.json` that lists every file with
its sha256, plus Parquet files for the points, duals and output datasets.

At startup the server reads the bundle from `DATA_DIR` (default `data/`). If
`DATASET_URL` is set, it first downloads the bundle from there, keeping files whose
checksum already matches, and refuses to start if any file is missing or corrupt.

| setting | meaning |
|---|---|
| `DATASET_URL` | bundle location, e.g. `https://github.com/EINS-TUDa/MGA-Compass/releases/download/<tag>` |
| `DATA_DIR` | where the bundle is (downloaded to), default `data/` |

## Publishing a dataset

On a machine with the network and a solver (not on the server):

```sh
cd mga-compass
uv run --extra scripts python scripts/generate_dataset.py --output-dir scripts/output
uv run python scripts/publish_dataset.py --data-dir scripts/output \
    --manifest scripts/manifests/model_energy.json --tag dataset-<yyyy-mm-dd>
```

Then set `DATASET_URL` on the server to the printed URL and restart it. Tags are
never reused, so rolling back is pointing `DATASET_URL` at an earlier tag.
