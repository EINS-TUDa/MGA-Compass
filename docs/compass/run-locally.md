# Run the platform locally

For development, run the backend and the frontend separately. The frontend's development
server forwards every request under `/api` to the backend on `127.0.0.1:8000`.

You need a dataset bundle: either a local directory (see [](build-dataset.md)) or the URL of
a published one, such as
`https://github.com/EINS-TUDa/MGA-Compass/releases/download/dataset-pypsa-eur-34-2026-11-04`.

## Backend

From `app/backend`:

```bash
uv sync --package mgaserver
```

Then start it with a local bundle:

```bash
DATA_DIR=/path/to/bundle uv run uvicorn mgaserver.main:app --reload --port 8000
```

or with a published one, which is downloaded into `data/` on startup:

```bash
DATASET_URL=https://github.com/EINS-TUDa/MGA-Compass/releases/download/dataset-pypsa-eur-34-2026-11-04 \
    uv run uvicorn mgaserver.main:app --reload --port 8000
```

On Windows PowerShell, set the variable first, e.g. `$env:DATA_DIR = "C:\path\to\bundle"`.

The server refuses to start if a file in the bundle is missing or its checksum doesn't match.
All settings are listed in [](../reference/server-settings.md).

## Frontend

From `app/frontend`, with [Node.js](https://nodejs.org/) installed:

```bash
npm install
npm run dev
```

Open the address Vite prints, `http://localhost:5173` by default.
