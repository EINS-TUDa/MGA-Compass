# Run the platform locally

For development, run the backend and the frontend separately. The frontend's development
server forwards every request under `/api` to the backend on `127.0.0.1:8000`.

You need a [near-optimal set](../reference/near-optimal-set-format.md): either a local
directory (see [](build-dataset.md)) or the URL of a published archive, such as
`https://github.com/EINS-TUDa/MGA-Compass/releases/download/<tag>/near-optimal-set.zip`.

## Backend

From `app/backend`:

```bash
uv sync --package mgaserver
```

Then start it with a local set:

```bash
DATA_DIR=/path/to/set uv run uvicorn mgaserver.main:app --reload --port 8000
```

or with a published one, which is downloaded and unpacked into `data/` on startup:

```bash
DATASET_URL=https://github.com/EINS-TUDa/MGA-Compass/releases/download/<tag>/near-optimal-set.zip \
    uv run uvicorn mgaserver.main:app --reload --port 8000
```

A local archive works too, as a `file://` URL. On Windows PowerShell, set the variable first,
e.g. `$env:DATA_DIR = "C:\path\to\set"`.

The server refuses to start if a file of the set is missing, its checksum doesn't match, or
the set breaks a rule of the format. All settings are listed in
[](../reference/server-settings.md).

## Frontend

From `app/frontend`, with [Node.js](https://nodejs.org/) installed:

```bash
npm install
npm run dev
```

Open the address Vite prints, `http://localhost:5173` by default.
