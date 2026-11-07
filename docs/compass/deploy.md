# Deploy the platform with Docker

`docker-compose.yml` runs the platform as two containers: the backend (FastAPI, port 8000)
and the frontend (nginx serving the built app, port 80, forwarding `/api` to the backend).
Neither image contains data: the backend downloads its dataset bundle at startup.

1. **Choose the dataset.** By default the backend serves the PyPSA-Eur dataset of the live
   demo, `dataset-pypsa-eur-34-2026-11-04`. To serve another published bundle, set
   `DATASET_URL` to it, for example the smaller `model_energy` dataset:

   ```bash
   export DATASET_URL=https://github.com/EINS-TUDa/MGA-Compass/releases/download/dataset-model-energy-2026-11-04
   ```

2. **Allow the frontend's origin.** `docker-compose.yml` sets `ALLOWED_ORIGINS` to
   `["http://localhost"]`. If the platform is served under another address, change it there
   to that address, as a JSON list.

3. **Build and start:**

   ```bash
   docker compose up --build
   ```

   The platform is then at `http://localhost`.

## Updating or rolling back the data

Bundles are never overwritten: each published dataset has its own release tag. To serve new
data, publish it under a new tag (see [](build-dataset.md)), point `DATASET_URL` at it and
restart the backend. To roll back, point `DATASET_URL` at the previous tag.

The backend keeps the whole dataset in memory, so size the backend's machine for the
dataset; the PyPSA-Eur dataset fits a small instance.
