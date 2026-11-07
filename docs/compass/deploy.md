# Deploy the platform with Docker

`docker-compose.yml` runs the platform as two containers: the backend (FastAPI, port 8000)
and the frontend (nginx serving the built app, port 80, forwarding `/api` to the backend).
Neither image contains data: the backend downloads its near-optimal set at startup.

1. **Choose the dataset.** Set `DATASET_URL` to the archive of a published
   [near-optimal set](../reference/near-optimal-set-format.md); `docker compose` refuses to
   start without it. [](build-dataset.md) shows how to build and publish one;
   `publish_dataset.py` prints the URL:

   ```bash
   export DATASET_URL=https://github.com/EINS-TUDa/MGA-Compass/releases/download/<tag>/near-optimal-set.zip
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

Published sets are never overwritten: each has its own release tag. To serve new data,
publish it under a new tag (see [](build-dataset.md)), point `DATASET_URL` at it and restart
the backend. To roll back, point `DATASET_URL` at the previous tag.

The backend keeps the points of the set in memory and reads its detailed outputs from disk,
only for the points a plot needs, so even large sets fit a small instance.
