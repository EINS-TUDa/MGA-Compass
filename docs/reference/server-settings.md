# Server settings

The MGA Compass server (`app/backend`) is configured through environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `DATASET_URL` | none | URL of a [near-optimal set](near-optimal-set-format.md) archive, e.g. `https://github.com/EINS-TUDa/MGA-Compass/releases/download/<tag>/near-optimal-set.zip`, or a `file://` URL. If set, the archive is downloaded and unpacked into `DATA_DIR` at startup, replacing what is there. `docker-compose.yml` requires it. |
| `DATA_DIR` | `data` | Directory holding the set, relative to the working directory. Without `DATASET_URL`, a set must already be there. |
| `ALLOWED_ORIGINS` | `["*"]` | Origins allowed to call the API, as a JSON list. |
| `SOLVER_NAME` | `highs` | LP solver for navigation and interpolation. |
| `SOLVER_FEASIBILITY_TOLERANCE` | `1e-7` | Primal feasibility tolerance passed to the solver. |
| `SOLVER_LOG_PATH` | `solver.log` | Where solver logs are written. |
| `DEBUG` | `false` | Debug mode. |

The server refuses to start if a file of the set is missing or fails its checksum, or if the
set breaks a rule of the format. It navigates over all points of the set, so over its largest
slack level.

The server keeps the points in memory and reads the outputs from `DATA_DIR` when a plot needs
them, only for the points the plot mixes.

## API

All endpoints are rate-limited per client address. Quantities are exchanged in the set's
display units (its manifest's `display`): `/init` reports them, the frontend sends values back
in them, and `/plot_data` returns them.

| Endpoint | Method | Limit | Does |
|---|---|---|---|
| `/init` | GET | 60/minute | Each dimension's range, description and unit, and the starting point (the cheapest one). |
| `/init_plot` | GET | 60/minute | The available plots and the category labels and colours. |
| `/navigate` | POST | 20/minute and 60/5 minutes | Run [navigation](../compass/navigation.md) and interpolation. |
| `/plot_data` | POST | 30/minute | Plot data along a traverse. |
| `/lower_bound` | POST | 60/minute | [Cost lower bound](../compass/traverse.md#a-lower-bound-on-the-cost) along a traverse. |
