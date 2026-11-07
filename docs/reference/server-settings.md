# Server settings

The MGA Compass server (`app/backend`) is configured through environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `DATASET_URL` | none | URL of a published [dataset bundle](dataset-bundle.md), e.g. `https://github.com/EINS-TUDa/MGA-Compass/releases/download/<tag>`. If set, the bundle is downloaded into `DATA_DIR` at startup. |
| `DATA_DIR` | `data` | Directory holding the bundle, relative to the working directory. Without `DATASET_URL`, a bundle must already be there. |
| `ALLOWED_ORIGINS` | `["*"]` | Origins allowed to call the API, as a JSON list. |
| `SOLVER_NAME` | `highs` | LP solver for navigation and interpolation. |
| `SOLVER_FEASIBILITY_TOLERANCE` | `1e-7` | Primal feasibility tolerance passed to the solver. |
| `SOLVER_LOG_PATH` | `solver.log` | Where solver logs are written. |
| `DEBUG` | `false` | Debug mode. |

The server refuses to start if any file of the bundle is missing or fails its checksum.

## API

All endpoints are rate-limited per client address.

| Endpoint | Method | Limit | Does |
|---|---|---|---|
| `/init` | GET | 60/minute | Each dimension's range, description and unit, and the starting point (the cheapest one). |
| `/init_plot` | GET | 60/minute | The available plots and the category labels and colours. |
| `/navigate` | POST | 20/minute and 60/5 minutes | Run [navigation](../compass/navigation.md) and interpolation. |
| `/plot_data` | POST | 30/minute | Plot data along a traverse. |
| `/lower_bound` | POST | 60/minute | [Cost lower bound](../compass/traverse.md#a-lower-bound-on-the-cost) along a traverse. |
