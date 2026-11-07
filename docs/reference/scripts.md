# Dataset scripts

The scripts in `mga-compass/scripts` build [near-optimal sets](near-optimal-set-format.md)
for the web platform and publish them. Run them from the `mga-compass` directory with the
`scripts` extra installed.

## generate_dataset.py

Solves PyPSA's `model_energy` example, explores it with MGA Surveyor at the 2%, 5% and 8%
slack levels (with properties of interest wind, solar, battery and hydrogen), and writes its
near-optimal set with each point's installed capacity and generation as outputs. Uses Gurobi
if a licence is available, HiGHS otherwise.

```bash
uv run python scripts/generate_dataset.py --tol 5 --max-iter 120 --max-parallel 4
```

| Option | Default | Meaning |
|---|---|---|
| `--tol` | required | Convergence tolerance, a percentage of each property's range (a plain distance with `--no-normalize`). |
| `--max-iter` | required | Iteration cap per slack level. |
| `--max-parallel` | required | Worker processes for the model solves. |
| `--output-dir` | `scripts/output` | Where to write the results and the set. |
| `--no-normalize` | normalize | Explore in raw units; `--tol` becomes a plain distance. |
| `--no-objective-cuts` | cuts on | Don't add a cost cut for every point found. |

Writes the set to `<output-dir>/set/` and its archive to `<output-dir>/near-optimal-set.zip`,
next to Surveyor's NetCDF files and a `run.log` of the console output. Names, units,
descriptions, plots and display units come from the [dataset config](#dataset-configs)
`scripts/datasets/model_energy.json`.

## europe/build_dataset_from_results.py

Builds the near-optimal set of a finished PyPSA-Eur run of `europe/run_custom_mga.py`, with
Europe-wide generation per carrier and snapshot and installed power capacity per carrier as
outputs, configured by `scripts/datasets/pypsa_eur_34.json`.

```bash
uv run --extra scripts python scripts/europe/build_dataset_from_results.py \
    --run-dir /path/to/mga_results --network /path/to/base_s_34_elec_.nc \
    --output-dir scripts/output_pypsa_eur
```

| Option | Default | Meaning |
|---|---|---|
| `--run-dir` | required | The run's output directory. |
| `--network` | required | The network the run explored, for its snapshots. |
| `--output-dir` | required | Where to write the set and its archive. |
| `--box-dir` | the run's own | Box-direction results to use instead of the run's own, e.g. ones solved again separately. |
| `--slack` | `0.05` | The slack level to build the set from. |

The run's slack level must have its run summary, `run.json`, which `generate_mga_files`
writes when a slack level finishes. A run without one is refused; see
[](../surveyor/resume.md#the-run-summary).

## publish_dataset.py

Checks a near-optimal set, packs it into its archive, checks that the archive unpacks to a
valid set, and optionally publishes the archive as a GitHub release.

```bash
uv run python scripts/publish_dataset.py --set-dir scripts/output/set --tag dataset-2026-11-06
```

| Option | Default | Meaning |
|---|---|---|
| `--set-dir` | required | The set's directory. Its archive is `near-optimal-set.zip` next to it; it is created if missing. |
| `--tag` | none | Publish the archive as the only asset of a new GitHub release with this tag. Needs the GitHub CLI. |
| `--repo` | `EINS-TUDa/MGA-Compass` | Repository for the release. |

With `--tag`, it prints the `DATASET_URL` to give the server:
`https://github.com/<repo>/releases/download/<tag>/near-optimal-set.zip`.

(dataset-configs)=
## Dataset configs

A dataset config, `scripts/datasets/<name>.json`, holds what MGA Surveyor does not know about a
dataset. The scripts apply it to the set they collect (see `scripts/dataset_config.py`).
Values are never converted: `display` only changes how the platform shows them.

| Field | Meaning |
|---|---|
| `cost` | `name` (the cost column), `unit`, and optional `display`. |
| `pois` | Per property of interest, keyed by its name in Surveyor's `dimensions`: `unit`, `description`, optional `name` (the name the platform shows instead) and optional `display`. |
| `sets` | Per index set: `description`, and optional `labels` and `colors` per member. |
| `outputs` | Per output: `unit`, `description` and optional `display`. |
| `plots` | The plots, as in the format's [presentation part](near-optimal-set-format.md#presentation-part-web-platform-only). |

`display` is `{"unit": …, "factor": …}`, the unit to show the quantity in and the factor to
multiply it by, e.g. `{"unit": "M€", "factor": 1e-6}` to show euros in million euros.
