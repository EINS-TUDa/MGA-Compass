# Near-optimal set format

Version 1. The format in which MGA Surveyor hands a computed near-optimal set to MGA Compass
(the web platform and the Python API).

A near-optimal set is a **directory** of Parquet files and one `manifest.json`:

```text
my-set/
├── manifest.json     # what the files are: PoIs, cost, run settings and results, checksums
├── points.parquet    # one row per point: slack, kind, key, PoI values, cost
├── duals.parquet     # one row per point: duals of the PoIs
├── sets/             # optional: index sets of the outputs, one file each
│   ├── snapshot.parquet
│   └── technology.parquet
└── outputs/          # optional: detailed outputs, one variable per file
    ├── capacity.parquet
    └── generation.parquet
```

Surveyor writes `manifest.json` (except its presentation part), `points.parquet` and
`duals.parquet`. The [index sets and outputs](#index-sets-and-outputs) and
the presentation part of the manifest are added for the web platform.

`points.parquet` and `duals.parquet` are **wide**: one row per point, one column per PoI.
Output files are **long**: one row per value, one column per index set.

All values are in the **model's units**, as recorded in the manifest. Nothing is rescaled.

## points.parquet

One row per near-optimal point, in no particular order.

| Column | Type | Meaning |
|---|---|---|
| `point` | int64 | Identifier, unique, `0 … N-1`. Every other file refers to points by it. |
| `slack` | float64 | The slack $\epsilon$ of the level the point was found at; `0.0` for the optimum. |
| `kind` | string | How it was found: `"optimum"`, `"box"` (maximizing or minimizing one PoI) or `"nearest"` (the nearest point to a trial point). |
| `key` | string, nullable | Optional. The producer's identifier for the solve behind the point, so its full result can be found again. Keys need not be unique. |
| one column per PoI | float64 | The point's PoI values $z$, named and ordered as in the manifest's `pois`. |
| the cost column | float64 | The point's cost $\nu(z)$: the cheapest system cost with the PoIs fixed at $z$. Named by the manifest's `cost.name`. |

MGA Surveyor writes `key` as the hash in a result file's name: a `box` point at slack
$\epsilon$ comes from `slack_<ε>/mga_furthest_result_<key>.nc` and a `nearest` point from
`slack_<ε>/mga_closest_result_<key>.nc`, under the run's output directory. The optimum has no
key; its result is `optimal.nc`. These names are Surveyor's convention, not part of the format.

A point found at slack $\epsilon'$ is near-optimal for every slack $\epsilon \ge \epsilon'$.
To work with the near-optimal set at slack $\epsilon$, use all points with
`slack` $\le \epsilon$.

## duals.parquet

Required. One row per point, with the same `point` identifiers as `points.parquet`. Every
point must have all its duals: a set whose duals are incomplete cannot be written, and is
refused when read.

| Column | Type | Meaning |
|---|---|---|
| `point` | int64 | As in `points.parquet`. |
| one column per PoI | float64 | The dual $\mu_i$ of fixing that PoI: a subgradient of $\nu$ at the point, in cost units per PoI unit. **Every value must be a real number**: no NaN. |

The duals give the cost lower bound $\nu(z) \ge \nu(z^{(k)}) + \mu^{(k)\top}(z - z^{(k)})$.

## manifest.json

The manifest has three parts:

- **Core** (required): written by Surveyor. Describes the points and their duals.
- **Outputs** (optional): `sets` and `outputs`, present when the set has
  [detailed outputs](#index-sets-and-outputs).
- **Presentation** (optional): `plots` and `display`, used only by the web platform.

A complete example with all three parts:

```json
{
  "format": "mga-near-optimal-set",
  "version": 1,
  "title": "PyPSA model_energy example",
  "description": "One electricity node and one hydrogen node over 2019, at 9-hourly resolution.",
  "created": "2026-11-06T12:00:00+00:00",
  "software": {"mga-surveyor": "0.1.0", "pypsa": "1.3.0"},
  "cost": {"name": "TOTEX", "unit": "EUR", "optimum": 3899079890.0},
  "pois": [
    {
      "name": "wind",
      "unit": "MW",
      "description": "Installed wind capacity.",
      "definition": {"Generator": {"p_nom": {"wind": 1}}}
    }
  ],
  "slack_levels": [
    {
      "slack": 0.05,
      "norm": "linf",
      "normalize": true,
      "tol": 5.0,
      "tol_unit": "percent",
      "vertex_method": "bb",
      "objective_cuts": true,
      "iterations": 31,
      "gap": 4.45,
      "converged": true
    }
  ],
  "fingerprint": {"network": "<sha256>"},
  "files": {
    "points.parquet": "<sha256>",
    "duals.parquet": "<sha256>",
    "sets/snapshot.parquet": "<sha256>",
    "sets/technology.parquet": "<sha256>",
    "outputs/generation.parquet": "<sha256>",
    "outputs/capacity.parquet": "<sha256>"
  },

  "sets": {
    "snapshot": {"description": "Time steps of the model."},
    "technology": {"description": "Generation and storage technologies."}
  },
  "outputs": {
    "generation": {"index": ["point", "snapshot", "technology"], "unit": "MW",
                   "description": "Electricity generation per technology."},
    "capacity": {"index": ["point", "technology"], "unit": "MW",
                 "description": "Installed capacity per technology."}
  },

  "plots": {
    "generation": {"type": "stacked_timeseries", "output": "generation",
                   "x": "snapshot", "stack": "technology"},
    "capacity": {"type": "bar", "output": "capacity", "x": "technology"}
  },
  "display": {
    "wind": {"unit": "GW", "factor": 0.001},
    "TOTEX": {"unit": "bn EUR", "factor": 1e-9},
    "generation": {"unit": "GW", "factor": 0.001}
  }
}
```

| Field | Part | Meaning |
|---|---|---|
| `format`, `version` | core | Always `"mga-near-optimal-set"` and the format version. A reader refuses versions it does not know. |
| `title` | optional | What the set is of, in a few words. Empty if not given. |
| `description` | optional | The model behind the set: what it covers and where it comes from. Empty if not given. |
| `created` | core | When the set was exported, ISO 8601 with time zone. |
| `software` | core | Versions of the packages that produced it. |
| `cost.name` | core | Name of the cost column in `points.parquet`. |
| `cost.unit` | core | Unit of the cost. |
| `cost.optimum` | core | The optimal cost $\nu^*$. |
| `pois` | core | The properties of interest, in column order: name, unit, description, and the definition as Surveyor's `dimensions` entry (component class → attribute → {component: coefficient}). |
| `slack_levels` | core | One entry per slack level that was explored: the settings of the run (`norm`, `normalize`, `tol` and its unit, `vertex_method`, `objective_cuts`) and its result: `iterations` (nearest-point solves in total, including earlier runs that were resumed), the final `gap` (in the unit of `tol`) and whether it `converged` (`gap` below `tol`). |
| `fingerprint.network` | core | Identifies the input network, so the set cannot be used with a different one; see [](#network-fingerprint). The PoIs need no fingerprint: their full definitions are in `pois`. |
| `files` | core | sha256 of every other file in the directory, keyed by its path relative to the directory, including those under `sets/` and `outputs/`. |
| `sets` | outputs | The index sets; see [](#in-the-manifest). |
| `outputs` | outputs | The outputs; see [](#in-the-manifest). |
| `plots` | presentation | The plots the web platform shows; see [](#presentation-part-web-platform-only). |
| `display` | presentation | The units the web platform shows quantities in; see [](#presentation-part-web-platform-only). |

A reader ignores the optional parts it does not use: the Python API needs only the core, the
web platform needs all three.

## Index sets and outputs

Detailed outputs, such as the generation of every technology at every time step, are
**variables over index sets**. The sets are defined once and shared by all outputs.

### Index sets

Each index set is one file, `sets/<name>.parquet`:

| Column | Type | Meaning |
|---|---|---|
| `<name>` | string, integer or timestamp | The set's members: unique, in display order. |
| further columns | any | Optional attributes of the members, e.g. `label` (display name) and `color` (any CSS colour). |

For example, `sets/technology.parquet`:

| technology | label | color |
|---|---|---|
| onwind | Onshore Wind | #235ebc |
| solar | Solar | #f9d002 |

`point` is a built-in set whose members are the `point` column of `points.parquet`; it has no
file of its own.

### Outputs

Each output is one variable, stored in `outputs/<name>.parquet` in **long** form:

| Column | Type | Meaning |
|---|---|---|
| one column per index set | the set's type | The keys, named after the sets, in the order the manifest gives. |
| `value` | float64 or float32 | The value. float32 keeps about 7 significant digits in half the space: enough for large outputs such as time series. |

For example, `outputs/generation.parquet`:

| point | snapshot | technology | value |
|---|---|---|---|
| 0 | 2013-01-01 00:00 | onwind | 1234.5 |
| 0 | 2013-01-01 00:00 | solar | 0.0 |

Rules:

- Every output's index starts with `point`, and every other set it uses is declared in the
  manifest's `sets`.
- Every key is a member of its set. A missing combination means the value does not exist (for
  example, a technology that is not built at a node), not zero.
- Rows are sorted by `point`, and each point's rows form one Parquet row group, so that a
  reader can load the outputs of a few points without reading the whole file.

### In the manifest

See the [complete example](#manifestjson) above.

| Field | Meaning |
|---|---|
| `sets` | {name: {`description`}}: the index sets, each with its file `sets/<name>.parquet`. |
| `outputs` | {name: {`index`, `unit`, `description`}}: the outputs, each with its file `outputs/<name>.parquet`. `index` lists its sets in column order. |

### Presentation part (web platform only)

| Field | Meaning |
|---|---|
| `plots` | {name: {`type`, `output`, `x`, `stack`}}: the plots the platform shows. `type` is `"bar"`, `"timeseries"`, `"stacked_bar"` or `"stacked_timeseries"`; `output` names an output; `x` is the set on the x-axis; `stack` is the set stacked as categories, required for the `stacked_` types. Labels and colours come from the sets' `label` and `color` attributes. |
| `display` | {name: {`unit`, `factor`}}: how the platform shows a quantity, keyed by a PoI, the cost column or an output. The platform shows the stored values times `factor` (greater than 0), labelled `unit`. A quantity without an entry is shown as stored, in its own unit. This changes only what is shown: the stored values stay in model units. |

## Network fingerprint

`fingerprint.network` is the sha256 of the network's **input data**:

- the snapshots and their weightings;
- every component's static attributes marked as input in PyPSA;
- every component's time series marked as input in PyPSA.

Optimization results, such as optimized capacities, dispatch and prices, are left out, so
solving the network again does not change the fingerprint, while changing any cost, limit or
time series does. Two sets with the same fingerprint were computed from the same model.

## Archive

To publish a set as a single file, for example as a GitHub release asset, it is packed into a
zip archive, conventionally named `near-optimal-set.zip`, that holds the directory's files at
their relative paths (`manifest.json`, `points.parquet`, `sets/…`, `outputs/…`). Unpacking it
gives the directory back, and the set is checked against the manifest's checksums as when
reading a directory.

Sets are immutable: a new set is a new archive under a new name or release, never an
overwrite of an old one.

## Functions

The `mga-format` package implements the format; see [](format-api.md).

| Function | Does |
|---|---|
| {py:func}`~mga_format.near_optimal_set.write` | Write a set, with its outputs, to a new directory. Refuses a set that breaks a rule above, such as missing duals. |
| {py:func}`~mga_format.near_optimal_set.read` | Read a set's manifest, points, duals and index sets, after checking every file's checksum. Outputs are not loaded. |
| {py:func}`~mga_format.near_optimal_set.read_output` | Read one output, optionally only for some points; reads only those points' row groups. |
| {py:func}`~mga_format.archive.pack`, {py:func}`~mga_format.archive.unpack` | Pack a set directory into an archive, and unpack one. |
| {py:func}`~mga_format.archive.fetch` | Download an archive from a URL and unpack it. |

MGA Surveyor writes a set with {py:func}`~mga_surveyor.export.export_near_optimal_set`, or
collects it in memory with {py:func}`~mga_surveyor.export.collect_near_optimal_set` to add
outputs first.
