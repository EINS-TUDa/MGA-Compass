# Near-optimal set format

Version 1. The format in which MGA Surveyor hands a computed near-optimal set to MGA Compass
(the web platform and the Python API).

A near-optimal set is a **directory** of Parquet files and one `manifest.json`:

```text
my-set/
├── manifest.json     # what the files are: PoIs, cost, run settings and results, checksums
├── points.parquet    # one row per point: slack, kind, PoI values, cost
├── duals.parquet     # one row per point: duals of the PoIs
├── outer.parquet     # one row per halfspace of the outer approximation
├── sets/             # optional: index sets of the outputs, one file each
│   ├── snapshot.parquet
│   └── technology.parquet
└── outputs/          # optional: detailed outputs, one variable per file
    ├── capacity.parquet
    └── generation.parquet
```

Surveyor writes `manifest.json` (except its presentation part), `points.parquet`,
`duals.parquet` and `outer.parquet`. The [index sets and outputs](#index-sets-and-outputs) and
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
| one column per PoI | float64 | The point's PoI values $z$, named and ordered as in the manifest's `pois`. |
| the cost column | float64 | The point's cost $\nu(z)$: the cheapest system cost with the PoIs fixed at $z$. Named by the manifest's `cost.name`. |

A point found at slack $\epsilon'$ is near-optimal for every slack $\epsilon \ge \epsilon'$.
To work with the near-optimal set at slack $\epsilon$, use all points with
`slack` $\le \epsilon$.

## duals.parquet

One row per point, with the same `point` identifiers as `points.parquet`.

| Column | Type | Meaning |
|---|---|---|
| `point` | int64 | As in `points.parquet`. |
| one column per PoI | float64 | The dual $\mu_i$ of fixing that PoI: a subgradient of $\nu$ at the point, in cost units per PoI unit. **NaN if not computed**, never 0. |

The duals give the cost lower bound $\nu(z) \ge \nu(z^{(k)}) + \mu^{(k)\top}(z - z^{(k)})$.

## outer.parquet

One row per halfspace $a^\top z \le b$ of the outer approximation, for each slack level.
Every point of the near-optimal set at that slack satisfies all of its halfspaces.

| Column | Type | Meaning |
|---|---|---|
| `slack` | float64 | The slack level the halfspace belongs to. |
| `kind` | string | `"box"` (a bound of one PoI), `"projection"` (a projection cut) or `"cost"` (a cost cut). |
| one column per PoI | float64 | The coefficients $a$. |
| `rhs` | float64 | The right-hand side $b$. |

## manifest.json

The manifest has three parts:

- **Core** (required): written by Surveyor. Describes the points, duals and outer
  approximation.
- **Outputs** (optional): `sets` and `outputs`, present when the set has
  [detailed outputs](#index-sets-and-outputs).
- **Presentation** (optional): `plots`, used only by the web platform.

A complete example with all three parts:

```json
{
  "format": "mga-near-optimal-set",
  "version": 1,
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
  "fingerprint": {"network": "<sha256>", "pois": "<sha256>"},
  "files": {
    "points.parquet": "<sha256>",
    "duals.parquet": "<sha256>",
    "outer.parquet": "<sha256>",
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
  }
}
```

| Field | Part | Meaning |
|---|---|---|
| `format`, `version` | core | Always `"mga-near-optimal-set"` and the format version. A reader refuses versions it does not know. |
| `created` | core | When the set was exported, ISO 8601 with time zone. |
| `software` | core | Versions of the packages that produced it. |
| `cost.name` | core | Name of the cost column in `points.parquet`. |
| `cost.unit` | core | Unit of the cost. |
| `cost.optimum` | core | The optimal cost $\nu^*$. |
| `pois` | core | The properties of interest, in column order: name, unit, description, and the definition as Surveyor's `dimensions` entry (component class → attribute → {component: coefficient}). |
| `slack_levels` | core | One entry per slack level that was explored: the settings of the run (`norm`, `normalize`, `tol` and its unit, `vertex_method`, `objective_cuts`) and its result: `iterations` (nearest-point solves in total, including earlier runs that were resumed), the final `gap` (in the unit of `tol`) and whether it `converged` (`gap` below `tol`). |
| `fingerprint.network` | core | Identifies the input network, so the set cannot be used with a different one. |
| `fingerprint.pois` | core | Identifies the PoI definitions. |
| `files` | core | sha256 of every other file in the directory, keyed by its path relative to the directory, including those under `sets/` and `outputs/`. |
| `sets` | outputs | The index sets; see [](#in-the-manifest). |
| `outputs` | outputs | The outputs; see [](#in-the-manifest). |
| `plots` | presentation | The plots the web platform shows; see [](#presentation-part-web-platform-only). |

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
| `value` | float64 | The value. |

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

## Open questions

- **Fingerprints:** exactly what goes into each hash. The network fingerprint should cover the
  network's input data but not its optimization results, so that solving it again does not
  change it.
