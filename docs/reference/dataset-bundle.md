# Dataset bundle

A dataset bundle is the data the MGA Compass server serves. It is one directory, or one URL
prefix such as a GitHub release, holding:

| File | Content |
|---|---|
| `manifest.json` | The [manifest](#manifest): what the other files are and how to plot them, with their checksums. |
| `points.parquet` | One row per near-optimal point, one column per property of interest, plus the cost column. |
| `duals.parquet` | Optional. The duals of each point, same rows and columns as `points.parquet`. |
| one Parquet file per output dataset | Detailed outputs per point, such as capacities or generation time series. |

Bundles are immutable: a new dataset is a new bundle, never an overwrite of an old one.

(manifest)=
## Manifest

| Field | Type | Meaning |
|---|---|---|
| `obj_label` | string, required | Name of the cost column in `points.parquet`. |
| `dimension_info` | {name: text} | Description of each column, shown in the platform. |
| `dimension_unit` | {name: unit} | Unit of each column, e.g. `"MW"`. |
| `points` | string | File name of the points. Default `points.parquet`. |
| `duals` | string or `null` | File name of the duals. Default `duals.parquet`; `null` for none. |
| `output_datasets` | {name: [output dataset](#output-datasets)} | Detailed outputs available for plotting. |
| `plots` | {name: [plot](#plots)} | The plots the platform shows. |
| `category_labels` | {category: label} | Optional display name per plot category, e.g. per technology. |
| `category_colors` | {category: CSS colour} | Optional colour per plot category. |
| `files` | {file name: sha256} | Checksum of every other file. Written by `finalize_bundle`; leave empty in a template. |

(output-datasets)=
### Output datasets

| Field | Meaning |
|---|---|
| `path` | File name inside the bundle. |
| `dims` | Dimension names in storage order, from `index`, `snapshot`, `technology`, `node`. The first is `index`, the point. |

Each file stores one `xarray.DataArray` densely, as written by
{py:func}`~mga_compass.dataset.write_dataarray`: one row per combination of its leading
dimensions (one column each), and one value column per coordinate of its last dimension,
whose coordinates must be strings.

(plots)=
### Plots

| Field | Meaning |
|---|---|
| `type` | `"bar"`, `"timeseries"`, `"stacked_bar"` or `"stacked_timeseries"`. |
| `dataset` | Name of an entry in `output_datasets`. |
| `x_dim` | Dimension on the x-axis. |
| `categories_dim` | Dimension stacked as categories. Required for the `stacked_` types. |

## Example

The manifest template of the example dataset, `mga-compass/scripts/manifests/model_energy.json`,
shortened:

```json
{
  "obj_label": "TOTEX",
  "dimension_unit": {"wind": "MW", "solar": "MW", "TOTEX": "€"},
  "output_datasets": {
    "capacity": {"path": "installed_capacity.parquet", "dims": ["index", "technology"]},
    "generation": {"path": "generation_over_snapshots.parquet", "dims": ["index", "snapshot", "technology"]}
  },
  "plots": {
    "capacity": {"type": "bar", "dataset": "capacity", "x_dim": "technology"},
    "generation": {"type": "stacked_timeseries", "dataset": "generation",
                   "x_dim": "snapshot", "categories_dim": "technology"}
  }
}
```

## Functions

| Function | Does |
|---|---|
| {py:func}`~mga_compass.dataset.write_frame`, {py:func}`~mga_compass.dataset.write_dataarray` | Write the points, duals and output datasets. |
| {py:func}`~mga_compass.dataset.finalize_bundle` | Record every file's checksum in the manifest and check that every referenced file exists. |
| {py:func}`~mga_compass.dataset.fetch_bundle` | Download a bundle from a URL, keeping files whose checksum already matches. |
| {py:func}`~mga_compass.dataset.load_bundle` | Verify a local bundle and load it. |
