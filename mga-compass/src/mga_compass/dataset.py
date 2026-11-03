"""Dataset bundles: the data the server reads, kept out of git and out of the image.

A bundle is one directory (or one URL prefix, e.g. a GitHub release) holding:

- ``manifest.json``: the :class:`~mga_compass.manifest.Manifest`, whose ``files``
  maps every other file in the bundle to its sha256.
- ``points.parquet`` (and ``duals.parquet``): one row per MGA point.
- one Parquet file per ``output_datasets`` entry, each an ``xr.DataArray`` stored
  densely: one row per combination of its leading dims (one column each), one
  value column per coordinate of its last dim (e.g. per technology).

Bundles are immutable: a new dataset is a new bundle (a new release tag), never an
overwrite, so a server can be pointed back at an old one to roll back.
"""
import hashlib
import json
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from .manifest import Manifest, load_manifest

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
#: Rows per Parquet row group for datasets: read_dataarray decodes one row group
#: at a time, so this bounds its working memory beyond the array itself.
_ROW_GROUP_SIZE = 65536


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frame(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=True)


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_dataarray(da: xr.DataArray, path: Path, decimals: int | None = None, float32: bool = False) -> None:
    """Store `da` densely, zstd-compressed: one row per combination of its leading
    dims in C order (one column per leading dim), one value column per coordinate
    of its last dim, which must be strings (e.g. technology names).

    Dense, so :func:`read_dataarray` only reshapes the value columns -- no per-value
    index to rebuild, which for a long-form table of millions of values takes many
    times the array's own memory. Lossless by default. `decimals` rounds the values
    and `float32` stores them in single precision (and they load as float32) -- for
    large plot-only data, e.g. MW time series, where 0.1 MW is far below what a plot
    shows; together they shrink such files ~3x.
    """
    *lead, last = da.dims
    columns = [str(c) for c in da[last].values]
    if list(columns) != list(da[last].values):
        raise ValueError(f"last dim {last!r} must have string coordinates (they become column names)")
    values = da.transpose(*lead, last).values.reshape(-1, len(columns))
    if decimals is not None:
        values = values.round(decimals)
    if float32:
        values = values.astype("float32")
    df = pd.DataFrame(values, columns=columns)
    grids = np.meshgrid(*(da[d].values for d in lead), indexing="ij")
    for pos, (dim, grid) in enumerate(zip(lead, grids)):
        df.insert(pos, dim, grid.ravel())
    df.to_parquet(path, index=False, compression="zstd", row_group_size=_ROW_GROUP_SIZE)


def read_dataarray(path: Path, dims: list[str]) -> xr.DataArray:
    """Inverse of :func:`write_dataarray`. Coordinates keep the order they were
    written in, not a sorted one, so e.g. stacked plots keep their technology order;
    values keep their stored dtype."""
    *lead, last = dims
    # One row group at a time, single-threaded, straight into the final array:
    # reading the whole table, reading with threads (one decode buffer each) or
    # going through a DataFrame each costs several extra copies of the values --
    # measured ~4x the array on the largest dataset, versus ~2x this way, which is
    # the difference between fitting a small server instance's memory and not.
    parquet = pq.ParquetFile(path, pre_buffer=False, memory_map=False)
    columns = [c for c in parquet.schema_arrow.names if c not in lead]
    lead_table = parquet.read(columns=lead, use_threads=False) if lead else None
    coords = {dim: pd.unique(lead_table.column(dim).to_pandas()) for dim in lead}
    del lead_table
    shape = tuple(len(c) for c in coords.values())
    n_rows = parquet.metadata.num_rows
    if n_rows != int(np.prod(shape)):
        raise ValueError(f"{path}: {n_rows} rows is not a dense {dict(zip(lead, shape))} grid")
    dtype = np.result_type(*(parquet.schema_arrow.field(c).type.to_pandas_dtype() for c in columns))
    flat = np.empty((n_rows, len(columns)), dtype=dtype)
    row = 0
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        for j, name in enumerate(columns):
            flat[row:row + table.num_rows, j] = table.column(name).to_numpy()
        row += table.num_rows
        del table
    return xr.DataArray(flat.reshape(shape + (len(columns),)), dims=dims, coords={**coords, last: columns})


def finalize_bundle(bundle_dir: Path) -> Manifest:
    """Record the sha256 of every file in `bundle_dir` (except the manifest itself)
    in its manifest's `files`, and check every file the manifest refers to exists."""
    bundle_dir = Path(bundle_dir)
    manifest = load_manifest(bundle_dir / MANIFEST_NAME)
    referenced = {manifest.points, *(str(ds.path) for ds in manifest.output_datasets.values())}
    if manifest.duals:
        referenced.add(manifest.duals)
    missing = sorted(name for name in referenced if not (bundle_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"bundle {bundle_dir} is missing {missing}")
    manifest.files = {
        p.name: sha256_file(p) for p in sorted(bundle_dir.iterdir()) if p.is_file() and p.name != MANIFEST_NAME
    }
    (bundle_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_bundle(bundle_dir: Path, manifest: Manifest) -> None:
    """Raise unless every file listed in `manifest.files` is present with its sha256."""
    bundle_dir = Path(bundle_dir)
    if not manifest.files:
        raise ValueError(f"manifest in {bundle_dir} lists no files -- not a finalized bundle")
    bad = [name for name, digest in manifest.files.items()
           if not (bundle_dir / name).is_file() or sha256_file(bundle_dir / name) != digest]
    if bad:
        raise ValueError(f"bundle {bundle_dir}: missing or corrupt {bad}")


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
        dir=dest.parent, delete=False,
    ) as tmp:
        shutil.copyfileobj(response, tmp)
    Path(tmp.name).replace(dest)


def fetch_bundle(base_url: str, dest_dir: Path) -> Manifest:
    """Download the bundle at `base_url` (`<base_url>/manifest.json` and every file
    it lists) into `dest_dir` and verify it. Files already there with the right
    sha256 are kept, so a restart against the same bundle downloads only the
    manifest."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    base_url = base_url.rstrip("/")
    _download(f"{base_url}/{MANIFEST_NAME}", dest_dir / MANIFEST_NAME)
    manifest = load_manifest(dest_dir / MANIFEST_NAME)
    for name, digest in manifest.files.items():
        path = dest_dir / name
        if path.is_file() and sha256_file(path) == digest:
            continue
        logger.info("downloading %s/%s", base_url, name)
        _download(f"{base_url}/{name}", path)
    verify_bundle(dest_dir, manifest)
    return manifest


def load_bundle(bundle_dir: Path) -> tuple[Manifest, pd.DataFrame, dict[str, xr.DataArray]]:
    """Verify and load a local bundle: its manifest, its points and its output datasets."""
    bundle_dir = Path(bundle_dir)
    manifest = load_manifest(bundle_dir / MANIFEST_NAME)
    verify_bundle(bundle_dir, manifest)
    points = read_frame(bundle_dir / manifest.points)
    output_datasets = {
        name: read_dataarray(bundle_dir / ds.path, ds.dims) for name, ds in manifest.output_datasets.items()
    }
    return manifest, points, output_datasets


def write_manifest(manifest: dict | Manifest, bundle_dir: Path) -> None:
    data = manifest.model_dump(mode="json") if isinstance(manifest, Manifest) else manifest
    (Path(bundle_dir) / MANIFEST_NAME).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
