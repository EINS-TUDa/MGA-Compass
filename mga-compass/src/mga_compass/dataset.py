"""Dataset bundles: the data the server reads, kept out of git and out of the image.

A bundle is one directory (or one URL prefix, e.g. a GitHub release) holding:

- ``manifest.json``: the :class:`~mga_compass.manifest.Manifest`, whose ``files``
  maps every other file in the bundle to its sha256.
- ``points.parquet`` (and ``duals.parquet``): one row per MGA point.
- one Parquet file per ``output_datasets`` entry, each an ``xr.DataArray`` stored
  in long form (one column per dim plus ``value``).

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

import pandas as pd
import xarray as xr

from .manifest import Manifest, load_manifest

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
_VALUE_COLUMN = "value"


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


def write_dataarray(da: xr.DataArray, path: Path) -> None:
    """Store `da` in long form: one column per dim, plus `value`."""
    da.rename(_VALUE_COLUMN).to_series().reset_index().to_parquet(path, index=False)


def read_dataarray(path: Path, dims: list[str]) -> xr.DataArray:
    """Inverse of :func:`write_dataarray`. Each dim keeps the coordinate order it
    was written in (first appearance), not a sorted one, so e.g. stacked plots keep
    their technology order."""
    df = pd.read_parquet(path)
    da = df.set_index(list(dims))[_VALUE_COLUMN].to_xarray()
    return da.reindex({dim: pd.unique(df[dim]) for dim in dims})


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
