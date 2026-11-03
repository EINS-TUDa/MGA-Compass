"""Build a dataset bundle from generate_dataset.py's output and publish it.

The bundle (see mga_compass.dataset) is a directory holding the manifest, with
the sha256 of every file, plus the Parquet files it refers to. With --tag it is
uploaded as a GitHub release of that name, and the server is pointed at it with

    DATASET_URL=https://github.com/<repo>/releases/download/<tag>

A tag is never reused: publishing new data means a new tag, and rolling back means
pointing DATASET_URL at an old one.

    uv run python scripts/publish_dataset.py --data-dir scripts/output \\
        --manifest scripts/manifests/model_energy.json --tag dataset-2026-11-03

--from-pickles converts the pickles older generate_dataset.py versions wrote
(points.pkl, duals.pkl, installed_capacity.pkl, generation_over_snapshots.pkl).
"""
import argparse
import pickle
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from mga_compass import finalize_bundle, load_manifest, write_dataarray, write_frame
from mga_compass.dataset import MANIFEST_NAME

DEFAULT_REPO = "EINS-TUDa/MGA-Compass"


def convert_pickles(data_dir: Path, bundle_dir: Path) -> None:
    """Write the Parquet files of a bundle from legacy pickles. Pickles run code
    when loaded: only convert files you generated yourself."""
    for name in ("points", "duals"):
        if (data_dir / f"{name}.pkl").is_file():
            write_frame(pd.read_pickle(data_dir / f"{name}.pkl"), bundle_dir / f"{name}.parquet")
    for name in ("installed_capacity", "generation_over_snapshots"):
        if (data_dir / f"{name}.pkl").is_file():
            da = pickle.loads((data_dir / f"{name}.pkl").read_bytes())
            write_dataarray(da, bundle_dir / f"{name}.parquet")


def build_bundle(data_dir: Path, manifest_path: Path, bundle_dir: Path, from_pickles: bool) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(manifest_path, bundle_dir / MANIFEST_NAME)
    manifest = load_manifest(manifest_path)
    if from_pickles:
        convert_pickles(data_dir, bundle_dir)
    else:
        names = [manifest.points, *(str(ds.path) for ds in manifest.output_datasets.values())]
        if manifest.duals:
            names.append(manifest.duals)
        for name in names:
            shutil.copy(data_dir / name, bundle_dir / name)
    finalize_bundle(bundle_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="generate_dataset.py's output directory.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest template for this dataset.")
    parser.add_argument("--bundle-dir", type=Path, help="Where to build the bundle (default: <data-dir>/bundle).")
    parser.add_argument("--from-pickles", action="store_true", help="Convert legacy .pkl outputs to Parquet.")
    parser.add_argument("--tag", help="Upload the bundle as a GitHub release with this (new) tag.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Repository for the release (default {DEFAULT_REPO}).")
    args = parser.parse_args()

    bundle_dir = args.bundle_dir or args.data_dir / "bundle"
    build_bundle(args.data_dir, args.manifest, bundle_dir, args.from_pickles)
    files = sorted(p for p in bundle_dir.iterdir() if p.is_file())
    print(f"bundle {bundle_dir}: {', '.join(p.name for p in files)}")
    if args.tag:
        subprocess.run(
            ["gh", "release", "create", args.tag, *map(str, files), "--repo", args.repo,
             "--title", args.tag, "--notes", "Dataset bundle for the MGA Compass server (see mga_compass.dataset)."],
            check=True,
        )
        print(f"DATASET_URL=https://github.com/{args.repo}/releases/download/{args.tag}")


if __name__ == "__main__":
    main()
