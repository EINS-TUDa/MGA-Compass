"""Publish a near-optimal set for the MGA Compass server.

generate_dataset.py (or europe/build_dataset_from_results.py) writes a set to
<output-dir>/set and its archive to <output-dir>/near-optimal-set.zip. This
script checks the set, packs it again if the archive is missing, and with --tag
uploads the archive as the only asset of a new GitHub release. The server is
then pointed at it with

    DATASET_URL=https://github.com/<repo>/releases/download/<tag>/near-optimal-set.zip

A tag is never reused: publishing new data means a new tag, and rolling back
means pointing DATASET_URL at an old one.

    uv run python scripts/publish_dataset.py --set-dir scripts/output/set --tag dataset-2026-11-06
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import mga_format
from dataset_config import ARCHIVE_NAME

DEFAULT_REPO = "EINS-TUDa/MGA-Compass"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-dir", type=Path, required=True, help="The near-optimal set's directory.")
    parser.add_argument("--tag", help="Upload the archive as a GitHub release with this (new) tag.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Repository for the release (default {DEFAULT_REPO}).")
    args = parser.parse_args()

    nos = mga_format.read(args.set_dir)
    archive = args.set_dir.parent / ARCHIVE_NAME
    if not archive.exists():
        mga_format.pack(args.set_dir, archive)
    with tempfile.TemporaryDirectory(dir=args.set_dir.parent) as check:
        mga_format.unpack(archive, Path(check) / "set")  # the archive itself must unpack to a valid set
    print(f"{archive}: {len(nos.points)} points, outputs {sorted(nos.manifest.outputs)}, "
          f"{archive.stat().st_size / 1e6:.1f} MB")
    if args.tag:
        subprocess.run(
            ["gh", "release", "create", args.tag, str(archive), "--repo", args.repo, "--title", args.tag,
             "--notes", "Near-optimal set for the MGA Compass server (format: docs/reference/near-optimal-set-format.md)."],
            check=True,
        )
        print(f"DATASET_URL=https://github.com/{args.repo}/releases/download/{args.tag}/{ARCHIVE_NAME}")


if __name__ == "__main__":
    main()
