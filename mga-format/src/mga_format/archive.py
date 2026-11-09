"""A near-optimal set as one archive, for publishing and downloading.

A set is a directory with subdirectories (``sets/``, ``outputs/``), but places
such as GitHub releases hold only flat files. So a set is published as one zip
archive of its directory: :func:`pack` creates it, :func:`unpack` restores the
directory, and :func:`fetch` downloads and unpacks one. Both check the set with
:func:`~mga_format.near_optimal_set.read` before handing it over.
"""
from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .near_optimal_set import MANIFEST_NAME, NearOptimalSet, NearOptimalSetError, read


def pack(directory: str | Path, archive: str | Path) -> Path:
    """Write the set in `directory` to the zip file `archive`, after checking
    it. Files are stored uncompressed: Parquet files are already compressed."""
    directory, archive = Path(directory), Path(archive)
    read(directory)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(directory).as_posix())
    return archive


def _safe_members(zf: zipfile.ZipFile) -> list[str]:
    names = zf.namelist()
    for name in names:
        parts = PurePosixPath(name).parts
        if name.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
            raise NearOptimalSetError(f"archive member {name!r} points outside the set")
    if MANIFEST_NAME not in names:
        raise NearOptimalSetError(f"archive has no {MANIFEST_NAME}")
    return names


def unpack(archive: str | Path, directory: str | Path) -> NearOptimalSet:
    """Unpack the zip file `archive` into `directory`, replacing whatever is
    there, and return the set read from it. The set is unpacked and checked
    next to `directory` first, so a broken archive leaves `directory` as it was."""
    archive, directory = Path(archive), Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".unpack-", dir=directory.parent))
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging, members=_safe_members(zf))
        read(staging)
        if directory.exists():
            shutil.rmtree(directory)
        staging.rename(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return read(directory)


def fetch(url: str, directory: str | Path, timeout: float = 600) -> NearOptimalSet:
    """Download the archive at `url` and unpack it into `directory`, replacing
    whatever is there, and return the set."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory.parent, suffix=".zip", delete=False) as tmp:
        download = Path(tmp.name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, open(download, "wb") as fh:
            shutil.copyfileobj(response, fh)
        return unpack(download, directory)
    finally:
        download.unlink(missing_ok=True)
