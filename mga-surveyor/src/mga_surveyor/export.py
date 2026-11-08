"""Export a `generate_mga_files` run as a near-optimal set (``mga_format``).

`export_near_optimal_set` reads the result files `generate_mga_files` wrote to
its `output_dir` and writes them in the format MGA Compass reads: one point per
result file (the optimum, the box points and the nearest points of every slack
level), with its cost and duals and the cache key in its file name (``key``), plus the PoI definitions, the optimal cost,
each slack level's run summary and the network's fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from mga_format import Cost, Fingerprint, Manifest, NearOptimalSet, Poi, SlackLevel, write

from .pypsa_ext import RUN_SUMMARY_NAME, _read_meta

_PACKAGES = ("mga-surveyor", "mga-format", "pypsa", "linopy")


def network_fingerprint(n: pypsa.Network) -> str:
    """The sha256 of `n`'s input data: its snapshots and their weightings, and
    every component's static attributes and time series that PyPSA marks as
    input. Optimization results are left out, so solving `n` again does not
    change the fingerprint, while changing any cost, limit or time series does."""
    digest = hashlib.sha256()

    def add(label: str, df: pd.DataFrame) -> None:
        digest.update(label.encode())
        digest.update(json.dumps([str(c) for c in df.columns]).encode())
        if not df.empty:
            digest.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())

    add("snapshot_weightings", n.snapshot_weightings.sort_index(axis=1))
    for component in n.components:
        attrs = component.attrs
        inputs = set(attrs.index[attrs["status"].str.startswith("Input")])
        static = component.static
        add(f"{component.name}.static", static[sorted(c for c in static.columns if c in inputs)])
        for attr in sorted(component.dynamic):
            series = component.dynamic[attr]
            if attr in inputs and not series.empty:
                add(f"{component.name}.{attr}", series.sort_index(axis=1))
    return digest.hexdigest()


def _slack_dirs(output_dir: Path) -> list[tuple[float, Path]]:
    dirs = [(float(p.name.removeprefix("slack_")), p) for p in output_dir.glob("slack_*") if p.is_dir()]
    return sorted(dirs)


def _key(path: Path) -> str | None:
    """The cache key in a result file's name, ``mga_*_result_<key>.nc``; None for
    ``optimal.nc``."""
    _, separator, key = path.stem.partition("_result_")
    return key if separator else None


def _row(meta: dict, keys: list[str], coordinates_key: str) -> tuple[list[float], list[float], float]:
    coordinates = [float(meta[coordinates_key][k]) for k in keys]
    duals = [float(meta["duals"][k]) for k in keys] if "duals" in meta else [np.nan] * len(keys)
    return coordinates, duals, float(meta["obj"])


def _software() -> dict[str, str]:
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return versions


def export_near_optimal_set(
    output_dir: str | Path,
    directory: str | Path,
    *,
    units: dict[str, str],
    cost_unit: str,
    cost_name: str = "cost",
    descriptions: dict[str, str] | None = None,
) -> Manifest:
    """Write the results of a `generate_mga_files` run in `output_dir` as a
    near-optimal set in the new directory `directory`, and return its manifest.
    The set has no outputs; to add some, use :func:`collect_near_optimal_set`
    and write the set with ``mga_format.write`` yourself.

    `units` gives every PoI's unit, `cost_unit` the cost's; `descriptions` can
    describe the PoIs. The cost column is named `cost_name`.

    :raises FileNotFoundError: If `output_dir` has no `optimal.nc`, or a slack
        level has no run summary (its run did not finish; resuming
        `generate_mga_files` writes it).
    :raises ValueError: If a PoI has no unit, or a result has no duals
        (it was solved with `resolve_main_objective=False`).
    """
    nos, _ = collect_near_optimal_set(
        output_dir, units=units, cost_unit=cost_unit, cost_name=cost_name, descriptions=descriptions,
    )
    return write(directory, nos)


def collect_near_optimal_set(
    output_dir: str | Path,
    *,
    units: dict[str, str],
    cost_unit: str,
    cost_name: str = "cost",
    descriptions: dict[str, str] | None = None,
) -> tuple[NearOptimalSet, list[Path]]:
    """The results of a `generate_mga_files` run in `output_dir` as a
    near-optimal set in memory, and the result file of each point, in point
    order (``files[p]`` is the solved network behind point ``p``), for reading
    detailed outputs. Arguments and errors as for :func:`export_near_optimal_set`.
    """
    output_dir = Path(output_dir)
    descriptions = descriptions or {}
    optimal_nc = output_dir / "optimal.nc"
    if not optimal_nc.exists():
        msg = f"{optimal_nc} not found -- run generate_mga_files first"
        raise FileNotFoundError(msg)
    optimal = _read_meta(optimal_nc)
    definitions = optimal["dimensions"]
    keys = list(definitions)
    if missing := [k for k in keys if k not in units]:
        msg = f"no unit given for PoIs {missing}"
        raise ValueError(msg)

    rows = [(0.0, "optimum", *_row(optimal, keys, "coordinates"))]
    files = [optimal_nc]
    slack_levels = []
    for slack, slack_dir in _slack_dirs(output_dir):
        summary_path = slack_dir / RUN_SUMMARY_NAME
        if not summary_path.exists():
            msg = f"{summary_path} not found -- the run at slack {slack} did not finish; resume generate_mga_files"
            raise FileNotFoundError(msg)
        slack_levels.append(SlackLevel.model_validate(json.loads(summary_path.read_text(encoding="utf-8"))))
        for path in sorted(slack_dir.glob("mga_furthest_result_*.nc")):
            rows.append((slack, "box", *_row(_read_meta(path), keys, "coordinates")))
            files.append(path)
        for path in sorted(slack_dir.glob("mga_closest_result_*.nc")):
            rows.append((slack, "nearest", *_row(_read_meta(path), keys, "point")))
            files.append(path)

    index = pd.Index(range(len(rows)), name="point")
    points = pd.DataFrame([r[2] for r in rows], columns=keys, index=index)
    points.insert(0, "key", [_key(path) for path in files])
    points.insert(0, "kind", [r[1] for r in rows])
    points.insert(0, "slack", [r[0] for r in rows])
    points[cost_name] = [r[4] for r in rows]
    duals = pd.DataFrame([r[3] for r in rows], columns=keys, index=index)
    if duals.isna().any().any():
        missing_points = duals.index[duals.isna().any(axis=1)].tolist()
        msg = (
            f"no duals for points {missing_points[:10]}: their results were solved with "
            "resolve_main_objective=False, so they lack duals and their cost is not the cheapest "
            "for their PoI values; solve them again with resolve_main_objective=True"
        )
        raise ValueError(msg)

    manifest = Manifest(
        format="mga-near-optimal-set",
        version=1,
        created=datetime.now(timezone.utc),
        software=_software(),
        cost=Cost(name=cost_name, unit=cost_unit, optimum=float(optimal["obj"])),
        pois=[
            Poi(name=k, unit=units[k], description=descriptions.get(k, ""), definition=definitions[k])
            for k in keys
        ],
        slack_levels=slack_levels,
        fingerprint=Fingerprint(network=network_fingerprint(pypsa.Network(optimal_nc))),
    )
    nos = NearOptimalSet(manifest=manifest, points=points, duals=duals)
    nos.validate()
    return nos, files
