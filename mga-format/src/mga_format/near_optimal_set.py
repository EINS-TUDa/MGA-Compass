"""The near-optimal set format, version 1: reading, validating and writing.

A near-optimal set is a directory of Parquet files and a ``manifest.json``; the
format is specified in ``docs/reference/near-optimal-set-format.md``. This module
holds the manifest's models, reads a set while checking it against its manifest,
and writes one.

In memory a set is a :class:`NearOptimalSet`:

- ``points``: indexed by ``point``; columns ``slack``, ``kind``, optionally
  ``key``, one per PoI, and the cost column.
- ``duals``: indexed by ``point``; one column per PoI.
- ``sets``: {name: DataFrame}, each with the members in a column named after the
  set, plus attribute columns.

Outputs are not loaded with the set: they can be large, so :func:`read_output`
reads one output, optionally for a few points only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

FORMAT = "mga-near-optimal-set"
VERSION = 1
MANIFEST_NAME = "manifest.json"
POINTS_FILE = "points.parquet"
DUALS_FILE = "duals.parquet"
#: The built-in index set whose members are the points.
POINT = "point"
POINT_KINDS = ("optimum", "box", "nearest")
#: Column names of points.parquet besides the PoIs and the cost.
_POINT_COLUMNS = (POINT, "slack", "kind", "key")
#: Optional column of points.parquet: the producer's identifier of the solve
#: behind each point (for MGA Surveyor, the hash in its result file's name).
KEY = "key"


class NearOptimalSetError(ValueError):
    """A near-optimal set that does not match the format or its own manifest."""


# ---------------------------------------------------------------- manifest models


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Cost(_Model):
    name: str
    unit: str
    optimum: float


class Poi(_Model):
    name: str
    unit: str
    description: str = ""
    #: Surveyor's ``dimensions`` entry: component class -> attribute -> {component: coefficient}.
    definition: dict[str, dict[str, dict[str, float]]]


class SlackLevel(_Model):
    slack: float = Field(gt=0)
    norm: Literal["l1", "l2", "linf"]
    normalize: bool
    tol: float = Field(gt=0)
    tol_unit: Literal["percent", "model"]
    vertex_method: Literal["bb", "cdd"]
    objective_cuts: bool
    iterations: int = Field(ge=0)
    gap: float = Field(ge=0)
    converged: bool


class Fingerprint(_Model):
    network: str


class IndexSet(_Model):
    description: str = ""


class Output(_Model):
    index: list[str] = Field(min_length=1)
    unit: str = ""
    description: str = ""

    @field_validator("index")
    @classmethod
    def _starts_with_point(cls, index: list[str]) -> list[str]:
        if index[0] != POINT:
            raise ValueError(f"an output's index must start with {POINT!r}, got {index}")
        if len(set(index)) != len(index):
            raise ValueError(f"an output's index must not repeat a set, got {index}")
        return index


class Plot(_Model):
    type: Literal["bar", "timeseries", "stacked_bar", "stacked_timeseries"]
    output: str
    x: str
    stack: str | None = None

    @model_validator(mode="after")
    def _stack_for_stacked(self) -> Plot:
        if self.type.startswith("stacked_") and not self.stack:
            raise ValueError(f"a {self.type!r} plot needs 'stack'")
        return self


class Display(_Model):
    """How the web platform shows a quantity: its values times `factor`, in `unit`."""
    unit: str
    factor: float = Field(gt=0)


class Manifest(_Model):
    format: Literal["mga-near-optimal-set"]
    version: Literal[1]
    #: What the set is of, in a few words, e.g. "PyPSA model_energy example".
    title: str = ""
    #: The model behind the set: what it covers and where it comes from.
    description: str = ""
    created: datetime
    software: dict[str, str] = {}
    cost: Cost
    pois: list[Poi] = Field(min_length=1)
    slack_levels: list[SlackLevel] = Field(min_length=1)
    fingerprint: Fingerprint
    files: dict[str, str] = {}
    sets: dict[str, IndexSet] = {}
    outputs: dict[str, Output] = {}
    plots: dict[str, Plot] = {}
    #: Display units, keyed by a PoI, the cost column or an output.
    display: dict[str, Display] = {}

    @property
    def poi_names(self) -> list[str]:
        return [poi.name for poi in self.pois]

    @property
    def slacks(self) -> list[float]:
        return [level.slack for level in self.slack_levels]

    @field_validator("created")
    @classmethod
    def _aware(cls, created: datetime) -> datetime:
        if created.tzinfo is None:
            raise ValueError("'created' needs a time zone")
        return created

    @model_validator(mode="after")
    def _consistent(self) -> Manifest:
        names = self.poi_names
        if len(set(names)) != len(names):
            raise ValueError(f"PoI names must be unique, got {names}")
        reserved = set(_POINT_COLUMNS) | {self.cost.name}
        if clash := reserved & set(names):
            raise ValueError(f"PoI names must not be {sorted(clash)}: reserved for other columns")
        if self.cost.name in _POINT_COLUMNS:
            raise ValueError(f"the cost column must not be named {self.cost.name!r}")
        if len(set(self.slacks)) != len(self.slacks):
            raise ValueError(f"slack levels must be unique, got {self.slacks}")
        if POINT in self.sets:
            raise ValueError(f"{POINT!r} is a built-in set and must not be declared")
        for name, output in self.outputs.items():
            unknown = [s for s in output.index[1:] if s not in self.sets]
            if unknown:
                raise ValueError(f"output {name!r} uses undeclared sets {unknown}")
        displayable = {*names, self.cost.name, *self.outputs}
        if unknown := sorted(set(self.display) - displayable):
            raise ValueError(f"display entries {unknown} are not a PoI, the cost or an output")
        for name, plot in self.plots.items():
            if plot.output not in self.outputs:
                raise ValueError(f"plot {name!r} refers to unknown output {plot.output!r}")
            index = self.outputs[plot.output].index
            for role, used in (("x", plot.x), ("stack", plot.stack)):
                if used is not None and used not in index:
                    raise ValueError(f"plot {name!r}: {role} {used!r} is not a set of output {plot.output!r}")
        return self


# ---------------------------------------------------------------- the set in memory


@dataclass
class NearOptimalSet:
    manifest: Manifest
    points: pd.DataFrame
    duals: pd.DataFrame
    sets: dict[str, pd.DataFrame] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise :class:`NearOptimalSetError` unless the tables match the manifest."""
        _validate_points(self.points, self.manifest)
        _validate_duals(self.duals, self.points, self.manifest)
        if set(self.sets) != set(self.manifest.sets):
            raise NearOptimalSetError(
                f"sets {sorted(self.sets)} do not match the manifest's {sorted(self.manifest.sets)}"
            )
        for name, members in self.sets.items():
            _validate_set(name, members)


def _require_columns(table: str, df: pd.DataFrame, expected: list[str]) -> None:
    if list(df.columns) != expected:
        raise NearOptimalSetError(f"{table}: columns must be {expected}, got {list(df.columns)}")


def _require_numeric(table: str, df: pd.DataFrame, columns: list[str], allow_nan: bool) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]):
            raise NearOptimalSetError(f"{table}: column {column!r} must be numeric, got {df[column].dtype}")
        if not allow_nan and df[column].isna().any():
            raise NearOptimalSetError(f"{table}: column {column!r} has missing values")


def _validate_points(points: pd.DataFrame, manifest: Manifest) -> None:
    if points.index.name != POINT:
        raise NearOptimalSetError(f"points: must be indexed by {POINT!r}, got {points.index.name!r}")
    if not pd.api.types.is_integer_dtype(points.index):
        raise NearOptimalSetError(f"points: {POINT!r} must be integers, got {points.index.dtype}")
    if sorted(points.index) != list(range(len(points))):
        raise NearOptimalSetError(f"points: {POINT!r} must be unique and run from 0 to {len(points) - 1}")
    key = [KEY] if KEY in points.columns else []
    _require_columns("points", points, ["slack", "kind", *key, *manifest.poi_names, manifest.cost.name])
    _require_numeric("points", points, ["slack", *manifest.poi_names, manifest.cost.name], allow_nan=False)
    if key and not points[KEY].dropna().map(lambda v: isinstance(v, str)).all():
        raise NearOptimalSetError(f"points: {KEY!r} must be strings or missing")
    bad_kind = ~points["kind"].isin(POINT_KINDS)
    if bad_kind.any():
        raise NearOptimalSetError(
            f"points: 'kind' must be one of {POINT_KINDS}, got {sorted(set(points.loc[bad_kind, 'kind']))}"
        )
    optimum = points["kind"] == "optimum"
    if (points.loc[optimum, "slack"] != 0).any() or (points.loc[~optimum, "slack"] == 0).any():
        raise NearOptimalSetError("points: 'slack' must be 0 exactly for the optimum")
    unknown = set(points.loc[~optimum, "slack"]) - set(manifest.slacks)
    if unknown:
        raise NearOptimalSetError(f"points: slack {sorted(unknown)} is not in the manifest's slack levels")


def _validate_duals(duals: pd.DataFrame, points: pd.DataFrame, manifest: Manifest) -> None:
    if duals.index.name != POINT:
        raise NearOptimalSetError(f"duals: must be indexed by {POINT!r}, got {duals.index.name!r}")
    if sorted(duals.index) != sorted(points.index):
        raise NearOptimalSetError(f"duals: {POINT!r} must be the same as in points")
    _require_columns("duals", duals, manifest.poi_names)
    _require_numeric("duals", duals, manifest.poi_names, allow_nan=True)
    missing = duals.index[duals.isna().any(axis=1)].tolist()
    if missing:
        raise NearOptimalSetError(
            f"duals: not available for points {missing[:10]}{' ...' if len(missing) > 10 else ''}; "
            "every point needs all its duals"
        )


def _validate_set(name: str, members: pd.DataFrame) -> None:
    if len(members.columns) == 0 or members.columns[0] != name:
        raise NearOptimalSetError(f"set {name!r}: its first column must be named {name!r}")
    column = members[name]
    if column.isna().any() or column.duplicated().any():
        raise NearOptimalSetError(f"set {name!r}: members must be unique and not missing")


# ---------------------------------------------------------------- files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_path(name: str) -> str:
    return f"sets/{name}.parquet"


def _output_path(name: str) -> str:
    return f"outputs/{name}.parquet"


def read_manifest(directory: str | Path) -> Manifest:
    """Read and validate ``manifest.json``, refusing other formats and versions."""
    path = Path(directory) / MANIFEST_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise NearOptimalSetError(f"{path}: not a near-optimal set (format {data.get('format')!r})")
    if data.get("version") != VERSION:
        raise NearOptimalSetError(f"{path}: version {data.get('version')!r} is not supported (only {VERSION})")
    try:
        return Manifest.model_validate(data)
    except ValidationError as error:
        raise NearOptimalSetError(f"{path}: invalid manifest\n{error}") from error


def _verify_files(directory: Path, manifest: Manifest) -> None:
    required = [POINTS_FILE, DUALS_FILE]
    required += [_set_path(name) for name in manifest.sets]
    required += [_output_path(name) for name in manifest.outputs]
    unlisted = [name for name in required if name not in manifest.files]
    if unlisted:
        raise NearOptimalSetError(f"{directory}: manifest 'files' does not list {unlisted}")
    bad = [
        name for name, digest in manifest.files.items()
        if not (directory / name).is_file() or sha256_file(directory / name) != digest
    ]
    if bad:
        raise NearOptimalSetError(f"{directory}: missing or corrupt {bad}")


def read(directory: str | Path) -> NearOptimalSet:
    """Read a near-optimal set: verify its checksums, load its points, duals and index
    sets, and check them against the manifest. Outputs are
    read separately, with :func:`read_output`."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    _verify_files(directory, manifest)
    nos = NearOptimalSet(
        manifest=manifest,
        points=pd.read_parquet(directory / POINTS_FILE).set_index(POINT),
        duals=pd.read_parquet(directory / DUALS_FILE).set_index(POINT),
        sets={name: pd.read_parquet(directory / _set_path(name)) for name in manifest.sets},
    )
    nos.validate()
    for name in manifest.outputs:
        _validate_output_schema(directory, manifest, name)
    return nos


def _validate_output_schema(directory: Path, manifest: Manifest, name: str) -> None:
    schema = pq.read_schema(directory / _output_path(name))
    expected = [*manifest.outputs[name].index, "value"]
    if schema.names != expected:
        raise NearOptimalSetError(f"output {name!r}: columns must be {expected}, got {schema.names}")
    if not pa.types.is_floating(schema.field("value").type) and not pa.types.is_integer(schema.field("value").type):
        raise NearOptimalSetError(f"output {name!r}: 'value' must be numeric, got {schema.field('value').type}")


def read_output(directory: str | Path, name: str, points: list[int] | None = None) -> pd.DataFrame:
    """Read output `name` in its long form, for all points or only `points`.

    Each point's rows are one row group, so reading a few points skips the rest of
    the file."""
    path = Path(directory) / _output_path(name)
    filters = [(POINT, "in", list(points))] if points is not None else None
    return pq.read_table(path, filters=filters).to_pandas()


def validate_output(directory: str | Path, nos: NearOptimalSet, name: str) -> None:
    """Fully check output `name`: its columns, that every key is a member of its
    set, and that it is sorted by point with one row group per point. Reads the
    whole output, so :func:`read` does not do it."""
    directory = Path(directory)
    manifest = nos.manifest
    _validate_output_schema(directory, manifest, name)
    df = read_output(directory, name)
    if df["value"].isna().any():
        raise NearOptimalSetError(f"output {name!r}: 'value' has missing values")
    members = {POINT: pd.Index(nos.points.index)}
    members.update({s: pd.Index(nos.sets[s][s]) for s in manifest.outputs[name].index[1:]})
    for set_name in manifest.outputs[name].index:
        unknown = ~df[set_name].isin(members[set_name])
        if unknown.any():
            raise NearOptimalSetError(
                f"output {name!r}: keys {list(df.loc[unknown, set_name].unique()[:5])} are not members of set {set_name!r}"
            )
    if df.duplicated(subset=manifest.outputs[name].index).any():
        raise NearOptimalSetError(f"output {name!r}: a key combination appears more than once")
    if not df[POINT].is_monotonic_increasing:
        raise NearOptimalSetError(f"output {name!r}: rows must be sorted by {POINT!r}")
    metadata = pq.ParquetFile(directory / _output_path(name)).metadata
    column = metadata.schema.names.index(POINT)
    seen = set()
    for group in range(metadata.num_row_groups):
        stats = metadata.row_group(group).column(column).statistics
        if stats is None or not stats.has_min_max or stats.min != stats.max or stats.min in seen:
            raise NearOptimalSetError(f"output {name!r}: each point's rows must form one row group")
        seen.add(stats.min)


def write(
    directory: str | Path,
    nos: NearOptimalSet,
    outputs: dict[str, pd.DataFrame] | None = None,
) -> Manifest:
    """Write `nos` and its `outputs` (long form, as in the format) to the new
    directory `directory`, record every file's checksum in the manifest, and
    return that manifest. The set is validated first; `directory` must not exist."""
    directory = Path(directory)
    outputs = outputs or {}
    nos.validate()
    if set(outputs) != set(nos.manifest.outputs):
        raise NearOptimalSetError(
            f"outputs {sorted(outputs)} do not match the manifest's {sorted(nos.manifest.outputs)}"
        )
    directory.mkdir(parents=True, exist_ok=False)
    nos.points.reset_index().to_parquet(directory / POINTS_FILE, index=False)
    nos.duals.reset_index().to_parquet(directory / DUALS_FILE, index=False)
    if nos.sets:
        (directory / "sets").mkdir()
    for name, members in nos.sets.items():
        members.to_parquet(directory / _set_path(name), index=False)
    if outputs:
        (directory / "outputs").mkdir()
    for name, df in outputs.items():
        _write_output(directory / _output_path(name), df, nos.manifest.outputs[name].index)

    files = sorted(
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    )
    manifest = nos.manifest.model_copy(update={"files": {f: sha256_file(directory / f) for f in files}})
    (directory / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    for name in outputs:
        validate_output(directory, NearOptimalSet(manifest, nos.points, nos.duals, nos.sets), name)
    return manifest


def _write_output(path: Path, df: pd.DataFrame, index: list[str]) -> None:
    """Write one output sorted by point, each point's rows as one row group."""
    expected = [*index, "value"]
    if list(df.columns) != expected:
        raise NearOptimalSetError(f"output {path.stem!r}: columns must be {expected}, got {list(df.columns)}")
    df = df.sort_values(POINT, kind="stable").reset_index(drop=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    starts = df.index[df[POINT].ne(df[POINT].shift())].tolist() + [len(df)]
    with pq.ParquetWriter(path, table.schema, compression="zstd") as writer:
        for start, stop in zip(starts[:-1], starts[1:]):
            writer.write_table(table.slice(start, stop - start), row_group_size=stop - start)
