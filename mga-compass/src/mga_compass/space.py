"""Exploring a near-optimal set from Python, e.g. in a notebook.

:func:`load` reads a set in the near-optimal set format (``mga_format``) and
returns a :class:`Space`: the set's designs in PoI space, ready to navigate.

    import mga_compass as mc

    space = mc.load("model-energy/")
    space            # the PoIs and the cost, with their units and ranges
    space.points     # one row per design
    space.optimum    # the cheapest design
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import pandas as pd

import mga_format

from .near_optimal import display_factor, display_units, navigable_points
from .solvers import default_solver_options

Units = Literal["display", "model"]
_URL_SCHEMES = ("http://", "https://", "file://")


def load(
    source: str | Path,
    directory: str | Path | None = None,
    *,
    units: Units = "display",
    slack: float | None = None,
    solver_name: str = "highs",
    solver_options: dict | None = None,
) -> Space:
    """Load a near-optimal set as a :class:`Space`.

    `source` is the set's directory, its zip archive, or the URL of a published
    archive (``https://`` or ``file://``). An archive is unpacked into
    `directory`, replacing whatever is there; the set's outputs are read from
    that directory later, so it has to stay. The set is checked against its
    manifest's checksums before anything is used.

    See :class:`Space` for `units`, `slack`, `solver_name` and `solver_options`.
    """
    text = str(source)
    if text.startswith(_URL_SCHEMES) or Path(source).suffix == ".zip":
        if directory is None:
            raise ValueError(f"{text} is an archive: pass `directory`, where to unpack it")
        if text.startswith(_URL_SCHEMES):
            nos = mga_format.fetch(text, directory)
        else:
            nos = mga_format.unpack(source, directory)
    else:
        if directory is not None:
            raise ValueError(f"{text} is a set's directory already: `directory` is only for archives")
        directory = source
        nos = mga_format.read(directory)
    return Space(nos, directory, units=units, slack=slack, solver_name=solver_name, solver_options=solver_options)


class Space:
    """The designs of a near-optimal set in PoI space, to navigate.

    Navigation moves within the convex hull of these designs: every mix of
    them is a design too, at most as expensive as the same mix of their costs.

    - `units`: ``"display"`` (default) gives every PoI and the cost in the unit
      the set says to show it in (its manifest's ``display``, e.g. GW instead of
      MW); ``"model"`` keeps the units of the model. Navigation finds the same
      designs either way; values and deltas are just in these units.
    - `slack`: the slack level to explore, one of the set's. By default its
      largest, using all designs. A smaller one keeps the optimum and the designs
      found at slack levels up to it.
    - `solver_name`, `solver_options`: the LP solver navigation uses; the options
      default to :func:`~mga_compass.solvers.default_solver_options`.
    """

    def __init__(
        self,
        nos: mga_format.NearOptimalSet,
        directory: str | Path,
        *,
        units: Units = "display",
        slack: float | None = None,
        solver_name: str = "highs",
        solver_options: dict | None = None,
    ) -> None:
        if units not in ("display", "model"):
            raise ValueError(f"units must be 'display' or 'model', got {units!r}")
        manifest = nos.manifest
        self.nos = nos
        self.directory = Path(directory)
        self.slack = _slack_level(manifest.slacks, slack)
        self.cost: str = manifest.cost.name
        self.pois: list[str] = manifest.poi_names
        self.solver_name = solver_name
        self.solver_options = default_solver_options(solver_name) if solver_options is None else solver_options

        points = navigable_points(nos)[nos.points["slack"] <= self.slack]
        if units == "display":
            points = points * pd.Series({column: display_factor(nos, column) for column in points.columns})
            self.units: dict[str, str] = display_units(nos)
        else:
            self.units = {poi.name: poi.unit for poi in manifest.pois} | {self.cost: manifest.cost.unit}
        self.points: pd.DataFrame = points
        """One row per design (index: its point id), one column per PoI plus the cost."""

    @property
    def optimum(self) -> pd.Series:
        """The cheapest design: its PoI values and its cost."""
        return self.points.loc[self.points[self.cost].idxmin()]

    def summary(self) -> pd.DataFrame:
        """Each PoI and the cost: its unit, the range of the designs, and its value
        at the optimum."""
        return pd.DataFrame({
            "unit": pd.Series(self.units),
            "min": self.points.min(),
            "max": self.points.max(),
            "optimum": self.optimum,
            "description": pd.Series({poi.name: poi.description for poi in self.nos.manifest.pois}),
        }).reindex([*self.pois, self.cost]).fillna({"description": ""})

    def _title(self) -> str:
        levels = ", ".join(f"{s:g}" for s in sorted(self.nos.manifest.slacks))
        return (f"Space: {len(self.points)} near-optimal designs in {len(self.pois)} PoIs, "
                f"slack {self.slack:g} (set's slack levels: {levels})")

    def _formatted_summary(self) -> pd.DataFrame:
        table = self.summary()
        for column in ("min", "max", "optimum"):
            table[column] = table[column].map(_format_number)
        if not table["description"].astype(bool).any():
            table = table.drop(columns="description")
        return table

    def __repr__(self) -> str:
        return f"{self._title()}\n{self._formatted_summary().to_string()}"

    def _repr_html_(self) -> str:
        return f"<p>{self._title()}</p>{self._formatted_summary().to_html()}"


def _slack_level(slacks: list[float], slack: float | None) -> float:
    if slack is None:
        return max(slacks)
    for level in slacks:
        if math.isclose(level, slack):
            return level
    raise ValueError(f"slack {slack:g} is not one of the set's slack levels {sorted(slacks)}")


def _format_number(value: float) -> str:
    """About four significant digits, with thousands separators: 18,092, 288.9, 0.0104."""
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    decimals = max(0, 3 - math.floor(math.log10(abs(value))))
    return f"{value:,.{decimals}f}"
