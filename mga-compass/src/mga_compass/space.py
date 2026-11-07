"""Exploring a near-optimal set from Python, e.g. in a notebook.

:func:`load` reads a set in the near-optimal set format (``mga_format``) and
returns a :class:`Space`: the set's designs in PoI space, ready to navigate::

    import mga_compass as mc

    space = mc.load("model-energy/")
    space            # the PoIs and the cost, with their units and ranges
    space.points     # one row per design
    space.optimum    # the cheapest design

    result = space.navigate(space.optimum, {
        "wind": ("increase", 6),     # by at least 6, in wind's unit
        "TOTEX": ("keep", "3%"),     # within 3% of the start's cost
    })
    result           # start and target side by side, and what had to be adjusted
    result.traverse  # the way from start to target: cost along it, with a lower bound
    space.navigate(result, {...})    # carry on from the target

    space.outputs()                       # the set's detailed outputs
    space.output("generation", result)    # one of them, for the target
"""
from __future__ import annotations

import html
import math
import numbers
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd

import mga_format

from .core import (
    alpha_2_point, describe_changes, interpolate, linear_lower_bounds, navigate, point_2_alpha, upper_envelope,
)
from .near_optimal import display_factor, display_units, mix_output, navigable_points
from .schemes import Alpha, Breakpoint, ConstraintChange, Constraints
from .solvers import default_solver_options

Units = Literal["display", "model"]
Direction = Literal["increase", "decrease", "keep"]
#: How far to move a PoI or the cost: a number in its unit, or a percentage of
#: its value at the start, such as ``"20%"``.
Delta = Union[float, str]
#: What to ask of one PoI or the cost: ``(direction, delta)``, or a direction
#: alone, meaning a delta of 0.
Request = Union[Direction, tuple[Direction, Delta]]

_URL_SCHEMES = ("http://", "https://", "file://")
_DIRECTIONS = {"increase": ">=", "decrease": "<=", "keep": "=="}
_WORDS = {">=": "increase by", "<=": "decrease by", "==": "keep within ±"}


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
        self.unit_system: Units = units
        self.slack = _slack_level(manifest.slacks, slack)
        self.cost: str = manifest.cost.name
        self.pois: list[str] = manifest.poi_names
        self.title: str = manifest.title
        """What the set is of, from its manifest; may be empty."""
        self.description: str = manifest.description
        """The model behind the set, from its manifest; may be empty."""
        self.solver_name = solver_name
        self.solver_options = default_solver_options(solver_name) if solver_options is None else solver_options

        points = navigable_points(nos)[nos.points["slack"] <= self.slack]
        duals = nos.duals.loc[points.index, self.pois]
        if units == "display":
            factors = pd.Series({column: display_factor(nos, column) for column in points.columns})
            points = points * factors
            # A dual is cost per PoI unit: scaled by the cost's factor over the PoI's.
            duals = duals * (factors[self.cost] / factors[self.pois])
            self.units: dict[str, str] = display_units(nos)
            self.optimal_cost = manifest.cost.optimum * factors[self.cost]
        else:
            self.units = {poi.name: poi.unit for poi in manifest.pois} | {self.cost: manifest.cost.unit}
            self.optimal_cost = manifest.cost.optimum
        self.points: pd.DataFrame = points
        """One row per design (index: its point id), one column per PoI plus the cost."""
        self.duals: pd.DataFrame = duals
        """Each design's duals (index: its point id), one column per PoI: a
        subgradient of the cost at the design, in cost units per PoI unit."""

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

    def navigate(self, start: pd.Series | int | Navigation, request: Mapping[str, Request]) -> Navigation:
        """Move from design `start` as `request` asks, to the cheapest design that
        meets it.

        `start` is a design (a row of :attr:`points`, :attr:`optimum`, or any
        mix of designs with all its values), a point id, or the result of an
        earlier navigation, to carry on from its target.

        `request` maps PoIs, and the cost if wanted, to ``(direction, delta)``:

        - ``("increase", d)``: raise it by at least `d`, then as far as possible;
        - ``("decrease", d)``: lower it by at least `d`, then as far as possible;
        - ``("keep", d)``: keep it within `d` of its start value, then as close
          to it as possible.

        `d` is in the unit of :attr:`units`, or a percentage of the start value
        such as ``"20%"``; a direction alone means ``d = 0``. The order of
        `request` is the priority order: every request is met first, then each
        is pushed ("as far", "as close") as well as the ones before it allow,
        so a keep listed before an increase limits it more than one after it.
        When a request cannot be met in full, it is met as far as possible and
        :attr:`Navigation.changes` says how. PoIs left out are free: they move
        however the requests make cheapest.

        The way from start to target is computed too: :attr:`Navigation.traverse`.
        """
        if isinstance(start, Navigation):
            start = start.target
        elif isinstance(start, numbers.Integral) and not isinstance(start, bool):
            start = self.points.loc[start]
        start = self._design(start)
        requested = self._constraints(start, request)
        applied, target = navigate(self.points, requested, self.cost, self.solver_name, self.solver_options)
        target = target[list(self.points.columns)].rename("target")
        return Navigation(self, start, target, requested, applied, self.traverse(start, target))

    def traverse(self, start: pd.Series, target: pd.Series) -> Traverse:
        """The way from design `start` to design `target`: see :class:`Traverse`."""
        start, target = self._design(start), self._design(target)
        breakpoints = interpolate(self.points, start, target, self.cost,
                                  solver_name=self.solver_name, solver_options=self.solver_options)
        return Traverse(self, breakpoints)

    def outputs(self) -> pd.DataFrame:
        """The set's detailed outputs, one row each: the unit (in this space's
        units), the index sets besides ``point``, and the description."""
        manifest = self.nos.manifest
        return pd.DataFrame({
            "unit": [self._output_unit(name) for name in manifest.outputs],
            "index": [", ".join(output.index[1:]) for output in manifest.outputs.values()],
            "description": [output.description for output in manifest.outputs.values()],
        }, index=pd.Index(list(manifest.outputs), name="output"))

    def output(self, name: str, design: Navigation | Alpha | pd.Series | int) -> pd.Series:
        """Detailed output `name` (see :meth:`outputs`) of a design, such as
        hourly generation per technology.

        A design here is a mix of the set's designs, and its output is the same
        mix of their outputs. `design` is a navigation result (its target), the
        weights of a design (e.g. ``result.alpha`` or
        ``result.traverse.alpha_at(0.5)``), a point id, or a design's PoI
        values, whose cheapest mix is found first.

        Returns a Series indexed by the output's sets in their order (unstack it
        for a table), in this space's units. Reads only the outputs of the
        designs in the mix."""
        outputs = self.nos.manifest.outputs
        if name not in outputs:
            raise ValueError(f"the set has no output {name!r}; it has {list(outputs)}")
        alpha = self._alpha(design)
        (mixed,) = mix_output(self.directory, self.nos, name, [alpha])
        factor = display_factor(self.nos, name) if self.unit_system == "display" else 1.0
        series = (mixed * factor).to_series().rename(name)
        series.attrs["unit"] = self._output_unit(name)
        return series

    def _output_unit(self, name: str) -> str:
        display = self.nos.manifest.display.get(name)
        if self.unit_system == "display" and display is not None:
            return display.unit
        return self.nos.manifest.outputs[name].unit

    def _alpha(self, design: Navigation | Alpha | pd.Series | int) -> Alpha:
        if isinstance(design, Navigation):
            return design.alpha
        if isinstance(design, Alpha):
            return design
        if isinstance(design, numbers.Integral) and not isinstance(design, bool):
            if design not in self.points.index:
                raise ValueError(f"point {design} is not a design of this space")
            return Alpha(pd.Series({design: 1.0}))
        if isinstance(design, pd.Series):
            return point_2_alpha(self._design(design), self.points, self.cost,
                                 solver_name=self.solver_name, solver_options=self.solver_options)
        raise TypeError(f"a design is a navigation result, weights, a point id or PoI values, got {type(design).__name__}")

    def _design(self, design: pd.Series) -> pd.Series:
        columns = list(self.points.columns)
        if missing := [c for c in columns if c not in design.index]:
            raise ValueError(f"the design has no value for {missing}")
        return design[columns].astype(float).rename("start")

    def _constraints(self, start: pd.Series, request: Mapping[str, Request]) -> Constraints:
        columns = list(self.points.columns)
        rows: dict[str, tuple[str, float]] = {}
        for name, spec in request.items():
            if name not in columns:
                raise ValueError(f"{name!r} is neither a PoI nor the cost; choose from {columns}")
            direction, delta = (spec, 0.0) if isinstance(spec, str) else _pair(name, spec)
            if direction not in _DIRECTIONS:
                raise ValueError(f"{name}: direction must be one of {list(_DIRECTIONS)}, got {direction!r}")
            rows[name] = (_DIRECTIONS[direction], _delta(name, delta, start[name]))
        for name in columns:
            rows.setdefault(name, ("", 0.0))  # free
        names = list(rows)
        return Constraints(pd.DataFrame({
            "direction": [rows[n][0] for n in names],
            "value": start[names].to_numpy(),
            "delta": [rows[n][1] for n in names],
        }, index=names))

    def _title(self) -> str:
        levels = ", ".join(f"{s:g}" for s in sorted(self.nos.manifest.slacks))
        return (f"Space: {len(self.points)} near-optimal designs in {len(self.pois)} PoIs, "
                f"slack {self.slack:g} (set's slack levels: {levels})")

    def _formatted_summary(self) -> pd.DataFrame:
        table = self.summary()
        scales = self.points.abs().max()
        for column in ("min", "max", "optimum"):
            table[column] = [_format_number(v, scales[n]) for n, v in table[column].items()]
        if not table["description"].astype(bool).any():
            return table.drop(columns="description")
        width = table["description"].str.len().max()
        table["description"] = table["description"].str.ljust(width)  # text reads left-aligned
        return table

    def __repr__(self) -> str:
        lines = [self.title] if self.title else []
        lines += textwrap.wrap(self.description, 90)
        if lines:
            lines.append("")
        return "\n".join([*lines, self._title(), self._formatted_summary().to_string()])

    def _repr_html_(self) -> str:
        about = f"<h4>{html.escape(self.title)}</h4>" if self.title else ""
        about += f"<p>{html.escape(self.description)}</p>" if self.description else ""
        return f"{about}<p>{self._title()}</p>{self._formatted_summary().to_html()}"


@dataclass
class Navigation:
    """The result of :meth:`Space.navigate`."""

    space: Space
    #: The design navigation started from.
    start: pd.Series
    #: The design it reached: the cheapest one meeting the requests.
    target: pd.Series
    #: The requests, as constraints: one row per PoI and the cost, in priority
    #: order, with direction ``">="``, ``"<="``, ``"=="`` or ``""`` (free).
    requested: Constraints
    #: The constraints as navigation could meet them.
    applied: Constraints
    #: The way from start to target.
    traverse: Traverse
    #: How requests that could not be met in full were adjusted, in words.
    changes: list[str] = field(init=False)

    @property
    def alpha(self) -> Alpha:
        """The target as weights of the space's designs (point id -> weight)."""
        return self.traverse.breakpoints[-1].alpha

    def __post_init__(self) -> None:
        self.changes = [
            _describe(change, self.requested, self.space.units)
            for change in describe_changes(self.requested, self.applied)
            if not _negligible(change, self.space.points)
        ]

    def to_frame(self) -> pd.DataFrame:
        """Each PoI and the cost: its unit, start and target values, the change,
        and what was requested of it."""
        columns = list(self.space.points.columns)
        return pd.DataFrame({
            "unit": pd.Series(self.space.units),
            "start": self.start,
            "target": self.target,
            "change": self.target - self.start,
            "request": pd.Series({name: _request_text(self.requested.loc[name], self.space.units.get(name, ""))
                                  for name in self.requested.index}),
        }).reindex(columns)

    def _formatted(self) -> pd.DataFrame:
        table = self.to_frame()
        scales = self.space.points.abs().max()
        for column in ("start", "target"):
            table[column] = [_format_number(v, scales[n]) for n, v in table[column].items()]
        table["change"] = [_format_signed(v, scales[n]) for n, v in table["change"].items()]
        return table

    def _title(self) -> str:
        asked = sum(self.requested["direction"] != "")
        return f"Navigation: {asked} request{'s' * (asked != 1)}, {len(self.changes)} adjusted"

    def __repr__(self) -> str:
        lines = [self._title(), self._formatted().to_string()]
        if self.changes:
            lines += ["Adjusted:", *(f"  {change}" for change in self.changes)]
        lines.append(self.traverse._title())
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        html = f"<p>{self._title()}</p>{self._formatted().to_html()}"
        if self.changes:
            html += "<p>Adjusted:</p><ul>" + "".join(f"<li>{change}</li>" for change in self.changes) + "</ul>"
        return html + f"<p>{self.traverse._title()}</p>"


@dataclass
class Traverse:
    """The way from one design to another, computed by :meth:`Space.traverse`.

    The PoIs move in a straight line from start to target as `beta` goes from 0
    to 1. The cost along the way is the cheapest mix of the space's designs with
    those PoIs, refined at breakpoints until it improves by less than 0.1%: an
    upper bound on the true cost. :attr:`lower_bound` gives a matching lower
    bound from the designs' duals; the true cost lies between the two.
    """

    space: Space
    #: Where the cost curve bends, from ``beta = 0`` (the start) to ``beta = 1``
    #: (the target), each with its weights and its design.
    breakpoints: list[Breakpoint]

    @property
    def betas(self) -> list[float]:
        return [float(bp.beta) for bp in self.breakpoints]

    def to_frame(self) -> pd.DataFrame:
        """One row per breakpoint (index: beta): the PoIs, the cost (upper bound)
        and the cost's lower bound there."""
        frame = pd.DataFrame([bp.point for bp in self.breakpoints], index=pd.Index(self.betas, name="beta"))
        frame = frame[list(self.space.points.columns)]
        frame["lower bound"] = [self.lower_bound_at(beta) for beta in self.betas]
        return frame

    def at(self, beta: float) -> pd.Series:
        """The design at `beta` (0 to 1): its PoIs and cost, between the
        neighbouring breakpoints."""
        return alpha_2_point(self.alpha_at(beta), self.space.points).rename(beta)

    def alpha_at(self, beta: float) -> Alpha:
        """The weights of the design at `beta`: the neighbouring breakpoints'
        weights, mixed linearly."""
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be between 0 and 1, got {beta}")
        betas = self.betas
        right = next(i for i, b in enumerate(betas) if b >= beta)
        if betas[right] == beta or right == 0:
            return self.breakpoints[right].alpha
        left = right - 1
        share = (beta - betas[left]) / (betas[right] - betas[left])
        mixed = ((1 - share) * self.breakpoints[left].alpha).add(share * self.breakpoints[right].alpha, fill_value=0)
        return Alpha(mixed / mixed.sum())

    @cached_property
    def lower_bound(self) -> pd.Series:
        """A lower bound on the cost along the way: its value (index: beta) where
        it bends, at 0, at 1 and in between; it is linear in between.

        The largest of the bounds the designs' duals give, cost >= cost_i +
        dual_i . (PoIs - PoIs_i), for each design with a weight at some
        breakpoint, and of the optimal cost: no design is cheaper."""
        bounds = linear_lower_bounds(self.space.points, self.breakpoints, self.space.duals,
                                     self.space.cost, optimum=self.space.optimal_cost)
        envelope = upper_envelope(bounds)
        return pd.Series([p.value for p in envelope], index=pd.Index([p.beta for p in envelope], name="beta"),
                         name="lower bound")

    def lower_bound_at(self, beta: float) -> float:
        """The cost's lower bound at `beta`."""
        return float(np.interp(beta, self.lower_bound.index, self.lower_bound.to_numpy()))

    def plot(self, ax=None):
        """Plot the cost along the way, with its lower bound below it. Needs
        matplotlib (``pip install mga-compass[notebook]``). Returns the axes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("plotting needs matplotlib: pip install 'mga-compass[notebook]'") from None
        if ax is None:
            _, ax = plt.subplots()
        cost, unit = self.space.cost, self.space.units.get(self.space.cost, "")
        upper = [bp.point[cost] for bp in self.breakpoints]
        ax.plot(self.betas, upper, marker="o", label=f"{cost} (cheapest mix of designs)")
        ax.plot(self.lower_bound.index, self.lower_bound.to_numpy(), linestyle="--", label="lower bound")
        ax.set_xlabel("β: share of the way from start to target")
        ax.set_ylabel(f"{cost} [{unit}]" if unit else cost)
        ax.set_xlim(0, 1)
        ax.legend()
        return ax

    def cost_at(self, beta: float) -> float:
        """The cost at `beta`: the upper bound, linear between breakpoints."""
        return float(np.interp(beta, self.betas, [bp.point[self.space.cost] for bp in self.breakpoints]))

    def largest_gap(self) -> float:
        """How far the cost is above its lower bound at most, anywhere on the way.
        Both are linear between their bends, so checking those is enough."""
        betas = sorted({*self.betas, *self.lower_bound.index})
        return max(self.cost_at(beta) - self.lower_bound_at(beta) for beta in betas)

    def _title(self) -> str:
        inner = len(self.breakpoints) - 2
        unit = self.space.units.get(self.space.cost, "")
        scale = self.space.points[self.space.cost].abs().max()
        return (f"Traverse: {inner} breakpoint{'s' * (inner != 1)} between start and target; "
                f"the cost is at most {_with_unit(self.largest_gap(), unit, scale)} above its lower bound")

    def _formatted(self) -> pd.DataFrame:
        table = self.to_frame()
        scales = self.space.points.abs().max()
        for column in table.columns:
            scale = scales.get(column, scales[self.space.cost])
            table[column] = [_format_number(v, scale) for v in table[column]]
        table.index = [f"{beta:.3g}" for beta in table.index]
        table.index.name = "beta"
        return table

    def __repr__(self) -> str:
        return f"{self._title()}\n{self._formatted().to_string()}"

    def _repr_html_(self) -> str:
        return f"<p>{self._title()}</p>{self._formatted().to_html()}"


def _pair(name: str, spec) -> tuple:
    try:
        direction, delta = spec
    except (TypeError, ValueError):
        raise ValueError(f"{name}: a request is (direction, delta) or a direction alone, got {spec!r}") from None
    return direction, delta


def _delta(name: str, delta: Delta, start: float) -> float:
    if isinstance(delta, str):
        text = delta.strip()
        try:
            if not text.endswith("%"):
                raise ValueError
            value = float(text[:-1]) / 100 * abs(start)
        except ValueError:
            raise ValueError(f"{name}: a delta given as text must be a percentage such as '20%', got {delta!r}") from None
    elif isinstance(delta, (int, float)) and not isinstance(delta, bool):
        value = float(delta)
    else:
        raise ValueError(f"{name}: delta must be a number or a percentage such as '20%', got {delta!r}")
    if not value >= 0:
        raise ValueError(f"{name}: delta must not be negative, got {delta!r}")
    return value


def _with_unit(value: float, unit: str, scale: float | None = None) -> str:
    return f"{_format_number(value, scale)} {unit}".rstrip()


def _request_text(row: pd.Series, unit: str) -> str:
    if row["direction"] == "":
        return ""
    if row["delta"] == 0:
        return {">=": "increase", "<=": "decrease", "==": "keep"}[row["direction"]]
    words = _WORDS[row["direction"]]
    return f"{words}{_with_unit(row['delta'], unit)}" if words.endswith("±") else f"{words} {_with_unit(row['delta'], unit)}"


def _describe(change: ConstraintChange, requested: Constraints, units: dict[str, str]) -> str:
    name, unit = change.dim, units.get(change.dim, "")
    value, asked = requested.loc[name, "value"], requested.loc[name, "delta"]
    direction = requested.loc[name, "direction"]
    if change.code == "delta_reduced":
        got = change.values[1]
        verb = "increase" if direction == ">=" else "decrease"
        if _format_number(got) == "0":
            return f"{name}: could not {verb} at all, given the requests before it"
        return f"{name}: could {verb} by {_with_unit(got, unit)} only, not {_with_unit(asked, unit)}"
    if change.code == "infeasible_lb":
        reached = change.values[-1]
        return (f"{name}: could not increase, given the requests before it; "
                f"the most possible is {_with_unit(reached, unit)}, {_with_unit(value - reached, unit)} below the start")
    if change.code == "infeasible_ub":
        reached = change.values[-1]
        return (f"{name}: could not decrease, given the requests before it; "
                f"the least possible is {_with_unit(reached, unit)}, {_with_unit(reached - value, unit)} above the start")
    if change.code == "delta_widened":
        widened = change.values[2]
        return f"{name}: could not be kept within ±{_with_unit(asked, unit)}; kept within ±{_with_unit(widened, unit)}"
    return f"{name}: {change.code}"


def _negligible(change: ConstraintChange, points: pd.DataFrame) -> bool:
    """A widened tolerance of a rounding error only, e.g. ``keep`` with delta 0."""
    if change.code != "delta_widened":
        return False
    asked, out = change.values[1], change.values[2]
    column = points[change.dim]
    return out - asked <= 1e-9 * max(column.abs().max(), 1e-300)


def _slack_level(slacks: list[float], slack: float | None) -> float:
    if slack is None:
        return max(slacks)
    for level in slacks:
        if math.isclose(level, slack):
            return level
    raise ValueError(f"slack {slack:g} is not one of the set's slack levels {sorted(slacks)}")


def _format_number(value: float, scale: float | None = None) -> str:
    """About four significant digits, with thousands separators: 18,092, 288.9,
    0.0104. A value within a rounding error of 0, relative to `scale` (the size
    of the quantity's values), is shown as 0."""
    if scale is not None and abs(value) <= 1e-9 * abs(scale):
        value = 0.0
    if value == 0 or not math.isfinite(value):
        return f"{abs(value):g}" if value == 0 else f"{value:g}"
    decimals = max(0, 3 - math.floor(math.log10(abs(value))))
    return f"{value:,.{decimals}f}"


def _format_signed(value: float, scale: float | None = None) -> str:
    text = _format_number(value, scale)
    return text if text.startswith("-") or text == "0" else f"+{text}"
