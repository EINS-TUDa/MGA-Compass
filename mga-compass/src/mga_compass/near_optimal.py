"""Using a near-optimal set (``mga_format``) for navigation and plots.

The format keeps outputs on disk, one row group per point. A navigated or
interpolated point is a mix of only a few sample points (its weights `alpha`),
so plotting it reads just those points' outputs: memory stays small however
large the set is.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from mga_format import NearOptimalSet, read_output

from .schemes import Alpha

POINT = "point"


def navigable_points(nos: NearOptimalSet) -> pd.DataFrame:
    """The points as navigation expects them: indexed by point, one column per
    PoI plus the cost column. All points are used, so the space navigated is the
    set's largest slack level."""
    return nos.points.drop(columns=["slack", "kind"])


def display_factor(nos: NearOptimalSet, name: str) -> float:
    """The factor quantity `name` (a PoI, the cost or an output) is shown with;
    1 when the manifest's ``display`` has no entry for it."""
    entry = nos.manifest.display.get(name)
    return entry.factor if entry is not None else 1.0


def display_units(nos: NearOptimalSet) -> dict[str, str]:
    """The unit each PoI and the cost is shown in: its ``display`` unit if it has
    one, else its own unit."""
    units = {poi.name: poi.unit for poi in nos.manifest.pois} | {nos.manifest.cost.name: nos.manifest.cost.unit}
    return {name: (nos.manifest.display[name].unit if name in nos.manifest.display else unit)
            for name, unit in units.items()}


def display_points(nos: NearOptimalSet) -> pd.DataFrame:
    """:func:`navigable_points` with every column multiplied by its display
    factor. Navigating in these units gives the same weights as in model units."""
    points = navigable_points(nos)
    factors = pd.Series({column: display_factor(nos, column) for column in points.columns})
    return points * factors


def category_styles(nos: NearOptimalSet) -> tuple[dict[str, str], dict[str, str]]:
    """Display labels and colours of set members, from every set's optional
    ``label`` and ``color`` columns, keyed by the member as a string."""
    labels, colors = {}, {}
    for name, members in nos.sets.items():
        for column, target in (("label", labels), ("color", colors)):
            if column in members:
                styled = members[[name, column]].dropna()
                target.update({str(m): str(v) for m, v in zip(styled[name], styled[column])})
    return labels, colors


def mix_output(directory: str | Path, nos: NearOptimalSet, output: str, alphas: list[Alpha]) -> list[xr.DataArray]:
    """Output `output` mixed with each of the weights `alphas`: one DataArray per
    alpha, over the output's sets without ``point``, in the sets' member order.
    Reads only the points with a weight in some alpha.

    A set member with no rows at all in these points does not exist for this
    output (e.g. a storage technology in a generation output) and is left out; a
    combination missing for some points only counts as zero for them."""
    index = nos.manifest.outputs[output].index
    used = sorted({int(p) for alpha in alphas for p in alpha.index})
    df = read_output(directory, output, points=used)
    da = df.set_index(index)["value"].to_xarray()
    for set_name in index[1:]:
        members = nos.sets[set_name][set_name]
        da = da.reindex({set_name: members[members.isin(df[set_name])].to_numpy()})
    da = da.reindex({POINT: used}).fillna(0.0)  # a missing combination contributes nothing
    mixed = []
    for alpha in alphas:
        weights = xr.DataArray(alpha.to_numpy(dtype=float), dims=[POINT], coords={POINT: [int(p) for p in alpha.index]})
        mixed.append(xr.dot(da.sel({POINT: weights[POINT]}), weights, dim=POINT))
    return mixed


def _axis_values(values: np.ndarray) -> list:
    """Axis values as the frontend expects them: timestamps as integer
    nanoseconds, everything else as plain values."""
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.DatetimeIndex(values).as_unit("ns").asi8.tolist()
    return [str(v) if isinstance(v, str) else v for v in values.tolist()]


def plot_data(
    directory: str | Path, nos: NearOptimalSet, plot_name: str, alphas: list[Alpha], display: bool = False,
) -> dict:
    """The data of plot `plot_name` for each of the weights `alphas`:
    ``{"x_dim": [...], "data": [...]}``, one entry of ``data`` per alpha. An entry
    is ``{category: values over x}`` for a plot with ``stack``, else the values
    over x. Sets other than ``x`` and ``stack`` are summed over. With `display`,
    values are multiplied by the output's display factor."""
    plot = nos.manifest.plots[plot_name]
    keep = [plot.x] + ([plot.stack] if plot.stack else [])
    factor = display_factor(nos, plot.output) if display else 1.0
    mixed_outputs = mix_output(directory, nos, plot.output, alphas)
    x_values = mixed_outputs[0][plot.x].to_numpy()  # the members that exist in this output
    data = []
    for da in (factor * mixed for mixed in mixed_outputs):
        summed = da.sum([d for d in da.dims if d not in keep]) if set(da.dims) - set(keep) else da
        if plot.stack:
            data.append({
                str(member): summed.sel({plot.stack: member}).transpose(plot.x).values.tolist()
                for member in summed[plot.stack].values
            })
        else:
            data.append(summed.transpose(plot.x).values.tolist())
    return {"x_dim": _axis_values(x_values), "data": data}
