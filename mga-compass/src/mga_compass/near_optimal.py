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
    Reads only the points with a weight in some alpha."""
    index = nos.manifest.outputs[output].index
    used = sorted({int(p) for alpha in alphas for p in alpha.index})
    df = read_output(directory, output, points=used)
    da = df.set_index(index)["value"].to_xarray()
    for set_name in index[1:]:
        da = da.reindex({set_name: nos.sets[set_name][set_name].to_numpy()})
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


def plot_data(directory: str | Path, nos: NearOptimalSet, plot_name: str, alphas: list[Alpha]) -> dict:
    """The data of plot `plot_name` for each of the weights `alphas`:
    ``{"x_dim": [...], "data": [...]}``, one entry of ``data`` per alpha. An entry
    is ``{category: values over x}`` for a plot with ``stack``, else the values
    over x. Sets other than ``x`` and ``stack`` are summed over."""
    plot = nos.manifest.plots[plot_name]
    keep = [plot.x] + ([plot.stack] if plot.stack else [])
    data = []
    for da in mix_output(directory, nos, plot.output, alphas):
        summed = da.sum([d for d in da.dims if d not in keep]) if set(da.dims) - set(keep) else da
        if plot.stack:
            data.append({
                str(member): summed.sel({plot.stack: member}).transpose(plot.x).values.tolist()
                for member in summed[plot.stack].values
            })
        else:
            data.append(summed.transpose(plot.x).values.tolist())
    x_values = nos.sets[plot.x][plot.x].to_numpy()
    return {"x_dim": _axis_values(x_values), "data": data}
