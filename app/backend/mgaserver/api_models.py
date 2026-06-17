from typing import Literal

from pydantic import BaseModel


class ConstraintRow(BaseModel):
    delta: float
    direction: Literal["==", ">=", "<=", ""]
    value: float


class ConstraintsPayload(BaseModel):
    constraints: dict[str, ConstraintRow]


class BreakpointResult(BaseModel):
    beta: float
    alpha: dict[str, float]
    point: dict[str, float]


class ConstraintChangeItem(BaseModel):
    code: str
    message: str
    values: list[float]
    dim: str


class NavigateResponse(BaseModel):
    constraints: dict[str, ConstraintRow]
    breakpoints: list[BreakpointResult]
    changes: list[ConstraintChangeItem]
    point: dict[str, float]


class DimensionRange(BaseModel):
    min: float
    max: float


class DimensionEntry(BaseModel):
    range: DimensionRange
    info: str = ""
    unit: str = ""


class InitResponse(BaseModel):
    obj_label: str
    dimensions: dict[str, DimensionEntry]
    point: dict[str, float]


class InitPlotResponse(BaseModel):
    datasets: list[str]
    obj_label: str


class BetaAlphaItem(BaseModel):
    beta: float
    alpha: dict[str, float]  # point_index (as str) -> weight


class PlotRequest(BaseModel):
    name: str
    breakpoints: list[BetaAlphaItem]


class PlotResponse(BaseModel):
    type: Literal["bar", "timeseries", "stacked_bar", "stacked_timeseries"]
    betas: list[float]
    x_dim: list
    data: list[dict[str, list[float]] | list[float]]


class LowerBoundRequest(BaseModel):
    breakpoints: list[BreakpointResult]


class LowerBoundPointResult(BaseModel):
    beta: float
    value: float


class LowerBoundResponse(BaseModel):
    points: list[LowerBoundPointResult]
