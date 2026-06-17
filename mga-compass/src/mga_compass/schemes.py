from typing import NamedTuple
import pandas as pd
from pandera.typing import DataFrame, Series


class ConstraintsIndexError(ValueError):
    pass


class ConstraintsColumnsError(ValueError):
    pass


class ConstraintsDirectionError(ValueError):
    pass


class ConstraintsDeltaError(ValueError):
    pass


class ConstraintsValueError(ValueError):
    pass


class Constraints(pd.DataFrame):
    _REQUIRED_COLUMNS = {"value", "direction", "delta"}

    @property
    def _constructor(self):
        return pd.DataFrame

    def copy(self, deep: bool = True) -> "Constraints":
        return Constraints(super().copy(deep=deep))

    def changed(self, other: "Constraints") -> pd.Series:
        direction_changed = self["direction"] != other["direction"]
        delta_changed = self["delta"] != other["delta"]
        return direction_changed | delta_changed

    def __repr__(self) -> str:
        col_w = max(len(str(i)) for i in self.index)
        header = f"  {'':>{col_w}}  {'direction':>9}  {'value':>8}  {'delta':>8}"
        rows = [header, "  " + "-" * (col_w + 31)]
        for func in self.index:
            direction = self.loc[func, "direction"]
            display_dir = "✗" if direction == "" else direction
            rows.append(
                f"  {str(func):>{col_w}}  "
                f"{display_dir:>9}  "
                f"{self.loc[func, 'value']:>8.2f}  "
                f"{self.loc[func, 'delta']:>8.2f}"
            )
        return "Constraints(\n" + "\n".join(rows) + "\n)"

    def validate(self, points: pd.DataFrame) -> None:
        expected_index = set(points.columns)
        actual_index = set(self.index)
        if actual_index != expected_index:
            missing = expected_index - actual_index
            extra = actual_index - expected_index
            msg_parts = []
            if missing:
                msg_parts.append(f"missing: {missing}")
            if extra:
                msg_parts.append(f"unexpected: {extra}")
            raise ConstraintsIndexError(
                f"Constraints index must match points columns. {', '.join(msg_parts)}"
            )
        missing_cols = self._REQUIRED_COLUMNS - set(self.columns)
        if missing_cols:
            raise ConstraintsColumnsError(
                f"Constraints is missing required columns: {missing_cols}"
            )

        valid_directions = {"==", ">=", "<=", ""}
        invalid = self[~self["direction"].isin(valid_directions)]
        if not invalid.empty:
            raise ConstraintsDirectionError(
                f"Invalid direction values: {invalid['direction'].to_dict()}"
            )

        negative_delta = self[self["delta"] < 0]
        if not negative_delta.empty:
            raise ConstraintsDeltaError(
                f"Delta must be non-negative: {negative_delta['delta'].to_dict()}"
            )

        for func in self.index:
            col_min = points[func].min()
            col_max = points[func].max()
            val = self.loc[func, "value"]
            if not (col_min <= val <= col_max):
                raise ConstraintsValueError(
                    f"Value for '{func}' is {val}, outside the points range [{col_min}, {col_max}]"
                )


class NegativeAlphaError(ValueError):
    pass


class AlphaSumError(ValueError):
    pass


class Alpha(pd.Series):
    def __init__(self, data=None, *args, **kwargs):
        series = pd.Series(data, *args, **kwargs)
        if (series < 0).any():
            raise NegativeAlphaError(f"Alpha values must be non-negative, got {series[series < 0].to_dict()}")
        if abs(series.sum() - 1.0) >= 1e-9:
            raise AlphaSumError(f"Alpha values must sum to 1, got {series.sum()}")
        series = series[series != 0]
        super().__init__(series)


    @property
    def _constructor(self):
        return pd.Series  # arithmetic ops return plain Series, not Alpha

    def __repr__(self) -> str:
        idx_w = max(len("index"), max(len(str(i)) for i in self.index))
        rows = [f"  {'index':>{idx_w}}   {'weight':<6}", "  " + "-" * (idx_w + 11)]
        for i, v in self.items():
            rows.append(f"  {str(i):>{idx_w}}   {v:<6.3f}")
        return "Alpha(\n" + "\n".join(rows) + "\n)"


Point = Series[float]
Points = DataFrame[float]
Costs = Series[float]

class Breakpoint(NamedTuple):
    beta: float
    alpha: Alpha
    point: Point

    def __repr__(self) -> str:
        alpha_str = ", ".join(f"{k}: {v:.2f}" for k, v in self.alpha.items())
        point_str = ", ".join(f"{k}: {v:.2f}" for k, v in self.point.items())
        return f"Breakpoint(beta={self.beta:.2f}, alpha={{{alpha_str}}}, point={{{point_str}}})"


class LinearBound(NamedTuple):
    slope: float
    intercept: float


class LowerBoundPoint(NamedTuple):
    beta: float
    value: float


class ConstraintChange(NamedTuple):
    code: str
    message: str
    values: list[float]
    dim: str


