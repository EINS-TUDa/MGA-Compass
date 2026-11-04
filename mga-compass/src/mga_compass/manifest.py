import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

OutputDimension = Literal["index", "snapshot", "technology", "node"]


class OutputDataset(BaseModel):
    path: Path
    dims: list[OutputDimension]


class Plot(BaseModel):
    type: Literal["bar", "timeseries", "stacked_bar", "stacked_timeseries"]
    dataset: str
    x_dim: str
    categories_dim: str | None = None

    @model_validator(mode="after")
    def validate_stacked_plot_fields(self) -> "Plot":
        if self.type.startswith("stacked_") and not self.categories_dim:
            raise ValueError("categories_dim is required when type starts with 'stacked_'")
        return self


class Manifest(BaseModel):
    obj_label: str
    dimension_info: dict[str, str] = {}
    dimension_unit: dict[str, str] = {}
    # File names inside the dataset bundle (see mga_compass.dataset); output
    # dataset paths are relative to the bundle too.
    points: str = "points.parquet"
    duals: str | None = "duals.parquet"
    output_datasets: dict[str, OutputDataset] = {}
    plots: dict[str, Plot] = {}
    # Display name and color (any CSS color) per plot category, keyed by its
    # coordinate value in the output datasets (e.g. a technology name). Optional:
    # a category without one is shown by its own name, in the chart's default color.
    category_labels: dict[str, str] = {}
    category_colors: dict[str, str] = {}
    # sha256 of every file in the bundle except the manifest itself, written by
    # mga_compass.dataset.finalize_bundle and checked before the server loads it.
    files: dict[str, str] = {}


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
