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
    output_datasets: dict[str, OutputDataset] = {}
    plots: dict[str, Plot] = {}


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
