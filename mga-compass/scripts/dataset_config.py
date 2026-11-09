"""Turn a generate_mga_files run into a near-optimal set for the web platform.

A dataset config (``scripts/datasets/<name>.json``) holds everything about a
dataset that MGA Surveyor does not know: its title and the model's description, the PoIs' names, units and
descriptions, the cost's name and unit, the index sets' labels and colours, the
outputs' units and descriptions, the plots, and the units the platform shows
quantities in. Values are never converted: units are the model's, and
``display`` only changes what the platform shows.

Used by generate_dataset.py and europe/build_dataset_from_results.py:

    config = DatasetConfig.load(path)
    nos, files = collect_near_optimal_set(run_dir, **config.collect_arguments())
    outputs, members = ...  # read each point's outputs from files[point]
    nos, outputs = config.apply(nos, outputs, members)
    write_set(nos, outputs, directory)  # the set plus its zip archive
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

import mga_format
from mga_format import Display, IndexSet, NearOptimalSet, Output, Plot

ARCHIVE_NAME = "near-optimal-set.zip"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Shown(_Model):
    unit: str
    factor: float = Field(gt=0)


class PoiConfig(_Model):
    #: Name the platform uses; defaults to Surveyor's dimension key.
    name: str | None = None
    unit: str
    description: str = ""
    display: Shown | None = None


class CostConfig(_Model):
    name: str
    unit: str
    display: Shown | None = None


class SetConfig(_Model):
    description: str = ""
    labels: dict[str, str] = {}
    colors: dict[str, str] = {}


class OutputConfig(_Model):
    unit: str
    description: str = ""
    display: Shown | None = None


class DatasetConfig(_Model):
    #: The set's title and the model's description, copied to the manifest.
    title: str = ""
    description: str = ""
    cost: CostConfig
    #: Keyed by Surveyor's dimension key.
    pois: dict[str, PoiConfig]
    sets: dict[str, SetConfig] = {}
    outputs: dict[str, OutputConfig] = {}
    plots: dict[str, Plot] = {}

    @classmethod
    def load(cls, path: str | Path) -> DatasetConfig:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def collect_arguments(self) -> dict:
        """Arguments for mga_surveyor.collect_near_optimal_set."""
        return {
            "units": {key: poi.unit for key, poi in self.pois.items()},
            "descriptions": {key: poi.description for key, poi in self.pois.items()},
            "cost_unit": self.cost.unit,
            "cost_name": self.cost.name,
        }

    def apply(
        self,
        nos: NearOptimalSet,
        outputs: dict[str, pd.DataFrame],
        members: dict[str, list],
    ) -> tuple[NearOptimalSet, dict[str, pd.DataFrame]]:
        """Add the title, description, outputs, their index sets, plots and display
        units to `nos`, and rename the PoIs. `outputs` are long tables (``point``, one column per
        set, ``value``); `members` lists each set's members in display order."""
        if missing := sorted(set(outputs) - set(self.outputs)):
            raise ValueError(f"outputs {missing} have no entry in the dataset config")
        renames = {key: poi.name for key, poi in self.pois.items() if poi.name}
        pois = [
            poi.model_copy(update={"name": renames.get(poi.name, poi.name)}) for poi in nos.manifest.pois
        ]
        sets = {
            name: IndexSet(description=self.sets[name].description if name in self.sets else "")
            for name in members
        }
        output_specs = {
            name: Output(index=[c for c in df.columns if c != "value"], unit=self.outputs[name].unit,
                         description=self.outputs[name].description)
            for name, df in outputs.items()
        }
        display = {}
        for key, poi in self.pois.items():
            if poi.display:
                display[poi.name or key] = Display(**poi.display.model_dump())
        if self.cost.display:
            display[self.cost.name] = Display(**self.cost.display.model_dump())
        for name in outputs:
            if self.outputs[name].display:
                display[name] = Display(**self.outputs[name].display.model_dump())
        manifest = nos.manifest.model_copy(update={
            "title": self.title, "description": self.description,
            "pois": pois, "sets": sets, "outputs": output_specs, "plots": dict(self.plots), "display": display,
        })
        manifest = mga_format.Manifest.model_validate(manifest.model_dump())  # re-check the cross-references
        set_tables = {name: self._set_table(name, values) for name, values in members.items()}
        nos = NearOptimalSet(manifest, nos.points.rename(columns=renames), nos.duals.rename(columns=renames), set_tables)
        return nos, outputs

    def _set_table(self, name: str, values: list) -> pd.DataFrame:
        table = pd.DataFrame({name: values})
        config = self.sets.get(name)
        if config and config.labels:
            table["label"] = [config.labels.get(str(v)) for v in values]
        if config and config.colors:
            table["color"] = [config.colors.get(str(v)) for v in values]
        return table


def long_output(da, point_dim: str = "index") -> pd.DataFrame:
    """An xarray output indexed by `point_dim` as the format's long table."""
    df = da.to_dataframe(name="value").reset_index().rename(columns={point_dim: "point"})
    return df[["point", *[d for d in da.dims if d != point_dim], "value"]]


def write_set(nos: NearOptimalSet, outputs: dict[str, pd.DataFrame], directory: Path) -> Path:
    """Write the set to `directory` and pack it next to it as ``near-optimal-set.zip``,
    the file to publish. Returns the archive's path."""
    mga_format.write(directory, nos, outputs)
    return mga_format.pack(directory, directory.parent / ARCHIVE_NAME)
