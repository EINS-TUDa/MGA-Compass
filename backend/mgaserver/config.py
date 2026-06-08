from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings


_SOLVER_SILENCE: dict[str, dict] = {
    "gurobi": {"OutputFlag": 1},
    "highs": {"output_flag": False},
}

_SOLVER_FEASIBILITY_TOLERANCE_KEY: dict[str, str] = {
    "gurobi": "FeasibilityTol",
    "highs": "primal_feasibility_tolerance",
}

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


class Settings(BaseSettings):
    app_name: str = "MGA Server"
    debug: bool = False
    obj_label: str = "TOTEX"
    data_path: Path = Path("data/points.pkl")
    duals_path: Path = Path("data/duals.pkl")
    outer_approximation_path: Path = Path("data/outer_approximation.pkl")
    solver_log_path: Path = Path("solver.log")
    solver_name: str = "highs"
    solver_feasibility_tolerance: float = 1e-4
    output_datasets: dict[str, OutputDataset] = {
        "capacity": OutputDataset(path=Path("data/installed_capacity_xr.pkl"), dims=["index", "technology"]),
        "generation": OutputDataset(path=Path("data/generation_over_snapshots.pkl"), dims=["index", "snapshot", "technology"]),
        # "wind_generation": OutputDataset(path=Path("data/wind_generation.pkl"), dims=["snapshot", "technology"]),
        # "cost": OutputDataset(path=Path("data/cost_summary.pkl"), dims=["technology"]),
        # "price": OutputDataset(path=Path("data/price.pkl"), dims=["snapshot", "node"]),
    }
    
    plots: dict[str, Plot] = {
        "generation": Plot(
            type="stacked_timeseries",
            dataset="generation",
            x_dim="snapshot",
            categories_dim="technology",
        ),
        "capacity": Plot(
            type="bar",
            dataset="capacity",
            x_dim="technology",
        ),
        "total generation": Plot(
            type="timeseries",
            dataset="generation",
            x_dim="snapshot",
        )
    }
    

    @property
    def solver_options(self) -> dict:
        opts = dict(_SOLVER_SILENCE.get(self.solver_name, {}))
        tol_key = _SOLVER_FEASIBILITY_TOLERANCE_KEY.get(self.solver_name)
        if tol_key:
            opts[tol_key] = self.solver_feasibility_tolerance
        return opts

    # model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
