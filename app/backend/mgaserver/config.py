from pathlib import Path

from pydantic_settings import BaseSettings

from mga_compass import Manifest, OutputDataset, Plot, load_manifest, default_solver_options

__all__ = ["Manifest", "OutputDataset", "Plot", "load_manifest", "Settings", "settings"]


class Settings(BaseSettings):
    app_name: str = "MGA Server"
    debug: bool = False
    data_dir: Path = Path("data")
    solver_log_path: Path = Path("solver.log")
    solver_name: str = "highs"
    solver_feasibility_tolerance: float = 1e-4
    allowed_origins: list[str] = ["*"]

    @property
    def data_path(self) -> Path:
        return self.data_dir / "points.pkl"

    @property
    def duals_path(self) -> Path:
        return self.data_dir / "duals.pkl"

    @property
    def outer_approximation_path(self) -> Path:
        return self.data_dir / "outer_approximation.pkl"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def solver_options(self) -> dict:
        return default_solver_options(self.solver_name, self.solver_feasibility_tolerance)

    # model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
