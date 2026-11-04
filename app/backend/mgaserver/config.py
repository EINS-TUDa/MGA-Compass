from pathlib import Path

from pydantic_settings import BaseSettings

from mga_compass import Manifest, OutputDataset, Plot, load_manifest, default_solver_options
from mga_compass.solvers import FEASIBILITY_TOLERANCE
from mga_compass.dataset import MANIFEST_NAME

__all__ = ["Manifest", "OutputDataset", "Plot", "load_manifest", "Settings", "settings"]


class Settings(BaseSettings):
    app_name: str = "MGA Server"
    debug: bool = False
    # Local dataset bundle (see mga_compass.dataset). When dataset_url is set,
    # the bundle at that URL (e.g. a GitHub release's download prefix,
    # https://github.com/<org>/<repo>/releases/download/<tag>) is downloaded
    # into data_dir at startup; otherwise data_dir must already hold one.
    data_dir: Path = Path("data")
    dataset_url: str | None = None
    solver_log_path: Path = Path("solver.log")
    solver_name: str = "highs"
    solver_feasibility_tolerance: float = FEASIBILITY_TOLERANCE
    allowed_origins: list[str] = ["*"]

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / MANIFEST_NAME

    @property
    def solver_options(self) -> dict:
        return default_solver_options(self.solver_name, self.solver_feasibility_tolerance)

    # model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
