from pathlib import Path

from pydantic_settings import BaseSettings

from mga_compass import default_solver_options
from mga_compass.solvers import FEASIBILITY_TOLERANCE

__all__ = ["Settings", "settings"]


class Settings(BaseSettings):
    app_name: str = "MGA Server"
    debug: bool = False
    # Local near-optimal set (see mga_format). When dataset_url is set, the set's
    # archive at that URL (e.g. a GitHub release asset,
    # https://github.com/<org>/<repo>/releases/download/<tag>/near-optimal-set.zip)
    # is downloaded and unpacked into data_dir at startup; otherwise data_dir must
    # already hold one.
    data_dir: Path = Path("data")
    dataset_url: str | None = None
    solver_log_path: Path = Path("solver.log")
    solver_name: str = "highs"
    solver_feasibility_tolerance: float = FEASIBILITY_TOLERANCE
    allowed_origins: list[str] = ["*"]

    @property
    def solver_options(self) -> dict:
        return default_solver_options(self.solver_name, self.solver_feasibility_tolerance)

    # model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
