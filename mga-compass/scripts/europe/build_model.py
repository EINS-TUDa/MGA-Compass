"""Provision pypsa-eur in its own environment and solve an electricity-only
network from it -- phase 1 of the two-phase pipeline (see run_pipeline.py,
which aggregates this with run_custom_mga.py's own phase 2). Standalone-
runnable on its own (just produces a solved network on disk, no MGA
analysis), or import build_model()/solved_network_path() from another script.

Clones pypsa-eur if it isn't already checked out, `pixi install`s its env if
not already built, deploys config.template.yaml -> <pypsa-eur-dir>/config/
config.yaml (unless one already exists there -- an existing config is never
overwritten without --force-config, since it may hold a user's own
customization), then runs pypsa-eur's own `snakemake ... solve_elec_networks`
via `pixi run` (subprocess, pypsa-eur's own pixi environment -- pinned to an
older pypsa than mga_surveyor needs, so this module never imports pypsa
itself, only shells out to that environment's own snakemake).

Idempotent: an existing pypsa-eur checkout/pixi env/config is reused as-is,
and Snakemake's own DAG caching means a re-run only redoes what's missing --
so re-running after an interruption resumes rather than restarts (an
interrupted run needed a manual `--unlock` after a stale lock from a killed
process during this module's own development; build_model does that
automatically on a LockException before its one retry).

Run standalone with (from the MGA-Dashboard repo root, using its own venv):
    .venv\\Scripts\\python.exe mga-compass\\scripts\\europe\\build_model.py --resolution-elec 3h
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

#: This file lives at <MGA-Dashboard>/mga-compass/scripts/europe/build_model.py,
#: so three parents up is the MGA-Dashboard repo root -- computed from the
#: script's own location (not hardcoded) so this default is portable across
#: machines/checkouts.
_MGA_DASHBOARD_ROOT = Path(__file__).resolve().parents[3]
_EUROPE_DIR = Path(__file__).resolve().parent
DEFAULT_PYPSA_EUR_DIR = _MGA_DASHBOARD_ROOT.parent / "pypsa-eur"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_model")

PYPSA_EUR_REPO = "https://github.com/pypsa/pypsa-eur.git"
RUN_NAME = "elec-country-nodes"
CLUSTERS = 34
OPTS = ""


def _snakemake_exe(pypsa_eur_dir: Path) -> Path:
    subdir = "Scripts" if sys.platform == "win32" else "bin"
    suffix = ".exe" if sys.platform == "win32" else ""
    return pypsa_eur_dir / ".pixi" / "envs" / "default" / subdir / f"snakemake{suffix}"


def ensure_pypsa_eur_checkout(pypsa_eur_dir: Path) -> None:
    """Clone pypsa-eur if `pypsa_eur_dir` doesn't exist yet. If it exists,
    it's left exactly as-is (no pull/reset) -- an existing checkout may hold
    in-progress config/results from an earlier run of this same pipeline,
    and force-syncing it could discard that."""
    if pypsa_eur_dir.exists():
        if not (pypsa_eur_dir / ".git").exists():
            msg = f"{pypsa_eur_dir} already exists but isn't a git checkout -- refusing to touch it."
            raise SystemExit(msg)
        logger.info("Reusing existing pypsa-eur checkout at %s", pypsa_eur_dir)
        return
    logger.info("Cloning pypsa-eur into %s", pypsa_eur_dir)
    subprocess.run(["git", "clone", PYPSA_EUR_REPO, str(pypsa_eur_dir)], check=True)


def ensure_pixi_env(pypsa_eur_dir: Path) -> None:
    if _snakemake_exe(pypsa_eur_dir).exists():
        logger.info("Reusing existing pixi environment")
        return
    logger.info("Building pypsa-eur's pixi environment (this can take a while)")
    subprocess.run(["pixi", "install"], cwd=str(pypsa_eur_dir), check=True)


def deploy_config(pypsa_eur_dir: Path, resolution_elec: str, force: bool) -> None:
    """Render config.template.yaml -> <pypsa_eur_dir>/config/config.yaml,
    unless one already exists there and `force` isn't set -- an existing
    config.yaml may already hold a user's own customization on top of (or
    instead of) this pipeline's own run."""
    config_path = pypsa_eur_dir / "config" / "config.yaml"
    if config_path.exists() and not force:
        logger.info("Reusing existing %s (pass --force-config to overwrite)", config_path)
        return
    template = (_EUROPE_DIR / "config.template.yaml").read_text(encoding="utf-8")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        template.format(run_name=RUN_NAME, clusters=CLUSTERS, resolution_elec=resolution_elec),
        encoding="utf-8",
    )
    logger.info("Wrote %s", config_path)


def run_snakemake(pypsa_eur_dir: Path, cores: int) -> None:
    # Invoked via `pixi run`, not the env's snakemake.exe directly: calling
    # the exe by its raw path skips pixi's own environment activation,
    # which is what sources conda-forge packages' etc/conda/activate.d/
    # scripts (e.g. gdal-activate.bat, setting GDAL_DATA/GDAL_DRIVER_PATH)
    # -- confirmed to matter in practice, not just in theory: without it,
    # GDAL logs "Cannot find gdalvrt.xsd/header.dxf (GDAL_DATA is not
    # defined)" and determine_availability_matrix's raster/shapefile reads
    # (natura.tiff, corine.tif, the administrative-boundary geojsons) fail
    # outright with CPLE_OpenFailedError. `pixi run` finds pixi.toml from
    # cwd, same as any other subprocess call here.
    # --rerun-incomplete: a job killed mid-write (interrupted process, crash,
    # this very script being stopped and restarted) leaves Snakemake unable
    # to trust that job's own output files, so a plain re-run refuses to
    # proceed with an IncompleteFilesException rather than silently trusting
    # possibly-truncated data -- this tells it to just regenerate those
    # specific files instead of erroring, matching the "re-running after an
    # interruption resumes rather than restarts" idempotency this module's
    # own docstring promises. A no-op on a clean run: nothing to rerun if
    # nothing's incomplete.
    cmd = ["pixi", "run", "snakemake", "--cores", str(cores), "--rerun-incomplete", "solve_elec_networks"]

    result = subprocess.run(cmd, cwd=str(pypsa_eur_dir), capture_output=True, text=True)
    if result.returncode != 0 and "LockException" in result.stderr:
        # Same recovery this project needed by hand during development,
        # after a killed process left a stale lock behind.
        logger.warning("Stale Snakemake lock detected -- unlocking and retrying once")
        subprocess.run(["pixi", "run", "snakemake", "--unlock"], cwd=str(pypsa_eur_dir), check=True)
        result = subprocess.run(cmd, cwd=str(pypsa_eur_dir))
        if result.returncode != 0:
            raise SystemExit("snakemake failed again after --unlock -- see its own output above.")
    elif result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"snakemake failed (exit {result.returncode}) -- see its own output above.")
    else:
        logger.info(result.stdout[-4000:])  # tail -- the full log is long


def solved_network_path(pypsa_eur_dir: Path) -> Path:
    return pypsa_eur_dir / "results" / RUN_NAME / "networks" / f"base_s_{CLUSTERS}_elec_{OPTS}.nc"


def build_model(pypsa_eur_dir: Path, cores: int, resolution_elec: str, force_config: bool) -> Path:
    """Provision + solve in one call -- the single entry point a caller
    (run_pipeline.py's own aggregation, or this module's own CLI) needs for
    phase 1. Returns the solved network's path (see solved_network_path);
    doesn't check it actually exists afterward -- run_snakemake already
    raises on any snakemake failure, so reaching this function's return
    means the target file is there.
    """
    ensure_pypsa_eur_checkout(pypsa_eur_dir)
    ensure_pixi_env(pypsa_eur_dir)
    deploy_config(pypsa_eur_dir, resolution_elec, force_config)
    run_snakemake(pypsa_eur_dir, cores)
    return solved_network_path(pypsa_eur_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pypsa-eur-dir", type=Path, default=DEFAULT_PYPSA_EUR_DIR,
        help="Where to find/clone pypsa-eur. Reused as-is if it already exists, cloned fresh otherwise. "
             f"Defaults to a sibling of this repo (default: {DEFAULT_PYPSA_EUR_DIR}).",
    )
    parser.add_argument("--cores", type=int, default=8, help="Cores for pypsa-eur's own snakemake run")
    parser.add_argument("--resolution-elec", type=str, default="24h", help="Snapshot resampling for the config template")
    parser.add_argument("--force-config", action="store_true", default=False, help="Overwrite an existing config/config.yaml")
    args = parser.parse_args()

    network_path = build_model(args.pypsa_eur_dir, args.cores, args.resolution_elec, args.force_config)
    logger.info("Solved network: %s", network_path)


if __name__ == "__main__":
    main()
