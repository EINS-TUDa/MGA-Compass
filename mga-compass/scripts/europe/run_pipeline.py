"""End-to-end pipeline: aggregates build_model.py (phase 1: provision
pypsa-eur, solve an electricity-only network) and run_custom_mga.py (phase
2: our own mga_surveyor MGA analysis) into one run -- each phase is its own
standalone script, runnable independently in its own right; this one just
wires them together.

Pure orchestration, no duplicated logic -- everything pypsa-eur-specific is
a real file deployed alongside this script's own sibling files, not a string
template or a copy-pasted function:

1. **build_model.build_model()** (subprocess, pypsa-eur's own pixi
   environment): see that module's own docstring. Run standalone with
   build_model.py directly if you only want a solved network, no MGA
   analysis -- or pass --skip-solve here to reuse one already solved.

2. **MGA analysis** (subprocess, THIS project's own venv): deploys this
   repo's own run_custom_mga.py -> <pypsa-eur-dir>/scripts/run_custom_mga.py
   (always redeployed, so it can never drift out of sync with this repo's
   own version) and runs it with sys.executable (this script's own
   interpreter, i.e. mga_surveyor's required pypsa>=1.3.0) against the
   solved network -- see run_custom_mga.py's own docstring for what it does
   (dimension selection, generate_mga_files, consolidation).

Run with (from the MGA-Dashboard repo root, using its own venv). By default,
pypsa-eur is expected/cloned as a sibling of this repo (../pypsa-eur); pass
--pypsa-eur-dir to use a different checkout:
    .venv\\Scripts\\python.exe mga-compass\\scripts\\europe\\run_pipeline.py --output-dir mga-compass\\scripts\\europe\\output
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import build_model

_EUROPE_DIR = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run_pipeline")


def deploy_mga_script(pypsa_eur_dir: Path) -> Path:
    """Copy this repo's own run_custom_mga.py into <pypsa_eur_dir>/scripts/
    -- always redeployed (unlike config.yaml) so pypsa-eur's copy can never
    drift out of sync with this one, which is the single source of truth."""
    dest = pypsa_eur_dir / "scripts" / "run_custom_mga.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_EUROPE_DIR / "run_custom_mga.py", dest)
    logger.info("Deployed %s", dest)
    return dest


def run_mga_script(
    mga_script: Path, network_path: Path, output_dir: Path,
    slack_levels: str, max_iter: int, max_parallel: int,
) -> None:
    cmd = [
        sys.executable, str(mga_script), str(network_path),
        "--output-dir", str(output_dir),
        "--slack-levels", slack_levels,
        "--max-iter", str(max_iter),
        "--max-parallel", str(max_parallel),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pypsa-eur-dir", type=Path, default=build_model.DEFAULT_PYPSA_EUR_DIR,
        help="Where to find/clone pypsa-eur. Reused as-is if it already exists, cloned fresh otherwise. "
             f"Defaults to a sibling of this repo (default: {build_model.DEFAULT_PYPSA_EUR_DIR}).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write MGA results to")
    parser.add_argument("--cores", type=int, default=8, help="Cores for pypsa-eur's own snakemake run")
    parser.add_argument("--resolution-elec", type=str, default="24h", help="Snapshot resampling for the config template")
    parser.add_argument("--force-config", action="store_true", default=False, help="Overwrite an existing config/config.yaml")
    parser.add_argument("--slack-levels", type=str, default="0.05")
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--skip-solve", action="store_true", default=False, help="Skip phase 1 entirely (network already solved)")
    args = parser.parse_args()

    if args.skip_solve:
        network_path = build_model.solved_network_path(args.pypsa_eur_dir)
    else:
        network_path = build_model.build_model(args.pypsa_eur_dir, args.cores, args.resolution_elec, args.force_config)

    if not network_path.exists():
        raise SystemExit(f"Expected solved network at {network_path} but it doesn't exist.")

    mga_script = deploy_mga_script(args.pypsa_eur_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_mga_script(mga_script, network_path, args.output_dir, args.slack_levels, args.max_iter, args.max_parallel)


if __name__ == "__main__":
    main()
