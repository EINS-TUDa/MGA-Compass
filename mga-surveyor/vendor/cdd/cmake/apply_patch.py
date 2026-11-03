"""Idempotently apply a .patch file to a git submodule checkout.

Invoked by CMakeLists.txt via execute_process() at configure time, so a
plain reconfigure is safe to run repeatedly: if the patch is already
applied (checked via `git apply -R --check`), this is a no-op instead of
failing on "already applied" from a straight `git apply`.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def git_apply_check(repo_dir: Path, patch_path: Path, reverse: bool) -> bool:
    cmd = ["git", "apply", "--check", str(patch_path)]
    if reverse:
        cmd.insert(2, "-R")
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch", type=Path, help="Path to the .patch file")
    parser.add_argument("repo_dir", type=Path, help="Git submodule checkout to apply it to")
    args = parser.parse_args()

    patch_path = args.patch.resolve()
    repo_dir = args.repo_dir.resolve()

    if git_apply_check(repo_dir, patch_path, reverse=False):
        subprocess.run(["git", "apply", str(patch_path)], cwd=repo_dir, check=True)
        print(f"Applied {patch_path.name} to {repo_dir}")
        return 0

    if git_apply_check(repo_dir, patch_path, reverse=True):
        print(f"{patch_path.name} already applied to {repo_dir} -- skipping")
        return 0

    print(
        f"error: {patch_path.name} does not apply cleanly to {repo_dir}\n"
        "The submodule has likely drifted from the commit the patch was cut against, "
        "or the working tree has unrelated local changes. Rebase the patch or reset "
        "the submodule to its pinned commit.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
