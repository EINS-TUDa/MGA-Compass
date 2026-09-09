"""Builds the vendored, patched cddlib (mga-surveyor's ``cdd`` extra needs its
``cdd``/``cddgmp`` libraries) -- not a Python package installer. Run once per
machine before ``uv sync --extra cdd`` (from ``mga-surveyor/``); safe/idempotent
to re-run. Steps:

1. ``git submodule update --init`` the two vendored submodules.
2. Bootstrap vcpkg (reusing ``$VCPKG_ROOT`` if already set) and
   ``vcpkg install`` GMP for the host triplet, statically linked.
3. Configure/build/install cddlib's two libraries (``cdd``, ``cddgmp``) via
   CMake, against that GMP -- CMakeLists.txt applies the local patch to
   the cddlib submodule as part of configuring.

That's it -- this script never touches the Python environment. Afterwards, run
``uv sync --extra cdd`` yourself (from ``mga-surveyor/``): pycddlib's own
(unmodified) setuptools+Cython build then finds the ``cdd``/``cddgmp``/``gmp``
headers and libs this script just built, via the ``INCLUDE``/``LIB``
(``CPATH``/``LIBRARY_PATH`` on Unix) environment variables ``print_env()``
reports at the end -- see this module's own ``README.md`` for the exact
recommended 3-command sequence (``uv sync``, this script, ``uv sync --extra
cdd``).

Uses a native CMake fetched on demand via ``uvx`` rather than whatever
``cmake`` happens to be on PATH, since e.g. a Cygwin/MSYS cmake on Windows
offers no MSVC-compatible generator and would produce a binary
incompatible with a standard CPython build.
"""
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _vcpkg_work_root() -> Path:
    """Where vcpkg's own checkout, buildtrees, packages and install root live. Normally
    just VENDOR_DIR, alongside everything else -- except gmp's vcpkg port shells out to
    MSYS2 bash for autoreconf, which mis-tokenizes any embedded space in the paths it's
    given, including a Windows username with a space in it (e.g. "Sina Hajikazemi"),
    corrupting the invoked script's argv entirely (confirmed: a real `aclocal` invocation
    failing exactly this way). Windows' legacy 8.3 short name doesn't survive as a
    workaround -- vcpkg canonicalizes paths internally, which resolves a short name back
    to its long form before anything reaches MSYS bash -- so this relocates vcpkg's own
    work entirely outside the user profile instead, only when actually needed.
    """
    if platform.system() == "Windows" and " " in str(VENDOR_DIR):
        root = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "mga-surveyor-cdd-vcpkg"
        root.mkdir(parents=True, exist_ok=True)
        return root
    return VENDOR_DIR


VENDOR_DIR = Path(__file__).resolve().parent
BUILD_DIR = VENDOR_DIR / "_build"
INSTALL_DIR = VENDOR_DIR / "_install"
_VCPKG_WORK_ROOT = _vcpkg_work_root()
VCPKG_CACHE_DIR = _VCPKG_WORK_ROOT / "_vcpkg"
VCPKG_INSTALLED_DIR = _VCPKG_WORK_ROOT / "vcpkg_installed"

VCPKG_PINNED_TAG = "37bb045f3c7a747d3e5d1c13b6fe6a0aec4b5d00"  # bump deliberately; keeps GMP's build
# reproducible. A commit, not a tag: the newest tagged release (2026.07.29) predates
# microsoft/vcpkg#53437 ("[gmp] Update autoconf 2.71 download URL", 2026-08-15) -- gmp's port
# hardcodes an exact MSYS2 autoconf2.71 build, and the one every tag up to 2026.07.29 points at
# has since been pruned from every mirror (404), breaking the gmp build entirely until this fix.


def run(cmd: list[str], **kwargs) -> None:
    print(f"+ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True, **kwargs)


def detect_triplet() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    if system == "Windows":
        # static-md: static libs (no separate gmp.dll to ship/find at runtime),
        # linked against the dynamic MSVC CRT -- matches a standard CPython build.
        return f"{arch}-windows-static-md"
    if system == "Darwin":
        return f"{arch}-osx"
    if system == "Linux":
        return f"{arch}-linux"
    raise RuntimeError(f"Unsupported platform for the cdd vendor build: {system}")


def ensure_submodules() -> None:
    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=VENDOR_DIR, check=True, capture_output=True, text=True
    ).stdout.strip()
    run(
        ["git", "submodule", "update", "--init", "--", "mga-surveyor/vendor/cdd/cddlib", "mga-surveyor/vendor/cdd/pycddlib"],
        cwd=repo_root,
    )
    # cddlib's own incremental-append patch is applied by CMakeLists.txt at configure
    # time (it drives the cddlib build). pycddlib's side of the same feature --
    # exposing polyhedron_append_halfspaces/copy_generators_with_ids to Python -- has
    # no such hook, since pycddlib itself is built later by `uv sync --extra cdd`, not
    # by this script or CMake; apply it here instead, idempotently, same as the other.
    run([
        sys.executable, str(VENDOR_DIR / "cmake" / "apply_patch.py"),
        str(VENDOR_DIR / "patches" / "pycddlib-incremental-append.patch"),
        str(VENDOR_DIR / "pycddlib"),
    ])


def vcpkg_executable(vcpkg_root: Path) -> Path:
    return vcpkg_root / ("vcpkg.exe" if platform.system() == "Windows" else "vcpkg")


def ensure_vcpkg() -> Path:
    env_root = os.environ.get("VCPKG_ROOT")
    if env_root and vcpkg_executable(Path(env_root)).exists():
        return Path(env_root)

    if not VCPKG_CACHE_DIR.exists() or not any(VCPKG_CACHE_DIR.iterdir()):
        # `clone --branch` can't take an arbitrary commit (only tags/branches), and
        # VCPKG_PINNED_TAG is a commit here -- fetch it directly instead, still shallow. The
        # `iterdir` check also covers a leftover empty dir (e.g. one a prior failed run's
        # cleanup couldn't fully remove because something still had it locked).
        VCPKG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "-q", str(VCPKG_CACHE_DIR)])
        run(["git", "fetch", "--depth", "1", "https://github.com/microsoft/vcpkg.git", VCPKG_PINNED_TAG],
            cwd=VCPKG_CACHE_DIR)
        run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=VCPKG_CACHE_DIR)

    if not vcpkg_executable(VCPKG_CACHE_DIR).exists():
        bootstrap = VCPKG_CACHE_DIR / ("bootstrap-vcpkg.bat" if platform.system() == "Windows" else "bootstrap-vcpkg.sh")
        run([str(bootstrap), "-disableMetrics"])

    return VCPKG_CACHE_DIR


def vcpkg_install(vcpkg_root: Path, triplet: str) -> None:
    run([
        str(vcpkg_executable(vcpkg_root)), "install",
        f"--x-manifest-root={VENDOR_DIR}",
        f"--x-install-root={VCPKG_INSTALLED_DIR}",
        f"--triplet={triplet}",
    ])


def cmake_build(vcpkg_root: Path, triplet: str) -> None:
    toolchain = vcpkg_root / "scripts" / "buildsystems" / "vcpkg.cmake"
    run([
        "uvx", "cmake", "-S", str(VENDOR_DIR), "-B", str(BUILD_DIR),
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
        f"-DVCPKG_TARGET_TRIPLET={triplet}",
        f"-DVCPKG_INSTALLED_DIR={VCPKG_INSTALLED_DIR}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR}",
    ])
    run(["uvx", "cmake", "--build", str(BUILD_DIR), "--config", "Release"])
    run(["uvx", "cmake", "--install", str(BUILD_DIR), "--config", "Release"])


def build_paths(triplet: str) -> tuple[list[Path], list[Path]]:
    vcpkg_prefix = VCPKG_INSTALLED_DIR / triplet
    include_dirs = [INSTALL_DIR / "include", vcpkg_prefix / "include"]
    lib_dirs = [INSTALL_DIR / "lib", vcpkg_prefix / "lib"]
    return include_dirs, lib_dirs


def print_env(triplet: str) -> None:
    """pycddlib's own setup.py is deliberately unmodified (see module docstring), so
    the only way to hand its setuptools+Cython build the cdd/cddgmp/gmp headers and
    libs just built here is the same environment variables any C compiler driver
    reads for its search paths -- and since this script no longer runs `uv sync`
    itself, whoever runs it next needs to set these in their own shell first.
    """
    include_dirs, lib_dirs = build_paths(triplet)
    if platform.system() == "Windows":
        print("\nRun this (PowerShell), then `uv sync --extra cdd` (from mga-surveyor/):\n", file=sys.stderr)
        print(f'$env:INCLUDE = "{os.pathsep.join(map(str, include_dirs))};$env:INCLUDE"', file=sys.stderr)
        print(f'$env:LIB = "{os.pathsep.join(map(str, lib_dirs))};$env:LIB"', file=sys.stderr)
    else:
        print("\nRun this, then `uv sync --extra cdd` (from mga-surveyor/):\n", file=sys.stderr)
        print(f'export CPATH="{os.pathsep.join(map(str, include_dirs))}:$CPATH"', file=sys.stderr)
        print(f'export LIBRARY_PATH="{os.pathsep.join(map(str, lib_dirs))}:$LIBRARY_PATH"', file=sys.stderr)


def main() -> int:
    if shutil.which("uv") is None:  # needed for `uvx cmake` below, not for installing anything
        print("error: `uv` not found on PATH", file=sys.stderr)
        return 1

    triplet = detect_triplet()
    print(f"Using vcpkg triplet: {triplet}")

    ensure_submodules()
    vcpkg_root = ensure_vcpkg()
    vcpkg_install(vcpkg_root, triplet)
    cmake_build(vcpkg_root, triplet)

    print("\nBuild done: cdd/cddgmp/gmp are ready.", file=sys.stderr)
    print_env(triplet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
