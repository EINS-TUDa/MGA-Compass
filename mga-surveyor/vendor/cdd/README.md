# Vendored cddlib + pycddlib

This directory vendors [cddlib](https://github.com/cddlib/cddlib) and
[pycddlib](https://github.com/mcmtroffaes/pycddlib) as git submodules, pinned at
pristine upstream commits, plus two small patches (`patches/*.patch`) that add an
incremental halfspace-append entry point and stable per-vertex generator ids to
cddlib, and expose both to Python. They're consumed by
`mga_surveyor.methods.cdd.CddMethod`.

The patches are applied automatically (idempotently) by `CMakeLists.txt` when
cddlib is configured -- you never need to apply them by hand.

## Building

Prerequisites: git, Python, `uv`, a C compiler for your platform (MSVC on
Windows -- e.g. Visual Studio Build Tools with the "Desktop development with
C++" workload; gcc/clang on Linux/macOS). CMake and vcpkg are fetched
automatically.

`build.py` only builds the C libraries -- it doesn't touch the Python
environment. Run all three of these, in order, from the repo root:

```
uv sync                            # (from mga-surveyor/) fail fast on any base-dependency issue
                                    #  before committing to the much slower step below
python mga-surveyor/vendor/cdd/build.py
uv sync --extra cdd                # (from mga-surveyor/) with the env vars build.py printed set first
```

`build.py` builds GMP from source via [vcpkg](https://github.com/microsoft/vcpkg)
(statically linked, matching the host platform's usual C runtime), then builds
cddlib's `cdd`/`cddgmp` libraries against it via CMake. It prints the
`INCLUDE`/`LIB` (`CPATH`/`LIBRARY_PATH` on Unix) exports the last `uv sync`
needs to find them -- pycddlib's own setuptools+Cython build is unmodified, so
those environment variables are the only way to hand it those paths.

Safe to re-run any time (e.g. after pulling a newer patch, or on a fresh
machine) -- every step is idempotent.

## Verifying

```
cd mga-surveyor
uv run python -c "import cdd, cdd.gmp; cdd.polyhedron_append_halfspaces; cdd.gmp.polyhedron_append_halfspaces"
uv run pytest tests/test_cdd_incremental.py tests/test_cdd_vertex_id_stability.py -v
```

## Updating the patches

The patches were cut with a plain `git diff` against the pinned base commits
recorded in this repo's submodule pointers (not a proper commit in either
upstream clone). To update one: make your change in a local clone of the
relevant project checked out at the same pinned commit, then regenerate:

```
git diff > mga-surveyor/vendor/cdd/patches/cddlib-incremental-append.patch
```

To move to a newer upstream cddlib/pycddlib release: bump the submodule to the
new commit, re-apply the patch by hand to check for drift/fuzz, fix up as
needed, and regenerate the `.patch` file the same way.
