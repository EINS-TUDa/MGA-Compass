# Build the patched cddlib for CddMethod

`CddMethod` needs cddlib and its Python bindings, pycddlib, with two small patches that add
incremental halfspace updates and stable vertex ids. Both are vendored as git submodules
under `mga-surveyor/vendor/cdd/`, and the patches are applied automatically during the build.
You only need this for `vertex_method="cdd"`; the default `BBMethod` does not use cddlib.

**Prerequisites:** git, Python, uv, and a C compiler: MSVC on Windows (Visual Studio Build
Tools with the "Desktop development with C++" workload), gcc or clang on Linux and macOS.
CMake and vcpkg are fetched automatically.

1. Make sure the submodules are checked out:

   ```bash
   git submodule update --init --recursive
   ```

2. Check that the base dependencies install, before starting the slow build:

   ```bash
   cd mga-surveyor
   uv sync
   cd ..
   ```

3. Build GMP, cddlib and the patches:

   ```bash
   python mga-surveyor/vendor/cdd/build.py
   ```

   At the end it prints environment variables (`INCLUDE`/`LIB` on Windows,
   `CPATH`/`LIBRARY_PATH` on Linux and macOS). **Set them in your shell** before the next step.

4. Install pycddlib against the build:

   ```bash
   cd mga-surveyor
   uv sync --extra cdd
   ```

5. Check it:

   ```bash
   uv run python -c "import cdd, cdd.gmp; cdd.polyhedron_append_halfspaces; cdd.gmp.polyhedron_append_halfspaces"
   uv run pytest tests/test_cdd_incremental.py tests/test_cdd_vertex_id_stability.py
   ```

Every step can be re-run safely, for example after pulling an updated patch.

```{note}
`uv sync` removes packages you did not ask for. Once pycddlib is installed, keep passing
`--extra cdd` to `uv sync`, or use `uv run`, which never removes anything.
```
