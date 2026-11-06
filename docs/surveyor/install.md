# Install MGA Surveyor

MGA Surveyor is installed from a clone of the repository with
[uv](https://docs.astral.sh/uv/).

1. Clone the repository with its submodules:

   ```bash
   git clone --recurse-submodules https://github.com/EINS-TUDa/MGA-Compass.git
   cd MGA-Compass
   ```

2. Install the package and its dependencies:

   ```bash
   uv sync --package mga-surveyor
   ```

   This installs both solver interfaces, `gurobipy` and `highspy`. Gurobi additionally needs
   a licence; HiGHS is open source and needs nothing else.

3. Check the installation:

   ```bash
   uv run python -c "import mga_surveyor; print('ok')"
   ```

`CddMethod` needs one more, optional component, the patched cddlib. You only need it if you
use `vertex_method="cdd"` or `CddMethod` directly; the default, `BBMethod`, does not.
