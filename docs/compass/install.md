# Install the MGA Compass library

The `mga-compass` library is installed from a clone of the repository with
[uv](https://docs.astral.sh/uv/). It needs no commercial solver: navigation and
interpolation run on HiGHS.

1. Clone the repository:

   ```bash
   git clone https://github.com/EINS-TUDa/MGA-Compass.git
   cd MGA-Compass
   ```

2. Install the package and its dependencies:

   ```bash
   uv sync --package mga-compass
   ```

3. Check the installation:

   ```bash
   uv run python -c "import mga_compass; print('ok')"
   ```

To run notebooks with this environment as the kernel, e.g. in VS Code, and draw plots (see
[](explore-in-python.md)), add the `notebook` extra, which brings `ipykernel`, matplotlib and scipy:
`uv sync --package mga-compass --extra notebook`.

To build datasets from PyPSA networks you also need MGA Surveyor and PyPSA, which come with
the `scripts` extra: `uv sync --package mga-compass --extra scripts`.
