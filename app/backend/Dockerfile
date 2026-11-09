FROM python:3.14-slim

WORKDIR /repo

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# app/backend is a workspace member (see /pyproject.toml's [tool.uv.workspace])
# depending on mga-compass (see app/backend/pyproject.toml's [tool.uv.sources]).
# uv validates/resolves the whole workspace even when syncing just mgaserver, so
# every member needs its pyproject.toml present -- mga-surveyor only needs that
# (mgaserver doesn't depend on it, and mga-compass's own dependency on it is
# behind an extra mgaserver doesn't request), mga-compass and mga-format need
# their full source since they ARE editable dependencies of mgaserver.
COPY pyproject.toml uv.lock ./
COPY mga-surveyor/pyproject.toml ./mga-surveyor/pyproject.toml
COPY mga-format/ ./mga-format/
COPY mga-compass/ ./mga-compass/
COPY app/backend/pyproject.toml ./app/backend/pyproject.toml

# Install dependencies first (cached layer — only re-runs when any
# pyproject.toml/uv.lock changes, not on app source changes).
RUN uv sync --frozen --no-dev --package mgaserver

# Copy application code and install the project
COPY app/backend/mgaserver/ ./app/backend/mgaserver/
RUN uv sync --frozen --no-dev --package mgaserver

# No data in the image: at startup the server downloads the near-optimal set
# archive at $DATASET_URL into data/ (see mgaserver/config.py and mga_format.fetch).

WORKDIR /repo/app/backend

EXPOSE 8000

CMD ["/repo/.venv/bin/uvicorn", "mgaserver.main:app", "--host", "0.0.0.0", "--port", "8000"]
