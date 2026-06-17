FROM python:3.14-slim

WORKDIR /repo

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# mga-compass is a local path dependency of app/backend (see app/backend/pyproject.toml)
COPY mga-compass/ ./mga-compass/

WORKDIR /repo/app/backend

# Install dependencies first (cached layer — only re-runs when pyproject.toml/uv.lock change)
COPY app/backend/pyproject.toml app/backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code and install the project
COPY app/backend/mgaserver/ ./mgaserver/
RUN uv sync --frozen --no-dev

# Copy data files
COPY app/backend/data/ ./data/

EXPOSE 8000

CMD ["/repo/app/backend/.venv/bin/uvicorn", "mgaserver.main:app", "--host", "0.0.0.0", "--port", "8000"]
