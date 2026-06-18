# <img src="docs/mga_logo.svg" alt="MGA Compass logo" width="48" height="48" align="center" /> MGA Compass

![License](https://img.shields.io/github/license/EINS-TUDa/MGA-Compass)
[![Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://enormous-raphaela-eins-tu-darmstadt-215a3282.koyeb.app/)

**Live demo:** [enormous-raphaela-eins-tu-darmstadt-215a3282.koyeb.app](https://enormous-raphaela-eins-tu-darmstadt-215a3282.koyeb.app/)

**Presentation:** [eins-tuda.github.io/MGA-Compass](https://eins-tuda.github.io/MGA-Compass/)

Near-optimal solutions cost a small percentage more than the optimum. While slightly expensive, they can be preferable for political feasibility, social acceptance, or environmental impact. This platform helps you explore them in real time.

![MGA Compass](docs/MGA-compass.png)

## Installation

### Prerequisites

- Python 3.14+
- uv
- Node.js 20.19+ or 22.12+

### Backend

```bash
cd app/backend
uv sync
uv run uvicorn mgaserver.main:app --reload
```

### Frontend

```bash
cd app/frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

## Docker

### Docker prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Run locally

```bash
docker compose up --build
```

The app will be available at `http://localhost`.
