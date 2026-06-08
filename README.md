# MGA Dashboard

## Installation

### Prerequisites

- Python 3.14+
- Node.js 20.19+ or 22.12+

### Backend

```bash
cd backend
pip install -e ".[standard]"
uvicorn mgaserver.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.
