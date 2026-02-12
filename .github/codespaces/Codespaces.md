# GitHub Codespaces Setup Guide

## Quick Start

1. Open this repository in GitHub Codespaces
2. Wait for automatic setup to complete
3. Open a new terminal and run:

```bash
cd backend
```

## Manual Setup

If you need to set up manually:

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Set environment variables
cp backend/.env.example backend/.env
# Edit backend/.env and add your API keys

# Start the service
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Accessing the Service

Once running:
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Neon Database Setup

The project uses Neon (Serverless PostgreSQL) as the database. Connection is configured in `backend/.env`.

To verify Neon connection:
```bash
cd backend
python3 -c "from evolution.evolution_router_neon import get_neon_db; db = get_neon_db(); print('✅ Neon connected')"
```

## Troubleshooting

- If dependencies fail to install, try: `pip install --upgrade pip`
- If port 8000 is in use, try a different port: `python3 -m uvicorn main:app --port 8080`
- Check logs in `backend/logs/` directory
