#!/bin/bash
set -e

echo "⏳ Application des migrations Alembic…"
alembic upgrade head 2>/dev/null || echo "⚠️  Alembic non configuré, les tables seront créées au démarrage."

echo "🚀 Démarrage du serveur FastAPI…"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
