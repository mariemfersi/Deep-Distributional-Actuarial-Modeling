FROM python:3.11-slim AS builder

WORKDIR /build
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Image finale ─────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour psycopg2 et compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copier les packages installés
COPY --from=builder /install /usr/local

# Copier le code backend
COPY backend/app/ ./app/

# Copier les modules partagés (imports depuis services/)
COPY src/ ./src/
COPY config/ ./config/

# Coplier les modèles pré-entraînés (lecture seule)
COPY models/ ./models/

EXPOSE 8000

# Point d'entrée : migrations + serveur
COPY backend/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
