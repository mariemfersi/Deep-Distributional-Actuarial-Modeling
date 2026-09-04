-- ── Création de la base MLflow dédiée ──────────────────────────
-- Exécuté automatiquement au premier démarrage de PostgreSQL
-- (uniquement si le volume pgdata est vide)

CREATE DATABASE mlflow_tracking;
