"""
Configuration de l'application via variables d'environnement et fichier .env.
Utilise python-dotenv (déjà installé) — pas de dépendance dure sur pydantic-settings.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Charge le fichier .env s'il existe (backend/.env ou racine/.env)
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


class Settings:
    """Paramètres de configuration de la plateforme actuarielle."""

    # ── Base de données ─────────────────────────────────────────────
    DATABASE_URL: str = _env("DATABASE_URL", "postgresql://actuarial:changeme@db:5432/actuarial_platform")

    # ── Sécurité ────────────────────────────────────────────────────
    SECRET_KEY: str = _env("SECRET_KEY", "change-me-in-production-use-a-real-secret")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(_env("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    # ── MLflow ──────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = _env("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    # ── Environnement ───────────────────────────────────────────────
    ENVIRONMENT: str = _env("ENVIRONMENT", "development")  # development | staging | production

    # ── CORS ────────────────────────────────────────────────────────
    CORS_ORIGINS: str = _env("CORS_ORIGINS", "http://localhost:8080,http://localhost:3000,http://127.0.0.1:3000")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Singleton pour les settings (cacheé pendant la durée du process)."""
    return Settings()
