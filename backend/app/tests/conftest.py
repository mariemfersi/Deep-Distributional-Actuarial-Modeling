"""
Pytest configuration for backend tests.
"""

import os
import sys
from pathlib import Path

# Forcer SQLite pour les tests (empêche la connexion à PostgreSQL au démarrage)
os.environ["DATABASE_URL"] = "sqlite:///actuarial_test.db"

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from app.database import Base, get_db
from app.auth import get_current_user
from app.main import app

# ── Test database (SQLite fichier local, par fichier pour permettre plusieurs bases) ──
SQLALCHEMY_TEST_URL = os.environ["DATABASE_URL"]

test_engine = create_engine(
    SQLALCHEMY_TEST_URL,
    connect_args={"check_same_thread": False},
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Crée les tables de test une seule fois."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


def _override_get_db():
    """Remplace la DB par SQLite pour tous les Tests Clients."""
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


# Applique l'override globalement (concerne aussi le module-level client de test_integration)
app.dependency_overrides[get_db] = _override_get_db


# ── Mock user pour les tests (bypass JWT auth) ─────────────────────
class _MockUser:
    """Utilisateur factice pour les tests — satisfait get_current_user."""
    def __init__(self):
        self.id = 1
        self.username = "test_user"
        self.email = "test@example.com"
        self.is_active = True
        self.role = "admin"


async def _override_get_current_user():
    return _MockUser()


app.dependency_overrides[get_current_user] = _override_get_current_user


@pytest.fixture
def db_session():
    """Fournit une session DB isolée par test (tables nettoyées avant/après)."""
    # Nettoyer les tables pour éviter la contamination entre tests
    for table in reversed(Base.metadata.sorted_tables):
        table.drop(test_engine)
    Base.metadata.create_all(test_engine)

    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    """FastAPI TestClient utilisant la DB de test (via l'override global)."""
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


# ── Vérification des modèles requis ───────────────────────────────
MODELS_DIR = project_root / "models"

REQUIRED_MODELS = [
    "glm_poisson.pkl",
    "glm_gamma.pkl",
    "cann_group_interaction.pt",
    "ngboost_severity.pkl",
    "fraud_best_model.pkl",
    "fraud_encoders.pkl",
    "fraud_normalization_stats.pkl",
    "fraud_default_values.pkl",
    "deep_triangle.pt",
    "cann_stats.pkl",
]

for model_file in REQUIRED_MODELS:
    assert (MODELS_DIR / model_file).exists(), f"Required model file missing: {model_file}"
