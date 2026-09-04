"""
Configuration de la base de données — SQLAlchemy engine, session et Base.

L'engine est créé de manière paresseuse (lazy) : l'import du module ne
connexionne pas la base. Le driver est détecté automatiquement selon l'URL :
- postgresql:// → psycopg2 (production, installé via requirements.txt)
- sqlite:///    → sqlite3 intégré (développement / tests)
"""

from typing import Generator

from sqlalchemy import create_engine, text as sa_text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from app.config import get_settings

settings = get_settings()

_engine = None
_session_factory = None

Base = declarative_base()


def _get_engine():
    """Crée l'engine SQLAlchemy à la demande (lazy singleton)."""
    global _engine
    if _engine is None:
        connect_args = {}
        # Pour SQLite en tests : partager la connexion entre threads
        if settings.DATABASE_URL.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=not settings.DATABASE_URL.startswith("sqlite"),
            connect_args=connect_args,
        )
    return _engine


def _get_session_factory():
    """Session factory singleton liée à l'engine courant."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=_get_engine()
        )
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — fournit une session DB par requête."""
    db = _get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Crée toutes les tables (utilisé au démarrage si pas d'Alembic).

    Protégé par un advisory lock PostgreSQL pour éviter les conflits
    multi-workers (create_all n'est pas idempotent sur les sequences).
    SQLite (tests) n'a pas besoin de lock.
    """
    from app.models import Base as model_base  # import des modèles
    engine = _get_engine()
    is_postgres = settings.DATABASE_URL.startswith("postgresql")

    if not is_postgres:
        model_base.metadata.create_all(bind=engine)
        return

    # PostgreSQL : advisory lock pour serialiser create_all entre workers
    with engine.connect() as conn:
        conn.execute(sa_text("SELECT pg_advisory_lock(42)"))
        try:
            model_base.metadata.create_all(bind=conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.execute(sa_text("SELECT pg_advisory_unlock(42)"))
            conn.commit()
