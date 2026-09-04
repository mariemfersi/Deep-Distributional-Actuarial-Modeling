"""
Modèles ORM — importés ici pour la découverte automatique par Alembic.
"""

from app.database import Base  # noqa: F401
from app.models.policy import Policy  # noqa: F401
from app.models.prediction import Prediction  # noqa: F401
from app.models.triangle import ReservingRun  # noqa: F401
from app.models.user import User  # noqa: F401

__all__ = ["Base", "Policy", "Prediction", "ReservingRun", "User"]
