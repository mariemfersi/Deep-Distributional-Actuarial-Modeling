"""
Modèle ORM pour les runs de provisionnement (IBNR, triangles).
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from app.database import Base


class ReservingRun(Base):
    __tablename__ = "reserving_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # ── Identification ─────────────────────────────────────────────
    grcode = Column(Integer, nullable=False, index=True, comment="Code de la compagnie")
    evaluation_year = Column(Integer, nullable=False, comment="Année d'évaluation")

    # ── Estimation IBNR ────────────────────────────────────────────
    ibnr_estimate = Column(Float, nullable=False, comment="Estimation IBNR centrale")
    mack_lower = Column(Float, nullable=True, comment="Borne inf. intervalle Mack")
    mack_upper = Column(Float, nullable=True, comment="Borne sup. intervalle Mack")
    conformal_lower = Column(Float, nullable=True, comment="Borne inf. intervalle conformal")
    conformal_upper = Column(Float, nullable=True, comment="Borne sup. intervalle conformal")

    # ── Données détaillées ────────────────────────────────────────
    triangle_json = Column(JSON, nullable=True, comment="Triangle de development")
    ldfs_json = Column(JSON, nullable=True, comment="Loss development factors")
    model_version = Column(String(50), default="mack_conformal_v1")

    # ── Métadonnées ────────────────────────────────────────────────
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
