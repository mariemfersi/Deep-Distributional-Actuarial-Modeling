"""
Modèle ORM pour les prédictions (historique des appels API).
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # ── Identification ─────────────────────────────────────────────
    module = Column(String(20), nullable=False, index=True,
                    comment="pricing | fraud | reserving")
    model_version = Column(String(50), nullable=True, comment="Version du modèle utilisé")

    # ── Données ────────────────────────────────────────────────────
    request_json = Column(JSON, nullable=False, comment="Requête d'entrée (dict)")
    response_json = Column(JSON, nullable=False, comment="Réponse renvoyée (dict)")

    # ── Performance ────────────────────────────────────────────────
    latency_ms = Column(Float, nullable=True, comment="Latence de traitement (ms)")

    # ── Métadonnées ────────────────────────────────────────────────
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
