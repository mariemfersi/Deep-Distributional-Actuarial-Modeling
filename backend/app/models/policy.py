"""
Modèle ORM pour les polices d'assurance (données de tarification).
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime
from app.database import Base


class Policy(Base):
    __tablename__ = "policies"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # ── Caractéristiques véhicule ───────────────────────────────────
    veh_power = Column(Integer, nullable=False, comment="Puissance du véhicule")
    veh_age = Column(Integer, nullable=False, comment="Âge du véhicule (années)")
    veh_brand = Column(String(10), nullable=False, comment="Marque (B1–B14)")
    veh_gas = Column(String(10), nullable=False, comment="Carburant: Diesel / Regular")

    # ── Caractéristiques conducteur ─────────────────────────────────
    driv_age = Column(Integer, nullable=False, comment="Âge du conducteur")
    bonus_malus = Column(Integer, nullable=False, comment="Coefficient bonus-malus")

    # ── Géographie ─────────────────────────────────────────────────
    region = Column(String(50), nullable=False, comment="Région")
    area = Column(String(5), nullable=False, comment="Zone géographique (A–F)")
    density = Column(Float, nullable=False, comment="Densité de population")

    # ── Exposition ─────────────────────────────────────────────────
    exposure = Column(Float, default=1.0, comment="Exposition en années")

    # ── Métadonnées ────────────────────────────────────────────────
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
