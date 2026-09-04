"""
Tests pour les opérations CRUD sur la base de données.
"""

import sys
sys.path.insert(0, r"d:\Téléchargements\projet_actuariat\backend")

import pytest
from datetime import datetime

from app.models.policy import Policy
from app.models.prediction import Prediction
from app.models.triangle import ReservingRun
from app.models.user import User


class TestUserModel:
    """Tests pour le modèle User (nécessite passlib)."""

    def test_create_user(self, db_session):
        pytest.importorskip("passlib", reason="passlib requis pour le hachage")
        from app.auth import hash_password

        user = User(
            username="actuaire",
            email="actuaire@test.com",
            hashed_password=hash_password("secret123"),
            full_name="Marie Dupont",
            role="analyst",
        )
        db_session.add(user)
        db_session.commit()

        result = db_session.query(User).first()
        assert result.username == "actuaire"
        assert result.role == "analyst"
        assert result.is_active is True


class TestPolicyModel:
    """Tests pour le modèle Policy."""

    def test_create_policy(self, db_session):
        policy = Policy(
            veh_power=6, veh_age=5, veh_brand="B1", veh_gas="Diesel",
            driv_age=35, bonus_malus=60, region="Ile-de-France",
            area="A", density=1000.0, exposure=1.0,
        )
        db_session.add(policy)
        db_session.commit()

        result = db_session.query(Policy).first()
        assert result is not None
        assert result.veh_power == 6
        assert result.veh_brand == "B1"
        assert result.created_at is not None

    def test_policy_default_exposure(self, db_session):
        policy = Policy(
            veh_power=5, veh_age=3, veh_brand="B2", veh_gas="Regular",
            driv_age=40, bonus_malus=80, region="Lyon",
            area="B", density=500.0,
        )
        db_session.add(policy)
        db_session.commit()
        assert policy.exposure == 1.0


class TestPredictionModel:
    """Tests pour le modèle Prediction."""

    def test_create_pricing_prediction(self, db_session):
        pred = Prediction(
            module="pricing",
            model_version="glm_poisson_gamma_v1",
            request_json={"veh_power": 6, "veh_age": 5},
            response_json={"glm_baseline": {"pure_premium": 120.5}},
            latency_ms=45.2,
        )
        db_session.add(pred)
        db_session.commit()

        result = db_session.query(Prediction).filter_by(module="pricing").first()
        assert result is not None
        assert result.module == "pricing"
        assert result.request_json["veh_power"] == 6
        assert result.latency_ms == 45.2

    def test_prediction_index_on_module(self, db_session):
        for mod in ["pricing", "fraud", "reserving"]:
            db_session.add(Prediction(
                module=mod,
                request_json={},
                response_json={},
            ))
        db_session.commit()

        assert db_session.query(Prediction).filter_by(module="pricing").count() == 1
        assert db_session.query(Prediction).filter_by(module="fraud").count() == 1


class TestReservingRunModel:
    """Tests pour le modèle ReservingRun."""

    def test_create_reserving_run(self, db_session):
        run = ReservingRun(
            grcode=353,
            evaluation_year=1997,
            ibnr_estimate=150000.0,
            mack_lower=120000.0,
            mack_upper=180000.0,
            conformal_lower=115000.0,
            conformal_upper=185000.0,
        )
        db_session.add(run)
        db_session.commit()

        result = db_session.query(ReservingRun).first()
        assert result.grcode == 353
        assert result.ibnr_estimate == 150000.0
        assert result.mack_lower == 120000.0


