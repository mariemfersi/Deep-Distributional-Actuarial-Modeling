"""
Unit tests for Fraud module.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from app.services.fraud_service import (
    predict_fraud,
    _load_model,
)
from app.routers.fraud import get_methodology
from app.schemas.fraud import FraudRequest


class TestFraudRequest:
    """Test FraudRequest schema validation."""

    def test_valid_request(self):
        """Test that a valid request passes validation."""
        request = FraudRequest(
            week_of_month=2,
            age=35,
            fault="Policy Holder",
            policy_type="Sport - Liability",
            vehicle_category="Sport",
            base_policy="Liability",
            address_change_claim="1 year",
            days_policy_claim="more than 30",
            driver_rating=3,
            deductible=400,
        )
        assert request.week_of_month == 2
        assert request.fault == "Policy Holder"

    def test_invalid_week_of_month(self):
        """Test that invalid week_of_month raises validation error."""
        with pytest.raises(ValueError):
            FraudRequest(
                week_of_month=6,  # Must be 1-5
                age=35,
                fault="Policy Holder",
                policy_type="Sport - Liability",
                vehicle_category="Sport",
                base_policy="Liability",
                address_change_claim="1 year",
                days_policy_claim="more than 30",
                driver_rating=3,
                deductible=400,
            )

    def test_invalid_driver_rating(self):
        """Test that invalid driver_rating raises validation error."""
        with pytest.raises(ValueError):
            FraudRequest(
                week_of_month=2,
                age=35,
                fault="Policy Holder",
                policy_type="Sport - Liability",
                vehicle_category="Sport",
                base_policy="Liability",
                address_change_claim="1 year",
                days_policy_claim="more than 30",
                driver_rating=5,  # Must be 1-4
                deductible=400,
            )


class TestBuildFraudFeatures:
    """Test fraud feature construction."""

    def test_predict_fraud_returns_valid_structure(self):
        """Test that prediction returns expected structure."""
        request = FraudRequest(
            week_of_month=2,
            age=35,
            fault="Policy Holder",
            policy_type="Sport - Liability",
            vehicle_category="Sport",
            base_policy="Liability",
            address_change_claim="1 year",
            days_policy_claim="more than 30",
            driver_rating=3,
            deductible=400,
        )

        result = predict_fraud(request)

        # Check required fields
        assert hasattr(result, "fraud_probability")
        assert hasattr(result, "is_suspicious")
        assert hasattr(result, "feature_importance")
        assert hasattr(result, "model_version")

        # Check value ranges
        assert 0 <= result.fraud_probability <= 1
        assert isinstance(result.is_suspicious, bool)
        assert isinstance(result.feature_importance, dict)
        assert len(result.feature_importance) > 0


class TestPredictFraud:
    """Test fraud prediction."""

    @pytest.fixture
    def sample_request(self):
        """Create a sample fraud request."""
        return FraudRequest(
            week_of_month=2,
            age=35,
            fault="Policy Holder",
            policy_type="Sport - Liability",
            vehicle_category="Sport",
            base_policy="Liability",
            address_change_claim="1 year",
            days_policy_claim="more than 30",
            driver_rating=3,
            deductible=400,
        )

    def test_predict_fraud_returns_valid_structure(self, sample_request):
        """Test that prediction returns expected structure."""
        result = predict_fraud(sample_request)

        # Check required fields
        assert hasattr(result, "fraud_probability")
        assert hasattr(result, "is_suspicious")
        assert hasattr(result, "feature_importance")
        assert hasattr(result, "model_version")

        # Check value ranges
        assert 0 <= result.fraud_probability <= 1
        assert isinstance(result.is_suspicious, bool)
        assert isinstance(result.feature_importance, dict)
        assert len(result.feature_importance) > 0

    def test_predict_fraud_high_risk_profile(self):
        """Test prediction for high-risk profile."""
        # Profile with characteristics associated with fraud
        request = FraudRequest(
            week_of_month=1,
            age=22,
            fault="Third Party",
            policy_type="Sedan - All Perils",
            vehicle_category="Sport",
            base_policy="Collision",
            address_change_claim="1 year",
            days_policy_claim="1 to 7",
            driver_rating=1,
            deductible=100,
        )

        result = predict_fraud(request)

        assert 0 <= result.fraud_probability <= 1

    def test_predict_fraud_low_risk_profile(self):
        """Test prediction for low-risk profile."""
        # Profile with characteristics associated with legitimate claims
        request = FraudRequest(
            week_of_month=3,
            age=50,
            fault="Policy Holder",
            policy_type="Sport - Liability",
            vehicle_category="Utility",
            base_policy="Liability",
            address_change_claim="no change",
            days_policy_claim="more than 30",
            driver_rating=4,
            deductible=1000,
        )

        result = predict_fraud(request)

        assert 0 <= result.fraud_probability <= 1


class TestMethodology:
    """Test methodology endpoint."""

    def test_get_methodology_returns_structure(self):
        """Test that methodology returns expected structure."""
        result = get_methodology()

        assert "graph_attempts" in result
        assert "final_model" in result

        # Should document 4 failed attempts
        assert len(result["graph_attempts"]) == 4

        # Le benchmark doit contenir les 7 variantes et le meilleur modèle
        # est XGB + SMOTE (résultat mesuré, non codé en dur)
        assert len(result.get("benchmark_comparison", [])) == 7
        assert "XGB + SMOTE" in result["final_model"]["model"]

    def test_get_methodology_metrics_from_artifact(self):
        """Les chiffres du benchmark proviennent de fraud_metrics.json (aucune valeur codée en dur)."""
        from app.routers.fraud import _load_fraud_metrics
        m = _load_fraud_metrics()
        assert "benchmark" in m
        assert m["leakage_handling"] == "preprocessor_fit_on_train_only"
        # La colonne 'best' est marquée une seule fois
        best_count = sum(1 for r in m["benchmark"] if r.get("best"))
        assert best_count == 1
        # Les 7 variantes sont présentes
        assert len(m["benchmark"]) == 7


class TestNoDataLeakage:
    """Vérifie que le préprocessing fraude n'a pas de fuite de données
    (stats ajustées sur le train seul, appliquées au test)."""

    def test_preprocessor_fit_transform(self):
        """fit_fraud_preprocessor + apply_fraud_preprocessor doivent permettre
        d'ajuster sur train puis transformer train ET test sans fuite."""
        from src.fraud.data import (
            load_fraud_data, train_test_split_fraud,
            fit_fraud_preprocessor, apply_fraud_preprocessor, NUMERIC_COLS,
        )
        df = load_fraud_data()
        train_raw, test_raw = train_test_split_fraud(df, seed=123)
        encoders, norm_stats = fit_fraud_preprocessor(train_raw)
        train_f = apply_fraud_preprocessor(train_raw, encoders, norm_stats)
        test_f = apply_fraud_preprocessor(test_raw, encoders, norm_stats)

        # Les stats de normalisation du test doivent être celles du TRAIN,
        # pas celles du test lui-même.
        for col in NUMERIC_COLS:
            train_mean, train_std = norm_stats[col]
            test_own_mean = test_raw[col].mean()
            assert abs(train_mean - test_own_mean) > 0.0 or abs(train_std - test_raw[col].std()) > 0.0
            # La normalisation appliquée utilise (x - train_mean) / train_std
            expected = (test_raw[col] - train_mean) / train_std
            assert np.allclose(test_f[f"{col}_norm"], expected, atol=1e-6)

        # fraud_label présent sur les deux splits
        assert "fraud_label" in train_f.columns
        assert "fraud_label" in test_f.columns


class TestModelLoading:
    """Test that models load correctly."""

    def test_load_model(self):
        """Test that fraud model loads without error."""
        model, encoders, norm_stats, defaults, model_type = _load_model()

        assert model is not None
        assert encoders is not None
        assert norm_stats is not None
        assert defaults is not None
        assert model_type in ("best", "rf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])