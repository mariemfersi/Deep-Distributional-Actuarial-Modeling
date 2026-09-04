"""
Unit tests for Pricing module.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from app.services.pricing_service import (
    predict_pricing,
    get_severity_distribution,
    get_premium_with_copula,
    _build_feature_row,
)
from app.schemas.pricing import PricingRequest


class TestPricingRequest:
    """Test PricingRequest schema validation."""

    def test_valid_request(self):
        """Test that a valid request passes validation."""
        request = PricingRequest(
            veh_power=6,
            veh_age=5,
            driv_age=35,
            bonus_malus=60,
            veh_brand="B1",
            veh_gas="Diesel",
            region="Ile-de-France",
            area="A",
            density=1000,
            exposure=1.0,
        )
        assert request.veh_power == 6
        assert request.veh_gas == "Diesel"

    def test_invalid_veh_power(self):
        """Test that invalid veh_power raises validation error."""
        with pytest.raises(ValueError):
            PricingRequest(
                veh_power=0,  # Must be >= 1
                veh_age=5,
                driv_age=35,
                bonus_malus=60,
                veh_brand="B1",
                veh_gas="Diesel",
                region="Ile-de-France",
                area="A",
                density=1000,
                exposure=1.0,
            )

    def test_invalid_driv_age(self):
        """Test that invalid driv_age raises validation error."""
        with pytest.raises(ValueError):
            PricingRequest(
                veh_power=6,
                veh_age=5,
                driv_age=15,  # Must be >= 16
                bonus_malus=60,
                veh_brand="B1",
                veh_gas="Diesel",
                region="Ile-de-France",
                area="A",
                density=1000,
                exposure=1.0,
            )


class TestBuildFeatureRow:
    """Test feature row construction."""

    def test_build_feature_row(self):
        """Test that feature row has expected columns."""
        request = PricingRequest(
            veh_power=6,
            veh_age=5,
            driv_age=35,
            bonus_malus=60,
            veh_brand="B1",
            veh_gas="Diesel",
            region="Ile-de-France",
            area="A",
            density=1000,
            exposure=1.0,
        )
        row = _build_feature_row(request)

        # Check GLM bucket columns exist
        assert "DrivAge_bucket" in row.columns
        assert "VehAge_bucket" in row.columns
        assert "BM_bucket" in row.columns
        assert "Density_log" in row.columns

        # Check CANN normalization columns exist
        assert "VehPower_norm" in row.columns
        assert "VehAge_norm" in row.columns
        assert "VehGas_code" in row.columns
        assert "VehBrand_code" in row.columns

        # Check values are reasonable
        assert row["Density_log"].iloc[0] == np.log(1000)
        assert row["VehGas_code"].iloc[0] == 0  # Diesel = 0 (convention d'entraînement: Regular=1)


class TestPredictPricing:
    """Test pricing prediction endpoint logic."""

    @pytest.fixture
    def sample_request(self):
        """Create a sample pricing request."""
        return PricingRequest(
            veh_power=6,
            veh_age=5,
            driv_age=35,
            bonus_malus=60,
            veh_brand="B1",
            veh_gas="Diesel",
            region="Ile-de-France",
            area="A",
            density=1000,
            exposure=1.0,
        )

    def test_predict_pricing_returns_valid_structure(self, sample_request):
        """Test that prediction returns expected structure."""
        result = predict_pricing(sample_request)

        # Check required fields exist
        assert hasattr(result, "glm_baseline")
        assert hasattr(result, "improved_model")
        assert hasattr(result, "gini_improvement_pct")

        # Check GLM baseline
        assert "predicted_frequency" in result.glm_baseline
        assert "predicted_severity" in result.glm_baseline
        assert "pure_premium" in result.glm_baseline

        # Check improved model
        assert "predicted_frequency" in result.improved_model
        assert "predicted_severity" in result.improved_model
        assert "pure_premium" in result.improved_model

        # Check values are positive
        assert result.glm_baseline["predicted_frequency"] > 0
        assert result.glm_baseline["predicted_severity"] > 0
        assert result.glm_baseline["pure_premium"] > 0

    def test_predict_pricing_different_profiles(self):
        """Test predictions for different risk profiles."""
        # Young driver, high power
        request_young = PricingRequest(
            veh_power=10,
            veh_age=0,
            driv_age=22,
            bonus_malus=100,
            veh_brand="B10",
            veh_gas="Regular",
            region="Alsace",
            area="B",
            density=500,
            exposure=1.0,
        )

        # Experienced driver, low power
        request_experienced = PricingRequest(
            veh_power=4,
            veh_age=10,
            driv_age=55,
            bonus_malus=50,
            veh_brand="B6",
            veh_gas="Diesel",
            region="Provence-Alpes-Cotes-D'Azur",
            area="C",
            density=2000,
            exposure=0.5,
        )

        result_young = predict_pricing(request_young)
        result_experienced = predict_pricing(request_experienced)

        # Young driver should have higher frequency
        assert result_young.glm_baseline["predicted_frequency"] > result_experienced.glm_baseline["predicted_frequency"]


class TestSeverityDistribution:
    """Test severity distribution endpoint."""

    def test_get_severity_distribution(self):
        """Test that severity distribution returns percentiles."""
        request = PricingRequest(
            veh_power=6,
            veh_age=5,
            driv_age=35,
            bonus_malus=60,
            veh_brand="B1",
            veh_gas="Diesel",
            region="Ile-de-France",
            area="A",
            density=1000,
            exposure=1.0,
        )

        result = get_severity_distribution(request)

        assert "model" in result
        assert "percentiles" in result
        assert "p5" in result["percentiles"]
        assert "p25" in result["percentiles"]
        assert "p50" in result["percentiles"]
        assert "p75" in result["percentiles"]
        assert "p95" in result["percentiles"]

        # Percentiles should be ordered
        p = result["percentiles"]
        assert p["p5"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p95"]


class TestPremiumCopula:
    """Test premium with copula endpoint."""

    def test_get_premium_with_copula(self):
        """Test that copula premium returns distribution statistics."""
        request = PricingRequest(
            veh_power=6,
            veh_age=5,
            driv_age=35,
            bonus_malus=60,
            veh_brand="B1",
            veh_gas="Diesel",
            region="Ile-de-France",
            area="A",
            density=1000,
            exposure=1.0,
        )

        result = get_premium_with_copula(request)

        assert "premium_mean" in result
        assert "premium_std" in result
        assert "premium_percentiles" in result
        assert "copula_rho" in result
        assert "frequency_mean" in result
        assert "severity_mean" in result

        # Percentiles should be ordered
        p = result["premium_percentiles"]
        assert p["p5"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p95"]


class TestShapExplain:
    """Vérifie que l'explication SHAP du GLM est authentique et cohérente."""

    def test_shap_dimension_alignment(self):
        """Le design matrix doit avoir exactement autant de colonnes que de
        coefficients GLM (régression sur le bug de mismatch 47 vs 46)."""
        from app.services.explainability_service import (
            _build_design_matrix_row, _get_glm_coefficients,
        )
        request = PricingRequest(
            veh_power=12, veh_age=5, driv_age=35, bonus_malus=80,
            veh_brand="B10", veh_gas="Regular", region="Aquitaine",
            area="C", density=1000, exposure=1.0,
        )
        X = _build_design_matrix_row(request)
        params = _get_glm_coefficients()
        assert X.shape[1] == len(params)
        assert set(X.columns) == set(params.index)

    def test_shap_internal_consistency(self):
        """base_value + somme(SHAP) doit valoir X @ params (décomposition exacte
        pour un modèle linéaire avec coeff analytiques)."""
        from app.services.explainability_service import (
            explain_pricing, _build_design_matrix_row, _get_glm_coefficients,
        )
        request = PricingRequest(
            veh_power=12, veh_age=5, driv_age=35, bonus_malus=80,
            veh_brand="B10", veh_gas="Regular", region="Aquitaine",
            area="C", density=1000, exposure=1.0,
        )
        result = explain_pricing(request)
        log_pred = float((_build_design_matrix_row(request).values
                          @ _get_glm_coefficients().values).sum())
        shap_sum = sum(item["value"] for item in result["shap_values"])
        # Tolérance pour arrondi à 4 décimales des valeurs SHAP regroupées
        assert abs(result["base_value"] + shap_sum - log_pred) < 1e-2

    def test_shap_matches_prediction(self):
        """exp(log_pred) doit rejoindre la prédiction de fréquence du service."""
        from app.services.explainability_service import (
            _build_design_matrix_row, _get_glm_coefficients,
        )
        import numpy as np
        request = PricingRequest(
            veh_power=12, veh_age=5, driv_age=35, bonus_malus=80,
            veh_brand="B10", veh_gas="Regular", region="Aquitaine",
            area="C", density=1000, exposure=1.0,
        )
        log_pred = float((_build_design_matrix_row(request).values
                          @ _get_glm_coefficients().values).sum())
        freq = predict_pricing(request).glm_baseline["predicted_frequency"]
        assert abs(np.exp(log_pred) - freq) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])