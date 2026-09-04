"""
Integration tests for API endpoints.
"""
import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_model_status(self):
        """Test model status endpoint."""
        response = client.get("/models/status")
        assert response.status_code == 200
        data = response.json()
        assert "pricing" in data
        assert "reserving" in data
        assert "fraud" in data
        assert data["pricing"]["loaded"] is True
        assert data["reserving"]["loaded"] is True
        assert data["fraud"]["loaded"] is True


class TestPricingEndpoints:
    """Test pricing API endpoints."""

    @pytest.fixture
    def sample_request(self):
        """Create a sample pricing request."""
        return {
            "veh_power": 6,
            "veh_age": 5,
            "driv_age": 35,
            "bonus_malus": 60,
            "veh_brand": "B1",
            "veh_gas": "Diesel",
            "region": "Ile-de-France",
            "area": "A",
            "density": 1000,
            "exposure": 1.0,
        }

    def test_predict_endpoint(self, sample_request):
        """Test /pricing/predict endpoint."""
        response = client.post("/pricing/predict", json=sample_request)
        assert response.status_code == 200

        data = response.json()
        assert "glm_baseline" in data
        assert "improved_model" in data
        assert "gini_improvement_pct" in data

        # Check values are present and positive
        assert data["glm_baseline"]["predicted_frequency"] > 0
        assert data["glm_baseline"]["predicted_severity"] > 0
        assert data["glm_baseline"]["pure_premium"] > 0

    def test_severity_distribution_endpoint(self, sample_request):
        """Test /pricing/severity-distribution endpoint."""
        response = client.post("/pricing/severity-distribution", json=sample_request)
        assert response.status_code == 200

        data = response.json()
        assert "model" in data
        assert "percentiles" in data
        assert all(p in data["percentiles"] for p in ["p5", "p25", "p50", "p75", "p95"])

    def test_premium_copula_endpoint(self, sample_request):
        """Test /pricing/premium-copula endpoint."""
        response = client.post("/pricing/premium-copula", json=sample_request)
        assert response.status_code == 200

        data = response.json()
        assert "premium_mean" in data
        assert "premium_std" in data
        assert "premium_percentiles" in data
        assert "copula_rho" in data

    def test_explain_endpoint(self, sample_request):
        """Test /pricing/explain endpoint."""
        response = client.post("/pricing/explain", json=sample_request)
        assert response.status_code == 200

        data = response.json()
        assert "base_value" in data
        assert "shap_values" in data
        assert isinstance(data["shap_values"], list)


class TestReservingEndpoints:
    """Test reserving API endpoints."""

    def test_companies_endpoint(self):
        """Test /reserving/companies endpoint."""
        response = client.get("/reserving/companies")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert all("grcode" in c and "name" in c for c in data)

    def test_ibnr_endpoint(self):
        """Test /reserving/ibnr endpoint."""
        response = client.post("/reserving/ibnr", json={"grcode": 353, "evaluation_year": 1997})
        assert response.status_code == 200

        data = response.json()
        assert "ibnr_estimate" in data
        assert "mack_interval" in data
        assert "conformal_interval" in data
        assert data["ibnr_estimate"] > 0

    def test_deep_triangle_endpoint(self):
        """Test /reserving/predict endpoint."""
        response = client.post("/reserving/predict", json={
            "observed_increments": [100, 200, 150],
            "premium": 10000
        })
        assert response.status_code == 200

        data = response.json()
        assert "predicted_future_increments" in data
        assert "predicted_ibnr" in data
        assert isinstance(data["predicted_future_increments"], list)


class TestFraudEndpoints:
    """Test fraud API endpoints."""

    @pytest.fixture
    def sample_request(self):
        """Create a sample fraud request."""
        return {
            "week_of_month": 2,
            "age": 35,
            "fault": "Policy Holder",
            "policy_type": "Sport - Liability",
            "vehicle_category": "Sport",
            "base_policy": "Liability",
            "address_change_claim": "1 year",
            "days_policy_claim": "more than 30",
            "driver_rating": 3,
            "deductible": 400,
        }

    def test_predict_endpoint(self, sample_request):
        """Test /fraud/predict endpoint."""
        response = client.post("/fraud/predict", json=sample_request)
        assert response.status_code == 200

        data = response.json()
        assert "fraud_probability" in data
        assert "is_suspicious" in data
        assert "feature_importance" in data
        assert 0 <= data["fraud_probability"] <= 1
        assert isinstance(data["is_suspicious"], bool)

    def test_methodology_endpoint(self):
        """Test /fraud/methodology endpoint."""
        response = client.get("/fraud/methodology")
        assert response.status_code == 200

        data = response.json()
        assert "graph_attempts" in data
        assert "final_model" in data
        assert len(data["graph_attempts"]) == 4


class TestExplainabilityEndpoints:
    """Test explainability API endpoints."""

    @pytest.fixture
    def pricing_request(self):
        return {
            "veh_power": 6,
            "veh_age": 5,
            "driv_age": 35,
            "bonus_malus": 60,
            "veh_brand": "B1",
            "veh_gas": "Diesel",
            "region": "Ile-de-France",
            "area": "A",
            "density": 1000,
            "exposure": 1.0,
        }

    @pytest.fixture
    def fraud_request(self):
        return {
            "week_of_month": 2,
            "age": 35,
            "fault": "Policy Holder",
            "policy_type": "Sport - Liability",
            "vehicle_category": "Sport",
            "base_policy": "Liability",
            "address_change_claim": "1 year",
            "days_policy_claim": "more than 30",
            "driver_rating": 3,
            "deductible": 400,
        }

    def test_explain_pricing(self, pricing_request):
        """Test /explain/pricing endpoint."""
        response = client.post("/explain/pricing", json=pricing_request)
        assert response.status_code == 200

        data = response.json()
        assert "base_value" in data
        assert "shap_values" in data

    def test_explain_fraud(self, fraud_request):
        """Test /explain/fraud endpoint."""
        response = client.post("/explain/fraud", json=fraud_request)
        assert response.status_code == 200

        data = response.json()
        assert "base_value" in data
        assert "shap_values" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])