"""
Unit tests for Reserving module.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from app.services.reserving_service import (
    predict_ibnr,
    predict_reserving,
    _load_deep_triangle_model,
)
from app.routers.reserving import get_companies
from app.schemas.reserving import ReservingRequest


from app.schemas.reserving import ReservingRequest, ReservingIbnrRequest


class TestReservingSchemas:
    """Test Reserving schema validation."""

    def test_valid_ibnr_request(self):
        """Test that a valid IBNR request passes validation."""
        request = ReservingIbnrRequest(grcode=100, evaluation_year=2005)
        assert request.grcode == 100
        assert request.evaluation_year == 2005

    def test_invalid_grcode(self):
        """Test that invalid grcode raises validation error."""
        with pytest.raises(ValueError):
            ReservingIbnrRequest(grcode=-1, evaluation_year=2020)

    def test_valid_deep_triangle_request(self):
        """Test that a valid Deep Triangle request passes validation."""
        request = ReservingRequest(
            observed_increments=[100, 200, 150],
            premium=10000
        )
        assert len(request.observed_increments) == 3
        assert request.premium == 10000


class TestGetCompanies:
    """Test company list retrieval."""

    def test_get_companies_returns_list(self):
        """Test that get_companies returns a list of companies."""
        companies = get_companies()

        assert isinstance(companies, list)
        assert len(companies) > 0

        # Check structure
        for company in companies:
            assert "grcode" in company
            assert "name" in company
            assert isinstance(company["grcode"], int)
            assert isinstance(company["name"], str)


class TestPredictIBNR:
    """Test IBNR prediction."""

    def test_predict_ibnr_returns_valid_structure(self):
        """Test that IBNR prediction returns expected structure."""
        request = ReservingIbnrRequest(grcode=353, evaluation_year=1997)  # State Farm
        result = predict_ibnr(request)

        # Check required fields
        assert hasattr(result, "grcode")
        assert hasattr(result, "ibnr_estimate")
        assert hasattr(result, "mack_interval")
        assert hasattr(result, "conformal_interval")
        assert hasattr(result, "model_version")

        # Check interval structure
        assert "lower_90" in result.mack_interval
        assert "upper_90" in result.mack_interval
        assert "empirical_coverage" in result.mack_interval

        assert "lower_90" in result.conformal_interval
        assert "upper_90" in result.conformal_interval
        assert "empirical_coverage" in result.conformal_interval

        # Values should be reasonable
        assert result.ibnr_estimate > 0
        assert result.mack_interval["lower_90"] < result.mack_interval["upper_90"]
        assert result.conformal_interval["lower_90"] < result.conformal_interval["upper_90"]
        assert 0 <= result.mack_interval["empirical_coverage"] <= 1
        assert 0 <= result.conformal_interval["empirical_coverage"] <= 1

    def test_predict_ibnr_conformal_wider_than_mack(self):
        """Test that conformal intervals are typically wider than Mack."""
        request = ReservingIbnrRequest(grcode=353, evaluation_year=1997)
        result = predict_ibnr(request)

        mack_width = result.mack_interval["upper_90"] - result.mack_interval["lower_90"]
        conformal_width = result.conformal_interval["upper_90"] - result.conformal_interval["lower_90"]

        # Conformal should be wider (or equal) for guaranteed coverage
        assert conformal_width >= mack_width


class TestDeepTriangle:
    """Test Deep Triangle GRU predictions."""

    def test_predict_reserving_structure(self):
        """Test that Deep Triangle prediction returns expected structure."""
        request = ReservingRequest(
            observed_increments=[100, 200, 150, 80, 50],
            premium=10000
        )
        result = predict_reserving(request)

        # Check required fields (ReservingResponse is a Pydantic model)
        assert hasattr(result, "predicted_future_increments")
        assert hasattr(result, "predicted_ibnr")
        assert hasattr(result, "model_used")

        # Future increments should be a list
        assert isinstance(result.predicted_future_increments, list)
        assert len(result.predicted_future_increments) > 0

        # All increments should be positive
        for inc in result.predicted_future_increments:
            assert inc >= 0


class TestModelLoading:
    """Test that models load correctly."""

    def test_load_deep_triangle_model(self):
        """Test that Deep Triangle model loads without error."""
        model, scaler = _load_deep_triangle_model()
        assert model is not None
        # scaler is None (not used in current implementation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])