"""Schémas de validation pour l'API de détection de fraude."""

from pydantic import BaseModel, Field


class FraudRequest(BaseModel):
    week_of_month: int = Field(..., ge=1, le=5)
    age: int = Field(..., ge=16, le=100)
    fault: str = Field(..., description="'Policy Holder' ou 'Third Party'")
    policy_type: str
    vehicle_category: str
    base_policy: str
    address_change_claim: str = Field(..., description="ex. 'under 6 months', 'no change'")
    days_policy_claim: str
    driver_rating: int = Field(..., ge=1, le=4)
    deductible: float = Field(..., gt=0)


class FraudResponse(BaseModel):
    fraud_probability: float
    is_suspicious: bool
    feature_importance: dict = Field(description="Importance globale des features (Random Forest)")
    model_version: str = "random_forest_v1"
