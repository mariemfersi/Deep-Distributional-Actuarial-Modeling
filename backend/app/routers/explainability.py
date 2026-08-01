"""
Router pour l'API d'explicabilité (SHAP).
"""

from fastapi import APIRouter, HTTPException
from app.schemas.explainability import (
    PricingExplanationRequest, 
    FraudExplanationRequest, 
    ShapExplanation
)
from app.services.explainability_service import explain_pricing, explain_fraud

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.post("/pricing", response_model=ShapExplanation)
async def explain_pricing_prediction(request: PricingExplanationRequest):
    """
    Retourne les valeurs SHAP pour une prédiction de tarification.
    
    Les valeurs SHAP indiquent l'impact de chaque feature sur la prédiction
    par rapport à la valeur de base (prédiction moyenne).
    """
    try:
        # Convertir la requête en PricingRequest pour le service
        from app.schemas.pricing import PricingRequest
        pricing_request = PricingRequest(
            veh_power=request.veh_power,
            veh_age=request.veh_age,
            driv_age=request.driv_age,
            bonus_malus=request.bonus_malus,
            veh_brand=request.veh_brand,
            veh_gas=request.veh_gas,
            region=request.region,
            area=request.area,
            density=request.density,
            exposure=request.exposure
        )
        
        shap_result = explain_pricing(pricing_request)
        return ShapExplanation(**shap_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fraud", response_model=ShapExplanation)
async def explain_fraud_prediction(request: FraudExplanationRequest):
    """
    Retourne les valeurs SHAP pour une prédiction de fraude.
    
    Les valeurs SHAP indiquent l'impact de chaque feature sur la probabilité
    de fraude par rapport à la valeur de base.
    """
    try:
        request_data = {
            "fault": request.fault,
            "policy_type": request.policy_type,
            "vehicle_category": request.vehicle_category,
            "base_policy": request.base_policy,
            "address_change_claim": request.address_change_claim,
            "days_policy_claim": request.days_policy_claim,
            "driver_rating": request.driver_rating,
            "deductible": request.deductible,
            "week_of_month": request.week_of_month,
            "age": request.age
        }
        
        shap_result = explain_fraud(request_data)
        return ShapExplanation(**shap_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))