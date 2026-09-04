"""
Router pour l'API de tarification.
"""

import time
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.prediction import Prediction
from app.schemas.pricing import PricingRequest, PricingResponse
from app.services.pricing_service import predict_pricing, get_severity_distribution, get_premium_with_copula
from app.services.mlflow_service import log_prediction

router = APIRouter(prefix="/pricing", tags=["pricing"])


@router.post("/predict", response_model=PricingResponse)
def predict(request: PricingRequest, db: Session = Depends(get_db), _user: dict = Depends(get_current_user)):
    start = time.time()
    try:
        response = predict_pricing(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start) * 1000
    db.add(Prediction(
        module="pricing",
        model_version=response.model_version,
        request_json=request.model_dump(),
        response_json=response.model_dump(),
        latency_ms=round(latency_ms, 2),
    ))
    db.commit()

    # Log MLflow (best-effort) — ne bloque jamais la réponse
    log_prediction(
        module="pricing",
        request_data=request.model_dump(),
        response_data=response.model_dump(),
        latency_ms=latency_ms,
        model_version=response.model_version,
    )
    return response


@router.post("/severity-distribution")
def severity_distribution(request: PricingRequest, _user: dict = Depends(get_current_user)):
    """Retourne la distribution de sévérité avec percentiles (NGBoost ou GLM Gamma)."""
    try:
        return get_severity_distribution(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/premium-copula")
def premium_with_copula(request: PricingRequest, _user: dict = Depends(get_current_user)):
    """Retourne la distribution de prime avec dépendance fréquence-sévérité via copule gaussienne."""
    try:
        return get_premium_with_copula(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
def explain(request: PricingRequest, _user: dict = Depends(get_current_user)):
    """Retourne les valeurs SHAP calculees reellement pour le GLM Poisson.

    Utilise le service d'explicabilite qui calcule analytiquement
    phi_j = coef_j * (x_j - E[x_j]) pour chaque feature du GLM.
    """
    from app.services.explainability_service import explain_pricing
    try:
        return explain_pricing(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")
