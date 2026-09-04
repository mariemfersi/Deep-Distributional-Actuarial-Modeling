"""
Router pour l'API de provisionnement (IBNR).
"""

import time
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.prediction import Prediction
from app.models.triangle import ReservingRun
from app.schemas.reserving import (
    ReservingIbnrRequest,
    ReservingIbnrResponse,
    ReservingRequest,
    ReservingResponse,
)
from app.services.reserving_service import (
    predict_ibnr,
    predict_reserving,
    load_raw_reserving_data,
)
from app.services.mlflow_service import log_prediction

router = APIRouter(prefix="/reserving", tags=["reserving"])


@router.get("/companies")
def get_companies():
    """Liste des compagnies disponibles avec leurs GRCODE et noms."""
    try:
        df = load_raw_reserving_data()
        if "CompanyName" in df.columns:
            name_col = "CompanyName"
        elif "Company" in df.columns:
            name_col = "Company"
        else:
            companies = df[["GRCODE"]].drop_duplicates().sort_values("GRCODE")
            return [
                {"grcode": int(row["GRCODE"]), "name": f"Company {int(row['GRCODE'])}"}
                for _, row in companies.iterrows()
            ]

        companies = df[["GRCODE", name_col]].drop_duplicates().sort_values("GRCODE")
        return [
            {"grcode": int(row["GRCODE"]), "name": row[name_col]}
            for _, row in companies.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement compagnies: {e}")


@router.post("/ibnr", response_model=ReservingIbnrResponse)
def ibnr_endpoint(request: ReservingIbnrRequest, db: Session = Depends(get_db), _user: dict = Depends(get_current_user)):
    start = time.time()
    try:
        response = predict_ibnr(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start) * 1000

    # Persister la prédiction
    db.add(Prediction(
        module="reserving",
        model_version="mack_conformal_v1",
        request_json=request.model_dump(),
        response_json=response.model_dump(),
        latency_ms=round(latency_ms, 2),
    ))

    # Persister le run de provisionnement
    db.add(ReservingRun(
        grcode=request.grcode,
        evaluation_year=request.evaluation_year,
        ibnr_estimate=response.ibnr_estimate,
        mack_lower=response.mack_interval.get("lower_90") if response.mack_interval else None,
        mack_upper=response.mack_interval.get("upper_90") if response.mack_interval else None,
        conformal_lower=response.conformal_interval.get("lower_90") if response.conformal_interval else None,
        conformal_upper=response.conformal_interval.get("upper_90") if response.conformal_interval else None,
    ))
    db.commit()

    # Log MLflow (best-effort) — ne bloque jamais la réponse
    log_prediction(
        module="reserving",
        request_data=request.model_dump(),
        response_data=response.model_dump(),
        latency_ms=latency_ms,
        model_version="mack_conformal_v1",
    )
    return response


@router.post("/predict", response_model=ReservingResponse)
def predict_deep_triangle(request: ReservingRequest, db: Session = Depends(get_db), _user: dict = Depends(get_current_user)):
    start = time.time()
    try:
        response = predict_reserving(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start) * 1000
    db.add(Prediction(
        module="reserving",
        model_version="deep_triangle_v1",
        request_json=request.model_dump(),
        response_json=response.model_dump(),
        latency_ms=round(latency_ms, 2),
    ))
    db.commit()

    # Log MLflow (best-effort) — ne bloque jamais la réponse
    log_prediction(
        module="reserving",
        request_data=request.model_dump(),
        response_data=response.model_dump(),
        latency_ms=latency_ms,
        model_version="deep_triangle_v1",
    )
    return response
