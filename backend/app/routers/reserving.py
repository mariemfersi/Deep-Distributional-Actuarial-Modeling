from fastapi import APIRouter, HTTPException

from app.schemas.reserving import ReservingIbnrRequest, ReservingIbnrResponse, ReservingRequest, ReservingResponse
from app.services.reserving_service import predict_ibnr, predict_reserving, load_raw_reserving_data

router = APIRouter(prefix="/reserving", tags=["reserving"])


@router.get("/companies")
def get_companies():
    """Liste des compagnies disponibles avec leurs GRCODE et noms."""
    try:
        df = load_raw_reserving_data()
        # Check available columns and use the correct one
        if "CompanyName" in df.columns:
            name_col = "CompanyName"
        elif "Company" in df.columns:
            name_col = "Company"
        else:
            # Fallback to GRCODE only
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
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement des compagnies: {str(e)}")


@router.post("/ibnr", response_model=ReservingIbnrResponse)
def predict(request: ReservingIbnrRequest):
    try:
        return predict_ibnr(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=ReservingResponse)
def predict_deep_triangle(request: ReservingRequest):
    try:
        return predict_reserving(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
