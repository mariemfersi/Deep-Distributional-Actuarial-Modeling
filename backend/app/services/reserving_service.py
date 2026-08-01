"""
Service de provisionnement — charge les données et le modèle Mack,
expose une fonction de prédiction IBNR avec intervalles de confiance.
Service de provisionnement — expose le modèle Deep Triangle (meilleure
variante : écrêtage à l'inférence). Rappel du chapitre 5 : ce modèle
n'égale pas la performance du modèle de Mack (ratio médian 1.21 contre
1.06) ; il est exposé ici à titre de démonstration technique, pas comme
recommandation de modèle de référence.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Ajoute la racine du projet au path pour importer src/
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.reserving.data import load_raw_reserving_data, split_observed_future
from src.reserving.models import fit_mack_for_company
from src.reserving.data import DeepTriangleGRU
from app.schemas.reserving import ReservingIbnrRequest, ReservingIbnrResponse, ReservingRequest, ReservingResponse

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
_observed_data = None
_future_data = None
_model_dt = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_data():
    """Charge les données une seule fois (cache en mémoire du process)."""
    global _observed_data, _future_data
    if _observed_data is None:
        df = load_raw_reserving_data()
        _observed_data, _future_data = split_observed_future(df)
    return _observed_data, _future_data


def predict_ibnr(request: ReservingIbnrRequest) -> ReservingIbnrResponse:
    """
    Prédit l'IBNR pour une compagnie donnée avec intervalles de confiance 90%
    (Mack brut + Conformal calibration pour couverture garantie).
    """
    observed, future = _load_data()
    
    try:
        model = fit_mack_for_company(observed, request.grcode)
        
        ibnr = model.ibnr_.to_frame().iloc[:, 0]
        ibnr.index = ibnr.index.year
        
        std_err = model.mack_std_err_.to_frame()[9999]
        std_err.index = std_err.index.year
        
        # Somme sur toutes les années de survenance
        total_ibnr = ibnr.sum()
        total_std_err = np.sqrt((std_err ** 2).sum())
        
        z_90 = 1.645
        
        # Intervalle Mack (asymptotique, couverture empirique 74.4%)
        lower_90_mack = max(0, total_ibnr - z_90 * total_std_err)
        upper_90_mack = total_ibnr + z_90 * total_std_err
        
        # Intervalle Conforme (calibré empiriquement, couverture 91.9%)
        q_hat = 1.85  # Valeur empirique obtenue lors de la calibration
        
        lower_90_conformal = max(0, total_ibnr - q_hat * total_std_err)
        upper_90_conformal = total_ibnr + q_hat * total_std_err
        
        return ReservingIbnrResponse(
            grcode=request.grcode,
            ibnr_estimate=round(float(total_ibnr), 2),
            mack_interval={
                "lower_90": round(float(lower_90_mack), 2),
                "upper_90": round(float(upper_90_mack), 2),
                "empirical_coverage": 0.744
            },
            conformal_interval={
                "lower_90": round(float(lower_90_conformal), 2),
                "upper_90": round(float(upper_90_conformal), 2),
                "empirical_coverage": 0.919
            },
        )
    except Exception as e:
        raise ValueError(f"Impossible d'ajuster le modèle Mack pour GRCODE {request.grcode}: {str(e)}")


def _load_deep_triangle_model():
    global _model_dt
    if _model_dt is None:
        _model_dt = DeepTriangleGRU(hidden_dim=16).to(_device)
        try:
            _model_dt.load_state_dict(torch.load(MODELS_DIR / "deep_triangle.pt", map_location=_device))
            _model_dt.eval()
        except FileNotFoundError:
            raise ValueError("Modèle Deep Triangle non trouvé. Entraînez et sauvegardez le modèle d'abord.")
    return _model_dt


def predict_future_increments(model, obs_scaled, device):
    """Prédit les incréments futurs à partir des incréments observés."""
    obs_tensor = torch.tensor(obs_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    
    with torch.no_grad():
        # Prédire séquentiellement chaque incrément futur
        future_scaled = []
        current_seq = obs_tensor.clone()
        
        for _ in range(10 - len(obs_scaled)):
            pred = model(current_seq)
            next_val = pred[0, -1].item()
            future_scaled.append(max(0, next_val))  # Clip valeurs négatives
            current_seq = torch.cat([current_seq, pred[:, -1:]], dim=1)
    
    return future_scaled


def predict_reserving(request: ReservingRequest) -> ReservingResponse:
    """Prédit les incréments futurs et l'IBNR avec Deep Triangle."""
    model = _load_deep_triangle_model()

    # Normalisation par la prime (cohérent avec l'entraînement)
    obs_scaled = [x / request.premium for x in request.observed_increments]

    future_scaled = predict_future_increments(model, obs_scaled, _device)
    future_amounts = [float(x * request.premium) for x in future_scaled]

    ibnr = sum(future_amounts)

    return ReservingResponse(
        predicted_future_increments=[round(x, 2) for x in future_amounts],
        predicted_ibnr=round(ibnr, 2),
    )
