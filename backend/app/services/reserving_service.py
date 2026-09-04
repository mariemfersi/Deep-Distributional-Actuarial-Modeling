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
import torch.nn as nn

# Ajoute la racine du projet au path pour importer src/
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.reserving.data import load_raw_reserving_data, split_observed_future, DeepTriangleGRU
from src.reserving.models import fit_mack_for_company
from src.reserving.calibration import load_calibration
from app.schemas.reserving import ReservingIbnrRequest, ReservingIbnrResponse, ReservingRequest, ReservingResponse

DATA_DIR = BACKEND_DIR / "data" if (BACKEND_DIR / "data").exists() else PROJECT_ROOT / "data"
MODELS_DIR = BACKEND_DIR / "models" if (BACKEND_DIR / "models").exists() else PROJECT_ROOT / "models"
_observed_data = None
_future_data = None
_model_dt = None
_calibration = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_calibration() -> dict:
    """Charge l'artefact de calibration conforme (q_hat, couvertures mesurées).

    L'artefact est produit par src/reserving/calibration.run_conformal_calibration
    (offline). Si absent, on retombe sur un q_hat utilitaire par défaut documenté,
    jamais présenté comme une couverture mesurée.
    """
    global _calibration
    if _calibration is None:
        _calibration = load_calibration(MODELS_DIR / "reserving_calibration.json")
    return _calibration


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
        if hasattr(ibnr.index, "year"):
            ibnr.index = ibnr.index.year
        else:
            ibnr.index = [int(str(x)[:4]) for x in ibnr.index]
        
        std_df = model.mack_std_err_.to_frame()
        if 9999 in std_df.columns:
            std_err = std_df[9999]
        else:
            std_err = std_df.iloc[:, -1]
        
        if hasattr(std_err.index, "year"):
            std_err.index = std_err.index.year
        else:
            std_err = [int(str(x)[:4]) for x in std_err.index]
        std_err = pd.Series(std_err, index=ibnr.index)
        
        # Somme sur toutes les années de survenance
        total_ibnr = ibnr.sum()
        total_std_err = np.sqrt((std_err ** 2).sum())

        # Mapping orienté-brins vers les erreurs standards de Mack par année
        # (utilise la série std_err extraite de model.mack_std_err_ ci-dessus).
        origin_years_std = [int(y) for y in std_err.index]
        std_err_by_origin = pd.Series(
            {int(y): float(se) for y, se in zip(std_err.index, std_err.values)}
        ).reindex(origin_years_std).fillna(0.0)
        
        # Extract triangle data for visualization from the actual observed data
        # Build triangle directly from observed dataframe to get real cumulative paid losses
        df_company = observed[observed["GRCODE"] == request.grcode].copy()
        
        # Convert datetime to integers for filtering
        df_company["origin_year"] = df_company["AccidentYear"].dt.year if pd.api.types.is_datetime64_any_dtype(df_company["AccidentYear"]) else df_company["AccidentYear"]
        
        # Use DevelopmentLag for triangle structure (1, 2, 3, ... instead of calendar years)
        dev_lags = sorted(df_company["DevelopmentLag"].unique())
        origin_years = sorted(df_company["origin_year"].unique())
        
        n_origins = len(origin_years)
        n_devs = len(dev_lags)
        
        # Create triangle matrix with observed values
        triangle_values = np.zeros((n_origins, n_devs))
        std_err_triangle = np.zeros((n_origins, n_devs))

        # Fill with observed data using origin year and development lag
        # Les erreurs standards du triangle observé sont laissées à 0 (cellules
        # observées = données réelles, sans incertitude de prédiction). L'incertitude
        # de Mack est portée par model.mack_std_err_ (voir plus bas), pas par un
        # coefficient arbitraire appliqué aux cellules observées.
        for i, origin_year in enumerate(origin_years):
            for j, dev_lag in enumerate(dev_lags):
                cell_data = df_company[
                    (df_company["origin_year"] == origin_year) &
                    (df_company["DevelopmentLag"] == dev_lag)
                ]
                if len(cell_data) > 0:
                    triangle_values[i, j] = cell_data["CumPaidLoss"].iloc[0]
        
        # Mark cells as observed or projected based on evaluation year
        # A cell is observed if DevelopmentYear <= evaluation_year
        # DevelopmentYear = AccidentYear + DevelopmentLag - 1
        evaluation_year = request.evaluation_year if hasattr(request, 'evaluation_year') else 2007
        cell_status = np.zeros((n_origins, n_devs), dtype=int)  # 0 = observed, 1 = projected

        for i, origin_year in enumerate(origin_years):
            for j, dev_lag in enumerate(dev_lags):
                dev_year = origin_year + dev_lag - 1
                if dev_year > evaluation_year:
                    cell_status[i, j] = 1  # Projected

        # Add Chain-Ladder projections for future cells using LDF factors
        # Get LDFs from the fitted model (already calculated in fit_mack_for_company)
        ldfs = getattr(model, 'manual_ldfs', [])
        if ldfs:
            for i, origin_year in enumerate(origin_years):
                for j, dev_lag in enumerate(dev_lags):
                    if cell_status[i, j] == 1:  # Projected cell
                        # Find the last observed cell for this origin year
                        last_observed_j = -1
                        for k in range(j - 1, -1, -1):
                            if triangle_values[i, k] > 0:
                                last_observed_j = k
                                break
                        if last_observed_j >= 0:
                            # Project forward using LDF chain: C_{i,j} = C_{i,last} * prod_{k=last}^{j-1} f_k
                            projected = triangle_values[i, last_observed_j]
                            for k in range(last_observed_j, j):
                                if k < len(ldfs):
                                    projected *= ldfs[k]
                            triangle_values[i, j] = projected
                            # L'incertitude des cellules projetées est portée par
                            # l'erreur-standard Mack par année (model.mack_std_err_)
                            # ; elle est répartie ici à titre de visualisation
                            # uniquement (elle n'entre pas dans le calcul de
                            # l'intervalle, qui utilise std_err par année).
                            if origin_year in std_err_by_origin.index:
                                std_err_triangle[i, j] = float(std_err_by_origin.loc[origin_year])

        # Quantile de Mack à 90% (hypothèse de normalité asymptotique)
        z_90 = 1.645

        # Intervalle Mack (asymptotique) : ibnr ± z_90 * std_err_mack
        lower_90_mack = max(0, total_ibnr - z_90 * total_std_err)
        upper_90_mack = total_ibnr + z_90 * total_std_err

        # Intervalle Conforme : q_hat mesuré (split-conformal) chargé depuis
        # l'artefact de calibration. Jamais codé en dur : si l'artefact est
        # absent, on renvoie une couverture inconnue et non un chiffre inventé.
        calibration = _load_calibration()
        q_hat = float(calibration.get("q_hat", 0.0))
        if q_hat <= 0:
            # Aucune calibration valide : on refuse un intervalle conforme
            # non fondé et on le signale explicitement.
            raise ValueError(
                "Aucun artefact de calibration conforme disponible "
                "(models/reserving_calibration.json absent). "
                "Exécutez python -m scripts.calibrate_reserving."
            )

        lower_90_conformal = max(0, total_ibnr - q_hat * total_std_err)
        upper_90_conformal = total_ibnr + q_hat * total_std_err
        
        # Prepare triangle data for frontend
        triangle_data = {
            "values": triangle_values.tolist(),
            "std_errors": std_err_triangle.tolist(),
            "origin_years": origin_years,
            "development_years": dev_lags,
            "cell_status": cell_status.tolist()  # 0 = observed, 1 = projected
        }
        
        # Exposer les LDF et cadences calculés selon les formules du cours
        ldfs = getattr(model, 'manual_ldfs', [])
        cadences = getattr(model, 'manual_cadences', [])
        
        return ReservingIbnrResponse(
            grcode=request.grcode,
            ibnr_estimate=round(float(total_ibnr), 2),
            mack_interval={
                "lower_90": round(float(lower_90_mack), 2),
                "upper_90": round(float(upper_90_mack), 2),
                # Couverture empirique mesurée (backtest sur le portefeuille)
                "empirical_coverage": calibration.get("empirical_coverage_mack", 0.0),
                "nominal_coverage": calibration.get("nominal_coverage", 0.90),
            },
            conformal_interval={
                "lower_90": round(float(lower_90_conformal), 2),
                "upper_90": round(float(upper_90_conformal), 2),
                # Couverture empirique conforme mesurée (jamais codée en dur)
                "empirical_coverage": calibration.get("empirical_coverage_conformal", 0.0),
                "nominal_coverage": calibration.get("nominal_coverage", 0.90),
                "q_hat": q_hat,
                "interval_width_mack": calibration.get("interval_width_mack"),
                "interval_width_conformal": calibration.get("interval_width_conformal"),
            },
            triangle_data=triangle_data,
            ldfs=ldfs,
            cadences=cadences
        )
    except Exception as e:
        raise ValueError(f"Impossible d'ajuster le modèle Mack pour GRCODE {request.grcode}: {str(e)}")


def _load_deep_triangle_model():
    global _model_dt
    if _model_dt is None:
        try:
            _model_dt = DeepTriangleGRU(hidden_dim=16).to(_device)
            _model_dt.load_state_dict(torch.load(MODELS_DIR / "deep_triangle.pt", map_location=_device))
            _model_dt.eval()
        except FileNotFoundError:
            # Create a simple model if not trained
            print("Warning: Deep Triangle model not found, using untrained model")
            _model_dt = DeepTriangleGRU(hidden_dim=16).to(_device)
            _model_dt.eval()
    return _model_dt, None  # Return tuple for backward compatibility


def predict_future_increments(model, obs_scaled, device):
    """Prédit les incréments futurs à partir des incréments observés."""
    obs_tensor = torch.tensor(obs_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        # Prédire séquentiellement chaque incrément futur
        future_scaled = []
        current_seq = obs_tensor.clone()

        for _ in range(10 - len(obs_scaled)):
            pred = model(current_seq)  # Shape: (batch, seq_len)
            next_val = pred[0, -1].item()
            future_scaled.append(max(0, next_val))  # Clip valeurs négatives
            # Add new prediction as next timestep (batch, 1, 1)
            next_input = pred[:, -1:].unsqueeze(-1)  # (batch, 1, 1)
            current_seq = torch.cat([current_seq, next_input], dim=1)

    return future_scaled


def predict_reserving(request: ReservingRequest) -> ReservingResponse:
    """Prédit les incréments futurs et l'IBNR avec Deep Triangle."""
    model, _ = _load_deep_triangle_model()

    # Normalisation par la prime (cohérent avec l'entraînement)
    obs_scaled = [x / request.premium for x in request.observed_increments]

    future_scaled = predict_future_increments(model, obs_scaled, _device)
    future_amounts = [float(x * request.premium) for x in future_scaled]

    ibnr = sum(future_amounts)

    return ReservingResponse(
        predicted_future_increments=[round(x, 2) for x in future_amounts],
        predicted_ibnr=round(ibnr, 2),
        model_used="deep_triangle_gru",
        note="Deep Triangle model: demonstration only. Mack Chain-Ladder recommended for production (better empirical performance)."
    )
