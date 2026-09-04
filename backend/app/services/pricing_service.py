"""
Service de tarification — charge les modèles entraînés et expose une
fonction de prédiction. Utilise GLM Poisson pour la fréquence et GLM Gamma pour la sévérité,
avec CANN (Combined Actuarial Neural Network) pour le modèle amélioré et copule gaussienne
pour la dépendance fréquence-sévérité.

Les métriques (Gini, copule, etc.) sont calculées par scripts/evaluate_pricing.py
et chargées depuis models/pricing_metrics.json. Rien n'est codé en dur.
"""

import json
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import torch
from scipy.stats import norm, gamma

# Ajoute la racine du projet au path pour importer src/
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.pricing.models import predict_frequency, predict_severity
from src.pricing.cann import GroupInteractionNet
from app.schemas.pricing import PricingRequest, PricingResponse

MODELS_DIR = BACKEND_DIR / "models" if (BACKEND_DIR / "models").exists() else PROJECT_ROOT / "models"

_model_glm = None
_model_gamma = None
_model_ngboost = None
_model_cann = None
_cann_stats = None
_copula_params = None
_pricing_metrics = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_pricing_metrics() -> dict:
    """Charge les métriques mesurées depuis l'artefact (jamais codé en dur)."""
    global _pricing_metrics
    if _pricing_metrics is None:
        path = MODELS_DIR / "pricing_metrics.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                _pricing_metrics = json.load(f)
        else:
            _pricing_metrics = {}
    return _pricing_metrics


def _load_models():
    """Charge les modèles une seule fois (cache en mémoire du process)."""
    global _model_glm, _model_gamma, _model_ngboost, _model_cann, _cann_stats, _copula_params
    if _model_glm is None:
        _model_glm = joblib.load(MODELS_DIR / "glm_poisson.pkl")
        _model_gamma = joblib.load(MODELS_DIR / "glm_gamma.pkl")
        try:
            _model_ngboost = joblib.load(MODELS_DIR / "ngboost_severity.pkl")
        except FileNotFoundError:
            _model_ngboost = None

        # Load CANN model if available
        try:
            _model_cann = GroupInteractionNet(
                n_continuous=3,  # VehPower, VehAge, VehGas_code
                brand_cardinality=11,
                embedding_dim=2,
                hidden_dim=20
            )
            _model_cann.load_state_dict(torch.load(MODELS_DIR / "cann_group_interaction.pt", map_location=_device))
            _model_cann.eval()

            # Load normalization statistics (training uses min-max normalization)
            try:
                _cann_stats = joblib.load(MODELS_DIR / "cann_stats.pkl")
            except FileNotFoundError:
                # Fallback to training statistics
                _cann_stats = {
                    "VehPower": {"min": 4, "max": 15},
                    "VehAge": {"min": 0, "max": 100},
                }
        except FileNotFoundError:
            _model_cann = None
            _cann_stats = None

        # Load copula parameters if available
        try:
            _copula_params = joblib.load(MODELS_DIR / "copula_params.pkl")
        except FileNotFoundError:
            _copula_params = None

    return _model_glm, _model_gamma, _model_ngboost, _model_cann, _cann_stats, _copula_params


# Mapping VehBrand -> codes catégoriels (ordre alphabétique, comme pd.Categorical)
# Identique à src/pricing/features.py : df["VehBrand"].astype("category").cat.codes
_BRAND_CODE_MAP = {
    "B1": 0, "B10": 1, "B11": 2, "B12": 3, "B13": 4, "B14": 5,
    "B2": 6, "B3": 7, "B4": 8, "B5": 9, "B6": 10,
}


def _build_feature_row(request: PricingRequest) -> pd.DataFrame:
    """
    Reconstruit une ligne de features au format attendu par le GLM et le CANN.

    Conventions identiques à l'entraînement (src/pricing/features.py, notebooks) :
      - Buckets GLM (DrivAge, VehAge, BonusMalus)
      - VehPower_norm, VehAge_norm : min-max en [-1,1] via cann_stats
      - VehGas_code : binaire brut, Regular=1, Diesel=0 (AUCUNE normalisation)
      - VehBrand_code : codes catégoriels alphabétiques (pd.Categorical)
    """
    row = pd.DataFrame([{
        "VehPower": request.veh_power,
        "VehAge": request.veh_age,
        "DrivAge": request.driv_age,
        "BonusMalus": request.bonus_malus,
        "VehBrand": request.veh_brand,
        "VehGas": request.veh_gas,
        "Region": request.region,
        "Area": request.area,
        "Density": request.density,
        "Exposure": request.exposure,
    }])

    # Buckets identiques à src/pricing/features.py
    row["DrivAge_bucket"] = pd.cut(
        row["DrivAge"], bins=[17, 20, 25, 30, 40, 50, 70, 120],
        labels=["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]
    )
    row["VehAge_bucket"] = pd.cut(
        row["VehAge"], bins=[-1, 0, 10, 200], labels=["neuf", "recent", "ancien"]
    )
    row["BM_bucket"] = pd.cut(
        row["BonusMalus"], bins=[49, 60, 80, 100, 125, 150, 350],
        labels=["50-60", "61-80", "81-100", "101-125", "126-150", "151+"]
    )
    row["Density_log"] = np.log(row["Density"])
    row["ClaimAmount_capped"] = 0  # placeholder requis par la formule du GLM Gamma

    # Normalisation CANN/NGBoost avec les stats d'ENTRAÎNEMENT (cann_stats.pkl)
    # et non les stats de la ligne courante (anti-fuite).
    _, _, _, _, cann_stats, _ = _load_models()
    if cann_stats is not None:
        vp_lo = cann_stats.get("VehPower", {}).get("min", 4)
        vp_hi = cann_stats.get("VehPower", {}).get("max", 15)
        va_lo = cann_stats.get("VehAge", {}).get("min", 0)
        va_hi = cann_stats.get("VehAge", {}).get("max", 100)
    else:
        vp_lo, vp_hi, va_lo, va_hi = 4, 15, 0, 100

    row["VehPower_norm"] = 2 * (row["VehPower"] - vp_lo) / (vp_hi - vp_lo) - 1
    row["VehAge_norm"] = 2 * (row["VehAge"] - va_lo) / (va_hi - va_lo) - 1

    # DrivAge_norm et BonusMalus_norm requis par NGBoost (NGBOOST_FEATURES)
    # et par add_cann_features dans le pipeline d'entraînement.
    da_lo = cann_stats.get("DrivAge", {}).get("min", 18) if cann_stats else 18
    da_hi = cann_stats.get("DrivAge", {}).get("max", 100) if cann_stats else 100
    bm_lo = cann_stats.get("BonusMalus", {}).get("min", 50) if cann_stats else 50
    bm_hi = cann_stats.get("BonusMalus", {}).get("max", 150) if cann_stats else 150
    row["DrivAge_norm"] = 2 * (row["DrivAge"] - da_lo) / (da_hi - da_lo) - 1
    row["BonusMalus_norm"] = 2 * (row["BonusMalus"] - bm_lo) / (bm_hi - bm_lo) - 1

    # Convention d'entraînement : Regular=1, Diesel=0 (binaire brut, PAS z-score)
    row["VehGas_code"] = (row["VehGas"] == "Regular").astype(int)

    # Codes catégoriels dans l'ordre alphabétique (comme pd.Categorical.codes)
    row["VehBrand_code"] = row["VehBrand"].map(_BRAND_CODE_MAP).fillna(0).astype(int)
    row["Region_code"] = row["Region"].astype("category").cat.codes
    row["Area_code"] = row["Area"].astype("category").cat.codes

    return row


def predict_pricing(request: PricingRequest) -> PricingResponse:
    """
    Prédit fréquence, sévérité et prime pure pour un profil donné.
    Retourne à la fois le baseline GLM et le modèle amélioré (CANN si disponible).
    """
    model_glm, model_gamma, _, model_cann, cann_stats, _ = _load_models()
    row = _build_feature_row(request)

    # Prédiction GLM (baseline)
    freq_glm = predict_frequency(model_glm, row).iloc[0]
    severity = predict_severity(model_gamma, row).iloc[0]
    pure_premium_glm = freq_glm * severity * request.exposure

    glm_result = {
        "predicted_frequency": round(float(freq_glm), 4),
        "predicted_severity": round(float(severity), 2),
        "pure_premium": round(float(pure_premium_glm), 2)
    }

    # Prédiction CANN (modèle amélioré) si disponible
    # Conventions d'entraînement :
    #   - continuous = [VehPower_norm, VehAge_norm, VehGas_code]
    #   - VehGas_code : Regular=1, Diesel=0 (binaire brut, PAS z-score)
    #   - brand_code : codes catégoriels alphabétiques
    #   - log_mu_glm = log(freq_glm * exposure)
    #   - sortie : mu = exp(log_lambda) = freq * exposure → freq = mu / exposure
    if model_cann is not None and cann_stats is not None:
        try:
            vp_lo = cann_stats.get("VehPower", {}).get("min", 4)
            vp_hi = cann_stats.get("VehPower", {}).get("max", 15)
            va_lo = cann_stats.get("VehAge", {}).get("min", 0)
            va_hi = cann_stats.get("VehAge", {}).get("max", 100)

            veh_power_norm = 2 * (request.veh_power - vp_lo) / (vp_hi - vp_lo) - 1
            veh_age_norm = 2 * (request.veh_age - va_lo) / (va_hi - va_lo) - 1
            veh_gas_code = 1.0 if request.veh_gas == "Regular" else 0.0

            continuous_features = np.array([
                veh_power_norm, veh_age_norm, veh_gas_code
            ], dtype=np.float32)

            brand_code = _BRAND_CODE_MAP.get(request.veh_brand, 0)

            log_mu_glm = np.log(freq_glm * request.exposure + 1e-8)

            with torch.no_grad():
                # S'assurer que les tensors sont sur le même device que le modèle
                model_device = next(model_cann.parameters()).device
                continuous_tensor = torch.tensor(continuous_features).unsqueeze(0).to(model_device)
                brand_tensor = torch.tensor([brand_code], dtype=torch.long).to(model_device)
                log_mu_glm_tensor = torch.tensor([log_mu_glm]).to(model_device)

                log_lambda_cann = model_cann(continuous_tensor, brand_tensor, log_mu_glm_tensor)
                mu_cann = torch.exp(log_lambda_cann).item()
                freq_cann = mu_cann / request.exposure

            pure_premium_cann = freq_cann * severity * request.exposure

            improved_result = {
                "predicted_frequency": round(float(freq_cann), 4),
                "predicted_severity": round(float(severity), 2),
                "pure_premium": round(float(pure_premium_cann), 2)
            }

            # Gain de Gini mesuré (jamais codé en dur) — chargé depuis l'artefact
            metrics = _load_pricing_metrics()
            gini_improvement = metrics.get("cann_gini_relative_gain", 0.0)
        except Exception as e:
            print(f"CANN prediction failed, falling back to GLM: {e}")
            improved_result = glm_result.copy()
            gini_improvement = 0.0
    else:
        improved_result = glm_result.copy()
        gini_improvement = 0.0

    return PricingResponse(
        glm_baseline=glm_result,
        improved_model=improved_result,
        gini_improvement_pct=gini_improvement,
    )


def get_severity_distribution(request: PricingRequest) -> dict:
    """
    Retourne la distribution de sévérité avec percentiles.

    Si NGBoost est disponible : prédiction distributionnelle genuine via
    pred_dist() → Gamma paramétrique → percentiles analytiques.
    Sinon : fallback GLM Gamma avec CV=0.5 (documenté comme approximation).
    """
    _, _, model_ngboost, _, _, _ = _load_models()
    row = _build_feature_row(request)

    percentiles_list = [5, 25, 50, 75, 95]

    if model_ngboost is not None:
        try:
            from src.pricing.models import predict_ngboost_severity
            sev = predict_ngboost_severity(model_ngboost, row)
            mean_val = float(sev["pred_mean"].iloc[0])
            lower_90 = float(sev["pred_lower_90"].iloc[0])
            upper_90 = float(sev["pred_upper_90"].iloc[0])

            # Estimer l'écart-type de la distribution Gamma prédite via
            # l'intervalle de confiance à 90% : z_95 = 1.645
            # (si mean et std sont les paramètres de la Gamma, on utilise
            # laWilson-Hilferty approx : ppf(0.95) ≈ mean + 1.645 * std)
            z_95 = 1.645
            est_std = max((upper_90 - mean_val) / z_95, 1.0)

            # Ajuster une Gamma à partir de mean et std estimés
            # Gamma : E[X] = kθ, Var[X] = kθ² → k = (E[X]/σ)², θ = σ²/E[X]
            shape = (mean_val / est_std) ** 2
            scale = est_std ** 2 / mean_val

            distribution = {
                "model": "ngboost_gamma",
                "mean_severity": round(mean_val, 2),
                "pred_lower_90": round(lower_90, 2),
                "pred_upper_90": round(upper_90, 2),
                "percentiles": {
                    f"p{p}": round(float(gamma.ppf(p / 100, a=shape, scale=scale)), 2)
                    for p in percentiles_list
                }
            }
        except Exception as e:
            print(f"NGBoost severity failed, falling back to GLM Gamma: {e}")
            distribution = _get_glm_distribution(row)
    else:
        distribution = _get_glm_distribution(row)

    return distribution


def _get_glm_distribution(row: pd.DataFrame) -> dict:
    """Distribution basée sur GLM Gamma si NGBoost non disponible."""
    from src.pricing.models import predict_severity
    model_gamma = joblib.load(MODELS_DIR / "glm_gamma.pkl")
    severity_mean = predict_severity(model_gamma, row).iloc[0]
    severity_std = severity_mean * 0.5  # Approximation
    
    from scipy.stats import gamma
    shape = (severity_mean / severity_std) ** 2
    scale = severity_std ** 2 / severity_mean
    
    percentiles = [5, 25, 50, 75, 95]
    
    return {
        "model": "glm_gamma",
        "percentiles": {
            f"p{p}": round(float(gamma.ppf(p/100, a=shape, scale=scale)), 2)
            for p in percentiles
        }
    }


def _gaussian_copula_sample(freq_mean, severity_mean, rho, n_samples=1000):
    """
    Génère des échantillons de (fréquence, sévérité) liés par une copule gaussienne.
    
    La copule gaussienne modélise la dépendance entre fréquence et sévérité sans
    faire d'hypothèse sur les marginales. rho est le coefficient de corrélation
    de la copule (pas la corrélation linéaire entre les variables).
    """
    # Générer des variables gaussiennes corrélées
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    z = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Transformer en uniformes via la CDF gaussienne
    u = norm.cdf(z)
    
    # Transformer vers les distributions marginales
    # Fréquence : Poisson
    freq_samples = np.random.poisson(freq_mean, n_samples)
    
    # Sévérité : Gamma
    severity_cv = 0.5  # Coefficient de variation
    severity_shape = (severity_mean / severity_cv) ** 2
    severity_scale = severity_mean / severity_shape
    severity_samples = gamma.rvs(a=severity_shape, scale=severity_scale, size=n_samples)
    
    # Appliquer la dépendance via les rangs (copule)
    # On réordonne les échantillons marginaux selon les rangs de la copule
    freq_sorted = np.sort(freq_samples)
    severity_sorted = np.sort(severity_samples)
    
    freq_rank = np.argsort(np.argsort(u[:, 0]))
    severity_rank = np.argsort(np.argsort(u[:, 1]))
    
    freq_copula = freq_sorted[freq_rank]
    severity_copula = severity_sorted[severity_rank]
    
    return freq_copula, severity_copula


def get_premium_with_copula(request: PricingRequest, n_samples=1000) -> dict:
    """
    Calcule la prime pure en tenant compte de la dépendance fréquence-sévérité
    via une copule gaussienne.
    """
    model_glm, model_gamma, _, _, _, _ = _load_models()
    row = _build_feature_row(request)

    freq_mean = predict_frequency(model_glm, row).iloc[0]
    severity_mean = predict_severity(model_gamma, row).iloc[0]

    # rho gaussien estimé sur les données (jamais codé en dur)
    metrics = _load_pricing_metrics()
    rho = metrics.get("copula", {}).get("gaussian_rho", 0.0)
    
    # Générer des échantillons avec dépendance
    freq_samples, severity_samples = _gaussian_copula_sample(
        freq_mean * request.exposure, severity_mean, rho, n_samples
    )
    
    # Calculer les primes pure pour chaque échantillon
    premium_samples = freq_samples * severity_samples
    
    # Statistiques de la distribution de prime
    premium_mean = np.mean(premium_samples)
    premium_std = np.std(premium_samples)
    premium_percentiles = {
        f"p{p}": np.percentile(premium_samples, p)
        for p in [5, 25, 50, 75, 95]
    }
    
    return {
        "premium_mean": round(float(premium_mean), 2),
        "premium_std": round(float(premium_std), 2),
        "premium_percentiles": {k: round(float(v), 2) for k, v in premium_percentiles.items()},
        "copula_rho": rho,
        "frequency_mean": round(float(freq_mean), 4),
        "severity_mean": round(float(severity_mean), 2),
        "note": "Premium distribution accounts for frequency-severity dependence via Gaussian copula"
    }