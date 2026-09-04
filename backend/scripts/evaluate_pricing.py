"""
Évaluation empirique authentique du module tarification.

Calcule, sur le jeu de test (split 60/20/20, seed 123), les métriques
réellement mesurées, puis les persiste dans models/pricing_metrics.json :

  - Gini GLM (fréquence)
  - Gini CANN (fréquence), si le CANN est disponible
  - gain relatif de Gini CANN vs GLM (mesuré, jamais codé en dur)
  - distribution NGBoost de sévérité (mean/median/std/quantiles) sur le
    sous-échantillon attritionnel de test
  - paramètre de dépendance copule (rho) estimé sur les données
  - déviance Poisson (fréquence)

Usage (conteneur backend) :
    python -m scripts.evaluate_pricing

Aucune valeur de ce script n'est codée en dur : tout est calculé à partir des
données, des modèles et des évaluations réelles.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

MODELS_DIR = BACKEND_DIR / "models" if (BACKEND_DIR / "models").exists() else PROJECT_ROOT / "models"
SEED = 123


def _load_models():
    import joblib
    from src.pricing.models import predict_frequency, predict_severity
    from src.pricing.cann import GroupInteractionNet

    model_glm = joblib.load(MODELS_DIR / "glm_poisson.pkl")
    model_gamma = joblib.load(MODELS_DIR / "glm_gamma.pkl")
    ngboost = joblib.load(MODELS_DIR / "ngboost_severity.pkl")
    cann_stats = joblib.load(MODELS_DIR / "cann_stats.pkl")

    cann = None
    pt = MODELS_DIR / "cann_group_interaction.pt"
    if pt.exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cann = GroupInteractionNet(
            n_continuous=3, brand_cardinality=11,
            embedding_dim=2, hidden_dim=20,
        )
        cann.load_state_dict(torch.load(pt, map_location=device))
        cann.eval().to(device)

    return model_glm, model_gamma, ngboost, cann, cann_stats


BRAND_MAPPING = {
    "B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4,
    "B6": 5, "B10": 6, "B11": 7, "B12": 8, "B13": 9, "B14": 10,
}


def _prepare_features(df: pd.DataFrame, cann_stats: dict) -> pd.DataFrame:
    """Applique le feature engineering de production.

    - buckets GLM (via add_glm_features-config)
    - normalisation CANN/NGBoost avec les statistiques d'ENTRAÎNEMENT persistées
      (cann_stats.pkl) et non celles du jeu de test (anti-fuite).
    - codes catégoriels dans l'ordre observé (add_cann_features, cohérent avec
      l'entraînement).
    """
    from src.pricing.features import add_glm_features

    out = add_glm_features(df)
    out["ClaimAmount_capped"] = out["ClaimAmount"].clip(upper=100000)

    # Normalisation min-max avec les stats d'entraînement (donne des [−1,1])
    for col in ["VehPower", "VehAge", "DrivAge", "BonusMalus"]:
        lo = cann_stats[col]["min"]
        hi = cann_stats[col]["max"]
        out[f"{col}_norm"] = 2 * (out[col] - lo) / (hi - lo) - 1

    # Codes catégoriels (même convention qu'à l'entraînement : Regular=1)
    out["VehBrand_code"] = out["VehBrand"].astype("category").cat.codes
    out["Region_code"] = out["Region"].astype("category").cat.codes
    out["Area_code"] = out["Area"].astype("category").cat.codes
    out["VehGas_code"] = (out["VehGas"] == "Regular").astype(int)
    return out


def _cann_predict(cann, cann_stats, df: pd.DataFrame, log_mu_glm: np.ndarray, device):
    """Prédit mu = E[X] (nombre attendu de sinistres) via le GroupInteractionNet.

    Conventions identiques à l'entraînement (notebook 03b, src/pricing/features.py) :
      - VehPower_norm, VehAge_norm : min-max en [-1,1] via cann_stats
      - VehGas_code : binaire brut, Regular=1, Diesel=0 (AUCUNE normalisation)
      - brand_code : codes pandas catégoriels (ordre alphabétique)
      - log_mu_glm : log(freq_glm * exposure) = log(mu_GLM)

    Retourne mu = exp(log_lambda_cann), c'est-à-dire le nombre attendu de
    sinistres (pas la fréquence annuelle). Pour obtenir la fréquence,
    diviser par exposure en aval.
    """
    vp = df["VehPower"].values
    va = df["VehAge"].values

    veh_power_norm = 2 * (vp - cann_stats["VehPower"]["min"]) / (
        cann_stats["VehPower"]["max"] - cann_stats["VehPower"]["min"]) - 1
    veh_age_norm = 2 * (va - cann_stats["VehAge"]["min"]) / (
        cann_stats["VehAge"]["max"] - cann_stats["VehAge"]["min"]) - 1
    # Convention d'entraînement : Regular=1, Diesel=0, binaire brut
    veh_gas_code = (df["VehGas"] == "Regular").astype(float).values

    # Codes catégoriels pandas (ordre alphabétique, comme à l'entraînement)
    brand_codes = df["VehBrand"].astype("category").cat.codes.values

    freqs = []
    for i in range(len(df)):
        cont = torch.tensor([[veh_power_norm[i], veh_age_norm[i], veh_gas_code[i]]],
                            dtype=torch.float32).to(device)
        br = torch.tensor([brand_codes[i]], dtype=torch.long).to(device)
        lg = torch.tensor([log_mu_glm[i]], dtype=torch.float32).to(device)
        with torch.no_grad():
            ll = cann(cont, br, lg).item()
        # ll est sur l'échelle mu (log du nombre attendu de sinistres)
        freqs.append(float(np.exp(ll)))
    return np.array(freqs)


def main() -> None:
    from src.pricing.data import build_pricing_dataset, train_valid_test_split, get_severity_subset
    from src.pricing.evaluate import compute_gini_index
    from src.pricing.models import predict_frequency, predict_ngboost_severity, NGBOOST_FEATURES
    from scipy.stats import spearmanr
    import joblib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_glm, model_gamma, ngboost, cann, cann_stats = _load_models()

    # 1) Dataset + split 60/20/20 (seed 123)
    df = build_pricing_dataset()
    train, valid, test = train_valid_test_split(df)
    print(f"Dataset: {len(df):,} | train {len(train):,} | valid {len(valid):,} | test {len(test):,}")

    # 2) Features + prédictions fréquence GLM sur le test
    #    _prepare_features utilise cann_stats.pkl pour la normalisation
    #    (anti-fuite) et add_glm_features pour les buckets.
    test_f = _prepare_features(test, cann_stats)
    freq_glm = predict_frequency(model_glm, test_f).values
    exposure = test["Exposure"].values
    claimnb = test["ClaimNb"].values

    # Le Gini se calcule en ordonnant par la fréquence annuelle (λ) :
    #   y_pred = freq (PAS freq*exposure), comme dans le rayon de référence.
    gini_glm = compute_gini_index(claimnb, freq_glm, exposure)

    metrics = {
        "dataset": "freMTPL2 (freq+sev)",
        "split": "train/valid/test = 60/20/20 (seed 123)",
        "test_size": int(len(test)),
        "glm_gini_test": round(float(gini_glm), 4),
        "cann_available": cann is not None,
    }

    # 3) Gini CANN si disponible
    #    Le CANN prend en entrée log(λ_glm * E + eps) en tant que log_mu_glm
    #    et corrige le log-λ avec une perturbation apprise.
    #    _cann_predict retourne mu = E[X] = freq * exposure. On divise par
    #    exposure pour obtenir la fréquence annuelle λ, cohérente avec le Gini.
    if cann is not None:
        log_mu_glm = np.log(np.maximum(freq_glm * exposure, 1e-8))
        mu_cann = _cann_predict(cann, cann_stats, test_f, log_mu_glm, device)
        freq_cann = mu_cann / np.maximum(exposure, 1e-8)
        # Même convention que le GLM : y_pred = λ (fréquence annuelle)
        gini_cann = compute_gini_index(claimnb, freq_cann, exposure)
        rel_gain = (gini_cann - gini_glm) / abs(gini_glm) if abs(gini_glm) > 1e-10 else 0.0
        metrics["cann_gini_test"] = round(float(gini_cann), 4)
        metrics["cann_gini_relative_gain"] = round(float(rel_gain), 4)
        print(f"Gini GLM={gini_glm:.4f} | CANN={gini_cann:.4f} | gain rel={rel_gain:.4%}")

    # 4) Déviance Poisson GLM (fréquence)
    from src.pricing.data import compute_poisson_deviance
    dev = compute_poisson_deviance(claimnb, freq_glm, exposure)
    metrics["glm_poisson_deviance_test"] = round(float(dev), 5)

    # 5) Distribution NGBoost de sévérité sur le sous-échantillon attritionnel
    #    Les features normalisées (VehPower_norm, ...) et codes catégoriels
    #    sont fournis par _prepare_features avec les stats d'entraînement.
    sev_test = get_severity_subset(test)
    sev_test = _prepare_features(sev_test, cann_stats)
    sev_dist = predict_ngboost_severity(ngboost, sev_test)
    mean = sev_dist["pred_mean"]
    lower = sev_dist["pred_lower_90"]
    upper = sev_dist["pred_upper_90"]
    metrics["ngboost"] = {
        "n_observations": int(len(sev_test)),
        "mean_severity": round(float(mean.mean()), 2),
        "median_severity": round(float(np.median(sev_test["ClaimAmount_capped"])), 2),
        "mean_lower_90": round(float(lower.mean()), 2),
        "mean_upper_90": round(float(upper.mean()), 2),
        "std_severity": round(float(mean.std()), 2),
    }

    # 6) Paramètre de dépendance copule (rho de Spearman → rho gaussien)
    pos = test[(test["ClaimNb"] > 0) & (test["ClaimAmount"] > 0)]
    if len(pos) >= 10:
        rho_s, p_s = spearmanr(pos["ClaimNb"], pos["ClaimAmount"])
        rho_gauss = 2 * np.sin(np.pi * rho_s / 6)
        metrics["copula"] = {
            "spearman_rho": round(float(rho_s), 4),
            "spearman_p": round(float(p_s), 4),
            "gaussian_rho": round(float(rho_gauss), 4),
            "n_observations": int(len(pos)),
            "note": "Dépendance estimée sur polices sinistrées (Rank).",
        }
    else:
        metrics["copula"] = {"n_observations": 0, "note": "Pas assez de sinistres pour estimer rho."}

    # Persister l'artefact
    out = MODELS_DIR / "pricing_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True, default=str)
    print(f"Metriques persistees -> {out}")
    print(json.dumps(metrics, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
