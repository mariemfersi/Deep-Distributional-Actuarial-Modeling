"""
Script de démonstration MLflow — réentraîne les GLM de tarification
(fréquence Poisson + sévérité Gamma) sur le split train, évalue sur le
split test (indice de Gini, RMSE, MAE) et loggue le run dans MLflow
via `app.services.mlflow_service.log_model_training`.

Usage (depuis le conteneur backend):
    python -m scripts.train_demo_mlflow  --experiment actuarial_platform

Le but est de peupler l'UI MLflow (http://localhost:5000) avec un run
de référence reproductible, intégré au pipeline de feature engineering
existant (src/pricing/*).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.common.config import load_config
from src.pricing.data import (
    build_pricing_dataset,
    train_valid_test_split,
    get_severity_subset,
)
from src.pricing.features import build_features
from src.pricing.models import (
    fit_glm_poisson,
    fit_glm_gamma,
    predict_frequency,
    predict_severity,
    compute_pure_premium,
    FREQUENCY_FORMULA,
    SEVERITY_FORMULA,
)
from src.pricing.evaluate import compute_gini_index
from app.services.mlflow_service import log_model_training


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne les GLM et loggue un run MLflow")
    parser.add_argument("--experiment", default="actuarial_platform",
                        help="Nom de l'expérience MLflow (défaut: actuarial_platform)")
    args = parser.parse_args()

    config = load_config()
    run_name = "glm_pricing_demo"

    # 1. Pipeline complet : data -> features
    df = build_features(build_pricing_dataset(config), config)
    print(f"Dataset: {len(df):,} polices")

    train_df, valid_df, test_df = train_valid_test_split(df, config)

    # 2. Entraînement des GLM
    print("Entraînement GLM Poisson (fréquence)...")
    poisson_model = fit_glm_poisson(train_df)

    train_sev = get_severity_subset(train_df)
    valid_sev = get_severity_subset(valid_df)
    test_sev = get_severity_subset(test_df)

    print("Entraînement GLM Gamma (sévérité)...")
    gamma_model = fit_glm_gamma(train_sev)

    # 3. Évaluation sur le split test
    test_freq = predict_frequency(poisson_model, test_df)
    test_sev_pred = predict_severity(gamma_model, test_sev)

    # Gini sur les fréquences observées (polices en risque), exposure pondérée
    gini = compute_gini_index(
        y_true=test_df["ClaimNb"].values,
        y_pred=test_freq.values,
        exposure=test_df["Exposure"].values,
    )

    # Errur de fréquence (RMSE/MAE sur ClaimNb vs lambda prédite)
    freq_rmse = _rmse(test_df["ClaimNb"].values, (test_freq * test_df["Exposure"]).values)
    freq_mae = _mae(test_df["ClaimNb"].values, (test_freq * test_df["Exposure"]).values)

    # Erreur de sévérité (sur le sous-échantillon attritionnel)
    sev_rmse = _rmse(test_sev["ClaimAmount_capped"].values, test_sev_pred.values)
    sev_mae = _mae(test_sev["ClaimAmount_capped"].values, test_sev_pred.values)

    metrics = {
        "test_gini_index": round(gini, 4),
        "test_freq_rmse": round(freq_rmse, 4),
        "test_freq_mae": round(freq_mae, 4),
        "test_severity_rmse": round(sev_rmse, 2),
        "test_severity_mae": round(sev_mae, 2),
        "train_n_policies": int(len(train_df)),
        "test_n_policies": int(len(test_df)),
    }

    params = {
        "frequency_model": "GLM-Poisson",
        "severity_model": "GLM-Gamma (link log)",
        "frequency_formula": FREQUENCY_FORMULA,
        "severity_formula": SEVERITY_FORMULA,
        "train_ratio": 0.6,
        "valid_ratio": 0.2,
        "test_ratio": 0.2,
        "seed": 123,
    }

    print("\nMétriques test:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 4. Log MLflow (les modèles entraînés restent en mémoire ; on ne réécrit
    #    pas les modèles déployés — le script est un run de démonstration MLflow)
    run_id = log_model_training(
        experiment_name=args.experiment,
        run_name=run_name,
        params=params,
        metrics=metrics,
        tags={"source": "demo_script", "module": "pricing"},
    )
    print(f"\nRun MLflow loggué: {run_id}")
    print(f"Voir http://localhost:5000/#/experiments/{args.experiment}/runs/{run_id}")


if __name__ == "__main__":
    main()
