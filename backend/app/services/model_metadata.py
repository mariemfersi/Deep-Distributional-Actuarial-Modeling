"""
Enregistrement des métadonnées des modèles dans MLflow.

À chaque redémarrage du backend, on vérifie si l'expérience de chaque module
contient déjà un run "model_registry" (métriques de performance + paramètres
du modèle). Si non, on le crée une seule fois (idempotent).

Les métriques sont CHARGÉES depuis les artefacts mesurés (pricing_metrics.json,
fraud_metrics.json, reserving_calibration.json). Aucune valeur n'est codée en dur.
"""

import json
import logging
from pathlib import Path
from typing import Any

from app.services.mlflow_service import log_model_training

logger = logging.getLogger("actuarial.model_metadata")

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models" if (BACKEND_DIR / "models").exists() else PROJECT_ROOT / "models"


def _load_json(filename: str) -> dict:
    """Charge un fichier JSON depuis le répertoire des modèles."""
    path = MODELS_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _build_metadata() -> dict[str, dict[str, Any]]:
    """Construit les métadonnées dynamiquement depuis les artefacts mesurés."""
    pricing_m = _load_json("pricing_metrics.json")
    fraud_m = _load_json("fraud_metrics.json")
    reserving_m = _load_json("reserving_calibration.json")

    # ── Fraud ───────────────────────────────────────────────────────
    fraud_best = next(
        (r for r in fraud_m.get("benchmark", []) if r.get("best")),
        fraud_m.get("benchmark", [{}])[0] if fraud_m.get("benchmark") else {},
    )
    fraud_cv = fraud_m.get("cross_validation", {})
    xgb_smote_cv = fraud_cv.get("XGB + SMOTE", {})

    fraud_metrics = {
        "test_auc_roc": fraud_best.get("auc_roc", 0.0),
        "test_pr_auc": fraud_best.get("pr_auc", 0.0),
        "test_precision": fraud_best.get("precision", 0.0),
        "test_recall": fraud_best.get("recall", 0.0),
        "test_f1": fraud_best.get("f1", 0.0),
        "cv_5fold_auc_roc": xgb_smote_cv.get("mean_auc_roc", 0.0),
    }

    # ── Pricing ─────────────────────────────────────────────────────
    pricing_metrics = {
        "test_gini_index": pricing_m.get("glm_gini_test", 0.0),
        "cann_gini_test": pricing_m.get("cann_gini_test", 0.0),
        "cann_gini_relative_gain": pricing_m.get("cann_gini_relative_gain", 0.0),
    }

    # ── Reserving ───────────────────────────────────────────────────
    reserving_metrics = {
        "empirical_coverage_conformal": reserving_m.get("empirical_coverage_conformal", 0.0),
        "empirical_coverage_mack": reserving_m.get("empirical_coverage_mack", 0.0),
        "q_hat": reserving_m.get("q_hat", 0.0),
        "nominal_coverage": reserving_m.get("nominal_coverage", 0.90),
        "n_observations": reserving_m.get("n_observations_conformal_test", 0),
    }

    return {
        "fraud": {
            "params": {
                "model": fraud_m.get("best_model", "XGB + SMOTE"),
                "model_version": "fraud_best_v1",
                "features": 30,
                "imbalance_handling": "SMOTE (imblearn Pipeline, dans les folds)",
                "feature_selection": "Boruta (8 features confirmées)",
                "threshold": 0.20,
            },
            "metrics": fraud_metrics,
            "tags": {"source": "model_registry", "module": "fraud"},
        },
        "pricing": {
            "params": {
                "frequency_model": "GLM-Poisson",
                "severity_model": "GLM-Gamma (link log)",
                "model_version": "glm_poisson_gamma_v1",
                "frequency_features": "DrivAge_bucket, VehAge_bucket, BM_bucket, VehGas, VehBrand, Region, Density_log",
                "offset": "log(Exposure)",
            },
            "metrics": pricing_metrics,
            "tags": {"source": "model_registry", "module": "pricing"},
        },
        "reserving": {
            "params": {
                "method": "Mack Chain-Ladder + Conformal Prediction",
                "model_version": "mack_conformal_v1",
                "interval_approach": "distribution-free (conformal)",
                "backtesting": "10 années d'accident (1988-1997)",
                "companies": 10,
            },
            "metrics": reserving_metrics,
            "tags": {"source": "model_registry", "module": "reserving"},
        },
    }


def register_all_model_metadata() -> None:
    """Enregistre (idempotent) les métadonnées de modèle de chaque module dans MLflow."""
    metadata = _build_metadata()
    for module, meta in metadata.items():
        _register_one(module, meta)


def _register_one(module: str, meta: dict[str, Any]) -> None:
    """Crée le run model_registry du module s'il n'existe pas déjà."""
    experiment_name = module
    run_name = "model_registry"
    try:
        from app.services.mlflow_service import _ensure_init, _get_active_mlflow_run
        _ensure_init(experiment_name)
        # Idempotence : on ne crée le run que s'il n'y en a pas déjà un du même nom.
        if _get_active_mlflow_run(experiment_name, run_name) is not None:
            logger.debug("model_registry déjà présent pour %s — ignoré", module)
            return

        run_id = log_model_training(
            experiment_name=experiment_name,
            run_name=run_name,
            params=meta["params"],
            metrics=meta["metrics"],
            tags=meta["tags"],
        )
        if run_id:
            logger.info("Métadonnées modèle MLflow enregistrées pour %s (run %s)", module, run_id)
    except Exception as e:
        logger.warning("Échec enregistrement métadonnées MLflow (%s) : %s", module, e)
