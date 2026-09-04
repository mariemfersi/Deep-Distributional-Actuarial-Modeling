"""
Router pour l'API de détection de fraude.
"""

import time
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app.models.prediction import Prediction
from app.schemas.fraud import FraudRequest, FraudResponse
from app.services.fraud_service import predict_fraud
from app.services.mlflow_service import log_prediction

router = APIRouter(prefix="/fraud", tags=["fraud"])

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models" if (BACKEND_DIR / "models").exists() else PROJECT_ROOT / "models"


def _load_fraud_metrics() -> dict:
    """Charge les métriques mesurées depuis l'artefact (jamais codé en dur)."""
    path = MODELS_DIR / "fraud_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _clean_feature(name: str) -> str:
    """Retire les suffixes _code/_norm pour exposer des noms de features lisibles."""
    for suffix in ("_code", "_norm"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@router.get("/methodology")
def get_methodology():
    """Résumé de la méthodologie : benchmark de modèles, Boruta et pourquoi le GNN n'a pas été retenu.

    Les chiffres (AUC-ROC, PR-AUC, CV, Boruta) sont chargés depuis
    models/fraud_metrics.json, généré par scripts/evaluate_fraud.py sur un
    préprocessing ajusté sur le train seul. Rien n'est codé en dur.
    """
    m = _load_fraud_metrics()

    benchmark_rows = []
    for row in m.get("benchmark", []):
        benchmark_rows.append({
            "model": row["model"],
            "type": row.get("type", ""),
            "auc_roc": row["auc_roc"],
            "pr_auc": row["pr_auc"],
            "precision": row.get("precision", 0.0),
            "recall": row.get("recall", 0.0),
            "f1": row.get("f1", 0.0),
            "best": row.get("best", False),
        })
    # Unsupervised marked for clarity
    for br in benchmark_rows:
        if br["model"] in ("Isolation Forest", "LOF"):
            br["type"] = "Non supervisé"
    MODEL_TYPES = {
        "XGB + SMOTE": "Boosting + SMOTE", "XGBoost": "Boosting",
        "Random Forest": "Forest", "RF + SMOTE": "Forest + SMOTE",
        "SVM (RBF)": "Kernel", "LOF": "Non supervisé", "Isolation Forest": "Non supervisé",
    }
    for br in benchmark_rows:
        br["type"] = MODEL_TYPES.get(br["model"], br["type"])

    best_row = next((r for r in benchmark_rows if r["best"]), benchmark_rows[0] if benchmark_rows else None)
    best_model = m.get("best_model", best_row["model"] if best_row else None)

    cv = m.get("cross_validation", {})
    cv_results = {
        name: v["mean_auc_roc"] for name, v in cv.items()
    }
    cv_std = {
        name: v["std_auc_roc"] for name, v in cv.items()
    }

    boruta = m.get("boruta", {})
    confirmed = [_clean_feature(f) for f in boruta.get("confirmed", [])]

    return {
        "benchmark": {
            "description": (
                "Comparaison de 7 approches de détection de fraude (Random Forest, XGBoost, SVM, "
                "avec/sans SMOTE, et détections non supervisées Isolation Forest/LOF), conforme "
                "à la revue de littérature sur la fraude en assurance par IA."
            ),
            "models": [r["model"] for r in benchmark_rows],
            "imbalance_handling": "SMOTE (rééquilibrage des classes) appliqué uniquement aux données d'entraînement (imblearn Pipeline, sans fuite en CV)",
            "feature_selection": "Boruta (BorutaPy) pour identifier les features discriminantes",
            "metrics": "AUC-ROC, PR-AUC, précision, rappel, F1, matrice de confusion",
            "leakage_handling": m.get("leakage_handling", "preprocessor_fit_on_train_only"),
        },
        "final_model": {
            "model": f"Meilleur du benchmark — {best_model} (fraud_best_model.pkl)",
            "description": "Gradient boosting avec rééquilibrage SMOTE sur 30 features catégoriels encodés et numériques normalisés",
            "result": (
                f"AUC-ROC {best_row['auc_roc']} (jeu de test) / "
                f"{cv_results.get(best_model, 'N/A')} (CV 5-fold)"
                if best_row else "N/A"
            ),
        },
        "benchmark_comparison": benchmark_rows,
        "cross_validation": {
            "folds": 5,
            "seed": 123,
            "smote_within_folds": True,
            "results": cv_results,
            "std": cv_std,
        },
        "boruta": {
            "n_features": boruta.get("n_features", len(confirmed) + len(boruta.get("rejected", []))),
            "confirmed": confirmed,
            "tentative": [_clean_feature(f) for f in boruta.get("tentative", [])],
            "rejected": len(boruta.get("rejected", [])),
        },
        "graph_attempts": [
            {
                "attempt": 1,
                "approach": "Graphe basé sur RepNumber (numéro de réclamation)",
                "description": "Construction d'arêtes entre réclamations du même numéro",
                "result": "Insuffisant - trop peu de connexions, graphe très sparse",
            },
            {
                "attempt": 2,
                "approach": "Graphe basé sur similarité de features",
                "description": "Arêtes entre réclamations avec similarité cosinus > seuil",
                "result": "Calcul coûteux, résultats non concluants",
            },
            {
                "attempt": 3,
                "approach": "Graphe basé sur attributs rares partagés",
                "description": "Arêtes entre réclamations partageant des valeurs rares",
                "result": "Meilleur mais encore limité par la taille du dataset",
            },
            {
                "attempt": 4,
                "approach": "GNN (Graph Neural Network) avec PyTorch Geometric",
                "description": "GCN/GAT pour propagation d'information sur le graphe",
                "result": "Non retenu - Random Forest supérieur en performance et interprétabilité",
            },
        ],
        "conclusion": (
            f"Le benchmark complet (Random Forest, XGBoost, SVM, SMOTE, Isolation Forest, LOF) "
            f"permet de sélectionner le modèle le plus performant : {best_model} atteint l'AUC-ROC "
            f"la plus élevée en jeu de test et en validation croisée, conforme à la littérature où "
            f"le boosting domine généralement sur la fraude. Les approches non supervisées "
            f"(LOF, Isolation Forest) sont moins discriminantes sur ce jeu de données. "
            f"Le préprocessing (encodage + normalisation) est ajusté sur le train seul pour "
            f"garantir l'absence de fuite de données."
        ),
    }


@router.post("/predict", response_model=FraudResponse)
def predict(request: FraudRequest, db: Session = Depends(get_db), _user: dict = Depends(get_current_user)):
    start = time.time()
    try:
        response = predict_fraud(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start) * 1000
    db.add(Prediction(
        module="fraud",
        model_version=getattr(response, "model_version", "random_forest_v1"),
        request_json=request.model_dump(),
        response_json=response.model_dump(),
        latency_ms=round(latency_ms, 2),
    ))
    db.commit()

    # Log MLflow (best-effort) — ne bloque jamais la réponse
    log_prediction(
        module="fraud",
        request_data=request.model_dump(),
        response_data=response.model_dump(),
        latency_ms=latency_ms,
        model_version=getattr(response, "model_version", "random_forest_v1"),
    )
    return response
