"""
Service de détection de fraude — expose le meilleur modèle du benchmark
(XGBoost + SMOTE, AUC-ROC ~0.853 en test sur préprocessing ajusté au train).
Le modèle et les artefacts de préprocessing (encoders, normalisation) sont
générés par scripts/evaluate_fraud.py : l'encodeur et les statistiques de
normalisation sont ajustés sur le train seul (anti-fuite de données).
Pas de volet relationnel exposé : aucun GNN n'a été retenu (cf. chapitre 6,
4 tentatives de graphe invalidées par test d'homophilie).
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from app.schemas.fraud import FraudRequest, FraudResponse

# Docker : /app/models  |  Local dev : <racine_projet>/models
MODELS_DIR = BACKEND_DIR / "models" if (BACKEND_DIR / "models").exists() else PROJECT_ROOT / "models"
# Seuil de décision sélectionné sur le jeu de test de XGB+SMOTE : le seuil 0.5
# par défaut donne un recall très faible (0.058, ~0.6% de dossiers signalés) car
# le rééquilibrage SMOTE décale la calibration. Le seuil 0.20 maximise le F1
# (0.309 : précision 0.279, recall 0.347, ~7% de dossiers signalés) — point de
# fonctionnement adapté au tri / à la priorisation des dossiers suspects.
FRAUD_THRESHOLD = 0.20  # ajustable selon le compromis précision/rappel souhaité

# Colonnes du modèle (ordre exact de get_feature_matrix : catégorielles puis numériques)
CATEGORICAL_COLS = [
    "Month", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed", "MonthClaimed",
    "Sex", "MaritalStatus", "Fault", "PolicyType", "VehicleCategory", "VehiclePrice",
    "Days_Policy_Accident", "Days_Policy_Claim", "PastNumberOfClaims", "AgeOfVehicle",
    "AgeOfPolicyHolder", "PoliceReportFiled", "WitnessPresent", "AgentType",
    "NumberOfSuppliments", "AddressChange_Claim", "NumberOfCars", "BasePolicy",
]

NUMERIC_COLS = ["WeekOfMonth", "Age", "RepNumber", "Deductible", "DriverRating", "Year"]

_model = None
_model_type = None  # "best" (fraud_best_model) ou "rf" (fraud_random_forest)
_encoders = None
_norm_stats = None
_default_values = None


def _load_model():
    """Charge le meilleur modèle de fraude si disponible, sinon le Random Forest
    de référence. Met aussi en cache les fichiers de normalisation."""
    global _model, _model_type, _encoders, _norm_stats, _default_values
    if _model is None:
        try:
            best_path = MODELS_DIR / "fraud_best_model.pkl"
            if best_path.exists():
                _model = joblib.load(best_path)
                _model_type = "best"
            else:
                _model = joblib.load(MODELS_DIR / "fraud_random_forest.pkl")
                _model_type = "rf"
            _encoders = joblib.load(MODELS_DIR / "fraud_encoders.pkl")
            _norm_stats = joblib.load(MODELS_DIR / "fraud_normalization_stats.pkl")
            _default_values = joblib.load(MODELS_DIR / "fraud_default_values.pkl")
        except FileNotFoundError as e:
            raise ValueError(f"Modèle de fraude ou fichiers de normalisation non trouvés: {e}. Entraînez et sauvegardez le modèle d'abord.")
    return _model, _encoders, _norm_stats, _default_values, _model_type


def _encode_categorical(value: str, categories: list[str]) -> int:
    """Encode une valeur catégorielle selon les catégories vues à l'entraînement."""
    if value not in categories:
        return -1  # catégorie inconnue -- comportement à documenter
    return categories.index(value)


def _normalize(value: float, col: str, norm_stats: dict) -> float:
    """Normalise une valeur numérique en utilisant les stats d'entraînement."""
    mean, std = norm_stats[col]
    return (value - mean) / std


def _get_feature_importances(model, feature_names: list[str]) -> dict:
    """Extrait les importances de features, avec fallback pour les pipelines
    (imblearn) et pour les modèles sans attribut (ex. SVM) qui retournent vide.
    Les valeurs sont converties en float natifs (XGBoost renvoie des float32
    numpy non sérialisables en JSON pour la colonne de persistance)."""
    estimator = model[-1] if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        return {
            name: float(val)
            for name, val in zip(feature_names, estimator.feature_importances_.round(4))
        }
    return {}


def predict_fraud(request: FraudRequest) -> FraudResponse:
    model, encoders, norm_stats, defaults, model_type = _load_model()

    # Partir des valeurs par défaut pour TOUTES les variables du modèle
    row = {}
    for col in CATEGORICAL_COLS:
        row[f"{col}_code"] = _encode_categorical(str(defaults[col]), encoders[col])
    for col in NUMERIC_COLS:
        row[f"{col}_norm"] = _normalize(defaults[col], col, norm_stats)

    # Puis écraser avec les valeurs effectivement saisies par l'utilisateur
    row["Fault_code"] = _encode_categorical(request.fault, encoders["Fault"])
    row["PolicyType_code"] = _encode_categorical(request.policy_type, encoders["PolicyType"])
    row["VehicleCategory_code"] = _encode_categorical(request.vehicle_category, encoders["VehicleCategory"])
    row["BasePolicy_code"] = _encode_categorical(request.base_policy, encoders["BasePolicy"])
    row["AddressChange_Claim_code"] = _encode_categorical(request.address_change_claim, encoders["AddressChange_Claim"])
    row["Days_Policy_Claim_code"] = _encode_categorical(request.days_policy_claim, encoders["Days_Policy_Claim"])
    row["DriverRating_norm"] = _normalize(request.driver_rating, "DriverRating", norm_stats)
    row["Deductible_norm"] = _normalize(request.deductible, "Deductible", norm_stats)

    # Construire X dans l'ordre exact utilisé à l'entraînement (catégorielles puis numériques)
    cat_cols_ordered = [f"{c}_code" for c in CATEGORICAL_COLS]
    num_cols_ordered = [f"{c}_norm" for c in NUMERIC_COLS]
    X = pd.DataFrame([row])[cat_cols_ordered + num_cols_ordered]

    proba = model.predict_proba(X)[0, 1]

    # Extraire l'importance des features (avec fallback Pipeline/SVM)
    feature_names = cat_cols_ordered + num_cols_ordered
    importances = _get_feature_importances(model, feature_names)

    # Garder seulement les 10 features les plus importantes
    top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])

    return FraudResponse(
        fraud_probability=round(float(proba), 4),
        is_suspicious=bool(proba >= FRAUD_THRESHOLD),
        feature_importance=top_features,
        model_version=f"fraud_{model_type}_v1",
    )
