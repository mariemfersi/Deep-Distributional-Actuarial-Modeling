"""
Service de détection de fraude — expose le Random Forest supervisé
(meilleur modèle du chapitre 6, AUC-ROC 0.815). Pas de volet relationnel
exposé : aucun GNN n'a été entraîné (cf. chapitre 6, 4 tentatives de
graphe invalidées par test d'homophilie).
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from app.schemas.fraud import FraudRequest, FraudResponse

MODELS_DIR = PROJECT_ROOT / "models"
FRAUD_THRESHOLD = 0.5  # seuil de décision, ajustable selon le compromis précision/rappel souhaité

# Colonnes du modèle (ordre exact de get_feature_matrix : catégorielles puis numériques)
CATEGORICAL_COLS = [
    "Month", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed", "MonthClaimed",
    "Sex", "MaritalStatus", "Fault", "PolicyType", "VehicleCategory", "VehiclePrice",
    "Days_Policy_Accident", "Days_Policy_Claim", "PastNumberOfClaims", "AgeOfVehicle",
    "AgeOfPolicyHolder", "PoliceReportFiled", "WitnessPresent", "AgentType",
    "NumberOfSuppliments", "AddressChange_Claim", "NumberOfCars", "BasePolicy",
]

NUMERIC_COLS = ["WeekOfMonth", "Age", "RepNumber", "Deductible", "DriverRating", "Year"]

_model_rf = None
_encoders = None
_norm_stats = None
_default_values = None


def _load_model():
    global _model_rf, _encoders, _norm_stats, _default_values
    if _model_rf is None:
        try:
            _model_rf = joblib.load(MODELS_DIR / "fraud_random_forest.pkl")
            _encoders = joblib.load(MODELS_DIR / "fraud_encoders.pkl")
            _norm_stats = joblib.load(MODELS_DIR / "fraud_normalization_stats.pkl")
            _default_values = joblib.load(MODELS_DIR / "fraud_default_values.pkl")
        except FileNotFoundError as e:
            raise ValueError(f"Modèle Random Forest ou fichiers de normalisation non trouvés: {e}. Entraînez et sauvegardez le modèle d'abord.")
    return _model_rf, _encoders, _norm_stats, _default_values


def _encode_categorical(value: str, categories: list[str]) -> int:
    """Encode une valeur catégorielle selon les catégories vues à l'entraînement."""
    if value not in categories:
        return -1  # catégorie inconnue -- comportement à documenter
    return categories.index(value)


def _normalize(value: float, col: str, norm_stats: dict) -> float:
    """Normalise une valeur numérique en utilisant les stats d'entraînement."""
    mean, std = norm_stats[col]
    return (value - mean) / std


def predict_fraud(request: FraudRequest) -> FraudResponse:
    model, encoders, norm_stats, defaults = _load_model()

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

    # Extraire l'importance des features du Random Forest
    feature_names = cat_cols_ordered + num_cols_ordered
    importances = dict(zip(feature_names, model.feature_importances_.round(4)))
    
    # Garder seulement les 10 features les plus importantes
    top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])

    return FraudResponse(
        fraud_probability=round(float(proba), 4),
        is_suspicious=bool(proba >= FRAUD_THRESHOLD),
        feature_importance=top_features,
    )
