"""
Service d'explicabilité — SHAP pour les modèles de tarification et de fraude.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap

# Ajoute la racine du projet au path pour importer src/
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from backend.app.schemas.pricing import PricingRequest

MODELS_DIR = PROJECT_ROOT / "models"

# Cache pour les explainers
_glm_explainer = None
_fraud_explainer = None
_glm_background = None
_fraud_background = None


def _load_glm_explainer():
    """Charge l'explainer SHAP pour le modèle GLM de tarification."""
    global _glm_explainer, _glm_background
    if _glm_explainer is None:
        try:
            # Charger le modèle GLM
            model_glm = joblib.load(MODELS_DIR / "glm_poisson.pkl")
            
            # Créer un dataset de fond (background) pour SHAP
            # On utilise les données d'entraînement pour créer un background représentatif
            try:
                from src.pricing.data import load_pricing_data
                df = load_pricing_data()
                
                # Préparer les features comme dans le service pricing
                df["DrivAge_bucket"] = pd.cut(
                    df["DrivAge"], bins=[17, 20, 25, 30, 40, 50, 70, 120],
                    labels=["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]
                )
                df["VehAge_bucket"] = pd.cut(
                    df["VehAge"], bins=[-1, 0, 10, 200], labels=["neuf", "recent", "ancien"]
                )
                df["BM_bucket"] = pd.cut(
                    df["BonusMalus"], bins=[49, 60, 80, 100, 125, 150, 350],
                    labels=["50-60", "61-80", "81-100", "101-125", "126-150", "151+"]
                )
                df["Density_log"] = np.log(df["Density"])
                df["ClaimAmount_capped"] = 0
                
                # Sélectionner les features utilisées par le modèle
                feature_cols = ["DrivAge_bucket", "VehAge_bucket", "BM_bucket", 
                              "VehGas", "VehBrand", "Region", "Density_log"]
                
                # Encoder les variables catégorielles
                X_background = pd.get_dummies(df[feature_cols], drop_first=False)
                
                # S'assurer que toutes les colonnes du modèle sont présentes
                model_features = model_glm.model.exog_names
                for col in model_features:
                    if col not in X_background.columns:
                        X_background[col] = 0
                
                X_background = X_background[model_features]
                
                # Prendre un échantillon de 100 observations pour le background
                X_background_sample = X_background.sample(n=min(100, len(X_background)), random_state=42)
                
                # Créer l'explainer SHAP
                _glm_explainer = shap.Explainer(model_glm.predict, X_background_sample)
                _glm_background = X_background_sample
            except Exception as e:
                # Fallback: créer un background synthétique si le chargement des données échoue
                print(f"Warning: Could not load pricing data for SHAP background: {e}")
                # Créer un background synthétique basé sur les features du modèle
                model_features = model_glm.model.exog_names
                X_background_sample = pd.DataFrame(
                    np.random.randn(100, len(model_features)),
                    columns=model_features
                )
                _glm_explainer = shap.Explainer(model_glm.predict, X_background_sample)
                _glm_background = X_background_sample
            
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement de l'explainer GLM: {e}")
    
    return _glm_explainer, _glm_background


def _build_feature_row_for_shap(request: PricingRequest) -> pd.DataFrame:
    """
    Construit une ligne de features au format attendu par le GLM pour SHAP.
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
    row["ClaimAmount_capped"] = 0

    return row


def explain_pricing(request: PricingRequest) -> dict:
    """
    Retourne les valeurs SHAP pour une prédiction de tarification.
    """
    explainer, background = _load_glm_explainer()
    
    # Construire les features
    row = _build_feature_row_for_shap(request)
    
    # Préparer les features comme pour le background
    feature_cols = ["DrivAge_bucket", "VehAge_bucket", "BM_bucket", 
                  "VehGas", "VehBrand", "Region", "Density_log"]
    X = pd.get_dummies(row[feature_cols], drop_first=False)
    
    # S'assurer que toutes les colonnes sont présentes
    model_features = background.columns.tolist()
    for col in model_features:
        if col not in X.columns:
            X[col] = 0
    
    X = X[model_features]
    
    # Calculer les valeurs SHAP
    shap_values = explainer(X)
    
    # Formater les résultats
    feature_names = X.columns.tolist()
    shap_values_array = shap_values.values[0]
    base_value = float(shap_values.base_values[0])
    
    # Créer un dictionnaire de résultats
    shap_dict = {
        "base_value": round(base_value, 4),
        "shap_values": [
            {
                "feature": name,
                "value": round(float(val), 4)
            }
            for name, val in zip(feature_names, shap_values_array)
        ]
    }
    
    return shap_dict


def _load_fraud_explainer():
    """Charge l'explainer SHAP pour le modèle Random Forest de fraude."""
    global _fraud_explainer, _fraud_background
    if _fraud_explainer is None:
        try:
            # Charger le modèle Random Forest
            model_rf = joblib.load(MODELS_DIR / "fraud_random_forest.pkl")
            
            # Charger les données de fond
            from src.fraud.data import load_fraud_data, prepare_fraud_features
            df = load_fraud_data()
            df = prepare_fraud_features(df)
            
            # Sélectionner les features du modèle
            cat_cols = [c for c in df.columns if c.endswith("_code")]
            num_cols = [c for c in df.columns if c.endswith("_norm")]
            feature_cols = cat_cols + num_cols
            
            X_background = df[feature_cols].sample(n=100, random_state=42)
            
            # Créer l'explainer SHAP
            _fraud_explainer = shap.TreeExplainer(model_rf)
            _fraud_background = X_background
            
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement de l'explainer fraude: {e}")
    
    return _fraud_explainer, _fraud_background


def explain_fraud(request_data: dict) -> dict:
    """
    Retourne les valeurs SHAP pour une prédiction de fraude.
    request_data doit contenir les mêmes champs que FraudRequest.
    """
    explainer, background = _load_fraud_explainer()
    
    # Charger les encodeurs et stats de normalisation
    encoders = joblib.load(MODELS_DIR / "fraud_encoders.pkl")
    norm_stats = joblib.load(MODELS_DIR / "fraud_normalization_stats.pkl")
    defaults = joblib.load(MODELS_DIR / "fraud_default_values.pkl")
    
    # Colonnes du modèle
    from src.fraud.data import CATEGORICAL_COLS, NUMERIC_COLS
    
    # Construire la ligne de features avec les valeurs par défaut
    row = {}
    for col in CATEGORICAL_COLS:
        categories = encoders.get(col, [])
        value = str(defaults.get(col, ""))
        if value in categories:
            row[f"{col}_code"] = categories.index(value)
        else:
            row[f"{col}_code"] = -1
    
    for col in NUMERIC_COLS:
        mean, std = norm_stats.get(col, (0, 1))
        value = defaults.get(col, 0)
        row[f"{col}_norm"] = (value - mean) / std
    
    # Écraser avec les valeurs de la requête
    if "fault" in request_data:
        categories = encoders.get("Fault", [])
        value = request_data["fault"]
        if value in categories:
            row["Fault_code"] = categories.index(value)
    
    if "policy_type" in request_data:
        categories = encoders.get("PolicyType", [])
        value = request_data["policy_type"]
        if value in categories:
            row["PolicyType_code"] = categories.index(value)
    
    if "vehicle_category" in request_data:
        categories = encoders.get("VehicleCategory", [])
        value = request_data["vehicle_category"]
        if value in categories:
            row["VehicleCategory_code"] = categories.index(value)
    
    if "base_policy" in request_data:
        categories = encoders.get("BasePolicy", [])
        value = request_data["base_policy"]
        if value in categories:
            row["BasePolicy_code"] = categories.index(value)
    
    if "address_change_claim" in request_data:
        categories = encoders.get("AddressChange_Claim", [])
        value = request_data["address_change_claim"]
        if value in categories:
            row["AddressChange_Claim_code"] = categories.index(value)
    
    if "days_policy_claim" in request_data:
        categories = encoders.get("Days_Policy_Claim", [])
        value = request_data["days_policy_claim"]
        if value in categories:
            row["Days_Policy_Claim_code"] = categories.index(value)
    
    if "driver_rating" in request_data:
        mean, std = norm_stats.get("DriverRating", (0, 1))
        row["DriverRating_norm"] = (request_data["driver_rating"] - mean) / std
    
    if "deductible" in request_data:
        mean, std = norm_stats.get("Deductible", (0, 1))
        row["Deductible_norm"] = (request_data["deductible"] - mean) / std
    
    # Construire X dans l'ordre exact
    cat_cols_ordered = [f"{c}_code" for c in CATEGORICAL_COLS]
    num_cols_ordered = [f"{c}_norm" for c in NUMERIC_COLS]
    feature_cols = cat_cols_ordered + num_cols_ordered
    
    X = pd.DataFrame([row])[feature_cols]
    
    # Calculer les valeurs SHAP
    shap_values = explainer.shap_values(X)
    
    # Pour Random Forest binary classification, shap_values est une liste
    if isinstance(shap_values, list):
        shap_values_array = shap_values[1][0]  # Classe positive (fraude)
    else:
        shap_values_array = shap_values[0]
    
    base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value)
    
    # Créer un dictionnaire de résultats
    shap_dict = {
        "base_value": round(base_value, 4),
        "shap_values": [
            {
                "feature": name,
                "value": round(float(val), 4)
            }
            for name, val in zip(feature_cols, shap_values_array)
        ]
    }
    
    return shap_dict
