"""
Service d'explicabilité — SHAP pour les modèles de tarification et de fraude.

GLM : SHAP analytique (phi_j = coef_j * (x_j - E[x_j])), exact pour un
      modèle linéaire avec lien log.
CANN : SHAP via KernelExplainer (estimation par approximation), plus coûteux.
Fraude : SHAP TreeExplainer (exact pour les arbres).
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

# Ajoute la racine du projet au path pour importer src/
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from app.schemas.pricing import PricingRequest
from app.services.pricing_service import _build_feature_row

MODELS_DIR = BACKEND_DIR / "models" if (BACKEND_DIR / "models").exists() else PROJECT_ROOT / "models"

# Cache pour les explainers
_fraud_explainer = None
_fraud_background = None
_mean_log_density = None  # moyenne de log(Density) sur le jeu d'entraînement

# Mapping des catégories pour les features catégorielles
# Doit correspondre EXACTEMENT aux paramètres du GLM (model.params).
# La première catégorie (alphabétique) est la référence (drop_first=True dans patsy).
# Les noms de colonnes utilisent la notation C() de patsy : C(Region)[T.Aquitaine].
# source : fréMTPL2freq, formula = ClaimNb ~ C(DrivAge_bucket) + C(VehAge_bucket) +
#   C(BM_bucket) + C(VehGas) + C(VehBrand) + C(Region) + Density_log
FEATURE_CATEGORIES = {
    "DrivAge_bucket": ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"],
    "VehAge_bucket": ["neuf", "recent", "ancien"],
    "BM_bucket": ["50-60", "61-80", "81-100", "101-125", "126-150", "151+"],
    "VehGas": ["Diesel", "Regular"],       # Diesel = référence (alphabetically first)
    "VehBrand": ["B1", "B10", "B11", "B12", "B13", "B14", "B2", "B3", "B4", "B5", "B6"],
    "Region": [
        "Alsace", "Aquitaine", "Auvergne", "Basse-Normandie",
        "Bourgogne", "Bretagne", "Centre", "Champagne-Ardenne", "Corse",
        "Franche-Comte", "Haute-Normandie", "Ile-de-France", "Languedoc-Roussillon", "Limousin",
        "Lorraine", "Midi-Pyrenees", "Nord-Pas-de-Calais", "Pays-de-la-Loire",
        "Picardie", "Poitou-Charentes", "Provence-Alpes-Cotes-D'Azur", "Rhone-Alpes"
    ],
}


def _get_glm_coefficients():
    """Extrait les coefficients du modèle GLM Poisson."""
    model_glm = joblib.load(MODELS_DIR / "glm_poisson.pkl")

    # Le modèle statsmodels GLM a des coefficients dans model_glm.params
    # Les paramètres incluent l'intercept et les coefficients pour chaque feature one-hot encodée
    params = model_glm.params
    return params


def _get_mean_log_density() -> float:
    """Calcule la moyenne de log(Density) sur le jeu d'entraînement (cache).

    Utilisée comme valeur de référence pour SHAP (E[x_ref] pour Density_log).
    """
    global _mean_log_density
    if _mean_log_density is None:
        try:
            from src.pricing.data import build_pricing_dataset, train_valid_test_split
            df = build_pricing_dataset()
            train, _, _ = train_valid_test_split(df)
            _mean_log_density = float(np.log(train["Density"]).mean())
        except Exception:
            # Fallback uniquement si les données ne sont pas accessibles
            _mean_log_density = 5.98  # valeur pré-calculée
    return _mean_log_density


def _build_design_matrix_row(request: PricingRequest) -> pd.DataFrame:
    """
    Construit la ligne de features one-hot encodée comme attendu par le GLM.

    IMPORTANT: Les colonnes sont dérivées directement de model.params
    (et non d'une liste catégorielle codée en dur), garantissant l'alignement
    dimensionnel exact entre X et les coefficients du modèle.
    """
    row = _build_feature_row(request)
    params = _get_glm_coefficients()

    # Mapping feature_name -> valeur dans la requête
    _feature_map = {
        "DrivAge_bucket": str(row["DrivAge_bucket"].iloc[0]),
        "VehAge_bucket": str(row["VehAge_bucket"].iloc[0]),
        "BM_bucket": str(row["BM_bucket"].iloc[0]),
        "VehGas": str(row["VehGas"].iloc[0]),
        "VehBrand": str(row["VehBrand"].iloc[0]),
        "Region": str(row["Region"].iloc[0]),
    }

    # Construire le design matrix en itérant sur les params du modèle
    # (garantit que X.shape[1] == len(params))
    design_data = {}
    for param_name in params.index:
        if param_name == "Intercept":
            design_data["Intercept"] = [1.0]
        elif param_name == "Density_log":
            design_data["Density_log"] = [float(row["Density_log"].iloc[0])]
        elif param_name.startswith("C("):
            # Format patsy : C(Feature)[T.Category]
            feature = param_name.split("[T.")[0][2:-1]  # extraire "Feature" depuis "C(Feature)"
            cat = param_name.split("[T.")[1].rstrip("]")
            value = _feature_map.get(feature, "")
            design_data[param_name] = [1.0 if value == cat else 0.0]

    return pd.DataFrame(design_data)


def explain_cann_interactions(request: PricingRequest) -> dict:
    """
    Diagnostique des interactions non-linéaires capturées par le CANN.

    NOTE : Ceci n'est PAS un calcul SHAP exact. Le CANN (réseau de neurones)
    nécessiterait un KernelExplainer SHAP pour des valeurs exactes, ce qui
    est trop coûteux pour une API en temps réel. Cette fonction calcule des
    métriques d'interaction approximatives (produits de features normalisés)
    pour illustrer les types d'effets non-linéaires que le CANN peut capturer
    mais que le GLM linéaire ne peut pas.
    """
    import torch
    from src.pricing.cann import GroupInteractionNet

    # Construire les features normalisées (conventions d'entraînement)
    row = _build_feature_row(request)
    vp_norm = float(row["VehPower_norm"].iloc[0])
    va_norm = float(row["VehAge_norm"].iloc[0])
    # Convention d'entraînement : Regular=1, Diesel=0 (binaire brut)
    vg_code = 1.0 if request.veh_gas == "Regular" else 0.0

    # Calculer le log_mu_glm (skip connection)
    from src.pricing.models import predict_frequency
    model_glm = joblib.load(MODELS_DIR / "glm_poisson.pkl")
    freq_glm = predict_frequency(model_glm, row).iloc[0]
    log_mu_glm = np.log(freq_glm * request.exposure + 1e-8)

    # Tenter de charger le vrai CANN pour des diagnostics plus précis
    cann_available = False
    cann_model = None
    try:
        cann_stats = joblib.load(MODELS_DIR / "cann_stats.pkl")
        cann_model = GroupInteractionNet(
            n_continuous=3, brand_cardinality=11,
            embedding_dim=2, hidden_dim=20
        )
        cann_model.load_state_dict(torch.load(
            MODELS_DIR / "cann_group_interaction.pt",
            map_location="cpu"
        ))
        cann_model.eval()
        cann_available = True
    except Exception:
        pass

    # Calculer les interactions diagnostiques
    interactions = {}

    power_age = vp_norm * va_norm
    interactions["VehPower x VehAge"] = {
        "value": round(float(power_age), 4),
        "interpretation": "Effet combine puissance + age vehicule",
        "strength": round(abs(float(power_age)), 4),
    }

    power_gas = vp_norm * vg_code
    interactions["VehPower x VehGas"] = {
        "value": round(float(power_gas), 4),
        "interpretation": "Effet combine puissance + type carburant",
        "strength": round(abs(float(power_gas)), 4),
    }

    age_gas = va_norm * vg_code
    interactions["VehAge x VehGas"] = {
        "value": round(float(age_gas), 4),
        "interpretation": "Effet combine age vehicule + carburant",
        "strength": round(abs(float(age_gas)), 4),
    }

    power_sq = vp_norm ** 2
    interactions["VehPower^2"] = {
        "value": round(float(power_sq), 4),
        "interpretation": "Effet non-lineaire de la puissance (quadratique)",
        "strength": round(abs(float(power_sq)), 4),
    }

    # Si le CANN est disponible, comparer sa prédiction au GLM
    residual_magnitude = 0.0
    if cann_available:
        try:
            device = next(cann_model.parameters()).device
            cont = torch.tensor([[vp_norm, va_norm, vg_code]], dtype=torch.float32).to(device)
            brand_code = _BRAND_CODE_MAP.get(request.veh_brand, 0) if hasattr(__import__('app.services.pricing_service', fromlist=['_BRAND_CODE_MAP']), '_BRAND_CODE_MAP') else 0
            from app.services.pricing_service import _BRAND_CODE_MAP
            brand_code = _BRAND_CODE_MAP.get(request.veh_brand, 0)
            br = torch.tensor([brand_code], dtype=torch.long).to(device)
            lg = torch.tensor([log_mu_glm], dtype=torch.float32).to(device)
            with torch.no_grad():
                log_lambda_cann = cann_model(cont, br, lg).item()
            # Le résidu = log_lambda_cann - log_mu_glm
            residual_magnitude = float(abs(log_lambda_cann - log_mu_glm))
        except Exception:
            pass

    sorted_interactions = dict(sorted(
        interactions.items(),
        key=lambda x: x[1]["strength"],
        reverse=True
    ))

    return {
        "interactions": sorted_interactions,
        "total_interaction_effect": round(sum(i["strength"] for i in interactions.values()), 4),
        "dominant_interaction": max(interactions.items(), key=lambda x: x[1]["strength"])[0],
        "cann_residual_magnitude": round(residual_magnitude, 4),
        "note": "Interactions approximatives (produits de features normalisées). Pas des SHAP interaction values exactes. Le residu CANN mesure l'ecart effectif entre GLM et CANN sur ce profil.",
        "model_version": "cann_diagnostics_v1"
    }


def explain_pricing(request: PricingRequest) -> dict:
    """
    Retourne les valeurs SHAP pour une prédiction de tarification (GLM Poisson).

    Pour un modèle linéaire (GLM avec lien log), les valeurs SHAP peuvent être calculées
    analytiquement : SHAP_i = coef_i * (x_i - E[x_i]) où E[x_i] est la valeur moyenne
    de la feature dans les données d'entraînement (background).

    Comme on n'a pas accès direct aux moyennes d'entraînement, on utilise l'approximation
    SHAP pour modèles linéaires : phi_i = coef_i * (x_i - x_ref_i) où x_ref est une
    valeur de référence (ex: la catégorie de référence ou la moyenne empirique).
    """
    # Charger les coefficients
    params = _get_glm_coefficients()

    # Construire le design matrix pour cette requête
    X = _build_design_matrix_row(request)

    # Calculer la prédiction (log scale pour le lien log)
    log_pred = float((X.values @ params.values).sum())

    # Valeur de base (prédiction pour un individu de référence)
    # Référence: toutes les features catégorielles à leur niveau de référence (0 en one-hot),
    # Density_log à sa moyenne empirique (approximée par log(5000) ~ 8.5)
    X_ref = X.copy()
    for col in X_ref.columns:
        if col != "Intercept" and col != "Density_log":
            X_ref[col] = 0.0
    X_ref["Density_log"] = _get_mean_log_density()
    base_log = float((X_ref.values @ params.values).sum())

    # Note: Pour VehGas, la référence est "Diesel" (première catégorie dans FEATURE_CATEGORIES),
    # ce qui correspond à toutes les colonnes one-hot VehGas[T.*] = 0

    # Calculer les contributions SHAP pour chaque feature
    # Pour un modèle linéaire: phi_j = beta_j * (x_j - x_ref_j)
    shap_contributions = {}

    # Grouper par feature originale (avant one-hot)
    feature_shap = {}

    for param_name, coef in params.items():
        if param_name == "Intercept":
            continue

        # Déterminer la feature originale et sa valeur
        # Les noms de params utilisent la notation C() de patsy : C(Region)[T.Aquitaine]
        if param_name.startswith("C(DrivAge_bucket)[T."):
            orig_feature = "DrivAge_bucket"
            cat = param_name.split("[T.")[1].rstrip("]")
            x_val = X[param_name].iloc[0]
            x_ref_val = 0.0
        elif param_name.startswith("C(VehAge_bucket)[T."):
            orig_feature = "VehAge_bucket"
            cat = param_name.split("[T.")[1].rstrip("]")
            x_val = X[param_name].iloc[0]
            x_ref_val = 0.0
        elif param_name.startswith("C(BM_bucket)[T."):
            orig_feature = "BM_bucket"
            cat = param_name.split("[T.")[1].rstrip("]")
            x_val = X[param_name].iloc[0]
            x_ref_val = 0.0
        elif param_name.startswith("C(VehGas)[T."):
            orig_feature = "VehGas"
            cat = param_name.split("[T.")[1].rstrip("]")
            x_val = X[param_name].iloc[0]
            x_ref_val = 0.0
        elif param_name.startswith("C(VehBrand)[T."):
            orig_feature = "VehBrand"
            cat = param_name.split("[T.")[1].rstrip("]")
            x_val = X[param_name].iloc[0]
            x_ref_val = 0.0
        elif param_name.startswith("C(Region)[T."):
            orig_feature = "Region"
            cat = param_name.split("[T.")[1].rstrip("]")
            x_val = X[param_name].iloc[0]
            x_ref_val = 0.0
        elif param_name == "Density_log":
            orig_feature = "Density_log"
            x_val = X[param_name].iloc[0]
            x_ref_val = _get_mean_log_density()
        else:
            continue

        # Contribution SHAP pour cette colonne one-hot
        phi = coef * (x_val - x_ref_val)

        # Accumuler par feature originale
        if orig_feature not in feature_shap:
            feature_shap[orig_feature] = 0.0
        feature_shap[orig_feature] += phi

    # Base value = log prediction for reference individual
    base_value = base_log

    # Formater les résultats
    shap_dict = {
        "base_value": round(base_value, 4),
        "shap_values": [
            {
                "feature": name,
                "value": round(float(val), 4)
            }
            for name, val in sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)
        ]
    }

    return shap_dict


def _load_fraud_explainer():
    """Charge l'explainer SHAP pour le modèle de fraude.

    Le modèle est un imblearn.Pipeline (SMOTE + XGBoost).  SHAP TreeExplainer
    ne supporte pas imblearn.Pipeline, on extrait donc le classifieur XGBoost
    depuis le pipeline.  SMOTE n'est actif qu'à l'entraînement (fit) — pendant
    la prédiction le pipeline se comporte comme le classifieur seul.
    """
    global _fraud_explainer, _fraud_background
    if _fraud_explainer is None:
        try:
            # Charger le meilleur modèle (XGBoost si présent), sinon le Random Forest
            best_path = MODELS_DIR / "fraud_best_model.pkl"
            raw_model = joblib.load(best_path) if best_path.exists() else joblib.load(MODELS_DIR / "fraud_random_forest.pkl")

            # Extraire le classifieur du pipeline imblearn si nécessaire
            from imblearn.pipeline import Pipeline as ImbPipeline
            if isinstance(raw_model, ImbPipeline):
                classifier = raw_model.named_steps.get("classifier", raw_model[-1])
            else:
                classifier = raw_model

            # Charger les données de fond
            from src.fraud.data import load_fraud_data, prepare_fraud_features
            df = load_fraud_data()
            df = prepare_fraud_features(df)

            # Sélectionner les features du modèle
            cat_cols = [c for c in df.columns if c.endswith("_code")]
            num_cols = [c for c in df.columns if c.endswith("_norm")]
            feature_cols = cat_cols + num_cols

            X_background = df[feature_cols].sample(n=100, random_state=42)

            # Créer l'explainer SHAP sur le classifieur brut (XGBoost ou RF)
            _fraud_explainer = shap.TreeExplainer(classifier)
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

    # Pour Random Forest binary classification, shap_values peut être:
    # - list[ndarray] (ancien API) : [classe_0, classe_1] chacun (n_samples, n_features)
    # - ndarray (nouvel API) : (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_values_array = shap_values[1][0]  # Classe positive (fraude)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_array = shap_values[0, :, 1]  # (n_features,) pour classe positive
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_values_array = shap_values[0]  # XGBoost binaire : un seul tableau pour la classe positive
    else:
        shap_values_array = shap_values[0]

    ev = explainer.expected_value
    base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) and np.ndim(ev) > 0 else ev)

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