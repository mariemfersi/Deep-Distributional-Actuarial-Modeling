"""
src/explainability/shap_fraud.py

Calcule les vraies valeurs SHAP pour le Random Forest de détection de fraude
via shap.TreeExplainer (exact pour les forêts aléatoires scikit-learn).
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"


def compute_shap_fraud_rf(df_sample: pd.DataFrame):
    """
    Calcule les valeurs SHAP pour le Random Forest de fraude.

    shap.TreeExplainer est exact pour les Random Forests sklearn.
    Retourne les SHAP values pour la classe positive (fraude = 1).

    Paramètres
    ----------
    df_sample : DataFrame avec les colonnes *_code et *_norm du dataset fraude

    Retourne
    --------
    shap_values : shap.Explanation — valeurs SHAP pour la classe fraude
    feature_names : liste des noms d'affichage (sans suffixe _code/_norm)
    X : np.ndarray des features
    """
    from src.fraud.data import CATEGORICAL_COLS, NUMERIC_COLS

    # Charger le meilleur modèle (XGBoost si présent), sinon le Random Forest
    best_path = MODELS_DIR / "fraud_best_model.pkl"
    model = joblib.load(best_path) if best_path.exists() else joblib.load(MODELS_DIR / "fraud_random_forest.pkl")

    # Colonnes dans l'ordre exact utilisé à l'entraînement
    cat_cols     = [f"{c}_code" for c in CATEGORICAL_COLS]
    num_cols     = [f"{c}_norm" for c in NUMERIC_COLS]
    feature_cols = cat_cols + num_cols

    X = df_sample[feature_cols].values.astype(float)

    # Noms lisibles (sans suffixe)
    feature_names = (
        [c.replace("_code", "") for c in cat_cols]
        + [c.replace("_norm", "") for c in num_cols]
    )

    # TreeExplainer exact sur RandomForest
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # shap_values.values a shape (n, n_features, 2) pour la classification binaire
    # On garde la classe 1 (fraude)
    if shap_values.values.ndim == 3:
        sv_fraud = shap.Explanation(
            values         = shap_values.values[:, :, 1],
            base_values    = shap_values.base_values[:, 1] if shap_values.base_values.ndim > 1
                             else shap_values.base_values,
            data           = shap_values.data,
            feature_names  = feature_names,
        )
    else:
        sv_fraud = shap.Explanation(
            values         = shap_values.values,
            base_values    = shap_values.base_values,
            data           = shap_values.data,
            feature_names  = feature_names,
        )

    return sv_fraud, feature_names, X
