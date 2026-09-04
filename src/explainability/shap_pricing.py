"""
src/explainability/shap_pricing.py

Calcule les vraies valeurs SHAP pour les modèles de tarification :
  - GLM Poisson (fréquence)  via shap.LinearExplainer (exact, modèle linéaire avec lien log)
  - GLM Gamma   (sévérité)   via shap.LinearExplainer
  - NGBoost     (sévérité)   via shap.TreeExplainer appliqué sur les base learners

Renvoie des DataFrame de valeurs SHAP prêts à être tracés.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Helpers : build design matrix identique à ce que patsy produit pour le GLM
# ---------------------------------------------------------------------------

def _patsy_design_matrix(df: pd.DataFrame, formula_rhs: str) -> pd.DataFrame:
    """Reconstruit la design matrix one-hot via patsy (même pipeline que l'entraînement)."""
    import patsy
    # On n'a besoin que du RHS (pas de la VD), offset non inclus dans la matrice X
    dm = patsy.dmatrix(formula_rhs, df, return_type="dataframe")
    return dm


GLM_FREQ_FORMULA_RHS = (
    "C(DrivAge_bucket) + C(VehAge_bucket) + C(BM_bucket) "
    "+ C(VehGas) + C(VehBrand) + C(Region) + Density_log"
)

GLM_SEV_FORMULA_RHS = GLM_FREQ_FORMULA_RHS  # même structure


# ---------------------------------------------------------------------------
# 1. SHAP pour GLM Poisson (fréquence)
# ---------------------------------------------------------------------------

def compute_shap_glm_frequency(df_sample: pd.DataFrame):
    """
    Calcule les valeurs SHAP pour le GLM Poisson de fréquence sur df_sample.

    Pour un modèle linéaire avec lien log :
        f(x) = exp(X @ beta)
    Les valeurs SHAP sont calculées dans l'espace log-linéaire via
    shap.LinearExplainer, ce qui est exact (pas d'approximation).

    Paramètres
    ----------
    df_sample : DataFrame avec les features brutes (DrivAge, VehAge, BonusMalus,
                VehGas, VehBrand, Region, Density, Area, Exposure, ...)

    Retourne
    --------
    shap_values : np.ndarray de shape (n_samples, n_features_one_hot)
    feature_names : liste des noms de colonnes one-hot
    base_value : float, valeur SHAP de base (prédiction log pour la moyenne)
    X_dm : DataFrame design matrix one-hot (pour le waterfall)
    """
    model = joblib.load(MODELS_DIR / "glm_poisson.pkl")

    # Design matrix identique à patsy lors de l'entraînement
    X_dm = _patsy_design_matrix(df_sample, GLM_FREQ_FORMULA_RHS)
    feature_names = X_dm.columns.tolist()

    # shap.LinearExplainer accepte un modèle sous la forme (coef, intercept)
    # Pour un modèle statsmodels, params inclut l'Intercept
    coef        = model.params.reindex(feature_names).fillna(0).values
    intercept   = 0.0   # déjà inclus dans coef[0] via patsy "Intercept"

    explainer   = shap.LinearExplainer(
        (coef, intercept),
        X_dm.values,
        feature_perturbation="interventional"
    )
    shap_values = explainer(X_dm.values)

    return shap_values, feature_names, explainer.expected_value, X_dm


def aggregate_shap_by_original_feature(shap_values_array: np.ndarray,
                                        feature_names: list[str]) -> pd.DataFrame:
    """
    Agrège les colonnes SHAP one-hot par variable originale.
    Ex : DrivAge_bucket[T.21-25], DrivAge_bucket[T.26-30] → DrivAge_bucket
    """
    import re
    mapping = {}
    for i, col in enumerate(feature_names):
        # patsy encode ex : "C(DrivAge_bucket)[T.21-25]" -> "DrivAge_bucket"
        m = re.match(r"C\((\w+)\)", col)
        orig = m.group(1) if m else col
        mapping.setdefault(orig, []).append(i)

    agg = {}
    for orig, idxs in mapping.items():
        agg[orig] = shap_values_array[:, idxs].sum(axis=1)

    return pd.DataFrame(agg)


# ---------------------------------------------------------------------------
# 2. SHAP pour NGBoost Sévérité
# ---------------------------------------------------------------------------

NGBOOST_FEATURES = [
    "VehPower_norm", "VehAge_norm", "DrivAge_norm", "BonusMalus_norm",
    "Density_log", "VehBrand_code", "Region_code", "Area_code", "VehGas_code",
]

NGBOOST_FEATURE_DISPLAY = [
    "VehPower", "VehAge", "DrivAge", "BonusMalus",
    "Density_log", "VehBrand", "Region", "Area", "VehGas",
]


def compute_shap_ngboost_severity(df_sample: pd.DataFrame):
    """
    Calcule les valeurs SHAP pour le NGBoost de sévérité.

    On utilise shap.KernelExplainer sur la fonction log(E[Y|X]) prédite
    par le modèle NGBoost complet (300 arbres composites).

    Paramètres
    ----------
    df_sample : DataFrame avec les features CANN normalisées

    Retourne
    --------
    shap_values : np.ndarray (n_samples, n_features)
    feature_display_names : liste des noms lisibles
    """
    model = joblib.load(MODELS_DIR / "ngboost_severity.pkl")

    X = df_sample[NGBOOST_FEATURES].values.astype(float)

    def ngboost_log_mean(X_):
        dist = model.pred_dist(X_)
        return np.log(np.clip(dist.mean(), 1e-6, None))

    # Background : 50 obs pour la vitesse
    n_bg = min(50, len(X))
    rng  = np.random.default_rng(42)
    bg   = X[rng.choice(len(X), n_bg, replace=False)]

    explainer   = shap.KernelExplainer(ngboost_log_mean, bg)
    shap_values = explainer.shap_values(X, nsamples=64, silent=True)

    return shap_values, NGBOOST_FEATURE_DISPLAY
