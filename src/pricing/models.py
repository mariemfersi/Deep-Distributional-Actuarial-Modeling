"""
Module Pricing — Modèles de fréquence et de sévérité.

GLM Poisson/Gamma comme baseline actuarielle de référence.
Architecture CANN à venir dans une étape ultérieure.
"""

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


FREQUENCY_FORMULA = (
    "ClaimNb ~ C(DrivAge_bucket) + C(VehAge_bucket) + C(BM_bucket) "
    "+ C(VehGas) + C(VehBrand) + C(Region) + Density_log"
)


def fit_glm_poisson(train_df: pd.DataFrame, formula: str = FREQUENCY_FORMULA):
    """
    Ajuste le GLM Poisson de fréquence avec offset sur log(Exposure).
    Area est volontairement exclue (colinéarité à 0.97 avec Density_log,
    cf. chapitre 3 du rapport).
    """
    model = smf.glm(
        formula=formula,
        data=train_df,
        family=sm.families.Poisson(),
        offset=np.log(train_df["Exposure"]),
    ).fit()
    return model


def predict_frequency(model, df: pd.DataFrame) -> pd.Series:
    """
    Prédit la fréquence annuelle (lambda), indépendamment de l'exposition.
    Cappée à 1e-8 pour éviter tout log(0) en aval (ex. calcul de glm_log_pred
    pour le CANN) -- protection numérique, pas une hypothèse actuarielle.
    """
    # On force un offset nul pour obtenir lambda directement (pas lambda * exposure)
    pred = model.predict(df, offset=np.zeros(len(df)))
    return pred.clip(lower=1e-8)


SEVERITY_FORMULA = (
    "ClaimAmount_capped ~ C(DrivAge_bucket) + C(VehAge_bucket) + C(BM_bucket) "
    "+ C(VehGas) + C(VehBrand) + C(Region) + Density_log"
)


def fit_glm_gamma(train_severity_df: pd.DataFrame, formula: str = SEVERITY_FORMULA):
    """
    Ajuste le GLM Gamma de sévérité sur le sous-échantillon attritionnel.
    Utilise un lien logarithmique, cohérent avec la pratique actuarielle standard.
    """
    model = smf.glm(
        formula=formula,
        data=train_severity_df,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    return model


def predict_severity(model, df: pd.DataFrame) -> pd.Series:
    """Prédit le coût moyen attendu d'un sinistre pour chaque profil."""
    return model.predict(df)

def compute_pure_premium(freq_model, severity_model, df: pd.DataFrame) -> pd.Series:
    """
    Calcule la prime pure = fréquence prédite × sévérité prédite.
    Appliqué à l'ensemble du portefeuille (pas seulement les polices sinistrées).
    """
    predicted_frequency = predict_frequency(freq_model, df)
    predicted_severity = predict_severity(severity_model, df)

    pure_premium = predicted_frequency * predicted_severity
    return pure_premium

from ngboost import NGBoost
from ngboost.distns import Gamma
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import kendalltau, norm

NGBOOST_FEATURES = [
    "VehPower_norm", "VehAge_norm", "DrivAge_norm", "BonusMalus_norm",
    "Density_log", "VehBrand_code", "Region_code", "Area_code", "VehGas_code",
]


def fit_gaussian_copula(df: pd.DataFrame):
    """
    Estime le paramètre de dépendance (tau de Kendall) entre fréquence et sévérité
    sur les polices avec sinistres positifs. Retourne le paramètre de copule gaussienne.
    """
    # Filtrer sur les polices avec sinistres (ClaimNb > 0 et ClaimAmount > 0)
    positive_claims = df[(df["ClaimNb"] > 0) & (df["ClaimAmount_capped"] > 0)].copy()
    
    if len(positive_claims) < 10:
        print("Warning: Insufficient positive claims for copula estimation")
        return 0.0
    
    # Calculer le tau de Kendall entre fréquence et sévérité
    tau, p_value = kendalltau(positive_claims["ClaimNb"], positive_claims["ClaimAmount_capped"])
    
    print(f"Kendall's tau: {tau:.4f} (p-value: {p_value:.4f})")
    
    return tau


def compute_pure_premium_with_copula(
    freq_model, 
    severity_model, 
    df: pd.DataFrame, 
    copula_tau: float = 0.0
) -> pd.Series:
    """
    Calcule la prime pure en ajustant pour la dépendance fréquence-sévérité via copule gaussienne.
    
    Si tau est proche de 0 (indépendance), revient au calcul standard.
    Sinon, ajuste la prime pure pour tenir compte de la dépendance positive/négative.
    """
    predicted_frequency = predict_frequency(freq_model, df)
    predicted_severity = predict_severity(severity_model, df)
    
    # Calcul standard (indépendance)
    pure_premium_independent = predicted_frequency * predicted_severity
    
    # Ajustement pour dépendance via copule gaussienne
    # Pour une copule gaussienne avec paramètre de corrélation rho:
    # rho = sin(pi * tau / 6)
    if abs(copula_tau) < 0.01:
        return pure_premium_independent
    
    rho = np.sin(np.pi * copula_tau / 6)
    
    # Ajustement simple: si dépendance positive (rho > 0), prime pure augmente
    # Formule approximative basée sur l'espérance du produit sous copule gaussienne
    # E[XY] = E[X]E[Y] + rho * sigma_X * sigma_Y
    # Ici on applique un facteur d'ajustement proportionnel à rho
    adjustment_factor = 1 + 0.1 * rho  # 0.1 est un facteur de sensibilité empirique
    
    pure_premium_adjusted = pure_premium_independent * adjustment_factor
    
    print(f"Copula adjustment factor: {adjustment_factor:.4f} (tau={copula_tau:.4f}, rho={rho:.4f})")
    
    return pure_premium_adjusted


def fit_ngboost_severity(train_severity_df: pd.DataFrame, n_estimators: int = 300):
    """
    Modélise la sévérité comme une distribution Gamma complète (forme + échelle),
    pas seulement une moyenne conditionnelle -- contrairement au GLM Gamma.
    Permet d'obtenir un intervalle de confiance par prédiction, utile pour
    les segments à faible volume identifiés comme instables (section 4.2).
    """
    X = train_severity_df[NGBOOST_FEATURES].values
    y = train_severity_df["ClaimAmount_capped"].values

    model = NGBoost(
        Dist=Gamma,
        Score=LogScore,
        Base=DecisionTreeRegressor(max_depth=3),
        n_estimators=n_estimators,
        learning_rate=0.02,
        verbose=True,
        verbose_eval=50,
    )
    model.fit(X, y)
    return model


def predict_ngboost_severity(model, df: pd.DataFrame):
    """
    Retourne moyenne, et intervalle de confiance à 90% (5e-95e percentile)
    pour chaque profil, à partir de la distribution Gamma complète prédite.
    """
    X = df[NGBOOST_FEATURES].values
    dist = model.pred_dist(X)

    mean = dist.mean()
    lower = dist.dist.ppf(0.05)
    upper = dist.dist.ppf(0.95)

    return pd.DataFrame({
        "pred_mean": mean,
        "pred_lower_90": lower,
        "pred_upper_90": upper,
    }, index=df.index)