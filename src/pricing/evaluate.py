"""
Module Pricing — Évaluation par indice de Gini et courbes de lift.

Méthode de la courbe de Lorenz ordonnée (Frees, Meyers & Cummings, 2011),
standard actuariel pour comparer le pouvoir discriminant de deux modèles
de tarification sur un même échantillon test.
"""

import numpy as np
import pandas as pd


def _trapezoid(y, x):
    """Wrapper compatible NumPy 1.x/2.x pour l'intégration trapézoïdale."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    # NumPy < 2.0
    return np.trapz(y, x)


def compute_gini_index(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray) -> float:
    """
    Indice de Gini via la courbe de Lorenz ordonnée par risque prédit croissant.

    y_true : sinistres observés (ClaimNb)
    y_pred : risque prédit (fréquence, en échelle comptage attendu = lambda * exposure)
    exposure : exposition de chaque police
    """
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    exposure_sorted = exposure[order]

    cum_exposure = np.cumsum(exposure_sorted) / exposure_sorted.sum()
    cum_claims = np.cumsum(y_true_sorted) / y_true_sorted.sum()

    # Aire sous la courbe de Lorenz (méthode des trapèzes)
    lorenz_area = _trapezoid(cum_claims, cum_exposure)

    # Gini = 2 * (aire sous la diagonale (0.5) - aire sous la courbe de Lorenz)
    gini = 1 - 2 * lorenz_area
    return gini


def compute_lorenz_curve(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray) -> pd.DataFrame:
    """Retourne les points de la courbe de Lorenz ordonnée, pour visualisation."""
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    exposure_sorted = exposure[order]

    cum_exposure = np.cumsum(exposure_sorted) / exposure_sorted.sum()
    cum_claims = np.cumsum(y_true_sorted) / y_true_sorted.sum()

    return pd.DataFrame({"cum_exposure": cum_exposure, "cum_claims": cum_claims})


def compute_lift_table(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Table de lift : regroupe les polices en déciles de risque prédit,
    compare fréquence observée vs prédite par décile.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "exposure": exposure})
    df["decile"] = pd.qcut(df["y_pred"], q=n_bins, labels=False, duplicates="drop")

    lift = df.groupby("decile").apply(
        lambda g: pd.Series({
            "n_policies": len(g),
            "exposure_sum": g["exposure"].sum(),
            "observed_freq": g["y_true"].sum() / g["exposure"].sum(),
            "predicted_freq": g["y_pred"].sum() / g["exposure"].sum(),
        })
    ).reset_index()

    return lift