"""
Module Reserving — Modèles de provisionnement.

Baseline Mack (chain-ladder stochastique) comme référence actuarielle
classique, à comparer aux approches Deep Learning (Deep Triangle,
conformal prediction) développées dans la suite du module.
"""

import pandas as pd
import numpy as np
import chainladder as cl


def fit_mack_for_company(observed: pd.DataFrame, grcode: int):
    """Ajuste le modèle de Mack pour une compagnie donnée."""
    df = observed[observed["GRCODE"] == grcode].copy()

    triangle = cl.Triangle(
        data=df,
        origin="AccidentYear",
        development="DevelopmentYear",
        columns=["CumPaidLoss"],
        cumulative=True,
    )

    model = cl.MackChainladder()
    model.fit(triangle)
    return model


def evaluate_mack_coverage(observed: pd.DataFrame, future: pd.DataFrame, grcode: int, z_90: float = 1.645):
    """
    Évalue la couverture de l'intervalle à 90% de Mack pour une compagnie,
    en comparant à la réalisation future réelle. Retourne None si le modèle
    échoue à converger (triangle trop petit/dégénéré pour cette compagnie).
    """
    try:
        model = fit_mack_for_company(observed, grcode)

        ibnr = model.ibnr_.to_frame().iloc[:, 0]
        ibnr.index = ibnr.index.year

        std_err = model.mack_std_err_.to_frame()[9999]
        std_err.index = std_err.index.year

        # Vérité terrain
        obs_company = observed[observed["GRCODE"] == grcode]
        fut_company = future[future["GRCODE"] == grcode]

        true_ultimate = fut_company[fut_company["DevelopmentLag"] == 10].set_index("AccidentYear")["CumPaidLoss"]
        paid_at_eval = obs_company.sort_values("DevelopmentLag").groupby("AccidentYear")["CumPaidLoss"].last()
        true_ibnr = true_ultimate - paid_at_eval

        results = pd.DataFrame({
            "ibnr_mack": ibnr,
            "ibnr_reel": true_ibnr,
            "std_err": std_err,
        }).dropna()

        if len(results) == 0:
            return None

        results["lower_90"] = results["ibnr_mack"] - z_90 * results["std_err"]
        results["upper_90"] = results["ibnr_mack"] + z_90 * results["std_err"]
        results["covered_90"] = (results["ibnr_reel"] >= results["lower_90"]) & (results["ibnr_reel"] <= results["upper_90"])
        results["grcode"] = grcode

        return results

    except Exception as e:
        return None



def split_conformal_calibration(full_results: pd.DataFrame, alpha: float = 0.10, calib_frac: float = 0.5, seed: int = 123):
    """
    Prédiction conforme normalisée (poids = erreur standard de Mack) sur les
    réserves IBNR. Corrige empiriquement le niveau de couverture de
    l'intervalle Mack, sans hypothèse de normalité asymptotique.
    """
    df = full_results.dropna(subset=["ibnr_mack", "ibnr_reel", "std_err"]).copy()
    df = df[df["std_err"] > 0]  # évite division par zéro

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_calib = int(calib_frac * len(df))

    calib = df.iloc[idx[:n_calib]]
    test = df.iloc[idx[n_calib:]].copy()

    # Scores de non-conformité normalisés sur le jeu de calibration
    scores = (calib["ibnr_reel"] - calib["ibnr_mack"]).abs() / calib["std_err"]

    # Quantile empirique avec correction à taille finie (Vovk et al., 2005)
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q_hat = scores.quantile(q_level)

    # Application au jeu de test
    test["lower_conformal"] = test["ibnr_mack"] - q_hat * test["std_err"]
    test["upper_conformal"] = test["ibnr_mack"] + q_hat * test["std_err"]
    test["covered_conformal"] = (test["ibnr_reel"] >= test["lower_conformal"]) & (test["ibnr_reel"] <= test["upper_conformal"])

    return test, q_hat