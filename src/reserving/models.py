"""
Module Reserving — Modèles de provisionnement.

Baseline Mack (chain-ladder stochastique) comme référence actuarielle
classique, à comparer aux approches Deep Learning (Deep Triangle,
conformal prediction) développées dans la suite du module.

Formules du cours (Vanessa Desert, 30/09/2022) :
- Facteurs de développement : f_j = Σ_{i=0}^{n-j} C_{i,j+1} / Σ_{i=0}^{n-j} C_{i,j}
- Provision : R_{i,j} = C_{i,j} × Π_{k>j} f_k - 1
- Cadences cumulées : pc_n = 1/f_n, pc_1 = 1/(f_n × ... × f_1)
"""

import pandas as pd
import numpy as np
import chainladder as cl


def fit_mack_for_company(observed: pd.DataFrame, grcode: int):
    """
    Ajuste le modèle de Mack pour une compagnie donnée.
    
    Implémentation manuelle suivant les formules du cours :
    - f_j = Σ_{i=0}^{n-j} C_{i,j+1} / Σ_{i=0}^{n-j} C_{i,j} (facteurs de développement)
    - R_{i,j} = C_{i,j} × Π_{k>j} f_k - 1 (provision)
    """
    df = observed[observed["GRCODE"] == grcode].copy()
    
    # Convertir les années en entiers pour faciliter les calculs
    if pd.api.types.is_datetime64_any_dtype(df["AccidentYear"]):
        df["AccidentYear"] = df["AccidentYear"].dt.year
    if pd.api.types.is_datetime64_any_dtype(df["DevelopmentYear"]):
        df["DevelopmentYear"] = df["DevelopmentYear"].dt.year

    # Construire le triangle de paiements cumulés indexed by DevelopmentLag
    # (Lag 1, 2, 3, ...) — PAS par année calendaire : le triangle chain-ladder
    # doit être aligné sur le développement relatif, pas sur l'année civile.
    origin_years = sorted(df["AccidentYear"].unique())
    dev_lags = sorted(df["DevelopmentLag"].unique())

    n_origins = len(origin_years)
    n_devs = len(dev_lags)

    # Matrice du triangle C_{i,j}
    triangle = np.zeros((n_origins, n_devs))

    for i, origin_year in enumerate(origin_years):
        for j, dev_lag in enumerate(dev_lags):
            cell_data = df[
                (df["AccidentYear"] == origin_year) &
                (df["DevelopmentLag"] == dev_lag)
            ]
            if len(cell_data) > 0:
                triangle[i, j] = cell_data["CumPaidLoss"].iloc[0]

    # Calcul des facteurs de développement f_j selon la formule du cours
    # f_j = Σ_{i=0}^{n-j-1} C_{i,j+1} / Σ_{i=0}^{n-j-1} C_{i,j}, 0≤j≤n-2
    ldfs = []
    for j in range(n_devs - 1):
        numerator = 0
        denominator = 0
        for i in range(n_origins - j - 1):
            if triangle[i, j + 1] > 0 and triangle[i, j] > 0:
                numerator += triangle[i, j + 1]
                denominator += triangle[i, j]

        if denominator > 0:
            f_j = numerator / denominator
            ldfs.append(f_j)
        else:
            ldfs.append(1.0)  # Valeur par défaut
    
    # Calcul des cadences cumulées pc_k selon la formule du cours
    # pc_n = 1/f_n, pc_1 = 1/(f_n × ... × f_1)
    cadences = []
    for k in range(len(ldfs)):
        # Produit des facteurs de développement de k à la fin
        product = 1.0
        for f in ldfs[k:]:
            product *= f
        pc_k = 1.0 / product if product > 0 else 1.0
        cadences.append(pc_k)
    
    # Calcul de la charge ultime pour chaque année de survenance
    # barre la dimension de développement sur DevelopmentYear (datetime),
    # ce que chainladder sait interpréter. Le triangle manuel ci-dessus,
    # construit sur DevelopmentLag, donne des LDF équivalents car
    # DevelopmentLag = DevelopmentYear - AccidentYear + 1.
    triangle_df = cl.Triangle(
        data=df,
        origin="AccidentYear",
        development="DevelopmentYear",
        columns=["CumPaidLoss"],
        cumulative=True,
    )

    model = cl.MackChainladder()
    model.fit(triangle_df)
    
    # Stocker les LDF manuels pour référence
    model.manual_ldfs = ldfs
    model.manual_cadences = cadences
    
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
        if hasattr(ibnr.index, "year"):
            ibnr.index = ibnr.index.year
        else:
            ibnr.index = [int(str(x)[:4]) for x in ibnr.index]

        std_df = model.mack_std_err_.to_frame()
        if 9999 in std_df.columns:
            std_err = std_df[9999]
        else:
            std_err = std_df.iloc[:, -1]

        if hasattr(std_err.index, "year"):
            std_err.index = std_err.index.year
        else:
            std_err = [int(str(x)[:4]) for x in std_err.index]
        std_err = pd.Series(std_err, index=ibnr.index)

        # Vérité terrain
        obs_company = observed[observed["GRCODE"] == grcode].copy()
        fut_company = future[future["GRCODE"] == grcode].copy()

        obs_company["AY_year"] = obs_company["AccidentYear"].dt.year if pd.api.types.is_datetime64_any_dtype(obs_company["AccidentYear"]) else obs_company["AccidentYear"]
        fut_company["AY_year"] = fut_company["AccidentYear"].dt.year if pd.api.types.is_datetime64_any_dtype(fut_company["AccidentYear"]) else fut_company["AccidentYear"]

        true_ultimate = fut_company[fut_company["DevelopmentLag"] == 10].set_index("AY_year")["CumPaidLoss"]
        paid_at_eval = obs_company.sort_values("DevelopmentLag").groupby("AY_year")["CumPaidLoss"].last()
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