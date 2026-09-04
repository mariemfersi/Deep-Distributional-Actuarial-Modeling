"""
Module Reserving — Chargement et préparation CAS Loss Reserving Database.

Structure du fichier : triangle complet 10x10 par compagnie (GRCODE), avec
la relation AccidentYear + DevelopmentLag - 1 = DevelopmentYear. Le triangle
"observé" (partie haute) est celui disponible à la date d'évaluation (2007
pour ppauto) ; le reste sert de vérité terrain pour l'évaluation objective
des méthodes de provisionnement (Mack, Deep Triangle).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

from src.common.config import load_config, get_project_root

EVALUATION_YEAR = 2007  # dernière AccidentYear du portefeuille ppauto


def load_raw_reserving_data(config: dict | None = None) -> pd.DataFrame:
    """Charge le fichier CAS Loss Reserving Database (ligne ppauto)."""
    if config is None:
        config = load_config()
    root = get_project_root()
    path = root / "data/raw/ppauto_pos.csv"
    df = pd.read_csv(path, sep=";")
    # Convert year columns to datetime for chainladder compatibility
    df["AccidentYear"] = pd.to_datetime(df["AccidentYear"], format="%Y")
    df["DevelopmentYear"] = pd.to_datetime(df["DevelopmentYear"], format="%Y")
    return df


def split_observed_future(df: pd.DataFrame, evaluation_year: int = EVALUATION_YEAR):
    """
    Sépare le triangle "observé" (connu à la date d'évaluation) du triangle
    "futur" (vérité terrain masquée, réservée à l'évaluation finale).
    """
    eval_date = pd.to_datetime(str(evaluation_year), format="%Y")
    observed = df[df["DevelopmentYear"] <= eval_date].copy()
    future = df[df["DevelopmentYear"] > eval_date].copy()
    return observed, future


def compute_incremental_paid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les paiements incrémentaux (par période de développement) à partir
    des cumuls CumPaidLoss -- nécessaire pour l'architecture séquentielle
    Deep Triangle, qui modélise les incréments plutôt que les cumuls.
    """
    df = df.sort_values(["GRCODE", "AccidentYear", "DevelopmentLag"]).copy()

    df["IncrementalPaid"] = df.groupby(["GRCODE", "AccidentYear"])["CumPaidLoss"].diff()
    # Premier développement (lag=1) : l'incrément = le cumul lui-même
    first_lag_mask = df["DevelopmentLag"] == 1
    df.loc[first_lag_mask, "IncrementalPaid"] = df.loc[first_lag_mask, "CumPaidLoss"]

    return df


def build_reserving_dataset(config: dict | None = None):
    """Pipeline complet : chargement -> incréments -> split observé/futur."""
    df = load_raw_reserving_data(config)
    df = compute_incremental_paid(df)
    observed, future = split_observed_future(df)
    return df, observed, future



def build_triangle_for_chainladder(observed: pd.DataFrame, grcode: int | None = None):
    """
    Construit un objet Triangle au format du package `chainladder`, à partir
    des données observées. Si grcode est fourni, restreint à une compagnie ;
    sinon agrège toutes les compagnies (triangle de portefeuille combiné).
    """
    import chainladder as cl

    df = observed.copy()
    if grcode is not None:
        df = df[df["GRCODE"] == grcode]

    triangle = cl.Triangle(
        data=df,
        origin="AccidentYear",
        development="DevelopmentYear",
        columns=["CumPaidLoss", "IncurredLosses"],
        cumulative=True,
    )
    return triangle

def get_all_grcodes(df: pd.DataFrame) -> list:
    """Retourne la liste des identifiants de compagnies présentes dans le triangle."""
    return df["GRCODE"].unique().tolist()



def build_sequences(full_df: pd.DataFrame, evaluation_year: int = EVALUATION_YEAR):
    """
    Construit, pour chaque (compagnie, année de survenance), la séquence des
    10 incréments de paiement normalisés par la prime acquise (échelle de
    ratio de sinistralité), ainsi que le masque des positions observées.
    """
    df = full_df.copy()
    df["ScaledIncr"] = df["IncrementalPaid"] / df["EarnedPremNet"].clip(lower=1.0)
    eval_date = pd.to_datetime(str(evaluation_year), format="%Y")
    df["is_observed"] = df["DevelopmentYear"] <= eval_date

    series_list, masks, keys, premiums = [], [], [], []

    for (grcode, ay), g in df.groupby(["GRCODE", "AccidentYear"]):
        g = g.sort_values("DevelopmentLag")
        series_list.append(g["ScaledIncr"].values)
        masks.append(g["is_observed"].values)
        keys.append((grcode, ay))
        premiums.append(g["EarnedPremNet"].iloc[0])

    return np.array(series_list), np.array(masks), keys, np.array(premiums)


def build_sequences(full_df: pd.DataFrame, evaluation_year: int = EVALUATION_YEAR, min_premium: float = 1000):
    """
    Construit, pour chaque (compagnie, année de survenance), la séquence des
    10 incréments de paiement normalisés par la prime acquise.

    Les compagnies dont la prime acquise nette est trop faible ou négative
    (ex. réassureurs avec primes cédées dominantes) sont exclues : le ratio
    incrément/prime devient statistiquement incohérent et non représentatif
    d'une vraie dynamique de sinistralité en dessous de ce seuil.
    """
    df = full_df.copy()

    # Filtrage des compagnies non exploitables (prime minimale observée trop faible)
    min_prem_by_company = df.groupby("GRCODE")["EarnedPremNet"].min()
    valid_grcodes = min_prem_by_company[min_prem_by_company >= min_premium].index
    n_excluded = df["GRCODE"].nunique() - len(valid_grcodes)
    print(f"Compagnies exclues (prime < {min_premium:,}$) : {n_excluded}")

    df = df[df["GRCODE"].isin(valid_grcodes)].copy()

    df["ScaledIncr"] = df["IncrementalPaid"] / df["EarnedPremNet"].clip(lower=1.0)
    eval_date = pd.to_datetime(str(evaluation_year), format="%Y")
    df["is_observed"] = df["DevelopmentYear"] <= eval_date

    series_list, masks, keys, premiums = [], [], [], []

    for (grcode, ay), g in df.groupby(["GRCODE", "AccidentYear"]):
        g = g.sort_values("DevelopmentLag")
        series_list.append(g["ScaledIncr"].values)
        masks.append(g["is_observed"].values)
        keys.append((grcode, ay))
        premiums.append(g["EarnedPremNet"].iloc[0])

    return np.array(series_list), np.array(masks), keys, np.array(premiums)


class DeepTriangleGRU(nn.Module):
    """GRU many-to-many, prédiction du prochain incrément à chaque pas (contraint positif)."""

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.Softplus()

    def forward(self, x):
        out, _ = self.gru(x)
        return self.activation(self.output(out)).squeeze(-1)