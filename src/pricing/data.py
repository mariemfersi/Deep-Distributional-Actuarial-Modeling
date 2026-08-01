"""
Module Pricing — Chargement, nettoyage et préparation freMTPL2.

Reproduit et formalise le pipeline validé manuellement lors de la phase
exploratoire : jointure fréquence/sévérité, correction des anomalies
documentées (Exposure > 1, ClaimNb aberrant), séparation du risque
attritionnel et du risque grave.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.common.config import load_config, get_project_root


def load_raw_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge freMTPL2freq et freMTPL2sev depuis les chemins définis en config."""
    root = get_project_root()
    freq = pd.read_csv(root / config["paths"]["raw_freq"])
    sev = pd.read_csv(root / config["paths"]["raw_sev"])
    return freq, sev


def merge_freq_sev(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les sinistres multiples par police puis joint au fichier de
    fréquence. Une police sans sinistre correspondant reçoit ClaimAmount=0.
    """
    sev_agg = sev.groupby("IDpol", as_index=False)["ClaimAmount"].sum()
    df = freq.merge(sev_agg, on="IDpol", how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)
    return df


def clean_anomalies(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Corrige les anomalies documentées :
      - Exposure tronquée au maximum théorique (1 an)
      - ClaimNb tronqué à la borne retenue dans la littérature
      - Exclusion des polices à exposition nulle (non exploitables pour un offset log)
    """
    df = df.copy()
    cap_exposure = config["pricing"]["exposure_cap"]
    cap_claimnb = config["pricing"]["claim_nb_cap"]

    df["Exposure"] = df["Exposure"].clip(upper=cap_exposure)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=cap_claimnb)
    df = df[df["Exposure"] > 0].copy()

    return df


def flag_large_claims(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Sépare risque attritionnel et risque grave selon le seuil configuré."""
    df = df.copy()
    threshold = config["pricing"]["large_claim_threshold"]

    df["is_large_claim"] = df["ClaimAmount"] > threshold
    df["ClaimAmount_capped"] = df["ClaimAmount"].clip(upper=threshold)

    return df


def build_pricing_dataset(config: dict | None = None) -> pd.DataFrame:
    """Pipeline complet : chargement -> jointure -> nettoyage -> flag grave."""
    if config is None:
        config = load_config()

    freq, sev = load_raw_data(config)
    df = merge_freq_sev(freq, sev)
    df = clean_anomalies(df, config)
    df = flag_large_claims(df, config)

    return df 
    


def train_valid_test_split(df: pd.DataFrame, config: dict | None = None):
    """
    Split 60/20/20 avec seed fixe pour reproductibilité.
    Convention standard en tarification actuarielle (cf. tutoriels Wüthrich et al.).
    """
    if config is None:
        config = load_config()

    split_cfg = config["pricing"]["train_test_split"]
    rng = np.random.default_rng(split_cfg["seed"])

    n = len(df)
    idx = rng.permutation(n)

    n_train = int(split_cfg["train"] * n)
    n_valid = int(split_cfg["valid"] * n)

    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx = idx[n_train + n_valid:]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[valid_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
        )


def get_severity_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre un dataset (train ou test) pour ne garder que les observations
    exploitables par le GLM Gamma de sévérité :
      - ClaimAmount > 0 (exclut les incohérences ClaimNb>0/Amount=0)
      - sinistres attritionnels uniquement (is_large_claim == False)
    """
    return df[(df["ClaimAmount"] > 0) & (~df["is_large_claim"])].copy()