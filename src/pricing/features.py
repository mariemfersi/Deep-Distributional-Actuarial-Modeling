"""
Module Pricing — Feature engineering.

Construit deux familles de représentations des variables de risque :
  - buckets discrets pour le GLM (interprétabilité)
  - variables normalisées + codes catégoriels pour le futur CANN (flexibilité)
"""

import pandas as pd
import numpy as np

from src.common.config import load_config


def add_glm_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Ajoute les variables discrétisées utilisées par le GLM (fréquence et sévérité)."""
    if config is None:
        config = load_config()

    df = df.copy()
    p = config["pricing"]

    df["DrivAge_bucket"] = pd.cut(
        df["DrivAge"], bins=p["driv_age_bins"], labels=p["driv_age_labels"]
    )
    df["VehAge_bucket"] = pd.cut(
        df["VehAge"], bins=p["veh_age_bins"], labels=p["veh_age_labels"]
    )
    df["BM_bucket"] = pd.cut(
        df["BonusMalus"], bins=p["bm_bins"], labels=p["bm_labels"]
    )
    df["Density_log"] = np.log(df["Density"])

    return df


def add_cann_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables normalisées et encodées pour le futur modèle CANN."""
    df = df.copy()

    for col in ["VehPower", "VehAge", "DrivAge", "BonusMalus"]:
        cmin, cmax = df[col].min(), df[col].max()
        df[f"{col}_norm"] = 2 * (df[col] - cmin) / (cmax - cmin) - 1

    df["VehBrand_code"] = df["VehBrand"].astype("category").cat.codes
    df["Region_code"] = df["Region"].astype("category").cat.codes
    df["Area_code"] = df["Area"].astype("category").cat.codes
    df["VehGas_code"] = (df["VehGas"] == "Regular").astype(int)

    return df


def build_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Pipeline complet de feature engineering (GLM + CANN)."""
    df = add_glm_features(df, config)
    df = add_cann_features(df)
    return df