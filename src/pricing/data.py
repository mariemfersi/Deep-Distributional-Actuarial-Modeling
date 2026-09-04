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


def out_of_time_split(df: pd.DataFrame, train_years: list[int], test_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporel pour validation out-of-time (Solvency II requirement).
    
    Entraîne sur des années de souscription antérieures, teste sur des années
    ultérieures pour évaluer la stabilité du modèle dans le temps.
    
    Args:
        df: Dataset complet avec colonne 'PolicyYear' (année de souscription)
        train_years: Liste des années d'entraînement (ex: [2003, 2004])
        test_years: Liste des années de test (ex: [2005])
    
    Returns:
        train_df, test_df: Datasets d'entraînement et test
    """
    df = df.copy()
    
    # Vérifier que PolicyYear existe
    if "PolicyYear" not in df.columns:
        # Si PolicyYear n'existe pas, créer une colonne basée sur une autre colonne temporelle
        # ou utiliser un split aléatoire comme fallback
        raise ValueError("Colonne 'PolicyYear' requise pour le split out-of-time")
    
    train_df = df[df["PolicyYear"].isin(train_years)].reset_index(drop=True)
    test_df = df[df["PolicyYear"].isin(test_years)].reset_index(drop=True)
    
    return train_df, test_df


def evaluate_out_of_time_performance(train_df: pd.DataFrame, test_df: pd.DataFrame, model, 
                                     exposure_col: str = "Exposure", claim_col: str = "ClaimNb") -> dict:
    """
    Évalue la performance out-of-time avec métriques actuarielles standard.
    
    Args:
        train_df: Données d'entraînement
        test_df: Données de test (années ultérieures)
        model: Modèle entraîné (GLM ou CANN)
        exposure_col: Nom de la colonne d'exposition
        claim_col: Nom de la colonne de nombre de sinistres
    
    Returns:
        Dictionnaire de métriques de performance out-of-time
    """
    from src.pricing.evaluate import compute_gini_index
    
    # Prédictions sur le test set
    X_test = test_df.drop(columns=[claim_col, exposure_col, "ClaimAmount", "is_large_claim", "ClaimAmount_capped"], errors="ignore")
    exposure_test = test_df[exposure_col].values
    claims_test = test_df[claim_col].values
    
    # Obtenir les prédictions du modèle
    if hasattr(model, 'predict'):
        pred_freq = model.predict(X_test)
    else:
        # Pour les modèles PyTorch/CANN
        pred_freq = model(X_test)
    
    # Calculer le Gini out-of-time
    gini_oot = compute_gini_index(claims_test, pred_freq * exposure_test, exposure_test)
    
    # Calculer la déviance out-of-time
    deviance_oot = compute_poisson_deviance(claims_test, pred_freq, exposure_test)
    
    # Calculer le ratio de sinistralité observé vs prédit
    observed_freq = claims_test.sum() / exposure_test.sum()
    predicted_freq = (pred_freq * exposure_test).sum() / exposure_test.sum()
    freq_ratio = observed_freq / predicted_freq if predicted_freq > 0 else 0
    
    return {
        "gini_out_of_time": gini_oot,
        "deviance_out_of_time": deviance_oot,
        "observed_frequency": observed_freq,
        "predicted_frequency": predicted_freq,
        "frequency_ratio": freq_ratio,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_years": sorted(train_df["PolicyYear"].unique()) if "PolicyYear" in train_df.columns else [],
        "test_years": sorted(test_df["PolicyYear"].unique()) if "PolicyYear" in test_df.columns else []
    }


def compute_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray) -> float:
    """
    Calcule la déviance de Poisson pour évaluer la qualité d'ajustement.
    
    Deviance = 2 * sum(y * log(y/mu) - (y - mu)) avec mu = pred * exposure
    """
    mu = y_pred * exposure
    # Éviter log(0) en ajoutant un petit epsilon
    epsilon = 1e-10
    deviance = 2 * np.sum(
        y_true * np.log((y_true + epsilon) / (mu + epsilon)) - (y_true - mu)
    )
    return deviance / exposure.sum()  # Normaliser par l'exposition totale