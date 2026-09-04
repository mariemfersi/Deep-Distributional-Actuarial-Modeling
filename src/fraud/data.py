"""
Module Fraude — Chargement et préparation fraud_oracle.csv.

Dataset validé académiquement (utilisé dans plusieurs travaux sur la
détection de fraude assurance auto, notamment par analyse relationnelle),
avec un signal discriminant confirmé sur plusieurs variables clés
(AddressChange_Claim, Fault, Days_Policy_Claim).
"""

import pandas as pd
import numpy as np

from src.common.config import get_project_root


CATEGORICAL_COLS = [
    "Month", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed", "MonthClaimed",
    "Sex", "MaritalStatus", "Fault", "PolicyType", "VehicleCategory", "VehiclePrice",
    "Days_Policy_Accident", "Days_Policy_Claim", "PastNumberOfClaims", "AgeOfVehicle",
    "AgeOfPolicyHolder", "PoliceReportFiled", "WitnessPresent", "AgentType",
    "NumberOfSuppliments", "AddressChange_Claim", "NumberOfCars", "BasePolicy",
]

NUMERIC_COLS = ["WeekOfMonth", "Age", "RepNumber", "Deductible", "DriverRating", "Year"]


def get_feature_columns() -> tuple[list[str], list[str]]:
    """Retourne (colonnes catégorielles encodées, colonnes numériques normalisées)
    dans l'ordre d'entraînement du modèle."""
    return [f"{c}_code" for c in CATEGORICAL_COLS], [f"{c}_norm" for c in NUMERIC_COLS]


def load_fraud_data() -> pd.DataFrame:
    root = get_project_root()
    df = pd.read_csv(root / "data/raw/fraud_oracle.csv", encoding="utf-8-sig")
    return df


def fit_fraud_preprocessor(df: pd.DataFrame) -> tuple[dict, dict]:
    """Ajuste l'encodeur catégoriel et les statistiques de normalisation
    sur une seule base (à utiliser sur le jeu d'ENTRAÎNEMENT uniquement).

    Retourne (encoders, norm_stats) où :
      - encoders[col]  = liste ordonnée des catégories (code = index)
      - norm_stats[col] = (mean, std) pour la normalisation z-score

    IMPORTANT (anti-fuite) : les statistiques doivent être apprises sur le
    train seul puis transformées sur train ET test. Les appliquer sur le
    dataset entier avant split fuite des statistiques du test (et des
    ordres de catégories) dans le préprocessing.
    """
    encoders = {
        col: df[col].astype("category").cat.categories.tolist()
        for col in CATEGORICAL_COLS
    }
    norm_stats = {
        col: (float(df[col].mean()), float(df[col].std()))
        for col in NUMERIC_COLS
    }
    return encoders, norm_stats


def apply_fraud_preprocessor(
    df: pd.DataFrame, encoders: dict, norm_stats: dict
) -> pd.DataFrame:
    """Transforme un dataframe en features encodées/normalisées en utilisant
    des (encoders, norm_stats) déjà ajustés (sur l'entraînement).

    Les catégories inconnues reçoivent le code -1 et les valeurs numériques
    sont normalisées avec les stats d'entraînement (cohérent avec l'API).
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        cats = encoders[col]
        df[f"{col}_code"] = df[col].map(
            lambda v: cats.index(v) if v in cats else -1
        ).astype(int)
    for col in NUMERIC_COLS:
        mean, std = norm_stats[col]
        df[f"{col}_norm"] = (df[col] - mean) / std
    df["fraud_label"] = df["FraudFound_P"].astype(int)
    return df


def prepare_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode et normalise un dataframe (compatibilité / usage exploratoire).

    NOTE (fuite de données) : cette fonction ajuste le préprocesseur sur le
    dataframe qui lui est passé. Si elle est appelée AVANT le split
    train/test, les statistiques du test fuient dans le préprocessing.
    Pour une évaluation sans fuite, préférer :
        enc, stats = fit_fraud_preprocessor(train_df)
        train_f = apply_fraud_preprocessor(train_df, enc, stats)
        test_f  = apply_fraud_preprocessor(test_df,  enc, stats)
    Dans le pipeline de production, le benchmark et les artefacts
    (fraud_encoders.pkl, fraud_normalization_stats.pkl) sont ajustés sur le
    train seul (cf. notebook 05_fraud et backend).
    """
    encoders, norm_stats = fit_fraud_preprocessor(df)
    return apply_fraud_preprocessor(df, encoders, norm_stats)


def train_test_split_fraud(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 123):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(test_frac * len(df))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)