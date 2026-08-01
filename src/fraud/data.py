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


def load_fraud_data() -> pd.DataFrame:
    root = get_project_root()
    df = pd.read_csv(root / "data/raw/fraud_oracle.csv", encoding="utf-8-sig")
    return df


def prepare_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in CATEGORICAL_COLS:
        df[f"{col}_code"] = df[col].astype("category").cat.codes

    for col in NUMERIC_COLS:
        df[f"{col}_norm"] = (df[col] - df[col].mean()) / df[col].std()

    df["fraud_label"] = df["FraudFound_P"].astype(int)

    return df


def train_test_split_fraud(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 123):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(test_frac * len(df))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)