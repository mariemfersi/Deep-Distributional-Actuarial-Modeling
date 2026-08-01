"""
Module Fraude — Modèles de détection.

Baseline Isolation Forest (non supervisé) comme référence, avant
extension vers une approche relationnelle par graphe (GNN).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from src.fraud.data import CATEGORICAL_COLS, NUMERIC_COLS


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Assemble la matrice de features (catégorielles encodées + numériques normalisées)."""
    cat_cols = [f"{c}_code" for c in CATEGORICAL_COLS]
    num_cols = [f"{c}_norm" for c in NUMERIC_COLS]
    return df[cat_cols + num_cols].values


def fit_isolation_forest(train_df: pd.DataFrame, contamination: float = 0.06, seed: int = 123):
    """
    Ajuste un Isolation Forest non supervisé. Le taux de contamination est
    fixé à la prévalence de fraude observée sur le portefeuille (6%), une
    pratique standard bien que l'entraînement n'utilise à aucun moment le
    label -- seule cette proportion sert d'hyperparamètre.
    """
    X = get_feature_matrix(train_df)
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    model.fit(X)
    return model


def evaluate_isolation_forest(model, test_df: pd.DataFrame) -> dict:
    """
    Évalue le modèle avec des métriques adaptées au déséquilibre de classe
    (AUC-ROC et surtout PR-AUC, plus informative quand la classe positive
    est rare -- l'accuracy serait trompeuse ici, cf. chapitre 3).
    """
    X = get_feature_matrix(test_df)
    y_true = test_df["fraud_label"].values

    # score_samples : plus négatif = plus anormal. On inverse pour que "élevé = suspect"
    anomaly_scores = -model.score_samples(X)

    auc_roc = roc_auc_score(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)

    return {
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "anomaly_scores": anomaly_scores,
        "y_true": y_true,
    }


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def fit_supervised_baseline(train_df: pd.DataFrame, seed: int = 123):
    """
    Baseline supervisée (Random Forest) exploitant les labels de fraude --
    point de comparaison plus pertinent que l'Isolation Forest non supervisé,
    puisque la fraude sur ce dataset est corrélée à des combinaisons de
    variables normales plutôt qu'à des anomalies statistiques génériques.
    """
    X = get_feature_matrix(train_df)
    y = train_df["fraud_label"].values

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=seed, n_jobs=-1
    )
    model.fit(X, y)
    return model


def evaluate_supervised(model, test_df: pd.DataFrame) -> dict:
    X = get_feature_matrix(test_df)
    y_true = test_df["fraud_label"].values

    proba = model.predict_proba(X)[:, 1]

    auc_roc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)

    return {"auc_roc": auc_roc, "pr_auc": pr_auc, "scores": proba, "y_true": y_true}