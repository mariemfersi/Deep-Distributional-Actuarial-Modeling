"""
Module Fraude — Modèles de détection.

Benchmark comparatif de 7 approches (cf. chapitre 6 et revue de littérature
sur la détection de fraude en assurance par IA) :
  - Apprentissage supervisé : Random Forest, XGBoost, SVM, avec/sans SMOTE
  - Apprentissage non supervisé (anomalie) : Isolation Forest, Local Outlier Factor
L'élément relationnel par graphe (GNN) a été écarté (test d'homophilie non concluant).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from boruta import BorutaPy
from xgboost import XGBClassifier

from src.fraud.data import get_feature_columns


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Assemble la matrice de features (catégorielles encodées + numériques normalisées)."""
    cat_cols, num_cols = get_feature_columns()
    return df[cat_cols + num_cols].values


def _get_model_feature_importances(model, feature_names: list[str]) -> dict:
    """Extrait les importances de features d'un modèle, qu'il soit brut ou un
    imblearn Pipeline, avec un retour vide pour les modèles sans attribut
    (ex. SVM)."""
    estimator = model[-1] if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        return dict(zip(feature_names, np.round(estimator.feature_importances_, 4)))
    return {}


# ── Non supervisé : détection d'anomalies ────────────────────────────────

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
        "model_name": "Isolation Forest",
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
        "anomaly_scores": anomaly_scores,
        "y_true": y_true,
    }


def fit_local_outlier_factor(train_df: pd.DataFrame, seed: int = 123):
    """
    Local Outlier Factor non supervisé pour la détection d'anomalies.
    `novelty=True` permet de scorer de nouvelles observations hors entraînement.
    Le taux de contamination est fixé à la prévalence de fraude (~6%).
    """
    X = get_feature_matrix(train_df)
    model = LocalOutlierFactor(contamination=0.06, novelty=True, n_neighbors=20)
    model.fit(X)
    return model


def evaluate_local_outlier_factor(model, test_df: pd.DataFrame) -> dict:
    """Évalue le LOF sur le jeu de test (score_samples inversé : élevé = suspect)."""
    X = get_feature_matrix(test_df)
    y_true = test_df["fraud_label"].values

    anomaly_scores = -model.score_samples(X)

    return {
        "model_name": "LOF",
        "auc_roc": roc_auc_score(y_true, anomaly_scores),
        "pr_auc": average_precision_score(y_true, anomaly_scores),
        "anomaly_scores": anomaly_scores,
        "y_true": y_true,
    }


# ── Supervisé : modèles avec predict_proba ───────────────────────────────

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


def fit_xgboost(train_df: pd.DataFrame, seed: int = 123):
    """
    XGBoost avec `scale_pos_weight` pour contrebalancer le déséquilibre
    de classes (ratio négatifs/positifs). XGBoost tend à surpasser le RF sur
    la détection de fraude en assurance (cf. Cherkaoui et al. 2024 ; Sagar 2025).
    """
    X = get_feature_matrix(train_df)
    y = train_df["fraud_label"].values

    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=n_neg / n_pos,
        random_state=seed,
        eval_metric="aucpr",
        use_label_encoder=False,
    )
    model.fit(X, y)
    return model


def fit_svm(train_df: pd.DataFrame, seed: int = 123):
    """
    Machine à Vecteurs de Support (RBF). `probability=True` est requis pour
    disposer de `predict_proba`. `class_weight="balanced"` compense
    partiellement le déséquilibre.
    """
    X = get_feature_matrix(train_df)
    y = train_df["fraud_label"].values

    model = SVC(
        kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
        probability=True, random_state=seed,
    )
    model.fit(X, y)
    return model


def fit_random_forest_smote(train_df: pd.DataFrame, seed: int = 123):
    """Random Forest avec SMOTE (rééquilibrage sur les données d'entraînement
    uniquement, via un imblearn Pipeline pour éviter toute fuite en CV)."""
    X = get_feature_matrix(train_df)
    y = train_df["fraud_label"].values

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=seed)),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )),
    ])
    pipeline.fit(X, y)
    return pipeline


def fit_xgboost_smote(train_df: pd.DataFrame, seed: int = 123):
    """XGBoost avec SMOTE. Comme SMOTE équilibre les classes, un seul
    `scale_pos_weight=1` convient ici (pas de ré-équilibrage supplémentaire)."""
    X = get_feature_matrix(train_df)
    y = train_df["fraud_label"].values

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=seed)),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1,
            random_state=seed,
            eval_metric="aucpr",
            use_label_encoder=False,
        )),
    ])
    pipeline.fit(X, y)
    return pipeline


def evaluate_classifier(model, test_df: pd.DataFrame, model_name: str = "") -> dict:
    """
    Évaluation unifiée des classifieurs supervisés (modèles bruts ou imblearn
    Pipeline). Retourne AUC-ROC, PR-AUC, ainsi que les métriques de seuil à 0.5
    (matrice de confusion, précision, rappel, F1, rapport de classification).
    """
    X = get_feature_matrix(test_df)
    y_true = test_df["fraud_label"].values

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    feature_names = get_feature_columns()[0] + get_feature_columns()[1]

    return {
        "model_name": model_name,
        "auc_roc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(y_true, pred, zero_division=0),
        "feature_importance": _get_model_feature_importances(model, feature_names),
        "scores": proba,
        "y_true": y_true,
    }


def cross_validate_models(train_df: pd.DataFrame, n_folds: int = 5, seed: int = 123) -> pd.DataFrame:
    """
    Validation croisée (AUC-ROC) des 5 variantes supervisées. SMOTE est
    intégré dans un imblearn Pipeline pour ne jamais échantillonner les
    folds de validation (pas de fuite).
    """
    X = get_feature_matrix(train_df)
    y = train_df["fraud_label"].values

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        random_state=seed, eval_metric="aucpr", use_label_encoder=False,
    )
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
              probability=True, random_state=seed)

    rf_smote = ImbPipeline([
        ("smote", SMOTE(random_state=seed)),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=seed, n_jobs=-1)),
    ])
    xgb_smote = ImbPipeline([
        ("smote", SMOTE(random_state=seed)),
        ("classifier", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=1, random_state=seed,
            eval_metric="aucpr", use_label_encoder=False)),
    ])

    models = {
        "Random Forest": rf,
        "RF + SMOTE": rf_smote,
        "XGBoost": xgb,
        "XGB + SMOTE": xgb_smote,
        "SVM (RBF)": svm,
    }

    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=n_folds, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": name, "mean_auc_roc": scores.mean(), "std_auc_roc": scores.std()})

    return pd.DataFrame(rows)


def run_boruta_selection(train_df: pd.DataFrame, seed: int = 123) -> dict:
    """
    Sélection de features par l'algorithme Boruta (Random Forest comme
    estimateur). Compare chaque feature à une ombre aléatoire : celles qui
    battent leur ombre de façon significative sont 'confirmées', les autres
    'rejetées'. Résultat purement informatif (méthodologie) : on n'élimine
    pas les features du modèle de production.
    """
    X = get_feature_matrix(train_df)
    y = train_df["fraud_label"].values
    cat_cols, num_cols = get_feature_columns()
    feature_names = cat_cols + num_cols

    estimator = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    boruta = BorutaPy(estimator, n_estimators="auto", random_state=seed, verbose=0)
    boruta.fit(X, y)

    mask = boruta.support_
    confirmed = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    rejected = [
        feature_names[i]
        for i in range(len(feature_names))
        if not mask[i] and not boruta.support_weak_[i]
    ]
    tentative = [
        feature_names[i]
        for i in range(len(feature_names))
        if boruta.support_weak_[i]
    ]

    return {
        "confirmed": confirmed,
        "tentative": tentative,
        "rejected": rejected,
        "selected_mask": mask,
    }
