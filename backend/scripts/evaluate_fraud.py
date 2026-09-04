"""
Évaluation empirique authentique du module détection de fraude.

Reproduit de manière autonome (sans dépendre du notebook) le benchmark
complet de 7 variantes et persiste les artefacts de production AVEC un
préprocessing ajusté sur le train seul (anti-fuite de données) :

  1. Split train/test sur les données BRUTES (60/40 80/20 par défaut).
  2. Ajustement de l'encodeur catégoriel et des stats de normalisation sur
     le train UNIQUEMENT, puis transformation de train et test via ces
     statistiques (aucune statistique du test ne fuit dans le préprocessing).
  3. Entraînement des 7 approches (RF, RF+SMOTE, XGB, XGB+SMOTE, SVM,
     Isolation Forest, LOF).
  4. Évaluation sur le jeu de test (AUC-ROC, PR-AUC, précision, rappel, F1).
  5. Persistance des artefacts de production cohérents avec ce préprocessing
     (fraud_encoders.pkl, fraud_normalization_stats.pkl,
     fraud_default_values.pkl, fraud_best_model.pkl) + modèles RF fallback.
  6. Persistance des métriques mesurées dans models/fraud_metrics.json.

Aucune valeur n'est codée en dur : tout est calculé à partir des données.

Usage (conteneur backend) :
    python -m scripts.evaluate_fraud [--seed 123] [--test-frac 0.2]

NOTE seed : pour une comparaison À L'IDENTIQUE avec les notebooks/audits
existants, utiliser --seed 123 (split 60/40 historique) ou --test-frac 0.2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

MODELS_DIR = BACKEND_DIR / "models" if (BACKEND_DIR / "models").exists() else PROJECT_ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    from src.fraud.data import (
        load_fraud_data, fit_fraud_preprocessor, apply_fraud_preprocessor,
        train_test_split_fraud, CATEGORICAL_COLS, NUMERIC_COLS,
    )

    # 1) Chargement + split sur les DONNÉES BRUTES (anti-fuite)
    df = load_fraud_data()
    train_raw, test_raw = train_test_split_fraud(df, test_frac=args.test_frac, seed=args.seed)
    print(f"Dataset: {len(df):,} | train {len(train_raw):,} | test {len(test_raw):,}"
          f" | fraude train {train_raw['FraudFound_P'].mean():.4f}")

    # 2) Préprocessing ajusté sur le train seul
    encoders, norm_stats = fit_fraud_preprocessor(train_raw)
    train_f = apply_fraud_preprocessor(train_raw, encoders, norm_stats)
    test_f = apply_fraud_preprocessor(test_raw, encoders, norm_stats)

    from src.fraud.models import (
        cross_validate_models,
        run_boruta_selection,
        fit_supervised_baseline,
        fit_xgboost,
        fit_svm,
        fit_random_forest_smote,
        fit_xgboost_smote,
        fit_isolation_forest,
        fit_local_outlier_factor,
        evaluate_classifier,
        evaluate_isolation_forest,
        evaluate_local_outlier_factor,
    )

    # 3) Validation croisée 5-fold (sur train)
    cv = cross_validate_models(train_f, n_folds=5, seed=args.seed)

    # 4) Boruta (méthodologie)
    boruta = run_boruta_selection(train_f, seed=args.seed)

    # 5) Entraînement des 7 variantes
    models = {
        "Random Forest":     fit_supervised_baseline(train_f, seed=args.seed),
        "RF + SMOTE":        fit_random_forest_smote(train_f, seed=args.seed),
        "XGBoost":           fit_xgboost(train_f, seed=args.seed),
        "XGB + SMOTE":       fit_xgboost_smote(train_f, seed=args.seed),
        "SVM (RBF)":         fit_svm(train_f, seed=args.seed),
        "Isolation Forest":  fit_isolation_forest(train_f, contamination=0.06, seed=args.seed),
        "LOF":               fit_local_outlier_factor(train_f, seed=args.seed),
    }

    # 6) Évaluation
    results = {}
    for name, model in models.items():
        if name == "Isolation Forest":
            results[name] = evaluate_isolation_forest(model, test_f)
        elif name == "LOF":
            results[name] = evaluate_local_outlier_factor(model, test_f)
        else:
            results[name] = evaluate_classifier(model, test_f, model_name=name)

    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "auc_roc": round(float(r["auc_roc"]), 4),
            "pr_auc": round(float(r.get("pr_auc", 0.0)), 4),
            "precision": round(float(r.get("precision", 0.0)), 4),
            "recall": round(float(r.get("recall", 0.0)), 4),
            "f1": round(float(r.get("f1", 0.0)), 4),
        })
    comparison = pd.DataFrame(rows).sort_values("auc_roc", ascending=False).reset_index(drop=True)
    print("\n=== Benchmark fraude — jeu de test (préprocessing train-seul) ===")
    print(comparison.to_string(index=False))

    best_name = comparison.iloc[0]["model"]
    best_is_unsupervised = best_name in {"Isolation Forest", "LOF"}

    # 7) CV + Boruta
    cv_dict = {
        str(r["model"]): {"mean_auc_roc": round(float(r["mean_auc_roc"]), 4),
                          "std_auc_roc": round(float(r["std_auc_roc"]), 4)}
        for _, r in cv.iterrows()
    }
    boruta_dict = {
        "n_features": len(boruta["confirmed"]) + len(boruta["tentative"]) + len(boruta["rejected"]),
        "confirmed": boruta["confirmed"],
        "tentative": boruta["tentative"],
        "rejected": boruta["rejected"],
    }

    # 8) Persistance des artefacts de production (préprocessing train-seul)
    if not best_is_unsupervised:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, MODELS_DIR / "fraud_encoders.pkl")
        joblib.dump(norm_stats, MODELS_DIR / "fraud_normalization_stats.pkl")
        default_values = {}
        for col in CATEGORICAL_COLS:
            default_values[col] = str(train_f[col].mode()[0])
        for col in NUMERIC_COLS:
            default_values[col] = float(train_f[col].median())
        joblib.dump(default_values, MODELS_DIR / "fraud_default_values.pkl")
        joblib.dump(models[best_name], MODELS_DIR / "fraud_best_model.pkl")
        joblib.dump(models["Random Forest"], MODELS_DIR / "fraud_random_forest.pkl")
        print(f"\nArtefacts de production (préprocessing train-seul) -> {MODELS_DIR}")
        print(f"  - fraud_encoders.pkl, fraud_normalization_stats.pkl, fraud_default_values.pkl")
        print(f"  - fraud_best_model.pkl ({best_name}), fraud_random_forest.pkl")

    # 9) Métriques mesurées
    sorted_rows = sorted(rows, key=lambda r: -r["auc_roc"])
    for r in sorted_rows:
        r["best"] = (r["model"] == best_name)
    metrics = {
        "dataset": "fraud_oracle.csv",
        "split": f"train/test = {int(1-args.test_frac)*100}/{int(args.test_frac*100)} (seed {args.seed}),"
                 " split sur données brutes AVANT préprocessing (anti-fuite)",
        "test_size": int(len(test_f)),
        "best_model": best_name,
        "leakage_handling": "preprocessor_fit_on_train_only",
        "benchmark": sorted_rows,
        "cross_validation": cv_dict,
        "boruta": boruta_dict,
    }
    out = MODELS_DIR / "fraud_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True, default=str)
    print(f"\nMetriques persistees -> {out}")


if __name__ == "__main__":
    main()
