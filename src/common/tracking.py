"""
Suivi d'expériences MLflow — utilitaire centralisé pour consigner les runs
des trois modules (pricing, reserving, fraud) de façon homogène.
"""

import mlflow
from pathlib import Path

from src.common.config import get_project_root


def init_mlflow(experiment_name: str):
    """Initialise le tracking MLflow en local (dossier mlruns/ à la racine du projet)."""
    root = get_project_root()
    tracking_dir = root / "mlruns"
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment(experiment_name)


def log_run(run_name: str, params: dict, metrics: dict, tags: dict | None = None):
    """Consigne un run complet (paramètres + métriques) en un seul appel."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)