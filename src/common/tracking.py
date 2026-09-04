"""
Suivi d'expériences MLflow — utilitaire centralisé pour consigner les runs
des trois modules (pricing, reserving, fraud) de façon homogène.
"""

import os
import mlflow
from pathlib import Path


def init_mlflow(experiment_name: str, tracking_uri: str | None = None):
    """Initialise le tracking MLflow.

    Args:
        experiment_name: Nom de l'expérience MLflow.
        tracking_uri: URI du serveur MLflow. Si None, lit la variable
                      d'environnement MLFLOW_TRACKING_URI, sinon défaut local.
    """
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        # Fallback : tracking local dans mlruns/
        root = Path(__file__).resolve().parents[2]
        tracking_uri = f"file://{root / 'mlruns'}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_run(run_name: str, params: dict, metrics: dict, tags: dict | None = None):
    """Consigne un run complet (paramètres + métriques) en un seul appel."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)