"""
Service MLflow — logging centralisé des prétraitions et suivi des modèles.
"""

import logging
from typing import Any

import mlflow

from app.config import get_settings

logger = logging.getLogger("actuarial.mlflow")
_settings = get_settings()

_initialized = False
_current_experiment = None


def _ensure_init(experiment_name: str = "actuarial_platform") -> None:
    """Initialise MLflow une seule fois par processus (tracking URI + set_experiment).

    On suit l'expérience courante séparément : chaque module (fraud/pricing/
    reserving) logue dans sa propre expérience MLflow, donc on ne « câche »
    pas le premier nom d'expérience comme si c'était définitif.
    """
    global _initialized, _current_experiment
    if not _initialized:
        try:
            mlflow.set_tracking_uri(_settings.MLFLOW_TRACKING_URI)
            _initialized = True
            logger.info("MLflow initialisé → %s", _settings.MLFLOW_TRACKING_URI)
        except Exception as e:
            logger.warning("Impossible d'initialiser MLflow : %s", e)
            return

    if experiment_name != _current_experiment:
        try:
            mlflow.set_experiment(experiment_name)
            _current_experiment = experiment_name
        except Exception as e:
            logger.warning("Impossible de sélectionner l'expérience %s : %s", experiment_name, e)


def log_prediction(
    module: str,
    request_data: dict[str, Any],
    response_data: dict[str, Any],
    latency_ms: float,
    model_version: str = "unknown",
) -> None:
    """Log une prédiction dans MLflow (best-effort)."""
    _ensure_init(module)
    try:
        with mlflow.start_run(run_name=f"{module}_api_call"):
            mlflow.log_params({
                "module": module,
                "model_version": model_version,
            })
            mlflow.log_metrics({
                "latency_ms": round(latency_ms, 2),
            })
            mlflow.set_tags({
                "source": "api",
                "module": module,
            })
    except Exception as e:
        logger.debug("MLflow logging ignoré : %s", e)


def _get_active_mlflow_run(experiment_name: str, run_name: str) -> str | None:
    """Retourne le run_id d'un run actif portant ce nom dans l'expérience, sinon None.

    Utilisé pour l'idempotence (ex. ne pas recréer un run model_registry à chaque
    redémarrage du backend). S'appuie sur le client MLflow interne plutôt que sur
    l'HTTP pour rester robuste aux changements de schéma API.
    """
    try:
        from mlflow.tracking import MlflowClient
        from mlflow.entities import ViewType
        cli = MlflowClient(tracking_uri=_settings.MLFLOW_TRACKING_URI)
        exp = cli.get_experiment_by_name(experiment_name)
        if exp is None:
            return None
        for run in cli.search_runs(
            experiment_ids=[exp.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
        ):
            # MLflow v3 : run.data.tags est déjà un dict {str: str}
            actual_name = run.data.tags.get("mlflow.runName", "") if run.data.tags else ""
            if actual_name == run_name:
                return run.info.run_id
        return None
    except Exception:
        return None


def log_model_training(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Log un entraînement complet et retourne le run_id."""
    _ensure_init(experiment_name)
    try:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, artifact_path=name)
            return run.info.run_id
    except Exception as e:
        logger.warning("Échec log training MLflow : %s", e)
        return None
