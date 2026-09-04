"""
Métriques Prometheus pour le suivi de l'API et des modèles.
"""

from prometheus_client import Counter, Histogram, Gauge

# ── Métriques HTTP (auto-instrumentées par prometheus-fastapi-instrumentator) ──

# ── Métriques métier ──────────────────────────────────────────────

prediction_counter = Counter(
    "actuarial_predictions_total",
    "Nombre total de prédictions",
    ["module", "model_version"],
)

prediction_latency = Histogram(
    "actuarial_prediction_latency_seconds",
    "Latence de traitement des prédictions",
    ["module"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

fraud_alert_counter = Counter(
    "actuarial_fraud_alerts_total",
    "Nombre de prédictions suspects (fraude)",
    ["model_version"],
)

active_users = Gauge(
    "actuarial_active_users",
    "Nombre d'utilisateurs actifs",
)

db_session_counter = Gauge(
    "actuarial_db_sessions",
    "Nombre de sessions DB actives",
)
