"""
Actuarial AI Platform API — point d'entrée FastAPI.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    HAS_METRICS = True
except ImportError:  # pragma: no cover
    HAS_METRICS = False

from app.config import get_settings
from app.database import init_db
from app.routers import pricing, reserving, fraud, explainability, auth

logger = logging.getLogger("actuarial")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Crée les tables au démarrage (développement). En prod, utiliser Alembic."""
    logger.info("Initialisation de la base de données…")
    init_db()
    logger.info("Base de données prête.")

    # Marque la révision Alembic courante pour que `alembic upgrade head`
    # fonctionne correctement sur les déploiements suivants.
    try:
        import subprocess, sys as _sys
        subprocess.run(
            [_sys.executable, "-m", "alembic", "stamp", "head"],
            cwd=str(Path(__file__).resolve().parents[1]),
            check=True, capture_output=True, timeout=10,
        )
        logger.info("Alembic stamp head → OK")
    except Exception as exc:
        logger.debug("Alembic stamp ignoré : %s", exc)

    # Enregistre (idempotent) les métadonnées des modèles dans MLflow
    # (métriques de performance + paramètres) pour chaque module.
    try:
        from app.services.model_metadata import register_all_model_metadata
        register_all_model_metadata()
    except Exception as e:
        logger.warning("Enregistrement des métadonnées MLflow ignoré : %s", e)
    yield
    logger.info("Arrêt de l'application.")


app = FastAPI(
    title="Actuarial AI Platform API",
    description="API de tarification, provisionnement et détection de fraude",
    version="0.3.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics ────────────────────────────────────────────
if HAS_METRICS:
    Instrumentator().instrument(app).expose(app)


# ── Middleware de logging ──────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s → %s (%.1f ms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


# ── Routers ────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(pricing.router)
app.include_router(reserving.router)
app.include_router(fraud.router)
app.include_router(explainability.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models/status")
def models_status():
    """État réel des artefacts de modèle (présence des fichiers sur disque).

    Chaque module est marqué `loaded: true` seulement si TOUS les artefacts
    requis existent. Aucune valeur n'est codée en dur : on liste les fichiers
    qui sont effectivement présents dans le répertoire des modèles.
    """
    from pathlib import Path
    from app.services.fraud_service import MODELS_DIR

    def _check(files: list[str]) -> dict:
        present = [f for f in files if (MODELS_DIR / f).exists()]
        missing = [f for f in files if not (MODELS_DIR / f).exists()]
        from datetime import datetime
        last_modified = None
        if present:
            latest = max((MODELS_DIR / f).stat().st_mtime for f in present)
            last_modified = datetime.utcfromtimestamp(latest).isoformat() + "Z"
        return {
            "loaded": len(missing) == 0,
            "last_modified": last_modified,
            "artifacts_present": present,
            "artifacts_missing": missing,
        }

    pricing = _check([
        "glm_poisson.pkl", "glm_gamma.pkl", "ngboost_severity.pkl",
        "cann_stats.pkl", "pricing_metrics.json",
    ])
    reserving = _check(["reserving_calibration.json", "deep_triangle.pt"])
    fraud = _check(["fraud_best_model.pkl", "fraud_encoders.pkl",
                    "fraud_normalization_stats.pkl", "fraud_metrics.json"])

    return {
        "pricing": pricing,
        "reserving": reserving,
        "fraud": fraud,
        "models_dir": str(MODELS_DIR),
        "generated": {
            "pricing": "scripts/evaluate_pricing.py",
            "reserving": "scripts/calibrate_reserving.py",
            "fraud": "scripts/evaluate_fraud.py",
        },
    }


@app.get("/")
def root():
    return {"status": "ok", "modules": ["pricing", "reserving", "fraud", "explainability"]}
