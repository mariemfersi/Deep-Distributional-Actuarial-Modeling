"""
Chapitre 8 — Architecture logicielle et implémentation

Génère exactement 5 figures :
    8.1 — Architecture générale de la plateforme
    8.2 — Backend : FastAPI + PostgreSQL
    8.3 — Frontend : Next.js + TypeScript
    8.4 — MLOps : MLflow + Docker
    8.5 — Choix techniques et justifications

Les figures sont destinées au mémoire et reflètent
l'architecture réelle du projet.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# PALETTE
# ================================================================

C_PRIC = "#4CAF50"       # Pricing
C_FRAUD = "#F44336"      # Fraud
C_RESP = "#FF9800"       # Reserving
C_EXPL = "#9C27B0"       # Explainability
C_BACK = "#1565C0"       # Backend
C_FRONT = "#00838F"      # Frontend
C_DEVOPS = "#6D4C41"     # MLOps
C_DB = "#37474F"         # Database
C_NEUT = "#607D8B"       # Neutral


LIGHT = {
    C_PRIC: "#E8F5E9",
    C_FRAUD: "#FFEBEE",
    C_RESP: "#FFF3E0",
    C_EXPL: "#F3E5F5",
    C_BACK: "#E3F2FD",
    C_FRONT: "#E0F7FA",
    C_DEVOPS: "#EFEBE9",
    C_DB: "#ECEFF1",
    C_NEUT: "#ECEFF1",
}


# ================================================================
# HELPERS
# ================================================================

def _box(
    ax,
    x,
    y,
    w,
    h,
    color,
    label,
    sublabel=None,
    fs=9,
    subfs=7,
    lw=1.6,
    dashed=False,
    text_color=None,
):
    """Dessine une boîte arrondie."""

    light = LIGHT.get(color, "#F5F5F5")

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08",
        facecolor="white" if dashed else light,
        edgecolor=color,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=2,
    )

    ax.add_patch(box)

    cx = x + w / 2
    cy = y + h / 2

    tc = text_color if text_color else color

    if sublabel:
        ax.text(
            cx,
            cy + 0.20,
            label,
            fontsize=fs,
            ha="center",
            va="center",
            color=tc,
            fontweight="bold",
            zorder=3,
        )

        ax.text(
            cx,
            cy - 0.27,
            sublabel,
            fontsize=subfs,
            ha="center",
            va="center",
            color="#666666",
            zorder=3,
        )

    else:
        ax.text(
            cx,
            cy,
            label,
            fontsize=fs,
            ha="center",
            va="center",
            color=tc,
            fontweight="bold",
            zorder=3,
        )


def _arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color="#455A64",
    lw=1.6,
    rad=0.0,
    ls="-",
):
    """Dessine une flèche."""

    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        color=color,
        linewidth=lw,
        mutation_scale=14,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=ls,
        zorder=4,
    )

    ax.add_patch(arrow)


def _title(ax, title, subtitle, width, height):
    """Titre uniforme pour les cinq figures."""

    ax.text(
        width / 2,
        height - 0.45,
        title,
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="center",
    )

    ax.text(
        width / 2,
        height - 1.05,
        subtitle,
        fontsize=9.5,
        ha="center",
        va="center",
        color="#666666",
    )


def _save(fig, filename):
    """Sauvegarde uniforme."""

    output = OUT_DIR / filename

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Figure générée : {output}")


# ================================================================
# FIGURE 8.1
# ARCHITECTURE GENERALE
# ================================================================

def fig8_1():

    W, H = 16, 10

    fig, ax = plt.subplots(figsize=(16, 9.5))

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _title(
        ax,
        "8.1 — Architecture générale de la plateforme",
        "Chaîne complète : données → modèles actuariels → API → dashboard",
        W,
        H,
    )

    # ------------------------------------------------------------
    # 1. DATA
    # ------------------------------------------------------------

    _box(
        ax,
        0.6,
        7.6,
        4.0,
        1.1,
        C_DB,
        "Sources de données",
        "freMTPL2 · CAS · Fraud claims",
        fs=10,
    )

    _box(
        ax,
        0.7,
        5.95,
        1.8,
        1.0,
        C_NEUT,
        "Pricing",
        "policies / claims",
        fs=8.5,
    )

    _box(
        ax,
        2.7,
        5.95,
        1.8,
        1.0,
        C_NEUT,
        "Reserving",
        "loss triangles",
        fs=8.5,
    )

    _box(
        ax,
        4.7,
        5.95,
        1.8,
        1.0,
        C_NEUT,
        "Fraud",
        "claim records",
        fs=8.5,
    )

    # ------------------------------------------------------------
    # 2. MODELISATION
    # ------------------------------------------------------------

    _box(
        ax,
        6.6,
        7.6,
        8.8,
        1.1,
        C_PRIC,
        "Couche de modélisation — src/",
        "prétraitement · features · entraînement · évaluation · inférence",
        fs=10,
    )

    _box(
        ax,
        6.8,
        5.95,
        2.0,
        1.0,
        C_PRIC,
        "Tarification",
        "GLM · CANN · NGBoost",
        fs=8.5,
    )

    _box(
        ax,
        9.0,
        5.95,
        2.0,
        1.0,
        C_RESP,
        "Provisionnement",
        "Mack · Conformal",
        fs=8.5,
    )

    _box(
        ax,
        11.2,
        5.95,
        2.0,
        1.0,
        C_FRAUD,
        "Fraude",
        "Random Forest",
        fs=8.5,
    )

    _box(
        ax,
        13.4,
        5.95,
        1.8,
        1.0,
        C_EXPL,
        "Explicabilité",
        "SHAP",
        fs=8.5,
    )

    # ------------------------------------------------------------
    # 3. BACKEND
    # ------------------------------------------------------------

    _box(
        ax,
        0.6,
        3.65,
        14.8,
        1.2,
        C_BACK,
        "Backend — FastAPI",
        "Routers · Services · Schémas Pydantic · modèles ML · tests",
        fs=10,
    )

    # ------------------------------------------------------------
    # 4. FRONTEND
    # ------------------------------------------------------------

    _box(
        ax,
        0.6,
        1.35,
        14.8,
        1.2,
        C_FRONT,
        "Frontend — Next.js + TypeScript",
        "Pricing · Reserving · Fraud · Explainability · Methodology",
        fs=10,
    )

    # ------------------------------------------------------------
    # 5. INFRASTRUCTURE
    # ------------------------------------------------------------

    _box(
        ax,
        0.6,
        0.15,
        4.3,
        0.8,
        C_DB,
        "PostgreSQL",
        "infrastructure locale via Docker Compose",
        fs=8.5,
    )

    _box(
        ax,
        5.2,
        0.15,
        4.3,
        0.8,
        C_DEVOPS,
        "MLflow",
        "tracking local des expériences",
        fs=8.5,
    )

    _box(
        ax,
        9.8,
        0.15,
        5.6,
        0.8,
        C_DEVOPS,
        "Docker / Compose",
        "reproductibilité de l'environnement",
        fs=8.5,
    )

    # ------------------------------------------------------------
    # ARROWS
    # ------------------------------------------------------------

    # Data → model layer
    for x in [1.6, 3.6, 5.6]:
        _arrow(ax, x, 5.95, x + 2.0, 6.9, rad=0.05)

    # Model layer → backend
    for x in [7.8, 10.0, 12.2, 14.2]:
        _arrow(ax, x, 5.95, x, 4.85)

    # Backend → frontend
    _arrow(
        ax,
        8.0,
        3.65,
        8.0,
        2.55,
        lw=2.2,
    )

    ax.text(
        8.25,
        3.0,
        "REST / JSON",
        fontsize=8,
        color="#455A64",
        style="italic",
    )

    # ------------------------------------------------------------
    # LEGEND
    # ------------------------------------------------------------

    handles = [
        mpatches.Patch(
            facecolor=LIGHT[C_PRIC],
            edgecolor=C_PRIC,
            label="Modélisation",
        ),
        mpatches.Patch(
            facecolor=LIGHT[C_BACK],
            edgecolor=C_BACK,
            label="Backend",
        ),
        mpatches.Patch(
            facecolor=LIGHT[C_FRONT],
            edgecolor=C_FRONT,
            label="Frontend",
        ),
        mpatches.Patch(
            facecolor=LIGHT[C_DB],
            edgecolor=C_DB,
            label="Données / Infrastructure",
        ),
        mpatches.Patch(
            facecolor=LIGHT[C_DEVOPS],
            edgecolor=C_DEVOPS,
            label="MLOps",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=7.5,
        ncol=5,
        frameon=False,
    )

    plt.tight_layout()

    _save(
        fig,
        "chapter8_01_architecture_globale.png",
    )


# ================================================================
# FIGURE 8.2
# BACKEND FASTAPI + POSTGRESQL
# ================================================================

def fig8_2():

    W, H = 16, 10

    fig, ax = plt.subplots(figsize=(16, 9.5))

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _title(
        ax,
        "8.2 — Backend : FastAPI et PostgreSQL",
        "Organisation de l'API en couches et intégration de la persistance",
        W,
        H,
    )

    # ------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------

    _box(
        ax,
        3.0,
        7.6,
        10.0,
        0.9,
        C_BACK,
        "Couche HTTP — FastAPI / Uvicorn",
        "REST · CORS · OpenAPI / Swagger",
        fs=9.5,
    )

    # ------------------------------------------------------------
    # ROUTERS
    # ------------------------------------------------------------

    routers = [
        (
            "Pricing",
            "/pricing",
            C_PRIC,
            "prediction\nseverity · copula",
        ),
        (
            "Reserving",
            "/reserving",
            C_RESP,
            "IBNR · companies\nprediction",
        ),
        (
            "Fraud",
            "/fraud",
            C_FRAUD,
            "prediction\nmethodology",
        ),
        (
            "Explainability",
            "/explain",
            C_EXPL,
            "pricing\ninteractions",
        ),
    ]

    router_y = 5.6

    for i, (name, prefix, color, endpoints) in enumerate(routers):

        x = 0.5 + i * 3.85

        _box(
            ax,
            x,
            router_y,
            3.25,
            1.0,
            color,
            f"{name} Router",
            prefix,
            fs=8.5,
        )

        _box(
            ax,
            x,
            4.0,
            3.25,
            1.1,
            color,
            "Endpoints",
            endpoints,
            fs=8,
            subfs=6.5,
        )

        _arrow(
            ax,
            x + 1.625,
            7.6,
            x + 1.625,
            6.6,
        )

        _arrow(
            ax,
            x + 1.625,
            5.6,
            x + 1.625,
            5.1,
        )

    # ------------------------------------------------------------
    # SERVICES
    # ------------------------------------------------------------

    _box(
        ax,
        0.5,
        2.65,
        15.0,
        0.9,
        C_NEUT,
        "Couche Services — logique métier",
        "pricing_service · reserving_service · fraud_service · explainability_service",
        fs=9.5,
    )

    # Router → service
    for x in [2.125, 5.975, 9.825, 13.675]:
        _arrow(
            ax,
            x,
            4.0,
            x,
            3.55,
        )

    # ------------------------------------------------------------
    # MODEL / INFERENCE
    # ------------------------------------------------------------

    _box(
        ax,
        0.5,
        1.25,
        7.2,
        0.9,
        C_PRIC,
        "Couche modèles / inférence",
        "artefacts ML · preprocessing · prédiction · explicabilité",
        fs=9,
    )

    # ------------------------------------------------------------
    # POSTGRESQL
    # ------------------------------------------------------------

    _box(
        ax,
        8.2,
        1.25,
        7.3,
        0.9,
        C_DB,
        "PostgreSQL 16 — infrastructure",
        "service Docker Compose · volume pgdata · port 5432",
        fs=9,
    )

    # ------------------------------------------------------------
    # SCHEMAS
    # ------------------------------------------------------------

    _box(
        ax,
        0.5,
        0.05,
        7.2,
        0.8,
        C_FRONT,
        "Schémas Pydantic",
        "validation · sérialisation · contrats request / response",
        fs=8.5,
    )

    _box(
        ax,
        8.2,
        0.05,
        7.3,
        0.8,
        C_FRAUD,
        "Tests",
        "validation des services et endpoints",
        fs=8.5,
    )

    # Service → model
    _arrow(
        ax,
        4.0,
        2.65,
        4.0,
        2.15,
    )

    # Service → DB
    _arrow(
        ax,
        11.0,
        2.65,
        11.0,
        2.15,
        ls="--",
    )

    ax.text(
        11.15,
        2.35,
        "persistance",
        fontsize=7,
        color="#666666",
    )

    # ------------------------------------------------------------
    # NOTE
    # ------------------------------------------------------------

    ax.text(
        8.2,
        0.88,
        "4 tables ORM (users, policies, predictions, reserving_runs) ; "
        "migrations Alembic pour le versionnage du schéma.",
        fontsize=7.2,
        color="#666666",
        style="italic",
    )

    plt.tight_layout()

    _save(
        fig,
        "chapter8_02_backend_fastapi_postgresql.png",
    )


# ================================================================
# FIGURE 8.3
# FRONTEND NEXT.JS + TYPESCRIPT
# ================================================================

def fig8_3():

    W, H = 16, 10

    fig, ax = plt.subplots(figsize=(16, 9.5))

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _title(
        ax,
        "8.3 — Frontend : Next.js + TypeScript",
        "Dashboard analytique modulaire pour l'exploitation des résultats actuariels",
        W,
        H,
    )

    # ------------------------------------------------------------
    # APP ROUTER
    # ------------------------------------------------------------

    _box(
        ax,
        5.9,
        7.6,
        4.2,
        0.9,
        C_FRONT,
        "Next.js App Router",
        "app/ · layout · pages · composants React",
        fs=9.5,
    )

    # ------------------------------------------------------------
    # PAGES
    # ------------------------------------------------------------

    pages = [
        (
            0.5,
            "/pricing",
            "Tarification",
            "GLM · CANN\nsimulation",
            C_PRIC,
        ),
        (
            4.1,
            "/reserving",
            "Provisionnement",
            "Mack · Conformal\nIBNR",
            C_RESP,
        ),
        (
            7.7,
            "/fraud",
            "Détection de fraude",
            "score · features\nSHAP",
            C_FRAUD,
        ),
        (
            11.3,
            "/methodology",
            "Méthodologie",
            "modèles · choix\nexplications",
            C_EXPL,
        ),
    ]

    for x, route, title, desc, color in pages:

        _box(
            ax,
            x,
            5.55,
            3.2,
            1.2,
            color,
            f"{route}",
            f"{title}\n{desc}",
            fs=8.5,
            subfs=6.5,
        )

        _arrow(
            ax,
            8.0,
            7.6,
            x + 1.6,
            6.75,
            rad=0.0,
        )

    # ------------------------------------------------------------
    # DASHBOARD
    # ------------------------------------------------------------

    _box(
        ax,
        3.0,
        3.9,
        10.0,
        0.95,
        C_FRONT,
        "Dashboard analytique",
        "cartes KPI · tableaux · graphiques · intervalles · explications",
        fs=9.5,
    )

    # ------------------------------------------------------------
    # COMPONENTS
    # ------------------------------------------------------------

    _box(
        ax,
        0.5,
        2.15,
        6.8,
        1.1,
        C_FRONT,
        "Composants de visualisation",
        "Recharts · SeverityDistribution · ConfidenceInterval · TriangleHeatmap · FraudDetection",
        fs=8.5,
    )

    _box(
        ax,
        7.7,
        2.15,
        7.8,
        1.1,
        C_FRONT,
        "Composants analytiques",
        "SHAP · CANN interactions · reliability diagram · reserving triangle",
        fs=8.5,
    )

    # Dashboard → components
    _arrow(
        ax,
        6.0,
        3.9,
        4.0,
        3.25,
    )

    _arrow(
        ax,
        10.0,
        3.9,
        11.0,
        3.25,
    )

    # ------------------------------------------------------------
    # API
    # ------------------------------------------------------------

    _box(
        ax,
        0.5,
        0.45,
        6.8,
        1.0,
        C_BACK,
        "Couche API — lib/api.ts",
        "fonctions fetch · interfaces TypeScript · appels REST",
        fs=8.5,
    )

    _box(
        ax,
        7.7,
        0.45,
        7.8,
        1.0,
        C_BACK,
        "Backend FastAPI",
        "REST / JSON · localhost:8000 · endpoints métier",
        fs=8.5,
    )

    # API → backend
    _arrow(
        ax,
        7.3,
        0.95,
        7.7,
        0.95,
        lw=2.0,
    )

    # Components → API
    _arrow(
        ax,
        3.9,
        2.15,
        3.9,
        1.45,
    )

    # ------------------------------------------------------------
    # STACK
    # ------------------------------------------------------------

    ax.text(
        12.0,
        4.35,
        "Stack : Next.js 16 · React 19 · TypeScript 5 · Tailwind CSS · Recharts",
        fontsize=7.5,
        ha="center",
        color="#666666",
        style="italic",
    )

    plt.tight_layout()

    _save(
        fig,
        "chapter8_03_frontend_nextjs_typescript.png",
    )


# ================================================================
# FIGURE 8.4
# MLOPS : MLFLOW + DOCKER
# ================================================================

def fig8_4():

    W, H = 16, 9.5

    fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _title(
        ax,
        "8.4 — MLOps : MLflow et Docker",
        "Traçabilité des expérimentations et reproductibilité de l'environnement",
        W,
        H,
    )

    # ============================================================
    # MLFLOW
    # ============================================================

    _box(
        ax,
        0.5,
        7.4,
        7.0,
        0.9,
        C_DEVOPS,
        "MLflow — Experiment Tracking",
        "src/common/tracking.py · tracking local",
        fs=9.5,
    )

    _box(
        ax,
        0.5,
        5.85,
        3.2,
        1.1,
        C_DEVOPS,
        "Tracking",
        "mlruns/\nexpériences locales",
        fs=8.5,
    )

    _box(
        ax,
        4.1,
        5.85,
        3.4,
        1.1,
        C_DEVOPS,
        "Runs",
        "params · metrics\nartifacts",
        fs=8.5,
    )

    # ------------------------------------------------------------
    # EXPERIMENTS
    # ------------------------------------------------------------

    experiments = [
        (
            0.5,
            "Pricing",
            "GLM · CANN · NGBoost",
            C_PRIC,
        ),
        (
            2.95,
            "Reserving",
            "Mack · Conformal\nDeep Triangle",
            C_RESP,
        ),
        (
            5.4,
            "Fraud",
            "Random Forest\nGNN expérimental",
            C_FRAUD,
        ),
    ]

    for x, name, desc, color in experiments:

        _box(
            ax,
            x,
            4.0,
            2.1,
            1.2,
            color,
            name,
            desc,
            fs=8.5,
            subfs=6.5,
        )

        _arrow(
            ax,
            x + 1.05,
            4.0,
            x + 1.05,
            4.95,
        )

    # Params / metrics
    _box(
        ax,
        0.5,
        2.1,
        7.0,
        1.2,
        C_DEVOPS,
        "Informations suivies",
        "hyperparamètres · métriques · versions · artefacts de modèles",
        fs=8.5,
    )

    _arrow(
        ax,
        3.5,
        5.85,
        3.5,
        5.15,
    )

    _arrow(
        ax,
        3.5,
        4.0,
        3.5,
        3.3,
    )

    # ============================================================
    # DOCKER
    # ============================================================

    _box(
        ax,
        8.0,
        7.4,
        7.5,
        0.9,
        C_BACK,
        "Docker / Docker Compose",
        "infrastructure locale reproductible",
        fs=9.5,
    )

    _box(
        ax,
        8.0,
        5.85,
        3.4,
        1.1,
        C_DB,
        "PostgreSQL",
        "postgres:16\nservice db",
        fs=8.5,
    )

    _box(
        ax,
        11.9,
        5.85,
        3.6,
        1.1,
        C_DB,
        "Volume pgdata",
        "/var/lib/postgresql/data\npersistance locale",
        fs=8.5,
    )

    _arrow(
        ax,
        9.7,
        5.85,
        11.9,
        5.85,
    )

    # ------------------------------------------------------------
    # ENVIRONNEMENT APPLICATION
    # ------------------------------------------------------------

    _box(
        ax,
        8.0,
        3.95,
        7.5,
        1.2,
        C_NEUT,
        "Applications",
        "FastAPI : 8000 · Next.js : 3000\nexécutées séparément de PostgreSQL",
        fs=8.5,
    )

    _box(
        ax,
        8.0,
        2.1,
        7.5,
        1.2,
        C_DEVOPS,
        "Reproductibilité",
        "versions de dépendances · configuration · environnement isolé",
        fs=8.5,
    )

    _arrow(
        ax,
        11.75,
        5.85,
        11.75,
        5.15,
    )

    _arrow(
        ax,
        11.75,
        3.95,
        11.75,
        3.3,
    )

    # ============================================================
    # BOTTOM NOTE
    # ============================================================

    _box(
        ax,
        0.5,
        0.35,
        15.0,
        0.9,
        C_NEUT,
        "Périmètre MLOps",
        "Monitoring (Prometheus + Grafana), CI/CD (GitHub Actions : tests + Docker build) et tracking MLflow — infrastructure MLOps complète.",
        fs=8.2,
        dashed=True,
    )

    plt.tight_layout()

    _save(
        fig,
        "chapter8_04_mlops_mlflow_docker.png",
    )


# ================================================================
# FIGURE 8.5
# CHOIX TECHNIQUES
# ================================================================

def fig8_5():

    W, H = 16, 11.5

    fig, ax = plt.subplots(figsize=(16, 10.5))

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _title(
        ax,
        "8.5 — Choix techniques et justifications",
        "Technologies retenues pour la modélisation, l'API, l'interface et l'infrastructure",
        W,
        H,
    )

    # ------------------------------------------------------------
    # TABLE CONFIG
    # ------------------------------------------------------------

    x1 = 0.5
    x2 = 5.3
    x3 = 9.0

    table_width = 15.0
    row_h = 0.82
    header_h = 0.75

    y = 9.35

    def table_cell(
        x,
        y,
        width,
        height,
        text,
        fontsize=8.0,
        color="#333333",
        bold=False,
        align="left",
    ):

        rect = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.015",
            facecolor="#F7F8FA",
            edgecolor="#D9DEE3",
            linewidth=0.7,
        )

        ax.add_patch(rect)

        ax.text(
            x + 0.14 if align == "left" else x + width / 2,
            y + height / 2,
            text,
            fontsize=fontsize,
            ha=align,
            va="center",
            color=color,
            fontweight="bold" if bold else "normal",
            wrap=True,
        )

    # ------------------------------------------------------------
    # HEADERS
    # ------------------------------------------------------------

    headers = [
        (x1, 4.6, "Technologie", C_BACK),
        (x2, 3.6, "Rôle", C_FRONT),
        (x3, 6.5, "Justification", C_DEVOPS),
    ]

    for x, width, text, color in headers:

        rect = FancyBboxPatch(
            (x, y),
            width,
            header_h,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor=color,
        )

        ax.add_patch(rect)

        ax.text(
            x + width / 2,
            y + header_h / 2,
            text,
            fontsize=9,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    # ------------------------------------------------------------
    # ROWS
    # ------------------------------------------------------------

    rows = [
        (
            "FastAPI + Uvicorn",
            "API REST",
            "Framework ASGI léger, validation Pydantic et documentation OpenAPI.",
        ),
        (
            "Pydantic",
            "Validation",
            "Structures typées pour contrôler les données d'entrée et de sortie de l'API.",
        ),
        (
            "scikit-learn",
            "Machine Learning",
            "Implémentation robuste des modèles tabulaires et des baselines.",
        ),
        (
            "PyTorch",
            "Deep Learning",
            "Flexibilité pour les architectures CANN et les expérimentations profondes.",
        ),
        (
            "PyTorch Geometric",
            "GNN expérimental",
            "Support de la modélisation en graphes et des expérimentations GNN.",
        ),
        (
            "Next.js + React",
            "Frontend",
            "Architecture web modulaire et composants React interactifs.",
        ),
        (
            "TypeScript",
            "Typage frontend",
            "Réduction des erreurs grâce au typage des données et des interfaces API.",
        ),
        (
            "Recharts",
            "Visualisation",
            "Composants graphiques React adaptés aux indicateurs actuariels.",
        ),
        (
            "PostgreSQL + Docker",
            "Infrastructure",
            "Base relationnelle open-source et environnement local reproductible.",
        ),
        (
            "MLflow",
            "MLOps",
            "Traçabilité des expérimentations, paramètres, métriques et artefacts.",
        ),
    ]

    y -= row_h

    for i, (technology, role, justification) in enumerate(rows):

        if i % 2 == 0:
            bg = "#F5F7F9"
        else:
            bg = "white"

        # background
        for x, width in [
            (x1, 4.6),
            (x2, 3.6),
            (x3, 6.5),
        ]:

            rect = FancyBboxPatch(
                (x, y),
                width,
                row_h - 0.04,
                boxstyle="round,pad=0.015",
                facecolor=bg,
                edgecolor="#D9DEE3",
                linewidth=0.6,
            )

            ax.add_patch(rect)

        ax.text(
            x1 + 0.14,
            y + row_h / 2,
            technology,
            fontsize=8.0,
            ha="left",
            va="center",
            color=C_BACK,
            fontweight="bold",
        )

        ax.text(
            x2 + 0.14,
            y + row_h / 2,
            role,
            fontsize=7.8,
            ha="left",
            va="center",
            color="#333333",
        )

        ax.text(
            x3 + 0.14,
            y + row_h / 2,
            justification,
            fontsize=7.5,
            ha="left",
            va="center",
            color="#333333",
        )

        y -= row_h

    # ------------------------------------------------------------
    # CONCLUSION
    # ------------------------------------------------------------

    ax.text(
        0.5,
        0.55,
        "Synthèse : l'architecture privilégie un écosystème Python/TypeScript cohérent, "
        "adapté à la modélisation actuarielle, à l'exposition des modèles et à leur visualisation.",
        fontsize=8.5,
        color="#455A64",
        style="italic",
    )

    plt.tight_layout()

    _save(
        fig,
        "chapter8_05_choix_techniques.png",
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("   GÉNÉRATION DES FIGURES — CHAPITRE 8")
    print("==============================================\n")

    fig8_1()
    fig8_2()
    fig8_3()
    fig8_4()
    fig8_5()

    print("\n==============================================")
    print("   5 FIGURES GÉNÉRÉES AVEC SUCCÈS")
    print("==============================================")

    print(f"\nDossier : {OUT_DIR}")