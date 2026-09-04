# Chapitre 8 — Architecture logicielle et implémentation

## 8.1 Architecture générale de la plateforme

La plateforme « Actuarial AI Platform » adopte une **architecture microservices** conteneurisée, séparant clairement les couchesprésentation, traitement et persistance. Le schéma global illustre les six services orchestrés par Docker Compose :

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Docker Compose Orchestration                       │
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐      │
│  │   Frontend   │───▶│     Backend      │───▶│    PostgreSQL 16     │      │
│  │  Next.js 14  │    │    FastAPI       │    │  actuarial_platform  │      │
│  │  Port 3000   │    │   Port 8000      │    │    Port 5432         │      │
│  └──────────────┘    └────────┬─────────┘    └──────────────────────┘      │
│                               │                                             │
│               ┌───────────────┼───────────────┐                            │
│               ▼               ▼               ▼                            │
│  ┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐                │
│  │     MLflow      │ │  Prometheus  │ │     Grafana      │                │
│  │  Port 5000      │ │  Port 9090   │ │  Port 3001       │                │
│  │  Experiment     │ │   Metrics    │ │   Dashboards     │                │
│  │  Tracking       │ │  Collection  │ │   Visualization  │                │
│  └─────────────────┘ └──────────────┘ └──────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Figure 8.1** — Architecture globale de la plateforme. Six services conteneurisés communiquent via le réseau interne Docker. Le backend FastAPI orchestre les appels aux modèles ML pré-entraînés et persiste chaque prédiction dans PostgreSQL.

### Principes architecturaux

| Principe | Implémentation |
|----------|---------------|
| **Séparation des préoccupations** | Frontend (présentation) / Backend (logique métier) / DB (persistance) |
| **Infrastructure as Code** | `docker-compose.yml` + Dockerfiles reproductibles |
| **Modèles en lecture seule** | Volume `./models:/app/models:ro` — jamais modifiés à l'exécution |
| **Dépendances 注入** | FastAPI `Depends()` pour DB sessions et authentification |
| **Observabilité** | Prometheus + Grafana pour métriques, MLflow pour expériences |

---

## 8.2 Backend — FastAPI et PostgreSQL

### 8.2.1 Structure modulaire

Le backend suit une architecture **layered** avec séparation nette entre routes, services et schémas :

```
backend/app/
├── main.py                  # Point d'entrée FastAPI, lifespan, CORS
├── config.py                # Pydantic BaseSettings (.env)
├── database.py              # SQLAlchemy engine, session, Base
├── auth.py                  # JWT tokens, hashage mot de passe
├── metrics.py               # Compteurs/histogrammes Prometheus
│
├── models/                  # ORM SQLAlchemy (persistance)
│   ├── policy.py            # Table policies (données tarifaires)
│   ├── prediction.py        # Table predictions (journalisation)
│   ├── triangle.py          # Table reserving_runs (provisionnement)
│   └── user.py              # Table users (authentification)
│
├── schemas/                 # Validation Pydantic (entrées/sorties)
│   ├── pricing.py           # PricingRequest, PricingResponse
│   ├── reserving.py         # ReservingIbnrRequest, ReservingIbnrResponse
│   ├── fraud.py             # FraudRequest, FraudResponse
│   ├── explainability.py    # PricingExplanationRequest, ShapExplanation
│   └── auth.py              # UserCreate, Token, UserResponse
│
├── routers/                 # Endpoints REST (couche HTTP)
│   ├── pricing.py           # POST /pricing/predict, /pricing/explain
│   ├── reserving.py         # POST /reserving/ibnr, GET /reserving/companies
│   ├── fraud.py             # POST /fraud/predict, GET /fraud/methodology
│   ├── explainability.py    # POST /explain/pricing, /explain/fraud
│   └── auth.py              # POST /auth/login, /auth/register, GET /auth/me
│
├── services/                # Logique métier (calculs ML)
│   ├── pricing_service.py   # GLM Poisson-Gamma + CANN
│   ├── reserving_service.py # Mack Chain-Ladder + Deep Triangle
│   ├── fraud_service.py     # Random Forest
│   ├── explainability_service.py  # SHAP values
│   └── mlflow_service.py    # Tracking MLflow
│
└── tests/                   # Suite de tests (43 tests, 2 sautés)
    ├── conftest.py          # Fixtures SQLite, TestClient
    ├── test_database.py     # CRUD ORM
    ├── test_integration.py  # Tests endpoints API
    ├── test_pricing.py      # Tests unitaires tarification
    ├── test_reserving.py    # Tests unitaires provisionnement
    ├── test_fraud.py        # Tests unitaires fraude
    └── test_auth.py         # Tests authentification
```

**Figure 8.2** — Arborescence du backend. La séparation `routers/` → `services/` → `models/` garantit que chaque couche ne dépend que de la couche inférieure.

### 8.2.2 Conception de l'API REST

L'API expose **14 endpoints** organisés par module métier :

| Module | Endpoint | Méthode | Description |
|--------|----------|---------|-------------|
| **Pricing** | `/pricing/predict` | POST | Prédiction prime pure (GLM + CANN) |
| | `/pricing/severity-distribution` | POST | Distribution NGBoost/Gamma |
| | `/pricing/premium-copula` | POST | Prime avec copule gaussienne |
| | `/pricing/explain` | POST | Valeurs SHAP tarification |
| **Reserving** | `/reserving/companies` | GET | Liste des compagnies |
| | `/reserving/ibnr` | POST | Estimation IBNR (Mack + Conforme) |
| | `/reserving/predict` | POST | Deep Triangle (démo) |
| **Fraud** | `/fraud/predict` | POST | Score de fraude Random Forest |
| | `/fraud/methodology` | GET | Résumé des approches testées |
| **Explain** | `/explain/pricing` | POST | SHAP tarification (global) |
| | `/explain/fraud` | POST | SHAP fraude |
| | `/explain/cann-interactions` | POST | Interactions CANN |
| **Auth** | `/auth/login` | POST | Authentification JWT |
| | `/auth/register` | POST | Inscription (admin) |
| | `/auth/me` | GET | Profil utilisateur |

### 8.2.3 Persistance et journalisation

Chaque appel à un endpoint de prédiction est **journalisé automatiquement** dans la table `predictions` :

```python
# Extrait de routers/pricing.py
@router.post("/predict", response_model=PricingResponse)
def predict(request: PricingRequest, db: Session = Depends(get_db)):
    start = time.time()
    response = predict_pricing(request)
    latency_ms = (time.time() - start) * 1000

    db.add(Prediction(
        module="pricing",
        model_version=response.model_version,
        request_json=request.model_dump(),
        response_json=response.model_dump(),
        latency_ms=round(latency_ms, 2),
    ))
    db.commit()
    return response
```

Cette journalisation permet de :
- **Auditer** les prédictions (traçabilité réglementaire)
- **Mesurer la latence** (monitoring performance)
- **Détecter les dérives** (comparing distributions temporelles)

### 8.2.4 Modèle de données

```
┌──────────────────────┐     ┌──────────────────────────┐
│      policies        │     │       predictions         │
├──────────────────────┤     ├──────────────────────────┤
│ id            PK     │     │ id                PK     │
│ veh_power            │     │ module                   │
│ veh_age              │     │ model_version            │
│ driv_age             │     │ request_json      JSON   │
│ bonus_malus          │     │ response_json     JSON   │
│ veh_brand            │     │ latency_ms               │
│ veh_gas              │     │ created_at        TS     │
│ region               │     └──────────────────────────┘
│ area                 │
│ density              │     ┌──────────────────────────┐
│ exposure             │     │     reserving_runs       │
│ created_at    TS     │     ├──────────────────────────┤
└──────────────────────┘     │ id                PK     │
                             │ grcode                   │
┌──────────────────────┐     │ evaluation_year          │
│       users          │     │ ibnr_estimate            │
├──────────────────────┤     │ mack_lower / upper       │
│ id            PK     │     │ conformal_lower / upper  │
│ username             │     │ triangle_json     JSON   │
│ hashed_password      │     │ ldfs_json         JSON   │
│ email                │     │ created_at        TS     │
│ role                 │     └──────────────────────────┘
│ is_active            │
│ created_at    TS     │
└──────────────────────┘
```

**Figure 8.3** — Schéma de la base de données. Quatre tables principales : `predictions` (journalisation), `policies` (données tarifaires), `reserving_runs` (provisionnement), `users` (authentification).

---

## 8.3 Frontend — Next.js et TypeScript

### 8.3.1 Structure du dashboard

Le frontend est une application **Next.js 14** (App Router) avec Tailwind CSS, organisée en pages correspondant aux modules actuariels :

```
frontend/
├── app/
│   ├── page.tsx              # Accueil — vue d'ensemble
│   ├── layout.tsx            # Layout racine (navbar, thème)
│   ├── dashboard.tsx         # Dashboard principal
│   ├── pricing/page.tsx      # Module tarification
│   ├── reserving/page.tsx    # Module provisionnement
│   ├── fraud/page.tsx        # Module détection de fraude
│   └── methodology/page.tsx  # Méthodologie détaillée
│
├── components/
│   ├── ConfidenceIntervalChart.tsx   # Intervalles Mack vs Conforme
│   ├── ReservingTriangleChart.tsx    # Triangle de développement
│   ├── TriangleHeatmap.tsx           # Heatmap du triangle
│   ├── SeverityDistributionChart.tsx # Distribution NGBoost
│   ├── CannInteractionChart.tsx      # Interactions CANN
│   ├── FraudDetectionChart.tsx       # Score de fraude
│   ├── FraudSubgraphChart.tsx        # Graphe relationnel
│   └── ReliabilityDiagram.tsx        # Diagramme de fiabilité
│
├── lib/
│   └── api.ts                # Client API typé (interfaces + fetch)
│
├── Dockerfile                # Multi-stage build
└── package.json
```

### 8.3.2 Client API typé

Le client API ([lib/api.ts](frontend/lib/api.ts)) définit des **interfaces TypeScript** pour chaque requête/réponse, garantissant la cohérence frontend-backend :

```typescript
// Interfaces typées pour le provisionnement
export interface ReservingIbnrResponse {
  grcode: number;
  ibnr_estimate: number;
  mack_interval: {
    lower_90: number;
    upper_90: number;
    empirical_coverage: number;
  };
  conformal_interval: {
    lower_90: number;
    upper_90: number;
    empirical_coverage: number;
  };
  triangle_data?: {
    values: number[][];
    std_errors: number[][];
    origin_years: number[];
    development_years: number[];
    cell_status: number[][];
  };
}
```

### 8.3.3 Composants de visualisation

Chaque module dispose de composants React dédiés aux visualisations actuarielles :

- **TriangleHeatmap** — Représentation visuelle du triangle de développement pertes
- **ConfidenceIntervalChart** — Comparaison visuelle intervalles Mack vs Conforme
- **SeverityDistributionChart** — Distribution de sévérité NGBoost avec percentiles
- **FraudDetectionChart** — Jauge de probabilité de fraude
- **CannInteractionChart** — Graphique radar des interactions non-linéaires

---

## 8.4 MLOps — MLflow et Docker

### 8.4.1 Conteneurisation

La conteneurisation utilise Docker avec des **images multi-stages** pour optimiser la taille :

**Backend** ([backend/Dockerfile](backend/Dockerfile)) :
```dockerfile
# Stage 1: Build
FROM python:3.11-slim AS builder
COPY backend/requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# Stage 2: Final
FROM python:3.11-slim
COPY --from=builder /install /usr/local
COPY backend/app/ ./app/
COPY src/ ./src/          # Modules partagés
COPY models/ ./models/    # Modèles (read-only)
```

**Frontend** ([frontend/Dockerfile](frontend/Dockerfile)) :
```dockerfile
# Stage 1: Build Next.js
FROM node:20-alpine AS builder
RUN npm ci && npm run build

# Stage 2: Serve (minimal)
FROM node:20-alpine
COPY --from=builder /app/.next ./.next
CMD ["npm", "start"]
```

### 8.4.2 MLflow — Tracking des expériences

MLflow est configuré pour tracker les métriques de chaque modèle :

| Composant | Configuration |
|-----------|--------------|
| **Backend store** | PostgreSQL (même base que l'app) |
| **Artifact store** | Volume Docker `/mlflow/artifacts` |
| **UI** | Port 5000 (accessible via `http://localhost:5000`) |

Le service `mlflow_service.py` automatise le logging :

```python
def log_prediction_metrics(module: str, latency_ms: float, **metrics):
    """Log les métriques de prédiction dans MLflow."""
    with mlflow.start_run(run_name=f"{module}_prediction"):
        mlflow.log_metric("latency_ms", latency_ms)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
```

### 8.4.3 Monitoring — Prometheus et Grafana

La pile de monitoring collecte les métriques applicables :

```
Backend FastAPI
    │
    ▼ (metrics endpoint)
Prometheus ──────▶ Grafana
  :9090              :3001
  scrape             dashboards
  15s                interactifs
```

**Métriques collectées** :
- `prediction_counter` — Nombre de prédictions par module
- `prediction_latency` — Histogramme de latence (ms)
- `model_version` — Version du modèle chargé
- `active_users` — Utilisateurs actifs

### 8.4.4 CI/CD — GitHub Actions

Le pipeline ([.github/workflows/test.yml](.github/workflows/test.yml)) automatise les vérifications :

```yaml
jobs:
  backend-tests:
    runs-on: ubuntu-latest
    services:
      postgres:  # Service PostgreSQL pour les tests
        image: postgres:16
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r backend/requirements.txt
      - run: pytest app/tests/ -v --cov=app

  frontend-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: cd frontend && npm ci && npm run build
```

---

## 8.5 Choix techniques et justifications

### Tableau comparatif des technologies

| Couche | Technologie choisie | Alternative considérée | Justification |
|--------|-------------------|----------------------|---------------|
| **Backend API** | FastAPI | Flask, Django REST | Async natif, auto-doc Swagger, validation Pydantic |
| **ORM** | SQLAlchemy 2.0 | Django ORM, Peewee | Déclaratif, async support, écosystème riche |
| **Base de données** | PostgreSQL 16 | SQLite, MySQL | JSONB natif, performances, production-ready |
| **Frontend** | Next.js 14 | React SPA, Vue.js | SSR/SSG, routing file-based, écosystème React |
| **Styling** | Tailwind CSS | Material UI, Bootstrap | Utility-first, bundle minimal, customisable |
| **ML Tracking** | MLflow | Weights & Biases, DVC | Open-source, self-hosted, intégration scikit-learn |
| **Conteneurisation** | Docker Compose | Kubernetes, Podman | Simple pour dev/staging, suffisant pour ce scope |
| **Monitoring** | Prometheus + Grafana | Datadog, New Relic | Open-source, auto-hébergé, pas de coût externe |
| **Authentification** | JWT (python-jose) | Session cookies, OAuth2 | Stateless, scalable, standard API |

### Justifications détaillées

**FastAPI vs Flask** : FastAPI génère automatiquement la documentation OpenAPI (Swagger), valide les entrées avec Pydantic, et supporte l'injection de dépendances — critique pour le pattern `Depends(get_db)` utilisé dans chaque route.

**PostgreSQL vs SQLite** : Bien que SQLite soit utilisé pour les tests (gratuit, sans serveur), PostgreSQL est retenu pour la production car il supporte nativement le type JSONB (stockage des requêtes/réponses ML) et offre le concurrent access nécessaire pour un service multi-utilisateurs.

**MLflow vs W&B** : MLflow est open-source et self-hosté, évitant la dépendance à un service cloud externe. L'intégration native avec scikit-learn et PyTorch couvre les besoins de tracking de ce projet.

**JWT vs sessions** : Les tokens JWT sont stateless (pas de stockage côté serveur), ce qui facilite le scaling horizontal. Chaque requête API est autonome — pas besoin de session store distribué.

### Métriques de qualité logicielle

| Métrique | Valeur | Commentaire |
|----------|--------|-------------|
| **Couverture de tests** | 43 tests, 2 sautés | Tests unitaires + intégration + CRUD |
| **Latence API** | ~50-200 ms | Selon le modèle (GLM rapide, Deep Triangle plus lent) |
| **Taille image backend** | ~450 MB | Multi-stage build avec dépendances système minimales |
| **Taille image frontend** | ~120 MB | Alpine + build Next.js optimisé |
| **Documentation API** | Auto-générée | Swagger UI via `/docs` |

---

## Résumé du chapitre

L'architecture de la plateforme repose sur six principes fondamentaux :

1. **Séparation frontend/backend** — Next.js (présentation) et FastAPI (logique métier) évoluent indépendamment
2. **Modularity** — Chaque module actuariel (pricing, reserving, fraud) est un router séparé avec son propre schéma et service
3. **Reproductibilité** — Docker Compose garantit que l'environnement de développement identique à la production
4. **Observabilité** — Chaque prédiction est journalisée (PostgreSQL), chaque latence est métriquée (Prometheus)
5. **Sécurité** — Authentification JWT, modèles en lecture seule, validation Pydantic sur toutes les entrées
6. **Qualité** — Tests automatisés (43 tests), CI/CD GitHub Actions, documentation auto-générée

Cette architecture permet à un actuaire de comparer visuellement les méthodes classiques (GLM, Mack) et les approches modernes (CANN, Deep Triangle, SHAP) via une interface web intuitive, tout en maintenant les garanties de traçabilité et de reproductibilité exigées en production.
