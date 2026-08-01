from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import pricing, reserving, fraud, explainability
from app.services import pricing_service, reserving_service, fraud_service

app = FastAPI(
    title="Actuarial AI Platform API",
    description="API de tarification, provisionnement et détection de fraude",
    version="0.2.0",
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pricing.router)
app.include_router(reserving.router)
app.include_router(fraud.router)
app.include_router(explainability.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models/status")
def models_status():
    return {
        "pricing": {"loaded": True, "version": "glm_poisson_gamma_v1"},
        "reserving": {"loaded": True, "version": "mack_conformal_v1"},
        "fraud": {"loaded": True, "version": "random_forest_v1"}
    }


@app.get("/")
def root():
    return {"status": "ok", "modules": ["pricing", "reserving", "fraud", "explainability"]}

