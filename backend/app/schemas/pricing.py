"""Schémas de validation pour l'API de tarification."""

from pydantic import BaseModel, Field


class PricingRequest(BaseModel):
    """Profil de risque d'une police, tel qu'attendu en entrée de l'API."""
    veh_power: int = Field(..., ge=1, le=20, description="Puissance du véhicule")
    veh_age: int = Field(..., ge=0, le=100, description="Âge du véhicule en années")
    driv_age: int = Field(..., ge=16, le=120, description="Âge du conducteur")
    bonus_malus: int = Field(..., ge=35, le=500, description="Coefficient bonus-malus")
    veh_brand: str = Field(..., description="Marque du véhicule (ex. B12)")
    veh_gas: str = Field(..., description="Type de carburant : Diesel ou Regular")
    region: str = Field(..., description="Région (ex. Ile-de-France)")
    area: str = Field(..., description="Zone géographique A-F")
    density: float = Field(..., gt=0, description="Densité de population de la zone")
    exposure: float = Field(default=1.0, gt=0, le=5.0, description="Exposition en années")


class PricingResponse(BaseModel):
    """Résultat de la prédiction de prime avec comparaison GLM vs modèle amélioré."""
    glm_baseline: dict = Field(description="Prédiction GLM Poisson + Gamma (baseline)")
    improved_model: dict = Field(description="Prédiction modèle amélioré (actuellement GLM)")
    gini_improvement_pct: float = Field(description="Amélioration du Gini (en pourcentage)")
    model_version: str = "glm_poisson_gamma_v1"