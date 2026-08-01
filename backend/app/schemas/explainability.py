"""
Schémas pour l'API d'explicabilité (SHAP).
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ShapValue(BaseModel):
    """Valeur SHAP pour une feature individuelle."""
    feature: str = Field(..., description="Nom de la feature")
    value: float = Field(..., description="Valeur SHAP (impact sur la prédiction)")


class ShapExplanation(BaseModel):
    """Explication SHAP complète pour une prédiction."""
    base_value: float = Field(..., description="Valeur de base (prédiction moyenne)")
    shap_values: List[ShapValue] = Field(..., description="Liste des valeurs SHAP par feature")


class PricingExplanationRequest(BaseModel):
    """Requête pour expliquer une prédiction de tarification."""
    veh_power: int = Field(..., description="Puissance du véhicule")
    veh_age: int = Field(..., description="Âge du véhicule")
    driv_age: int = Field(..., description="Âge du conducteur")
    bonus_malus: int = Field(..., description="Bonus-malus")
    veh_brand: str = Field(..., description="Marque du véhicule")
    veh_gas: str = Field(..., description="Type de carburant (Diesel/Regular)")
    region: str = Field(..., description="Région")
    area: str = Field(..., description="Zone (rurale/urbaine)")
    density: float = Field(..., description="Densité de population")
    exposure: float = Field(default=1.0, description="Exposition (années)")


class FraudExplanationRequest(BaseModel):
    """Requête pour expliquer une prédiction de fraude."""
    fault: str = Field(..., description="Responsabilité (Third Party/Policy Holder)")
    policy_type: str = Field(..., description="Type de police")
    vehicle_category: str = Field(..., description="Catégorie de véhicule")
    base_policy: str = Field(..., description="Police de base")
    address_change_claim: str = Field(..., description="Changement d'adresse lors du sinistre")
    days_policy_claim: str = Field(..., description="Jours entre police et sinistre")
    driver_rating: int = Field(..., description="Note du conducteur")
    deductible: int = Field(..., description="Franchise")
    week_of_month: Optional[int] = Field(default=3, description="Semaine du mois")
    age: Optional[int] = Field(default=40, description="Âge")
