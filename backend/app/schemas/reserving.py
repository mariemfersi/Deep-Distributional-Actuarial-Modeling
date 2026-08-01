"""Schémas de validation pour l'API de provisionnement."""



from pydantic import BaseModel, Field





class ReservingIbnrRequest(BaseModel):

    """Requête pour prédiction IBNR d'une compagnie."""

    grcode: int = Field(..., description="Identifiant de la compagnie (GRCODE)")

    evaluation_year: int = Field(default=2007, ge=1990, le=2010, description="Année d'évaluation")





class ReservingIbnrResponse(BaseModel):

    """Résultat de prédiction IBNR avec intervalles de confiance (Mack + Conformal)."""

    grcode: int

    ibnr_estimate: float

    mack_interval: dict = Field(description="Intervalle Mack (asymptotique, couverture 74.4%)")

    conformal_interval: dict = Field(description="Intervalle Conforme (calibré, couverture 91.9%)")

    model_version: str = "mack_conformal_v1"





class ReservingRequest(BaseModel):

    """Séquence de paiements observés pour une police/année de survenance."""

    observed_increments: list[float] = Field(..., min_length=1, max_length=9,

        description="Incréments de paiement déjà observés (échelle brute, pas normalisée)")

    premium: float = Field(..., gt=0, description="Prime acquise nette de la compagnie")





class ReservingResponse(BaseModel):

    predicted_future_increments: list[float]

    predicted_ibnr: float

    model_version: str = "deep_triangle_soft_clip_v1"



