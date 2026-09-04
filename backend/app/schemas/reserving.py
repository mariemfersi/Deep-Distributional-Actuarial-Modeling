"""Schémas de validation pour l'API de provisionnement."""



from pydantic import BaseModel, Field





class ReservingIbnrRequest(BaseModel):

    """Requête pour prédiction IBNR d'une compagnie."""

    grcode: int = Field(..., description="Identifiant de la compagnie (GRCODE)")

    evaluation_year: int = Field(default=2007, ge=1990, le=2010, description="Année d'évaluation")





class TriangleData(BaseModel):
    values: list[list[float]]
    std_errors: list[list[float]]
    origin_years: list[int]
    development_years: list[int]
    cell_status: list[list[int]]  # 0 = observed, 1 = projected


class ReservingIbnrResponse(BaseModel):

    """Résultat de prédiction IBNR avec intervalles de confiance (Mack + Conformal)."""

    grcode: int

    ibnr_estimate: float

    mack_interval: dict = Field(description="Intervalle Mack (asymptotique). Couverture empirique mesurée via backtest sur le portefeuille.")

    conformal_interval: dict = Field(description="Intervalle Conforme (split-conformal, Vovk et al.). q_hat et couverture empirique mesurés et chargés depuis l'artefact de calibration.")

    triangle_data: TriangleData | None = Field(default=None, description="Triangle values and standard errors for heatmap visualization")

    ldfs: list[float] = Field(default=[], description="Facteurs de développement (LDF) selon formule du cours: f_j = Σ C_{i,j+1} / Σ C_{i,j}")

    cadences: list[float] = Field(default=[], description="Cadences cumulées selon formule du cours: pc_k = 1/(f_k × ... × f_n)")

    model_version: str = "mack_conformal_v1"





class ReservingRequest(BaseModel):

    """Séquence de paiements observés pour une police/année de survenance."""

    observed_increments: list[float] = Field(..., min_length=1, max_length=9,

        description="Incréments de paiement déjà observés (échelle brute, pas normalisée)")

    premium: float = Field(..., gt=0, description="Prime acquise nette de la compagnie")





class ReservingResponse(BaseModel):

    predicted_future_increments: list[float]

    predicted_ibnr: float

    model_used: str = "deep_triangle_gru"

    note: str = ""



