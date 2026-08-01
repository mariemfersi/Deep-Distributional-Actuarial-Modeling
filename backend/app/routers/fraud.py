from fastapi import APIRouter, HTTPException

from app.schemas.fraud import FraudRequest, FraudResponse
from app.services.fraud_service import predict_fraud

router = APIRouter(prefix="/fraud", tags=["fraud"])


@router.get("/methodology")
def get_methodology():
    """Retourne un résumé des tentatives de modélisation par graphe et pourquoi le GNN n'a pas été retenu."""
    return {
        "baseline": {
            "model": "Isolation Forest",
            "description": "Approche non-supervisée pour détection d'anomalies",
            "result": "AUC-ROC non applicable (non-supervisé)"
        },
        "final_model": {
            "model": "Random Forest",
            "description": "Approche supervisée avec 30 features catégoriels et numériques",
            "result": "AUC-ROC 0.815"
        },
        "graph_attempts": [
            {
                "attempt": 1,
                "approach": "Graphe basé sur RepNumber (numéro de réclamation)",
                "description": "Construction d'arêtes entre réclamations du même numéro",
                "result": "Insuffisant - trop peu de connexions, graphe très sparse"
            },
            {
                "attempt": 2,
                "approach": "Graphe basé sur similarité de features",
                "description": "Arêtes entre réclamations avec similarité cosinus > seuil",
                "result": "Calcul coûteux, résultats non concluants"
            },
            {
                "attempt": 3,
                "approach": "Graphe basé sur attributs rares partagés",
                "description": "Arêtes entre réclamations partageant des valeurs rares (ex. même adresse inhabituelle)",
                "result": "Meilleur mais encore limité par la taille du dataset"
            },
            {
                "attempt": 4,
                "approach": "GNN (Graph Neural Network) avec PyTorch Geometric",
                "description": "GCN/GAT pour propagation d'information sur le graphe",
                "result": "Non retenu - Random Forest supérieur en performance et interprétabilité"
            }
        ],
        "conclusion": "Le Random Forest supervisé offre le meilleur compromis performance/interprétabilité pour ce dataset. Les approches par graphe n'ont pas démontré d'amélioration significative et ajoutent de la complexité sans gain mesurable."
    }


@router.post("/predict", response_model=FraudResponse)
def predict(request: FraudRequest):
    try:
        return predict_fraud(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
