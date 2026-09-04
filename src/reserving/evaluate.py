"""
Module Reserving — Évaluation des modèles de provisionnement.

Backtesting des intervalles de confiance sur plusieurs années de survenance
pour valider la robustesse des méthodes Mack et Conformal.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def backtest_coverage_multiple_years(observed: pd.DataFrame, future: pd.DataFrame, 
                                       alpha: float = 0.10, calib_frac: float = 0.5, 
                                       seed: int = 123) -> Dict:
    """
    Backtest la couverture des intervalles Mack et Conformal sur plusieurs compagnies.
    
    Cette fonction valide que la couverture conforme n'est pas due au hasard sur un
    seul triangle, mais robuste sur plusieurs années de survenance et compagnies.
    
    Args:
        observed: Données observées (triangle supérieur)
        future: Données futures (vérité terrain pour validation)
        alpha: Niveau de signification (0.10 pour intervalles 90%)
        calib_frac: Fraction des données pour calibration conformale
        seed: Seed pour reproductibilité
    
    Returns:
        Dictionnaire avec métriques de backtesting par compagnie et agrégées
    """
    from src.reserving.models import evaluate_mack_coverage, split_conformal_calibration
    
    # Obtenir toutes les compagnies disponibles
    companies = observed["GRCODE"].unique()
    
    results_by_company = {}
    all_mack_results = []
    all_conformal_results = []
    
    for grcode in companies:
        # Évaluer Mack pour cette compagnie
        mack_result = evaluate_mack_coverage(observed, future, grcode)
        
        if mack_result is None:
            continue  # Sauter les compagnies où le modèle échoue
        
        all_mack_results.append(mack_result)
        results_by_company[grcode] = {"mack": mack_result}
    
    # Combiner tous les résultats Mack pour calibration conformale
    if all_mack_results:
        full_mack_df = pd.concat(all_mack_results, ignore_index=True)
        
        # Calibration conformale sur l'ensemble
        conformal_test_df, q_hat = split_conformal_calibration(full_mack_df, alpha, calib_frac, seed)
        
        # Calculer les métriques de couverture
        mack_coverage = full_mack_df["covered_90"].mean()
        conformal_coverage = conformal_test_df["covered_conformal"].mean()
        
        # Distribution de la couverture par compagnie
        company_coverages = {}
        for grcode, data in results_by_company.items():
            mack_cov = data["mack"]["covered_90"].mean() if len(data["mack"]) > 0 else np.nan
            company_coverages[grcode] = {
                "mack_coverage": float(mack_cov),
                "n_years": len(data["mack"])
            }
        
        # Largeur moyenne des intervalles
        mack_avg_width = (full_mack_df["upper_90"] - full_mack_df["lower_90"]).mean()
        conformal_avg_width = (conformal_test_df["upper_conformal"] - conformal_test_df["lower_conformal"]).mean()
        
        return {
            "overall_mack_coverage": float(mack_coverage),
            "overall_conformal_coverage": float(conformal_coverage),
            "mack_avg_interval_width": float(mack_avg_width),
            "conformal_avg_interval_width": float(conformal_avg_width),
            "conformal_quantile": float(q_hat),
            "n_companies_tested": len(results_by_company),
            "n_total_observations": len(full_mack_df),
            "company_level_coverages": company_coverages,
            "target_coverage": 1 - alpha,
            "mack_improvement": float(conformal_coverage - mack_coverage)
        }
    else:
        return {
            "error": "No valid Mack results obtained for any company",
            "n_companies_tested": 0
        }


def evaluate_stability_by_accident_year(observed: pd.DataFrame, future: pd.DataFrame, 
                                         grcode: int) -> Dict:
    """
    Évalue la stabilité des intervalles par année de survenance individuelle.
    
    Permet d'identifier si certaines années de survenance sont plus difficiles
    à prédire que d'autres (ex: années avec catastrophe naturelle).
    
    Args:
        observed: Données observées
        future: Données futures
        grcode: Identifiant de la compagnie
    
    Returns:
        Métriques de couverture par année de survenance
    """
    from src.reserving.models import evaluate_mack_coverage
    
    mack_result = evaluate_mack_coverage(observed, future, grcode)
    
    if mack_result is None:
        return {"error": f"Mack model failed for GRCODE {grcode}"}
    
    # Analyser par année de survenance
    yearly_results = {}
    for year in mack_result.index:
        year_data = mack_result.loc[[year]]
        yearly_coverage = year_data["covered_90"].iloc[0] if len(year_data) > 0 else np.nan
        yearly_ibnr = year_data["ibnr_mack"].iloc[0] if len(year_data) > 0 else np.nan
        yearly_true = year_data["ibnr_reel"].iloc[0] if len(year_data) > 0 else np.nan
        
        yearly_results[int(year)] = {
            "mack_coverage": float(yearly_coverage),
            "ibnr_predicted": float(yearly_ibnr),
            "ibnr_actual": float(yearly_true),
            "prediction_error": float(abs(yearly_ibnr - yearly_true)) if not np.isnan(yearly_ibnr) and not np.isnan(yearly_true) else np.nan
        }
    
    return {
        "grcode": grcode,
        "yearly_results": yearly_results,
        "overall_coverage": mack_result["covered_90"].mean(),
        "n_years": len(yearly_results)
    }


def compute_interval_statistics(intervals: pd.DataFrame) -> Dict:
    """
    Calcule des statistiques descriptives sur les intervalles de confiance.
    
    Utile pour comprendre la distribution de l'incertitude et identifier
    les outliers ou les patterns inhabituels.
    
    Args:
        intervals: DataFrame avec colonnes lower_90, upper_90, ibnr_mack
    
    Returns:
        Statistiques descriptives des intervalles
    """
    intervals = intervals.copy()
    intervals["interval_width"] = intervals["upper_90"] - intervals["lower_90"]
    intervals["relative_width"] = intervals["interval_width"] / intervals["ibnr_mack"]
    
    return {
        "mean_width": float(intervals["interval_width"].mean()),
        "median_width": float(intervals["interval_width"].median()),
        "std_width": float(intervals["interval_width"].std()),
        "mean_relative_width": float(intervals["relative_width"].mean()),
        "median_relative_width": float(intervals["relative_width"].median()),
        "min_width": float(intervals["interval_width"].min()),
        "max_width": float(intervals["interval_width"].max()),
        "n_observations": len(intervals)
    }