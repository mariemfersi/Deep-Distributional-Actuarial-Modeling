"""
Module Pricing — Stress Testing et Sensitivity Analysis.

Tests de robustesse des modèles de tarification sous des scénarios de choc
plausibles (ex: inflation des coûts, changement de comportement des assurés).
Standard dans les rapports de validation actuarielle Solvency II.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class StressScenario:
    """Scénario de stress test pour les variables de tarification."""
    name: str
    description: str
    variable_shocks: Dict[str, float]  # variable -> percentage change


def define_standard_stress_scenarios() -> List[StressScenario]:
    """
    Définit des scénarios de stress standard pour l'assurance automobile.
    
    Ces scénarios sont basés sur des événements plausibles qui pourraient
    affecter la sinistralité et donc les primes.
    """
    scenarios = [
        StressScenario(
            name="inflation_couts",
            description="Inflation des coûts de réparation (+20%)",
            variable_shocks={"veh_power": 0.0, "veh_age": 0.0, "density": 0.20}
        ),
        StressScenario(
            name="vieillissement_parc",
            description="Vieillissement du parc automobile (+30% d'âge moyen)",
            variable_shocks={"veh_power": 0.0, "veh_age": 0.30, "density": 0.0}
        ),
        StressScenario(
            name="urbanisation",
            description="Urbanisation accrue (+25% densité)",
            variable_shocks={"veh_power": 0.0, "veh_age": 0.0, "density": 0.25}
        ),
        StressScenario(
            name="vehicules_puissants",
            description="Augmentation de la puissance moyenne (+15%)",
            variable_shocks={"veh_power": 0.15, "veh_age": 0.0, "density": 0.0}
        ),
        StressScenario(
            name="combined_stress",
            description="Stress combiné: inflation + urbanisation",
            variable_shocks={"veh_power": 0.0, "veh_age": 0.0, "density": 0.20}
        ),
        StressScenario(
            name="extreme_stress",
            description="Stress extrême: tous les facteurs défavorables",
            variable_shocks={"veh_power": 0.20, "veh_age": 0.20, "density": 0.30}
        )
    ]
    return scenarios


def apply_stress_scenario(base_profile: Dict, scenario: StressScenario) -> Dict:
    """
    Applique un scénario de stress à un profil de risque.
    
    Args:
        base_profile: Profil de risque original (dict de features)
        scenario: Scénario de stress à appliquer
    
    Returns:
        Profil de risque avec variables stressées
    """
    stressed_profile = base_profile.copy()
    
    for variable, shock_magnitude in scenario.variable_shocks.items():
        if variable in stressed_profile:
            original_value = stressed_profile[variable]
            
            # Appliquer le choc
            if variable == "veh_power":
                stressed_profile[variable] = min(20, original_value * (1 + shock_magnitude))
            elif variable == "veh_age":
                stressed_profile[variable] = min(30, original_value * (1 + shock_magnitude))
            elif variable == "density":
                stressed_profile[variable] = min(20000, original_value * (1 + shock_magnitude))
    
    return stressed_profile


def sensitivity_analysis(model, base_profile: Dict, feature_ranges: Dict[str, Tuple[float, float]], 
                          n_points: int = 20) -> Dict[str, np.ndarray]:
    """
    Analyse de sensibilité : impact de chaque variable sur la prédiction.
    
    Args:
        model: Modèle de tarification (GLM ou CANN)
        base_profile: Profil de risque de référence
        feature_ranges: Plages de variation pour chaque feature
        n_points: Nombre de points pour la courbe de sensibilité
    
    Returns:
        Dictionnaire avec courbes de sensibilité par feature
    """
    sensitivity_curves = {}
    
    for feature, (min_val, max_val) in feature_ranges.items():
        if feature not in base_profile:
            continue
        
        # Générer des valeurs pour la courbe
        values = np.linspace(min_val, max_val, n_points)
        predictions = []
        
        for val in values:
            test_profile = base_profile.copy()
            test_profile[feature] = val
            
            # Prédire avec le modèle stressé
            pred = predict_with_profile(model, test_profile)
            predictions.append(pred)
        
        sensitivity_curves[feature] = {
            "values": values,
            "predictions": np.array(predictions),
            "base_value": base_profile[feature],
            "base_prediction": predict_with_profile(model, base_profile),
            "sensitivity_range": np.max(predictions) - np.min(predictions),
            "relative_change": (np.max(predictions) - np.min(predictions)) / predict_with_profile(model, base_profile)
        }
    
    return sensitivity_curves


def predict_with_profile(model, profile: Dict) -> float:
    """
    Prédit la prime pour un profil donné.
    
    Cette fonction est un wrapper qui s'adapte au type de modèle
    (GLM statsmodels, CANN PyTorch, etc.)
    """
    # Adapter selon le type de modèle
    if hasattr(model, 'predict'):
        # Pour les modèles scikit-learn/statsmodels
        X = pd.DataFrame([profile])
        return float(model.predict(X)[0])
    elif hasattr(model, '__call__'):
        # Pour les modèles PyTorch/CANN
        return float(model(profile))
    else:
        raise ValueError("Type de modèle non supporté")


def run_stress_test_suite(model, base_profiles: List[Dict], 
                          scenarios: List[StressScenario] = None) -> Dict:
    """
    Exécute une suite complète de stress tests sur plusieurs profils.
    
    Args:
        model: Modèle de tarification
        base_profiles: Liste de profils de risque à tester
        scenarios: Scénarios de stress (si None, utilise les scénarios standard)
    
    Returns:
        Résultats des stress tests agrégés
    """
    if scenarios is None:
        scenarios = define_standard_stress_scenarios()
    
    results = {
        "scenarios": [],
        "profile_results": {},
        "aggregate_impacts": {}
    }
    
    # Résultats par profil
    for i, base_profile in enumerate(base_profiles):
        profile_name = f"profile_{i}"
        base_prediction = predict_with_profile(model, base_profile)
        
        profile_results = {
            "base_prediction": base_prediction,
            "scenario_results": {}
        }
        
        for scenario in scenarios:
            stressed_profile = apply_stress_scenario(base_profile, scenario)
            stressed_prediction = predict_with_profile(model, stressed_profile)
            
            impact_pct = ((stressed_prediction - base_prediction) / base_prediction) * 100
            
            profile_results["scenario_results"][scenario.name] = {
                "stressed_prediction": stressed_prediction,
                "impact_percentage": impact_pct,
                "impact_absolute": stressed_prediction - base_prediction
            }
        
        results["profile_results"][profile_name] = profile_results
    
    # Agréger les impacts par scénario
    for scenario in scenarios:
        scenario_impacts = []
        for profile_name, profile_data in results["profile_results"].items():
            scenario_impacts.append(profile_data["scenario_results"][scenario.name]["impact_percentage"])
        
        results["aggregate_impacts"][scenario.name] = {
            "mean_impact_pct": np.mean(scenario_impacts),
            "std_impact_pct": np.std(scenario_impacts),
            "min_impact_pct": np.min(scenario_impacts),
            "max_impact_pct": np.max(scenario_impacts),
            "n_profiles_tested": len(scenario_impacts)
        }
    
    results["scenarios"] = [{"name": s.name, "description": s.description} for s in scenarios]
    
    return results


def compute_robustness_metrics(stress_results: Dict) -> Dict:
    """
    Calcule des métriques de robustesse du modèle aux stress tests.
    
    Ces métriques aident à évaluer si le modèle se comporte de manière
    raisonnable sous des scénarios adverses.
    
    Args:
        stress_results: Résultats de run_stress_test_suite
    
    Returns:
        Métriques de robustesse
    """
    scenario_impacts = stress_results["aggregate_impacts"]
    
    # Identifier les scénarios les plus impactants
    max_impacts = {name: data["max_impact_pct"] for name, data in scenario_impacts.items()}
    worst_scenario = max(max_impacts.items(), key=lambda x: x[1])
    
    # Calculer la volatilité des réponses aux stress
    all_impacts = []
    for data in scenario_impacts.values():
        all_impacts.extend([data["mean_impact_pct"]])
    
    robustness_metrics = {
        "worst_case_scenario": worst_scenario[0],
        "worst_case_impact_pct": worst_scenario[1],
        "mean_scenario_impact_pct": np.mean(all_impacts),
        "std_scenario_impact_pct": np.std(all_impacts),
        "coefficient_of_variation": np.std(all_impacts) / np.mean(all_impacts) if np.mean(all_impacts) != 0 else np.inf,
        "n_scenarios_tested": len(scenario_impacts),
        "extreme_stress_threshold": 50.0  # Seuil d'alerte: +50% de variation
    }
    
    # Évaluer si le modèle passe les tests de robustesse
    robustness_metrics["passes_robustness_test"] = (
        robustness_metrics["worst_case_impact_pct"] < robustness_metrics["extreme_stress_threshold"]
    )
    
    return robustness_metrics


def generate_stress_test_report(stress_results: Dict, robustness_metrics: Dict) -> str:
    """
    Génère un rapport lisible des résultats de stress testing.
    
    Args:
        stress_results: Résultats de run_stress_test_suite
        robustness_metrics: Métriques de robustesse
    
    Returns:
        Rapport en format texte
    """
    report_lines = [
        "=" * 80,
        "RAPPORT DE STRESS TESTING - MODÈLE DE TARIFICATION",
        "=" * 80,
        "",
        "RÉSUMÉ EXÉCUTIF",
        "-" * 80,
        f"Scénarios testés: {robustness_metrics['n_scenarios_tested']}",
        f"Pire scénario: {robustness_metrics['worst_case_scenario']} ({robustress_metrics['worst_case_impact_pct']:.1f}%)",
        f"Impact moyen: {robustness_metrics['mean_scenario_impact_pct']:.1f}%",
        f"Test de robustesse: {'✅ PASSÉ' if robustness_metrics['passes_robustness_test'] else '❌ ÉCHOUÉ'}",
        "",
        "DÉTAILS PAR SCÉNARIO",
        "-" * 80
    ]
    
    for scenario_info in stress_results["scenarios"]:
        name = scenario_info["name"]
        desc = scenario_info["description"]
        impact = stress_results["aggregate_impacts"][name]
        
        report_lines.extend([
            f"\nScénario: {name}",
            f"Description: {desc}",
            f"  Impact moyen: {impact['mean_impact_pct']:.1f}%",
            f"  Impact min: {impact['min_impact_pct']:.1f}%",
            f"  Impact max: {impact['max_impact_pct']:.1f}%",
            f"  Écart-type: {impact['std_impact_pct']:.1f}%"
        ])
    
    report_lines.extend([
        "",
        "=" * 80,
        "INTERPRÉTATION",
        "-" * 80,
        "Un modèle robuste devrait montrer des variations raisonnables sous",
        "les scénarios de stress. Des variations extrêmes (>50%) peuvent indiquer",
        "une instabilité du modèle ou une sensibilité excessive à certains facteurs.",
        "=" * 80
    ])
    
    return "\n".join(report_lines)
