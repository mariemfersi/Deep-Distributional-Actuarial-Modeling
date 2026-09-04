"""
Module Pricing — Multivariate Copula Extension.

Modélisation de la dépendance entre plusieurs garanties sur le même contrat
(ex: responsabilité civile vs dommages propres) via des copules multivariées.
Extension naturelle du module existant de copule gaussienne fréquence-sévérité.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Guarantee:
    """Représente une garantie d'assurance avec ses caractéristiques."""
    name: str
    frequency_mean: float
    severity_mean: float
    exposure: float


@dataclass
class MultivariateCopulaResult:
    """Résultat d'une simulation de copule multivariée."""
    simulated_frequencies: np.ndarray  # Shape: (n_simulations, n_guarantees)
    simulated_severities: np.ndarray  # Shape: (n_simulations, n_guarantees)
    correlation_matrix: np.ndarray
    total_premium_distribution: np.ndarray
    marginal_premiums: Dict[str, np.ndarray]
    dependence_measures: Dict[str, float]


class GaussianCopula:
    """
    Copule gaussienne multivariée pour modéliser la dépendance entre garanties.
    
    Cette extension permet de capturer les effets de corrélation entre différents
    types de sinistres (ex: un accident grave génère souvent à la fois des dommages
    matériels et des blessures, affectant plusieurs garanties simultanément).
    """
    
    def __init__(self, correlation_matrix: np.ndarray):
        """
        Initialise la copule gaussienne avec une matrice de corrélation.
        
        Args:
            correlation_matrix: Matrice de corrélation (n x n, symétrique, définie positive)
        """
        self.correlation_matrix = correlation_matrix
        self.n_dimensions = correlation_matrix.shape[0]
        
        # Vérifier que la matrice est valide
        assert correlation_matrix.shape == (self.n_dimensions, self.n_dimensions)
        assert np.allclose(correlation_matrix, correlation_matrix.T)
        assert np.all(np.linalg.eigvals(correlation_matrix) > 0)
    
    def simulate(self, n_simulations: int) -> np.ndarray:
        """
        Simule des variables corrélées via la copule gaussienne.
        
        Args:
            n_simulations: Nombre de simulations Monte Carlo
        
        Returns:
            Variables uniformes corrélées (n_simulations x n_dimensions)
        """
        # Décomposition de Cholesky
        L = np.linalg.cholesky(self.correlation_matrix)
        
        # Générer des variables normales indépendantes
        Z = np.random.standard_normal((n_simulations, self.n_dimensions))
        
        # Appliquer la corrélation
        correlated_Z = Z @ L.T
        
        # Transformer en uniformes via la CDF normale
        U = stats.norm.cdf(correlated_Z)
        
        return U
    
    def fit_correlation(self, data: np.ndarray) -> np.ndarray:
        """
        Estime la matrice de corrélation à partir de données observées.
        
        Args:
            data: Données observées (n_observations x n_dimensions)
        
        Returns:
            Matrice de corrélation estimée
        """
        # Transformer les données en rangs (copula empirique)
        ranks = data.argsort(axis=0).argsort(axis=0) + 1
        uniform_ranks = ranks / (len(data) + 1)
        
        # Transformer en normales via la CDF inverse
        normal_scores = stats.norm.ppf(uniform_ranks)
        
        # Calculer la corrélation de Pearson sur les scores normaux
        correlation_matrix = np.corrcoef(normal_scores.T)
        
        return correlation_matrix


class MultivariatePricingModel:
    """
    Modèle de tarification multivarié avec copule pour dépendance entre garanties.
    
    Ce modèle étend l'approche existante fréquence-sévérité à plusieurs garanties
    simultanées, capturant les effets de dépendance qui sont importants pour
    la tarification des contrats multirisques.
    """
    
    def __init__(self, guarantees: List[Guarantee], correlation_matrix: np.ndarray):
        """
        Initialise le modèle multivarié.
        
        Args:
            guarantees: Liste des garanties à modéliser
            correlation_matrix: Matrice de corrélation entre garanties
        """
        self.guarantees = guarantees
        self.copula = GaussianCopula(correlation_matrix)
        self.n_guarantees = len(guarantees)
    
    def simulate_portfolio(self, n_simulations: int = 10000) -> MultivariateCopulaResult:
        """
        Simule un portefeuille avec dépendance entre garanties.
        
        Args:
            n_simulations: Nombre de simulations Monte Carlo
        
        Returns:
            Résultat de simulation avec distributions de primes
        """
        # Simuler les fréquences corrélées via copule
        U_freq = self.copula.simulate(n_simulations)
        
        # Transformer en fréquences de Poisson
        simulated_frequencies = np.zeros((n_simulations, self.n_guarantees))
        for i in range(self.n_guarantees):
            guarantee = self.guarantees[i]
            lambda_param = guarantee.frequency_mean * guarantee.exposure
            simulated_frequencies[:, i] = stats.poisson.ppf(U_freq[:, i], lambda_param)
        
        # Simuler les sévérités (indépendantes conditionnellement aux fréquences)
        simulated_severities = np.zeros((n_simulations, self.n_guarantees))
        for i in range(self.n_guarantees):
            guarantee = self.guarantees[i]
            # Sévérité Gamma (standard en assurance)
            shape = (guarantee.severity_mean ** 2) / 1000  # Paramètre de forme arbitraire
            scale = guarantee.severity_mean / shape
            simulated_severities[:, i] = stats.gamma.rvs(shape, scale=scale, size=n_simulations)
        
        # Calculer les primes par garantie
        marginal_premiums = {}
        for i, guarantee in enumerate(self.guarantees):
            premium = simulated_frequencies[:, i] * simulated_severities[:, i]
            marginal_premiums[guarantee.name] = premium
        
        # Prime totale (somme des garanties)
        total_premium = np.sum([marginal_premiums[g.name] for g in self.guarantees], axis=0)
        
        # Mesures de dépendance
        dependence_measures = self._compute_dependence_measures(simulated_frequencies, simulated_severities)
        
        return MultivariateCopulaResult(
            simulated_frequencies=simulated_frequencies,
            simulated_severities=simulated_severities,
            correlation_matrix=self.copula.correlation_matrix,
            total_premium_distribution=total_premium,
            marginal_premiums=marginal_premiums,
            dependence_measures=dependence_measures
        )
    
    def _compute_dependence_measures(self, frequencies: np.ndarray, severities: np.ndarray) -> Dict[str, float]:
        """Calcule des mesures de dépendance entre garanties."""
        measures = {}
        
        # Corrélation de Pearson entre fréquences
        freq_corr = np.corrcoef(frequencies.T)
        measures["mean_frequency_correlation"] = np.mean(freq_corr[np.triu_indices_from(freq_corr, k=1)])
        
        # Corrélation de Spearman (plus robuste)
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(frequencies)
        measures["mean_spearman_correlation"] = np.mean(spearman_corr[np.triu_indices_from(spearman_corr, k=1)])
        
        # Tail dependence (asymptotique)
        measures["tail_dependence_estimate"] = self._estimate_tail_dependence(frequencies)
        
        return measures
    
    def _estimate_tail_dependence(self, data: np.ndarray, threshold: float = 0.95) -> float:
        """Estime la dépendance de queue (tail dependence)."""
        n = len(data)
        threshold_idx = int(n * threshold)
        
        # Compter les événements extrêmes simultanés
        extreme_count = 0
        for i in range(self.n_guarantees):
            for j in range(i + 1, self.n_guarantees):
                # Vérifier si les deux variables sont dans la queue supérieure
                extreme_i = data[:, i] > np.percentile(data[:, i], threshold * 100)
                extreme_j = data[:, j] > np.percentile(data[:, j], threshold * 100)
                extreme_count += np.sum(extreme_i & extreme_j)
        
        # Normaliser
        total_pairs = self.n_guarantees * (self.n_guarantees - 1) / 2
        tail_dep = extreme_count / (n * total_pairs)
        
        return tail_dep
    
    def compare_with_independence(self, n_simulations: int = 10000) -> Dict:
        """
        Compare le modèle avec dépendance vs hypothèse d'indépendance.
        
        Cette comparaison montre l'impact de la dépendance sur la tarification.
        """
        # Simulation avec dépendance
        result_with_dep = self.simulate_portfolio(n_simulations)
        
        # Simulation avec indépendance (matrice de corrélation = identité)
        identity_corr = np.eye(self.n_guarantees)
        independent_model = MultivariatePricingModel(self.guarantees, identity_corr)
        result_independent = independent_model.simulate_portfolio(n_simulations)
        
        # Comparer les distributions de primes totales
        with_dep_mean = np.mean(result_with_dep.total_premium_distribution)
        independent_mean = np.mean(result_independent.total_premium_distribution)
        
        with_dep_std = np.std(result_with_dep.total_premium_distribution)
        independent_std = np.std(result_independent.total_premium_distribution)
        
        with_dep_var_95 = np.percentile(result_with_dep.total_premium_distribution, 95)
        independent_var_95 = np.percentile(result_independent.total_premium_distribution, 95)
        
        return {
            "mean_premium_with_dependence": with_dep_mean,
            "mean_premium_independent": independent_mean,
            "mean_premium_difference": with_dep_mean - independent_mean,
            "mean_premium_difference_pct": ((with_dep_mean - independent_mean) / independent_mean) * 100,
            "std_with_dependence": with_dep_std,
            "std_independent": independent_std,
            "std_difference": with_dep_std - independent_std,
            "var_95_with_dependence": with_dep_var_95,
            "var_95_independent": independent_var_95,
            "var_95_difference": with_dep_var_95 - independent_var_95,
            "conclusion": self._interpret_dependence_impact(with_dep_mean, independent_mean, with_dep_std, independent_std)
        }
    
    def _interpret_dependence_impact(self, dep_mean: float, ind_mean: float, 
                                    dep_std: float, ind_std: float) -> str:
        """Interprète l'impact de la dépendance sur la tarification."""
        mean_diff_pct = ((dep_mean - ind_mean) / ind_mean) * 100
        std_diff_pct = ((dep_std - ind_std) / ind_std) * 100
        
        if abs(mean_diff_pct) < 1 and abs(std_diff_pct) < 5:
            return "La dépendance a un impact minimal sur la tarification."
        elif abs(mean_diff_pct) < 5 and abs(std_diff_pct) < 15:
            return "La dépendance modère légèrement la variabilité des primes."
        elif abs(mean_diff_pct) < 10:
            return "La dépendance a un impact modéré sur la tarification et le risque."
        else:
            return "La dépendance a un impact significatif - la modélisation multivariée est recommandée."


def create_example_multivariate_contract() -> Tuple[List[Guarantee], np.ndarray]:
    """
    Crée un exemple de contrat multirisques automobile.
    
    Returns:
        Liste de garanties et matrice de corrélation
    """
    guarantees = [
        Guarantee(name="Responsabilité Civile", frequency_mean=0.10, severity_mean=5000, exposure=1.0),
        Guarantee(name="Dommages Véhicule", frequency_mean=0.05, severity_mean=3000, exposure=1.0),
        Guarantee(name="Blessures Corporelles", frequency_mean=0.02, severity_mean=15000, exposure=1.0),
        Guarantee(name="Vol", frequency_mean=0.01, severity_mean=8000, exposure=1.0)
    ]
    
    # Matrice de corrélation réaliste (basée sur le domaine)
    # RC et Dommages sont souvent corrélés (même accident)
    # Blessures sont plus rares mais corrélées avec accidents graves
    correlation_matrix = np.array([
        [1.00, 0.60, 0.40, 0.20],  # RC
        [0.60, 1.00, 0.50, 0.30],  # Dommages
        [0.40, 0.50, 1.00, 0.15],  # Blessures
        [0.20, 0.30, 0.15, 1.00]   # Vol
    ])
    
    return guarantees, correlation_matrix


def run_multivariate_analysis() -> Dict:
    """
    Exécute une analyse complète de tarification multivariée.
    
    Returns:
        Résultats complets de l'analyse
    """
    guarantees, correlation_matrix = create_example_multivariate_contract()
    
    model = MultivariatePricingModel(guarantees, correlation_matrix)
    
    # Simulation du portefeuille
    simulation_result = model.simulate_portfolio(n_simulations=10000)
    
    # Comparaison avec indépendance
    comparison = model.compare_with_independence(n_simulations=10000)
    
    return {
        "guarantees": [{"name": g.name, "frequency_mean": g.frequency_mean, 
                        "severity_mean": g.severity_mean} for g in guarantees],
        "correlation_matrix": correlation_matrix.tolist(),
        "simulation_summary": {
            "total_premium_mean": float(np.mean(simulation_result.total_premium_distribution)),
            "total_premium_std": float(np.std(simulation_result.total_premium_distribution)),
            "total_premium_median": float(np.median(simulation_result.total_premium_distribution)),
            "total_premium_95th": float(np.percentile(simulation_result.total_premium_distribution, 95)),
            "total_premium_99th": float(np.percentile(simulation_result.total_premium_distribution, 99))
        },
        "dependence_measures": simulation_result.dependence_measures,
        "independence_comparison": comparison
    }
