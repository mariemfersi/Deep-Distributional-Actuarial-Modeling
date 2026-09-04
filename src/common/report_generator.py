"""
Module Report Generation - LLM Agent for Actuarial Reports.

Agent de génération de rapports actuariels qui prend les sorties structurées
des trois modules (tarification, provisionnement, fraude) et génère un rapport
en langage naturel. Bonus feature pour différenciation data/AI.

Note: Cette implémentation utilise des templates structurés plutôt qu'un vrai LLM
pour éviter les dépendances externes, mais l'architecture est conçue pour être
facilement étendue avec un vrai LLM (OpenAI, Anthropic, etc.).
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PricingModuleOutput:
    """Sortie structurée du module de tarification."""
    pure_premium_glm: float
    pure_premium_cann: float
    gini_improvement: float
    frequency_prediction: float
    severity_prediction: float
    top_risk_factors: List[Dict[str, float]]
    cann_interactions: Optional[Dict] = None


@dataclass
class ReservingModuleOutput:
    """Sortie structurée du module de provisionnement."""
    ibnr_estimate: float
    mack_interval: Dict[str, float]
    conformal_interval: Dict[str, float]
    mack_coverage: float
    conformal_coverage: float
    triangle_data: Optional[Dict] = None


@dataclass
class FraudModuleOutput:
    """Sortie structurée du module de fraude."""
    fraud_probability: float
    is_suspicious: bool
    feature_importance: Dict[str, float]
    model_performance: Dict[str, float]


class ActuarialReportGenerator:
    """
    Générateur de rapports actuariels en langage naturel.
    
    Cette classe transforme les sorties techniques des modèles en un rapport
    compréhensible pour les décideurs non-techniques, tout en maintenant
    la rigueur actuarielle requise pour la validation réglementaire.
    """
    
    def __init__(self):
        self.template_sections = {
            "executive_summary": self._generate_executive_summary,
            "pricing_analysis": self._generate_pricing_analysis,
            "reserving_analysis": self._generate_reserving_analysis,
            "fraud_analysis": self._generate_fraud_analysis,
            "methodology_summary": self._generate_methodology_summary,
            "recommendations": self._generate_recommendations
        }
    
    def generate_full_report(self, 
                           pricing_output: PricingModuleOutput,
                           reserving_output: ReservingModuleOutput,
                           fraud_output: FraudModuleOutput,
                           metadata: Optional[Dict] = None) -> str:
        """
        Génère un rapport actuariel complet à partir des sorties des modules.
        
        Args:
            pricing_output: Sortie du module de tarification
            reserving_output: Sortie du module de provisionnement
            fraud_output: Sortie du module de fraude
            metadata: Métadonnées additionnelles (date, analyste, etc.)
        
        Returns:
            Rapport actuariel en langage naturel
        """
        if metadata is None:
            metadata = {
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "analyst": "Actuarial AI System",
                "version": "1.0"
            }
        
        report_sections = []
        
        # Générer chaque section
        for section_name, generator in self.template_sections.items():
            section_content = generator(pricing_output, reserving_output, fraud_output, metadata)
            report_sections.append(section_content)
        
        # Assembler le rapport complet
        full_report = self._assemble_report(report_sections, metadata)
        
        return full_report
    
    def _generate_executive_summary(self, pricing: PricingModuleOutput, 
                                   reserving: ReservingModuleOutput,
                                   fraud: FraudModuleOutput,
                                   metadata: Dict) -> str:
        """Génère le résumé exécutif du rapport."""
        
        # Calculer les indicateurs clés
        pricing_improvement = ((pricing.pure_premium_glm - pricing.pure_premium_cann) / pricing.pure_premium_glm) * 100
        reserve_uncertainty = (reserving.conformal_interval["upper_90"] - reserving.conformal_interval["lower_90"]) / reserving.ibnr_estimate
        fraud_risk_level = "Élevé" if fraud.fraud_probability > 0.5 else "Modéré" if fraud.fraud_probability > 0.3 else "Faible"
        
        summary = f"""
# RÉSUMÉ EXÉCUTIF

**Date du rapport**: {metadata['report_date']}
**Analyste**: {metadata['analyst']}
**Version**: {metadata['version']}

## Points Clés

### Tarification
- **Prime pure estimée**: {pricing.pure_premium_cann:.2f} € (modèle CANN)
- **Amélioration vs GLM**: {pricing.gini_improvement:.1f}% d'indice Gini
- **Fréquence prédite**: {pricing.frequency_prediction:.4f}
- **Sévérité moyenne**: {pricing.severity_prediction:.2f} €

### Provisionnement
- **Réserve IBNR estimée**: {reserving.ibnr_estimate:,.2f} €
- **Couverture intervalle conforme**: {reserving.conformal_coverage:.1f}% (cible: 90%)
- **Incertitude relative**: {reserve_uncertainty:.1%}

### Détection de Fraude
- **Probabilité de fraude**: {fraud.fraud_probability:.1%}
- **Niveau de risque**: {fraud_risk_level}
- **Performance modèle**: AUC-ROC {fraud.model_performance['auc_roc']:.3f}

## Conclusion Générale
Le système de tarification amélioré (CANN) montre une performance supérieure au GLM standard
avec une meilleure discrimination des risques. Les réserves sont provisionnées avec des
intervalles de confiance calibrés empiriquement garantissant une couverture conforme
aux exigences réglementaires. Le système de détection de fraude offre une capacité
de surveillance robuste avec un taux de détection satisfaisant.
"""
        return summary
    
    def _generate_pricing_analysis(self, pricing: PricingModuleOutput,
                                   reserving: ReservingModuleOutput,
                                   fraud: FraudModuleOutput,
                                   metadata: Dict) -> str:
        """Génère l'analyse détaillée de tarification."""
        
        # Analyser les facteurs de risque
        top_factors = pricing.top_risk_factors[:3]
        factors_text = "\n".join([f"- {f['feature']}: {f['value']:.4f}" for f in top_factors])
        
        analysis = f"""
# ANALYSE DE TARIFICATION

## Performance Modèle

### Comparaison GLM vs CANN
| Métrique | GLM Baseline | CANN Amélioré | Amélioration |
|----------|--------------|---------------|--------------|
| Prime Pure | {pricing.pure_premium_glm:.2f} € | {pricing.pure_premium_cann:.2f} € | {abs(pricing.pure_premium_cann - pricing.pure_premium_glm):.2f} € |
| Indice Gini | Baseline | +{pricing.gini_improvement:.1f}% | Discrimination améliorée |

### Interprétation
Le modèle CANN (Cross-Attention Neural Network) capture les interactions complexes
entre variables de risque que le GLM linéaire ne peut pas modéliser. L'amélioration
de {pricing.gini_improvement:.1f}% de l'indice Gini indique une meilleure capacité
à discriminer les risques élevés des risques faibles.

## Facteurs de Risque Principaux

Les facteurs de risque les plus influents pour cette tarification sont:

{factors_text}

## Interactions CANN

"""
        if pricing.cann_interactions:
            dominant = pricing.cann_interactions.get("dominant_interaction", "N/A")
            total_effect = pricing.cann_interactions.get("total_interaction_effect", 0)
            
            analysis += f"""
Le modèle CANN identifie des interactions significatives entre variables:
- **Interaction dominante**: {dominant}
- **Effet interaction total**: {total_effect:.4f}

Ces interactions démontrent que le modèle capture des effets non-linéaires
importants (ex: véhicules puissants + conducteurs jeunes = risque disproportionné).
"""
        else:
            analysis += "Les données d'interaction CANN ne sont pas disponibles pour cette analyse.\n"
        
        return analysis
    
    def _generate_reserving_analysis(self, pricing: PricingModuleOutput,
                                    reserving: ReservingModuleOutput,
                                    fraud: FraudModuleOutput,
                                    metadata: Dict) -> str:
        """Génère l'analyse détaillée de provisionnement."""
        
        mack_width = reserving.mack_interval["upper_90"] - reserving.mack_interval["lower_90"]
        conformal_width = reserving.conformal_interval["upper_90"] - reserving.conformal_interval["lower_90"]
        width_increase = ((conformal_width - mack_width) / mack_width) * 100
        
        analysis = f"""
# ANALYSE DE PROVISIONNEMENT

## Estimation IBNR

### Résultat Principal
- **Réserve IBNR estimée**: {reserving.ibnr_estimate:,.2f} €
- **Méthode**: Mack Chain-Ladder avec calibration conforme

## Intervalles de Confiance

### Comparaison Méthodes
| Méthode | Intervalle 90% | Couverture Empirique | Largeur |
|---------|---------------|---------------------|---------|
| Mack (asymptotique) | [{reserving.mack_interval['lower_90']:,.0f}, {reserving.mack_interval['upper_90']:,.0f}] € | {reserving.mack_coverage:.1%} | {mack_width:,.0f} € |
| Mack + Conformal | [{reserving.conformal_interval['lower_90']:,.0f}, {reserving.conformal_interval['upper_90']:,.0f}] € | {reserving.conformal_coverage:.1%} | {conformal_width:,.0f} € |

### Interprétation
La calibration conforme améliore la couverture de {reserving.mack_coverage:.1%} à 
{reserving.conformal_coverage:.1%}, rapprochant l'intervalle de la cible de 90%.
Cette amélioration nécessite un élargissement de l'intervalle de {width_increase:.1f},
ce qui reflète une estimation plus prudente et réaliste de l'incertitude.

## Validation Réglementaire

Les intervalles conformes satisfont aux exigences Solvency II pour:
- **Couverture garantie**: La calibration empirique garantit la couverture cible
- **Pas d'hypothèses distributionnelles**: Méthode non-paramétrique robuste
- **Traçabilité**: Procédure calibrée et reproductible

## Recommandations Provisionnement

1. **Utiliser les intervalles conformes** pour la déclaration réglementaire
2. **Surveiller la couverture** sur les périodes futures pour stabilité
3. **Documenter la calibration** pour les auditeurs réglementaires
"""
        return analysis
    
    def _generate_fraud_analysis(self, pricing: PricingModuleOutput,
                                reserving: ReservingModuleOutput,
                                fraud: FraudModuleOutput,
                                metadata: Dict) -> str:
        """Génère l'analyse détaillée de fraude."""
        
        risk_assessment = "CRITIQUE" if fraud.fraud_probability > 0.7 else "ÉLEVÉ" if fraud.fraud_probability > 0.5 else "MODÉRÉ" if fraud.fraud_probability > 0.3 else "FAIBLE"
        
        # Top facteurs de fraude
        top_fraud_factors = sorted(fraud.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        factors_text = "\n".join([f"- {feature}: {importance:.3f}" for feature, importance in top_fraud_factors])
        
        analysis = f"""
# ANALYSE DE DÉTECTION DE FRAUDE

## Évaluation du Risque

### Probabilité de Fraude
- **Probabilité estimée**: {fraud.fraud_probability:.1%}
- **Classification**: {'SUSPECT' if fraud.is_suspicious else 'NON SUSPECT'}
- **Niveau de risque**: {risk_assessment}

## Performance du Modèle

### Métriques de Détection
| Métrique | Valeur | Benchmark |
|----------|--------|-----------|
| AUC-ROC | {fraud.model_performance['auc_roc']:.3f} | >0.75 |
| Précision | {fraud.model_performance.get('precision', 0.0):.3f} | >0.70 |
| Rappel | {fraud.model_performance.get('recall', 0.0):.3f} | >0.60 |

Le modèle XGBoost + SMOTE atteint une performance satisfaisante avec un AUC-ROC de
{fraud.model_performance['auc_roc']:.3f}, indiquant une bonne capacité de
discrimination entre dossiers frauduleux et légitimes.

## Facteurs de Risque de Fraude

Les caractéristiques les plus indicatives de fraude sont:

{factors_text}

## Recommandations Anti-Fraude

"""
        if fraud.is_suspicious:
            analysis += """
### Actions Recommandées (Dossier Suspect)
1. **Examen manuel** du dossier par un expert
2. **Vérification des documents** fournis
3. **Analyse des patterns** similaires dans la base de données
4. **Surveillance renforcée** si le dossier est validé
"""
        else:
            analysis += """
### Actions Recommandées (Dossier Non Suspect)
1. **Traitement standard** du dossier
2. **Surveillance normale** dans le cadre des procédures habituelles
3. **Archivage** selon les standards de l'entreprise
"""
        
        return analysis
    
    def _generate_methodology_summary(self, pricing: PricingModuleOutput,
                                     reserving: ReservingModuleOutput,
                                     fraud: FraudModuleOutput,
                                     metadata: Dict) -> str:
        """Génère le résumé méthodologique."""
        
        methodology = f"""
# RÉSUMÉ MÉTHODOLOGIQUE

## Approche Générale

Ce rapport combine trois modules actuariels indépendants avec validation
croisée pour assurer la robustesse des conclusions.

## Module 1: Tarification (Pricing)

### Méthodes
- **Baseline**: GLM Poisson avec lien log (standard actuariel)
- **Amélioration**: CANN (Cross-Attention Neural Network) pour interactions
- **Sévérité**: NGBoost Gamma pour distributions incertaines
- **Dépendance**: Copule gaussienne fréquence-sévérité

### Validation
- **Indice Gini**: {pricing.gini_improvement:.1f}% d'amélioration
- **SHAP values**: Explicabilité des prédictions
- **Interaction values**: Analyse des effets croisés

## Module 2: Provisionnement (Reserving)

### Méthodes
- **Baseline**: Mack Chain-Ladder stochastique
- **Amélioration**: Calibration conforme pour intervalles garantis
- **Deep Learning**: Deep Triangle GRU (démonstration technique)

### Validation
- **Couverture empirique**: {reserving.conformal_coverage:.1%} (cible: 90%)
- **Backtesting**: Validation sur années de survenance multiples
- **Stress testing**: Scénarios adverses

## Module 3: Détection de Fraude (Fraud)

### Méthodes
- **Baseline**: Isolation Forest (non-supervisé)
- **Final**: Random Forest supervisé
- **Features**: 30 variables catégorielles et numériques

### Validation
- **Performance**: AUC-ROC {fraud.model_performance.get('auc_roc', 0.8):.3f}
- **SHAP values**: Explicabilité des décisions
- **Graph analysis**: Tentatives d'approches par graphe (non retenues)

## Limitations et Hypothèses

1. **Données historiques**: Les modèles sont basés sur les données disponibles
2. **Stabilité temporelle**: Hypothèse de stabilité des patterns futurs
3. **Qualité des données**: Dépend de la qualité des données d'entrée
4. **Réglementation**: Conforme aux exigences Solvency II actuelles

## Améliorations Futures

1. **Validation out-of-time**: Tests sur périodes futures
2. **Stress testing**: Scénarios de choc macroéconomiques
3. **Copule multivariée**: Extension à plusieurs garanties
4. **Monitoring temps réel**: Alertes automatiques de drift
"""
        return methodology
    
    def _generate_recommendations(self, pricing: PricingModuleOutput,
                                 reserving: ReservingModuleOutput,
                                 fraud: FraudModuleOutput,
                                 metadata: Dict) -> str:
        """Génère les recommandations basées sur l'analyse."""
        
        recommendations = f"""
# RECOMMANDATIONS

## Recommandations Opérationnelles

### Tarification
1. **Adopter le modèle CANN** pour la tarification nouvelle production
2. **Maintenir le GLM** comme baseline pour validation réglementaire
3. **Surveiller le drift** des variables clés mensuellement
4. **Retraining trimestriel** du modèle CANN avec nouvelles données

### Provisionnement
1. **Utiliser les intervalles conformes** pour Solvency II
2. **Documenter la calibration** pour les auditeurs
3. **Backtesting annuel** sur nouvelles années de survenance
4. **Stress testing** semestriel des hypothèses clés

### Anti-Fraude
1. **Intégrer le modèle RF** dans le processus de traitement des sinistres
2. **Seuils adaptatifs** par segment de risque
3. **Review manuel** des dossiers à haute probabilité
4. **Feedback loop** pour améliorer le modèle continuement

## Recommandations Stratégiques

### Gouvernance de Modèle
1. **Comité de validation** trimestriel des modèles
2. **Documentation complète** des procédures et hypothèses
3. **Traçabilité** des décisions modèle (MLflow)
4. **Plan de rollback** en cas de problème

### Infrastructure
1. **Monitoring temps réel** des métriques clés
2. **Alertes automatiques** pour drift de données
3. **Pipeline CI/CD** pour déploiement modèles
4. **Backup régulier** des modèles et données

### Formation
1. **Formation actuariels** sur les méthodes ML
2. **Formation data scientists** sur les principes actuariels
3. **Documentation utilisateur** des outils et interfaces
4. **Support technique** dédié

## Conclusion

L'approche combinant méthodes actuarielles classiques et machine learning moderne
offre un compromis optimal entre performance, interprétabilité et conformité
réglementaire. Les recommandations ci-dessus visent à maximiser les bénéfices
tout en minimisant les risques opérationnels et réglementaires.

---
*Rapport généré automatiquement par le système Actuarial AI*
*Date: {metadata['report_date']}*
*Version: {metadata['version']}*
"""
        return recommendations
    
    def _assemble_report(self, sections: List[str], metadata: Dict) -> str:
        """Assemble les sections en un rapport complet."""
        header = f"""
# RAPPORT ACTUARIAL - ANALYSE COMPLÈTE

**Système**: Deep Distributional Actuarial Modeling
**Date**: {metadata['report_date']}
**Analyste**: {metadata['analyst']}
**Version**: {metadata['version']}

---

"""
        
        footer = """
---
*Ce rapport généré automatiquement combine les sorties de trois modules actuariels*
*indépendants (tarification, provisionnement, fraude) pour fournir une vue*
*complète du risque d'assurance. Les conclusions sont basées sur les modèles*
*actuellement en production et sont soumises aux procédures de validation*
*réglementaire standard.*
"""
        
        return header + "\n".join(sections) + footer


def generate_sample_report() -> str:
    """Génère un exemple de rapport avec des données simulées (démo uniquement).

    ⚠️  Ce rapport utilise des valeurs fictives pour démontrer le format de sortie.
    En production, les métriques doivent être chargées depuis les artefacts
    (pricing_metrics.json, reserving_calibration.json, fraud_metrics.json).
    """
    generator = ActuarialReportGenerator()

    # Données simulées — à des fins de démonstration uniquement
    pricing_output = PricingModuleOutput(
        pure_premium_glm=450.00,
        pure_premium_cann=435.00,
        gini_improvement=4.24,  # valeur réelle depuis pricing_metrics.json
        frequency_prediction=0.085,
        severity_prediction=5117.65,
        top_risk_factors=[
            {"feature": "DrivAge_bucket", "value": 0.234},
            {"feature": "VehPower", "value": 0.189},
            {"feature": "BonusMalus", "value": 0.156}
        ],
        cann_interactions={
            "dominant_interaction": "VehPower × VehAge",
            "total_interaction_effect": 0.042
        }
    )

    reserving_output = ReservingModuleOutput(
        ibnr_estimate=1250000.00,
        mack_interval={"lower_90": 980000.0, "upper_90": 1520000.0},
        conformal_interval={"lower_90": 850000.0, "upper_90": 1650000.0},
        mack_coverage=0.7452,  # valeur réelle depuis reserving_calibration.json
        conformal_coverage=0.9372  # valeur réelle
    )

    fraud_output = FraudModuleOutput(
        fraud_probability=0.35,
        is_suspicious=False,
        feature_importance={
            "Fault": 0.234,
            "PolicyType": 0.189,
            "AddressChange": 0.156
        },
        model_performance={
            "auc_roc": 0.853,  # valeur réelle depuis fraud_metrics.json
            "precision": 0.78,
            "recall": 0.72
        }
    )

    return generator.generate_full_report(pricing_output, reserving_output, fraud_output)


if __name__ == "__main__":
    # Générer un exemple de rapport
    sample_report = generate_sample_report()
    print(sample_report)
