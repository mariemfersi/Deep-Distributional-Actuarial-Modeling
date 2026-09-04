# Chapitre 10 — Conclusion générale

## 10.1 Rappel des objectifs et de la démarche

Ce travail de mémoire avait pour ambition de concevoir et implémenter une plateforme actuarielle complète intégrant des méthodes classiques et des approches modernes de deep learning, tout en maintenant des standards de transparence et de reproductibilité applicables en production. Trois modules actuariels — tarification, provisionnement et détection de fraude — ont été traités selon une démarche méthodologique commune en cinq étapes :

1. **Baseline First** — Chaque module a commencé par une implémentation rigoureuse d'une méthode classique établie (GLM Poisson-Gamma, Mack Chain-Ladder, Isolation Forest), servant de point de comparaison factuel.
2. **ML Enhancement** — Sur cette base, une ou plusieurs approches avancées (CANN, Conformal Prediction, Random Forest supervisé) ont été entraînées et évaluées sur les mêmes données, avec les mêmes métriques.
3. **Empirical Validation** — Chaque gainML a été mesuré empiriquement sur un jeu de test indépendant (split 60/20/20, seed 123), sans recours à des hypothèses distributionnelles non vérifiées.
4. **Explainability** — Les prédictions des modèles retenus ont été analysées via SHAP (tarification, fraude) afin d'identifier les variables discriminantes et de garantir la transparence des décisions.
5. **Honest Reporting** — Les échecs méthodologiques ont été documentés avec la même rigueur que les succès (CANN générique non retenu, Deep Triangle instable, 4 tentatives GNN avortées), conformément à la philosophie de « science ouverte ».

Cette démarche a abouti à une plateforme « Actuarial AI Platform » conteneurisée (Docker Compose, 6 services), exposant une API REST FastAPI (14 endpoints) avec un frontend Next.js, une base PostgreSQL 16 (4 tables ORM), un tracking MLflow, un monitoring Prometheus/Grafana et un pipeline CI/CD GitHub Actions.

---

## 10.2 Principaux résultats et contributions

### Résultats par module

**Tarification** — Le GLM Poisson-Gamma constitue une baseline robuste (deviance/obs = 0.3179, fréquence prédite/observée = 0.984). Le CANN à interaction ciblée (VehPower/VehAge/VehGas/VehBrand) apporte un gain mesuré : Gini GLM 0.278 → CANN 0.290, soit **+4.2% en indice de Gini relatif**, sans toutefois que le CANN générique (sans sélection d'interactions a priori) parvienne à surpasser le GLM. La sévérité modélisée par NGBoost surpasse le GLM Gamma (log-likelihood −2.34 vs −2.51, CRPS 0.45 vs 0.52) avec une couverture empirique de 90.58% pour des intervalles à 90%, la copule gaussienne frequency-severity n'ayant pas été retenue (τ = 0.0784, dépendance négligeable).

**Provisionnement** — Le Mack Chain-Ladder produit des intervalles dont la couverture empirique (74.5%) est significativement inférieure au niveau nominal affiché (90%). La calibration conforme (Vovk et al., 2005), avec un quantile calibré q̂ = 4.89 (vs z = 1.645 pour Mack seul), corrige cette sous-couverture à **93.7%**, proche de la cible de 90% — le coût étant un élargissement significatif des intervalles. Le Deep Triangle GRU, bien que fonctionnel, s'est révélé instable (ratio prédit/réel oscillant entre 0.57 et 2.65 selon les runs) et a été relégué au statut de démonstration technique.

**Détection de fraude** — Un benchmark complet de 7 variantes (Random Forest, RF+SMOTE, XGBoost, XGBoost+SMOTE, SVM RBF, Isolation Forest, LOF) a été mené sur un préprocessing ajusté au train seul (anti-fuite). Les approches non supervisées sont inefficaces (Isolation Forest AUC-ROC = 0.527), la fraude étant corrélée à des combinaisons normales de variables plutôt qu'à des anomalies statistiques. Le **XGBoost + SMOTE** est retenu comme meilleur modèle (AUC-ROC 0.853 en test / 0.843 en CV), devançant le Random Forest (0.823) et le XGBoost (0.839). La sélection de features Boruta confirme 8 variables discriminantes (Fault, PolicyType, VehicleCategory, VehiclePrice, AddressChange_Claim, BasePolicy, Age, Deductible). Les quatre tentatives de construction de graphe et d'entraînement de GNN ont systématiquement échoué, l'homophilie mesurée (0.80–0.86) restant inférieure à la référence aléatoire (0.8875).

### Contributions méthodologiques

Deux contributions transversales méritent d'être soulignées :

**La conformal prediction appliquée au provisionnement actuariel** — L'intégration de la calibration conforme (Vovk et al., 2005) au Mack Chain-Ladder démontre qu'il est possible de garantir une couverture empirique fiable sans hypothèse distributionnelle, directement applicable à la conformité Solvency II et à l'ORSA. Cette combinaison n'est pas documentée dans la littérature actuarielle de manière systématique et constitue une contribution méthodologique du présent travail.

**L'explicabilité SHAP appliquée à un CANN** — L'analyse SHAP du résidu appris par le CANN (différence entre prédiction CANN et prédiction GLM) permet d'identifier précisément les variables sur lesquelles le réseau apprend des corrections non-linéaires (VehBrand, importance 0.197 ; VehAge, 0.172), validant la pertinence de la sélection d'interactions a priori.

### Infrastructure logicielle

La plateforme intègre un écosystème MLOps complet : PostgreSQL avec migrations Alembic (4 tables ORM, versionnage du schéma), MLflow pour le tracking des expériences, Prometheus et Grafana pour la collecte et la visualisation des métriques applicatives (latence, nombre de prédictions, versions de modèles), et GitHub Actions pour le CI/CD (tests automatisés sur PostgreSQL, build et push Docker vers GHCR). L'API expose 14 endpoints REST documentés via Swagger, avec journalisation automatique de chaque prédiction dans PostgreSQL et authentification JWT.

---

## 10.3 Apports personnels du stage : compétences techniques et métier

### Compétences techniques acquises

Ce stage a permis de développer des compétences techniques concrètes et directement applicables en environnement professionnel :

**Modélisation actuarielle avec deep learning** — Conception, entraînement et évaluation d'un CANN (Combined Actuarial Neural Network) avec architecture spécifique (skip connection du GLM, embedding VehBrand, interactions ciblées par paires), incluant la normalisation des features, l'optimisation GPU (2 heures d'entraînement) et l'encapsulation du modèle pour la prédiction. Maîtrise de NGBoost pour la modélisation distributionnelle de la sévérité (estimation de paramètres Gamma, intervalles de confiance natifs, CRPS comme métrique de calibration).

**Provisionnement et quantification de l'incertitude** — Implémentation du Mack Chain-Ladder stochastique avec calcul des erreurs standards, puis calibration conforme « distribution-free » pour garantir une couverture empirique cible. Compréhension des hypothèses asymptotiques de Mack et de leurs limites en échantillon fini (143 compagnies, triangles 10×10).

**Explicabilité ML** — Application de SHAP (SHapley Additive exPlanations) à un modèle de tarification actuarielle et à un modèle de fraude, interprétation des résultats en termes métier (variables discriminantes, interactions non-linéaires), et communication des résultats aux parties prenantes.

**Développement backend production** — Conception d'une API REST FastAPI avec architecture modulaire (routers/services/schemas), injection de dépendances pour les sessions DB, journalisation automatique des prédictions, authentification JWT, validation Pydantic des entrées/sorties, et documentation OpenAPI auto-générée.

**Infrastructure et MLOps** — Conteneurisation Docker multi-stages (backend Python, frontend Next.js), orchestration Docker Compose (6 services), tracking MLflow avec backend PostgreSQL, monitoring Prometheus/Grafana avec dashboards provisionnés, pipeline CI/CD GitHub Actions (tests avec service PostgreSQL, build et push Docker sur tags).

**Gestion de projet technique** — Versionnement Git avec convention de commits, structuration modulaire du code (src/pricing, src/reserving, src/fraud), rédaction de Model Cards (documentation éthique des modèles), et mise en place d'une suite de tests automatisés (43 tests, 2 sautés).

### Compétences métier acquises

**Conformité réglementaire** — Compréhension des exigences Solvency II et de l'ORSA en matière de Best Estimate Liability et de quantification de l'incertitude, application directe via la conformal prediction pour les provisions techniques.

**Analyse critique des modèles** — Capacité à évaluer empiriquement les gains d'un modèleML par rapport à une baseline classique, en tenant compte des contraintes opérationnelles (temps d'entraînement, reproductibilité, interprétabilité, coût de l'infrastructure).

**Communication scientifique** — Rédaction de rapports techniques structurés, documentation transparente des échecs méthodologiques (CANN générique, Deep Triangle instable, 4 tentatives GNN), et réflexion éthique sur l'utilisation de l'IA en assurance.

---

## 10.4 Perspectives

Les résultats ouvrent plusieurs pistes de développement à court et moyen terme.

### Développements à court terme

**Intégration du CANN dans l'API** — L'encapsulation des statistiques de normalisation (mean/std du jeu d'entraînement) dans le fichier de modèle pickle permettrait de déployer le CANN en production, sans dépendance externe. Cette modification est technique et ne nécessite pas de réentraînement.

**Validation out-of-time** — La mise en place d'un split temporel (entraînement sur les N premières années, test sur l'année N+1) permettrait d'évaluer la robustesse aux dérives de distribution et de fournir des métriques plus conservatrices, conformes aux standards de validation de Solvency II.

**Monitoring en production** — L'infrastructure Prometheus/Grafana est opérationnelle ; l'ajout de rules d'alerte (chute de l'AUC-ROC > 10%, dérive de la distribution d'entrée, augmentation de la latence) et de drift detection automatisé transformerait le projet de démonstration en système de monitoring opérationnel.

### Pistes de recherche à moyen terme

**GNN avec données relationnelles réelles** — Les quatre tentatives de construction de graphe ont échoué sur des données tabulaires sans structure relationnelle native. L'application de GNN à des données contenant des liens explicites (réseau de distribution, sinistres en cascade, fraude en réseau) pourrait révéler un gain que les données Kaggle ne permettent pas de capturer.

**Conformal prediction adaptative** — La calibration conforme actuelle est statique (q̂ = 4.89 calculé une fois). Une version adaptative, recalibrée périodiquement avec les données les plus récentes, permettrait de suivre les évolutions du portefeuille et de maintenir la couverture cible.

**Fine-tuning par compagnie** — Le provisionnement actuel utilise un modèle global (143 compagnies). Un fine-tuning par compagnie, en utilisant les données historiques spécifiques, pourrait améliorer la précision des estimations IBNR pour les portefeuilles atypiques.

**Expansion multibranche** — La modularité de la plateforme (src/pricing, src/reserving, src/fraud) permet d'ajouter de nouveaux modules actuariels (santé, habitation, vie) en réutilisant l'architecture backend/frontend existante, le pipeline MLOps et les bonnes pratiques documentées dans ce travail.

---

Ce travail a démontré que la combinaison de méthodes classiques éprouvées et d'approches modernes de deep learning, encadrée par une validation empirique rigoureuse et une explicabilité transparente, constitue une voie pertinente pour moderniser les pratiques actuarielles sans compromettre les garanties de reproductibilité et de conformité réglementaire exigées par l'industrie de l'assurance.
