# Chapitre 9 — Synthèse des résultats et discussion générale

## 9.1 Bilan comparatif des trois modules : méthode classique vs méthode avancée

Le tableau récapitulatif ci-dessous synthétise les résultats obtenus dans chacun des trois modules étudiés, en confrontant la baseline actuarielle classique au modèle final retenu :

| Module | Baseline classique | Modèle final retenu | Métrique principale | Gain mesuré |
|--------|-------------------|---------------------|---------------------|-------------|
| **Tarification** | GLM Poisson-Gamma | CANN (interaction ciblée) | Gini / deviance/obs | +4.2% Gini relatif (0.278 → 0.290), -1.56% deviance |
| **Provisionnement** | Mack Chain-Ladder | Mack + Conformal Prediction | Couverture empirique 90% | 74.5% → 93.7% |
| **Détection de fraude** | Isolation Forest | XGBoost + SMOTE | AUC-ROC | 0.527 → 0.853 |

**Tarification** — Le GLM Poisson constitue une baseline solide (deviance/obs = 0.3179 sur le jeu de test, fréquence prédite/observée = 0.984). Le CANN à interaction ciblée (VehPower/VehAge/VehGas/VehBrand) apporte un gain mesuré mais modeste : +4.2% en indice de Gini relatif (0.278 pour le GLM → 0.290 pour le CANN, soit +1.2 point absolu) et -1.56% en deviance normalisée (0.3130 vs 0.3179). Il est crucial de noter que le CANN « générique » (sans interaction ciblée, tentatives 1 et 2) n'a pas réussi à surpasser le GLM — seule la selection a priori des paires d'interactions pertinentes a permis cet improvement. L'analyse SHAP révèle que le réseau apprend principalement à corriger le GLM sur VehBrand (importance 0.197) et VehAge (0.172), confirmant que ces deux variables portent les interactions non-linéaires résiduelles. Pour la sévérité, NGBoost (log-likelihood -2.34, CRPS 0.45) surpasse le GLM Gamma (respectivement -2.51 et 0.52) tout en fournissant nativement des intervalles de confiance (couverture 90% empirique : 90.58%). La copule gaussienne frequency-severity n'a pas été retenue en raison d'une corrélation de Spearman négligeable (τ = 0.0784).

**Provisionnement** — Le Mack Chain-Ladder seul produit une couverture empirique de 74.5% pour un niveau nominal de 90%, soit une sous-couverture significative de 15.5 points. Cette déviation s'explique par l'hypothèse de normalité asymptotique de Mack, qui ne tient pas compte de la finitude de l'échantillon (143 compagnies, triangles 10×10). L'ajout de la calibration conforme (Vovk et al., 2005) avec un quantile calibré q̂ = 4.89 (vs z = 1.645 pour Mack) corrige cette sous-couverture : la couverture empirique atteint 93.7%, proche de la cible de 90%. Le coût de cette garantie est un élargissement de ~15% des intervalles. Le Deep Triangle GRU, bien que techniquement fonctionnel, s'est révélé instable (ratio médian prédit/réel oscillant entre 0.57 et 2.65 selon les runs) et n'égale pas la précision ponctuelle de Mack (ratio 1.06). Il a donc été relégué au statut de démonstration technique.

**Détection de fraude** — L'Isolation Forest non-supervisé s'est avéré inefficace (AUC-ROC = 0.527, à peine au-dessus du hasard), car la fraude dans ce jeu de données est corrélée à des combinaisons normales de variables plutôt qu'à des anomalies statistiques. Un benchmark complet de 7 variantes (Random Forest, RF+SMOTE, XGBoost, XGB+SMOTE, SVM RBF, Isolation Forest, LOF) a été mené sur un préprocessing ajusté au train seul (anti-fuite). Le **XGBoost + SMOTE** est retenu comme meilleur modèle (AUC-ROC 0.853 en test / 0.843 en CV 5-fold), devançant le XGBoost (0.839) et le Random Forest (0.823). L'Isolation Forest non-supervisé s'est avéré inefficace (AUC-ROC = 0.527), car la fraude dans ce jeu de données est corrélée à des combinaisons normales de variables plutôt qu'à des anomalies statistiques. La sélection de features Boruta confirme 8 variables discriminantes (Fault, PolicyType, VehicleCategory, VehiclePrice, AddressChange_Claim, BasePolicy, Age, Deductible). Quatre tentatives de construction de graphe et d'entraînement de GNN ont été systématiquement évaluées — l'homophilie mesurée (0.80 à 0.86) restant inférieure à la référence aléatoire (0.8875), invalidant l'hypothèse selon laquelle la structure relationnelle porterait le signal de fraude.

**Figure 9.1** — Synthèse des gains relatifs par module. Pour chaque module, le gain est exprimé en termes normalisés par rapport à la baseline.

---

## 9.2 Quand privilégier l'approche actuarielle classique, quand le ML apporte un gain mesuré

L'une des leçons centrales de ce travail est que le machine learning n'est pas systématiquement supérieur aux méthodes classiques — le gain doit être mesuré empiriquement et mis en balance avec la complexité ajoutée.

### Cas où la méthode classique reste suffisante

Le GLM Poisson-Gamma constitue un excellent modèle de tarification : interprétable, stable, rapide à entraîner et à déployer. Son deviance/obs de 0.3179 sur 135 000 observations de test démontre une capacité prédictive robuste. Lorsque l'objectif est la tarification de masse avec contraintes réglementaires d'interprétabilité (tarification non-discriminante, explication aux assurés), le GLM peut constituer le modèle de production, le CANN étant réservé aux analyses de sous-populations spécifiques.

De même, le Mack Chain-Ladder reste la méthode de référence pour le provisionnement standardsa, sa précision ponctuelle (ratio médian prédit/réel = 1.06) étant supérieure à celle du Deep Triangle (0.57–2.65). La simplicité méthodologique de Mack — ne nécessitant aucun entraînement, aucune hyperparamètre — en fait un choix pragmatique pour le reporting réglementaire courant.

### Cas où le ML apporte un gain clair et mesuré

Le provisionnement par conformal prediction représente le cas d'école d'un gain ML indiscutables : la couverture empirique passe de 74.5% (sous-couverture critique) à 93.7% (garantie), sans hypothèse distributionnelle. Ce gain est essentiel pour la conformité Solvency II et l'ORSA, où les Intervalles de Confiance doivent être fiables. La calibration conforme est par ailleurs simple à implémenter (une seule ligne de code, un quantile q̂ = 4.89) et n'ajoute aucune complexité d'entraînement.

Le XGBoost + SMOTE pour la fraude illustre un autre cas : l'approche non-supervisée (Isolation Forest, AUC-ROC 0.527) échoue car le signal de fraude n'est pas une anomalie statistique mais une corrélation de variables normales. Seul le supervisé (XGBoost + SMOTE, AUC-ROC 0.853) exploite efficacement les labels disponibles.

### Cas intermédiaire : le CANN

Le CANN se situe dans une zone grise. Son gain de +4.2% en Gini relatif (0.278 → 0.290) est réel mais modeste, et il introduit des contraintes opérationnelles significatives : nécessité de normaliser les features (mean/std du jeu d'entraînement), entraînement GPU long (~2 heures), complexité architecturelle (skip connection, embedding VehBrand). Pour une startup d'assurance avec un portefeuille homogène et des ressources data science dédiées, ce gain peut justifier l'investissement. Pour une compagnie traditionnelle avec des processus IT standards, le GLM peut rester le choix pragmatique.

**Figure 9.2** — Guide de décision : quand utiliser l'approche classique vs le machine learning, en fonction de la complexité du signal et de la criticité de l'application.

---

## 9.3 Apport de la quantification de l'incertitude pour la conformité réglementaire

La quantification de l'incertitude constitue un thread transversal de ce travail, avec des implications concrètes pour la conformité réglementaire.

### Le problème de la sous-couverture de Mack

Le Mack Chain-Ladder, bien que mathétement fondé ( théorème central limite appliqué aux LDF), produit des intervalles dont la couverture empirique réelle (74.5%) est significativement inférieure au niveau nominal affiché (90%). Cette sous-couverture n'est pas une surprise pour les actuaires expérimentés : l'asymptoticité de Mack suppose un nombre infini de garanties et des incréments indépendants, hypothèses rarement vérifiées en pratique. L'ampleur de la déviation (15.5 points) est cependant alarmante pour un reporting réglementaire.

### La conformal prediction comme correctif

La calibration conforme corrige empiriquement cette sous-couverture sans modifier la méthode de point estimation. Le facteur correcteur q̂ = 4.89 (vs z = 1.645 pour Mack seul) élargit les intervalles d'un facteur ~3.0, ce qui garantit une couverture de 93.7% — légèrement supérieure au 90% cible. Cette sur-couverture de 3.7 points est le prix à payer pour l'absence d'hypothèse distributionnelle : la méthode est « distribution-free » et ne dépend que de la recalibration empirique.

### Implications pour Solvency II et l'ORSA

Dans le cadre de Solvency II, les compagnies d'assurance doivent estimer la Best Estimate Liability (BEL) avec des provisions techniques adéquates. La conformal prediction fournit une garantie empirique de couverture qui renforce la crédibilité de ces estimations auprès des autorités de contrôle. L'ORSA (Own Risk and Solvency Assessment) requiert par ailleurs une quantification des risques sur un horizon pluriannuel : les intervalles conformes, recalibrables dynamiquement, s'adaptent naturellement à cette exigence.

### NGBoost et l'approche distributionnelle

Le modèle NGBoost de sévérité illustre une approche complémentaire : au lieu de produire des intervalles post-hoc, il modélise directement la distribution conditionnelle de la sévérité. La couverture 90% empirique de 90.58% et le CRPS de 0.45 (vs 0.52 pour le GLM Gamma) démontrent que l'approche distributionnelle produit des intervalles calibrés tout en améliorant la précision ponctuelle. Cette dualité (intervalle post-hoc via conformal vs intervalle natif via NGBoost) offre des choix méthodologiques complémentaires selon le contexte d'application.

---

## 9.4 Limites globales du projet

Malgré les résultats encourageants, plusieurs limites doivent être explicitement reconnues :

### Limites liées aux données

**Unicité des jeux de données** — La tarification repose exclusivement sur le jeu freMTPL2 (assurance automobile française, 2004–2008, 678 013 polices). Les résultats ne sont pas directement transférables à d'autres branches (habitation, santé, vie) ni à d'autres juridictions. Le provisionnement utilise la base CAS Loss Reserving Database (143 compagnies, 1988–1997), dont la structure en triangles cumulés est spécifique au reporting IARD. La fraude repose sur un dataset Kaggle dont la representativité par rapport aux données réelles d'assurance n'est pas garantie.

**Taille des échantillons** — Le jeu de fraude ne contient que 15 420 observations avec 6.16% de fraudes (950 cas), ce qui limite la fiabilité des métriques de rappel et de précision. Le provisionnement utilise des triangles 10×10 pour 10 compagnies dans l'analyse détaillée, un échantillon insuffisant pour des conclusions statistiquement robustes.

**Absence de validation temporelle** — Les splits train/test sont aléatoires (seed 123), sans séparation temporelle. Une validation out-of-time (année N+1) serait nécessaire pour évaluer la robustesse aux dérives de distribution.

### Limites liées au déploiement

**CANN hors production** — Le CANN interaction, bien que surpassant le GLM en notebook (+4.2% Gini relatif, 0.278 → 0.290), n'est pas déployé dans l'API. L'API utilise le GLM pour garantir la stabilité opérationnelle, le CANN nécessitant des statistiques de normalisation du jeu d'entraînement. Cette situation illustre un compromis fréquent en production ML : le modèle le plus performant n'est pas toujours le plus déployable.

**GNN non retenu** — Les quatre tentatives de construction de graphe et d'entraînement de GNN ont échoué, principalement en raison de l'absence d'homophilie dans les structures relationnelles testées (homophilie ≤ référence aléatoire de 0.8875). Le GraphSAGE entraîné sur le meilleur graphe (tentative 3) obtient un AUC-ROC de 0.790, inférieur au XGBoost + SMOTE (0.853). Cet échec méthodologique est documenté transparemment, conformément à la philosophie « Honest Reporting » du projet.

**Pas de monitoring temps réel** — Bien que l'infrastructure MLflow et Prometheus/Grafana soit configurée, le monitoring en production (drift detection, alertes de dégradation) n'est pas opérationnel.

### Limites méthodologiques

**Métriques de performance** — Les métriques de classification (precision, recall, F1) pour la fraude ne sont calculées qu'à un seuil unique (0.5) et ne sont pas rapportées par sous-groupes. La courbe ROC seule ne suffit pas à évaluer la performance en contexte de classes déséquilibrées.

**Reproductibilité des notebooks** — Les notebooks contiennent des résultats de run spécifiques (seed 123) qui peuvent diverger légèrement entre exécutions, en particulier pour le Deep Triangle GRU dont l'instabilité est documentée.

**Absence de comparaison multi-jeux** — L'absence de cross-validation k-fold sur le pricing limite la capacité à généraliser les résultats. Le 5-fold standard, mentionné dans les Model Cards, n'a pas été systématiquement appliqué.

---

## 9.5 Perspectives d'amélioration

Les résultats ouvrent plusieurs pistes de recherche et de développement :

### Améliorations techniques à court terme

**Intégration du CANN dans l'API** — L'encapsulation des statistiques de normalisation (mean/std du jeu d'entraînement) dans le fichier de modèle pickle permettrait de déployer le CANN en production sans dépendance externe. Cette modification est technique et ne nécessite pas de réentraînement.

**Validation out-of-time** — La mise en place d'un split temporel (entraînement sur les N premières années, test sur l'année N+1) permettrait d'évaluer la robustesse aux dérives de distribution et de fournir des métriques plus conservatrices.

**Fine-tuning par compagnie** — Le provisionnement actuel utilise un modèle global (143 compagnies). Un fine-tuning par compagnie, en utilisant les données historiques spécifiques, pourrait améliorer la précision des estimations IBNR pour les portefeuilles atypiques.

### Pistes de recherche à moyen terme

**GNN avec données relationnelles réelles** — Les quatre tentatives de construction de graphe ont échoué sur des données tabulaires sans structure relationnelle native. L'application de GNN à des données contenant des liens explicites (réseau de distribution, sinistres en cascade, fraude en réseau) pourrait révéler un gain que les données Kaggle ne permettent pas de capturer.

**Copule multivariée** — Le module `src/pricing/multivariate_copula.py` implémente déjà une copule pour 4 garanties (RC, Dommages, Blessures, Vol) avec une matrice de corrélation paramétrée. Son application à un portefeuille réel multi-garanties pourrait améliorer la tarification des bundles d'assurances.

**Conformal prediction adaptative** — La calibration conforme actuelle est statique (q̂ = 4.89 calculé une fois). Une version adaptative, recalibrée périodiquement avec les données les plus récentes, permettrait de suivre les évolutions du portefeuille et de maintenir la couverture cible.

### Perspectives industrielles

**Monitoring et drift detection** — L'infrastructure Prometheus/Grafana est en place ; l'ajout de rules d'alerte (chute de l'AUC-ROC > 10%, dérive de la distribution d'entrée, augmentation de la latence) transformerait le projet de démonstration en système de monitoring opérationnel.

**Pipeline MLOps complet** — Le CI/CD GitHub Actions est configuré pour les tests et le build Docker ; l'ajout d'un pipeline d'entraînement automatisé (triggered par l'arrivée de nouvelles données) et de promote-to-production compléterait le cycle MLOps.

**Expansion à d'autres branches** — La modularité de la plateforme (src/pricing, src/reserving, src/fraud) permet d'ajouter de nouveaux modules actuariels (santé, habitation, vie) en réutilisant l'architecture backend/frontend existante.

---

## Résumé du chapitre

Ce travail de synthèse permet de dégager trois enseignements majeurs :

1. **Le ML n'est pas une solution universelle** — Le gain doit être mesuré empiriquement. Le GLM reste un excellent modèle de tarification, le Mack Chain-Ladder reste précieux pour le provisionnement ponctuel, et XGBoost + SMOTE surpasse le deep learning (GNN) sur données tabulaires.

2. **La quantification de l'incertitude est un enjeu critique** — La sous-couverture de Mack (74.5% vs 90% cible) aurait pu passer inaperçue sans vérification empirique. La calibration conforme corrige ce problème de manière élégante et distribution-free, avec des implications directes pour Solvency II.

3. **La transparence des échecs est une force** — Les quatre tentatives GNN documentées, l'échec du CANN générique, et l'instabilité du Deep Triangle ne sont pas des faiblesses mais des preuves de rigueur méthodologique. La philosophie « Honest Reporting » garantit que chaque conclusion est étayée par des résultats empiriques, pas par des hypothèses non vérifiées.
