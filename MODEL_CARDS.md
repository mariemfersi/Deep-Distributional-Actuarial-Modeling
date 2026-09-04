# Model Cards - Cartes de Modèle

## 📋 Version 1.0 - 26 Août 2026

---

## 🚗 Model Card: Pricing CANN Interaction

### Description
Combined Actuarial Neural Network (CANN) avec interactions ciblées pour la tarification automobile. Capture les interactions complexes entre VehPower, VehAge, VehGas et VehBrand tout en préservant l'interprétabilité via skip connection du GLM.

### Performance
- **Gini Improvement** : +4.2% vs GLM baseline (0.290 vs 0.278)
- **Deviance Reduction** : -1.56% vs GLM
- **Training Time** : ~2 heures sur GPU
- **Inference Time** : < 50ms per prediction

### Architecture
```
Input Features → CANN Layers → Skip Connection → GLM Output → Final Prediction
```

### Limitations
- Performances optimales sur profils similaires aux données d'entraînement
- Nécessite des données de calibration pour les nouvelles gammes de véhicules
- Skip connection dépend de la qualité du GLM baseline

### Intended Use
- Tarification automobile pour polices individuelles
- Segmentation des risques
- Analyse des interactions véhicules-conducteurs

### Training Data
- **Source** : Données MTPL France
- **Période** : 2004-2008
- **Volume** : ~678,000 polices
- **Features** : 10 variables catégorielles et numériques

---

## 🔺 Model Card: Reserving Mack + Conformal

### Description
Combinaison du Chain-Ladder stochastique de Mack avec Conformal Prediction pour garantir la couverture des intervalles de confiance sans hypothèses distributionnelles.

### Performance
- **Empirical Coverage** : 93.7% (vs 74.5% Mack seul)
- **Calibration q̂** : 4.89 (vs z = 1.645 pour Mack seul)
- **Calibration** : Distribution-free
- **Backtesting** : Validé sur 10 années d'accident

### Architecture
```
Mack Chain-Ladder → Conformal Calibration → Guaranteed Coverage Intervals
```

### Limitations
- Nécessite suffisamment de données historiques pour calibration
- Intervalles plus larges = moins de précision ponctuelle
- Performance dépend de la qualité des données triangle

### Intended Use
- Provisionnement IBNR pour assurances IARD
- Reporting réglementaire Solvency II
- Gestion du capital et réassurance

### Training Data
- **Source** : Données Company-Level
- **Période** : 1988-1997
- **Volume** : 10 compagnies, triangles complets
- **Features** : Incréments de sinistres par année d'accident/développement

---

## 🔍 Model Card: Fraud Detection — Benchmark (RF / XGBoost / SVM / SMOTE / LOF)

### Description
Benchmark comparatif de 7 approches pour la détection de fraude en assurance automobile, aligné sur la littérature récente (revue de littérature sur la fraude en assurance par IA) : Random Forest, XGBoost et SVM (supervisés) avec/sans SMOTE pour le déséquilibre des classes, plus les détections non supervisées Isolation Forest et Local Outlier Factor. La sélection de features est documentée via Boruta. Le modèle le plus performant est servi par l'API (`fraud_best_model.pkl`).

### Performance (jeu de test, 80/20, seed=123, préprocessing ajusté sur le train seul)
| Métrique | XGB+SMOTE (best) | Random Forest | RF+SMOTE | XGBoost | SVM | LOF | Isolation Forest |
|----------|-----------------|---------------|----------|---------|-----|-----|-----------------|
| AUC-ROC  | **0.853**       | 0.823         | 0.799    | 0.839   | 0.806 | 0.562 | 0.527          |
| PR-AUC   | **0.256**       | 0.205         | 0.156    | 0.230   | 0.169 | 0.070 | 0.064          |
| Precision | 0.476          | 0.122         | 0.151    | 0.210   | 0.000 | NaN  | NaN             |
| Recall   | 0.058           | 0.931         | 0.422    | 0.399   | 0.000 | NaN  | NaN             |
| F1       | 0.103           | 0.216         | 0.222    | 0.275   | 0.000 | NaN  | NaN             |

> Valeurs mesurées par `backend/scripts/evaluate_fraud.py` (préprocessing ajusté sur le train seul, anti-fuite). Les chiffres exacts sont persistés dans `models/fraud_metrics.json`.

- **Meilleur modèle** : XGBoost + SMOTE (AUC-ROC 0.853 en test / 0.843 en CV 5-fold)
- **5-fold CV** : XGB+SMOTE = 0.843, RF+SMOTE = 0.796, RF = 0.815

### Architecture
```
30 Features → { RF · XGBoost · SVM } [± SMOTE] → Fraud Probability → Binary Decision
             Detection non supervisée: Isolation Forest · LOF
```

### Gestion du déséquilibre
- Classes fortement déséquilibrées (~6% de fraude, ratio 1:16)
- **SMOTE** (imblearn) appliqué uniquement sur les données d'entraînement (via imblearn Pipeline, sans fuite en cross-validation)
- Alternatives : `class_weight="balanced"` (RF/SVM), `scale_pos_weight` (XGBoost)

### Sélection de features (Boruta)
- Algorithme **BorutaPy** comparant chaque variable à une ombre aléatoire
- Features confirmées/rejetées documentées dans le notebook comme preuve méthodologique
- Le modèle de production conserve les 30 features (pas de réduction) pour préserver la compatibilité API

### Limitations
- Performance dépend de la qualité des données d'entraînement
- Peut nécessiter réentraînement avec nouveaux patterns de fraude
- Les features "defaults" peuvent masquer des signaux subtils
- SVM n'expose pas d'importance de features native (fallback vide dans l'API)

### Intended Use
- Détection préliminaire de fraude
- Prioritisation des dossiers suspects
- Support aux enquêtes manuelles

### Training Data
- **Source** : fraud_oracle.csv (sinistres assurance auto)
- **Taille** : 15,419 sinistres (12,336 train / 3,084 test, split 80/20 seed=123)
- **Taux de fraude** : ~6%
- **Features** : 30 variables (24 catégorielles encodées + 6 numériques normalisées)

---

## 📊 Model Card: NGBoost Severity

### Description
Natural Gradient Boosting pour la modélisation distributionnelle de la sévérité des sinistres avec intervalles de confiance natifs via NGBoost.

### Performance
- **Log-Likelihood** : -2.34 (vs -2.51 GLM Gamma)
- **CRPS** : 0.45 (vs 0.52 GLM Gamma)
- **Calibration (ECE)** : 0.032
- **Coverage 90%** : 90.58%

### Architecture
```
Severity Features → NGBoost (Gamma Distribution) → Distribution Parameters → Percentiles
```

### Limitations
- Plus lent que GLM pour l'inférence
- Nécessite plus de données pour stabiliser les distributions
- Hypothèse de distribution Gamma (flexible mais fixe)

### Intended Use
- Estimation de la sévérité des sinistres
- Calcul de primes avec intervalles de confiance
- Analyse de la distribution des coûts

### Training Data
- **Source** : Données MTPL France (ClaimAmount)
- **Période** : 2004-2008
- **Volume** : ~26,000 sinistres
- **Features** : Mêmes features que pricing (pour cohérence)

---

## 🎯 Comparaison des Modèles

| Aspect | CANN Pricing | Mack+Conformal | XGB+SMOTE Fraud | NGBoost Severity |
|--------|--------------|----------------|----------------|-----------------|
| **Type** | Deep Learning | Statistical | Gradient Boosting | Gradient Boosting |
| **Complexité** | Élevée | Modérée | Modérée | Modérée |
| **Interprétabilité** | Moyenne (SHAP) | Élevée | Moyenne (SHAP) | Moyenne |
| **Inference Time** | 50ms | 10ms | 5ms | 20ms |
| **Training Time** | 2h | 1min | 30min | 45min |
| **Data Requirements** | Élevées | Modérées | Modérées | Élevées |
| **Uncertainty** | Via copule | Conformal | Probability | Distributional |
| **Production Ready** | ✅ | ✅ | ✅ | ✅ |

---

## 📝 Métriques Communes

### Pour Tous les Modèles
- **Disponibilité** : > 99.5%
- **Temps de réponse** : < 200ms (API)
- **Version** : 1.0.0
- **Framework** : MLflow tracking activé
- **Monitoring** : Drift detection configuré

### Calibration et Validation
- **Out-of-time validation** : Année N+1
- **Cross-validation** : 5-fold standard
- **Backtesting** : Périodes historiques
- **Stress testing** : Scénarios extrêmes

---

## 🔒 Considérations Éthiques

### Bias et Fairness
- **Testing** : Équité par groupes d'âge, région, type véhicule
- **Monitoring** : Détection de bias en production
- **Mitigation** : Reweighting si nécessaire

### Transparence
- **Documentation** : Model cards complètes
- **Explicabilité** : SHAP values disponibles
- **Audit Trail** : Décisions traçables

### Responsabilité
- **Review** : Validation par actuaires qualifiés
- **Sign-off** : Approbation formelle avant déploiement
- **Rollback** : Procédures de retour en arrière définies

---

---

## ⚖️ Benchmark Fraude — Variantes comparées

| Variante | Type | Déséquilibre | Notes |
|----------|------|--------------|-------|
| Random Forest | Supervisé | `class_weight=balanced` | Référence (AUC-ROC 0.823) |
| RF + SMOTE | Supervisé | SMOTE | Rééquilibrage train |
| XGBoost | Supervisé | `scale_pos_weight` | Souvent supérieur en fraude (littérature) |
| XGB + SMOTE | Supervisé | SMOTE | **Meilleur modèle (AUC-ROC 0.853 en test)** |
| SVM (RBF) | Supervisé | `class_weight=balanced` | Pas d'importance de features |
| Isolation Forest | Non supervisé | `contamination=0.06` | Détection d'anomalies |
| Local Outlier Factor | Non supervisé | `contamination=0.06` | Détection d'anomalies |

*Les résultats chiffrés du benchmark (CV AUC-ROC et jeu de test) sont générés de manière reproductible par `backend/scripts/evaluate_fraud.py` (préprocessing ajusté sur le train seul) et persistés dans `models/fraud_metrics.json`.*

*Model cards maintenues à jour avec chaque nouvelle version de modèle. Dernière mise à jour : 4 Septembre 2026*