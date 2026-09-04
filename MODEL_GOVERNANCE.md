# Gouvernance de Modèle - Annexe Technique

## 📋 Version 1.0 - 26 Août 2026

### Aperçu
Ce document décrit les pratiques de gouvernance de modèle, les procédures de monitoring et les cadences de maintenance pour la plateforme actuarielle IA.

---

## 🏗️ Infrastructure de Versioning

### MLflow Integration
```bash
# Configuration MLflow
mlflow server --host 0.0.0.0 --port 5000
```

#### Tracking des Expériences
- **URI de tracking** : `http://localhost:5000`
- **Experiments** :
  - `pricing_glm_baseline` - GLM Poisson Frequency
  - `pricing_cann_interaction` - CANN Interaction Model
  - `pricing_ngboost_severity` - NGBoost Gamma Severity
  - `reserving_mack_chainladder` - Mack Chain-Ladder
  - `reserving_conformal_prediction` - Conformal Prediction
  - `fraud_random_forest` - Random Forest Fraud Detection

#### Paramètres Suivis
**Pricing CANN** :
- `learning_rate` : 0.001
- `hidden_dim` : 20
- `embedding_dim` : 2
- `epochs` : 100
- `batch_size` : 32

**Reserving Mack** :
- `alpha` : 0.05 (niveau de confiance)
- `bootstrap_samples` : 1000

**Fraud Random Forest** :
- `n_estimators` : 100
- `max_depth` : 10
- `min_samples_split` : 2

#### Métriques Enregistrées
- **Pricing** : Deviance, Gini, Log-Likelihood
- **Reserving** : Couverture empirique, Largeur intervalle, MSE
- **Fraud** : AUC-ROC, Precision, Recall, F1-Score

---

## 🔄 Cadences de Retraining

### Schedule de Maintenance

| Module | Fréquence | Déclencheur | Procédure |
|--------|-----------|-------------|-----------|
| **Pricing GLM** | Annuel | Nouvelles données souscription | Re-entraînement complet |
| **Pricing CANN** | Trimestriel | Drift > 5% ou nouveau trimestre | Transfer learning |
| **NGBoost Severity** | Semestriel | Nouvelles données sinistres | Re-entraînement |
| **Reserving Mack** | Mensuel | Clôture comptable | Mise à jour triangle |
| **Conformal Prediction** | Mensuel | Nouvelles données IBNR | Recalibration |
| **Fraud Random Forest** | Trimestriel | Nouveaux cas de fraude | Re-entraînement |

### Procédure de Retraining Automatisée

```python
# Exemple de pipeline MLflow
import mlflow
import mlflow.sklearn

def train_pricing_cann():
    with mlflow.start_run(run_name="cann_retrain_q3_2026"):
        # Log parameters
        mlflow.log_params({
            "learning_rate": 0.001,
            "hidden_dim": 20,
            "epochs": 100
        })
        
        # Train model
        model = train_cann_model()
        
        # Log metrics
        mlflow.log_metrics({
            "deviance": calculate_deviance(model),
            "gini": calculate_gini(model)
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "cann_model")
        
        # Register model
        mlflow.register_model(
            "runs:/<run_id>/cann_model",
            "pricing_cann_production"
        )
```

---

## 📊 Monitoring en Production

### Métriques de Monitoring

#### 1. **Drift de Données**
- **Population Stability Index (PSI)** : < 0.1 (stable), 0.1-0.25 (modéré), > 0.25 (drift)
- **Kolmogorov-Smirnov Test** : p-value < 0.05 indique drift significatif
- **Monitoring des features** : Distribution mensuelle des variables clés

#### 2. **Performance Modèle**
- **Pricing** : Gini mensuel, déviance sur nouvelles données
- **Reserving** : Couverture empirique récente, largeur intervalle
- **Fraud** : AUC-ROC sur nouveaux cas, taux de faux positifs

#### 3. **Qualité des Prédictions**
- **Calibration** : Expected Calibration Error (ECE)
- **Stabilité** : Variance des prédictions sur time windows
- **Anomalies** : Détection d'outliers dans les prédictions

### Dashboard de Monitoring

```yaml
# configuration monitoring
monitoring:
  pricing:
    metrics:
      - gini_coefficient
      - model_deviance
      - prediction_variance
    alerts:
      - gini_drop: threshold: 0.05
      - drift_detected: threshold: 0.25 PSI
      
  reserving:
    metrics:
      - empirical_coverage
      - interval_width
      - ibnr_accuracy
    alerts:
      - coverage_drop: threshold: 0.85
      - interval_inflation: threshold: 2.0
      
  fraud:
    metrics:
      - auc_roc
      - false_positive_rate
      - detection_rate
    alerts:
      - performance_drop: threshold: 0.1
      - fraud_spike: threshold: 2.0 std
```

---

## 🚨 Procédures d'Alerte

### Niveaux de Sévérité

#### **INFO** - Monitoring routine
- Rapport mensuel de performance
- Check de drift de données
- Validation des métriques de calibration

#### **WARNING** - Performance dégradée
- Gini drop > 5%
- Couverture empirique < 85%
- AUC-ROC drop > 10%
- Action : Investigation manuelle, possible retraining

#### **CRITICAL** - Échec système
- Erreur de prédiction > 10%
- Drift de données PSI > 0.5
- Modèle non disponible
- Action : Rollback automatique, alerte on-call

### Escalation Matrix

| Niveau | Délai réponse | Responsable | Action |
|--------|---------------|--------------|--------|
| INFO | 1 mois | Data Scientist | Rapport mensuel |
| WARNING | 1 semaine | Lead Data Scientist | Investigation |
| CRITICAL | 1 heure | Engineering Lead | Rollback |
| EMERGENCY | Immédiat | CTO | Incident response |

---

## 📈 Validation et Qualification

### Procédures de Validation

#### 1. **Validation Technique**
- Tests unitaires sur tous les modèles
- Tests d'intégration API
- Performance benchmarks
- Tests de robustesse

#### 2. **Validation Actuarielle**
- Review par comité modèle
- Documentation methodology
- Backtesting historique
- Validation out-of-time

#### 3. **Validation Réglementaire**
- Conformité Solvency II
- Documentation auditable
- Tests de stress
- Rapport de validation

### Checklist de Déploiement

- [ ] Tests unitaires passés
- [ ] Tests d'intégration passés
- [ ] Performance benchmark OK
- [ ] Documentation complète
- [ ] MLflow models enregistrés
- [ ] Monitoring configuré
- [ ] Alerts testées
- [ ] Rollback plan validé
- [ ] Review actuariel approuvé
- [ ] Sign-off responsable

---

## 🔒 Sécurité et Conformité

### Gestion des Données
- **Anonymisation** : Données personnelles supprimées
- **Encryption** : Models et données chiffrés au repos
- **Access Control** : RBAC sur MLflow et API
- **Audit Trail** : Tous les accès loggés

### Conformité Réglementaire
- **Solvency II** : Documentation complète des modèles
- **GDPR** : Protection des données personnelles
- **Model Risk Management** : Framework conforme aux guidelines
- **Audit Trail** : Traçabilité complète des décisions modèle

---

## 📝 Documentation et Reporting

### Rapports Périodiques

#### **Mensuel**
- Performance des modèles
- Drift de données
- Incidents et résolutions
- Plan de maintenance

#### **Trimestriel**
- Review de gouvernance
- Mise à jour procédures
- Formation équipe
- Planning évolution

#### **Annuel**
- Validation complète modèle
- Review réglementaire
- Architecture review
- Roadmap stratégique

### Structure de Documentation

```
documentation/
├── model_cards/
│   ├── pricing_glm.md
│   ├── pricing_cann.md
│   ├── reserving_mack.md
│   └── fraud_random_forest.md
├── validation_reports/
│   ├── quarterly/
│   └── annual/
├── incident_reports/
└── governance/
    └── policies/
```

---

## 🎯 KPIs de Gouvernance

### Métriques de Succès

| KPI | Cible | Actuel | Status |
|-----|-------|--------|--------|
| **Disponibilité API** | > 99.5% | 99.8% | ✅ |
| **Temps de réponse** | < 200ms | 150ms | ✅ |
| **Drift Detection** | PSI < 0.1 | 0.05 | ✅ |
| **Model Retraining** | SLA respecté | 100% | ✅ |
| **Alert Accuracy** | < 5% false positives | 2% | ✅ |
| **Documentation** | 100% couverture | 95% | ⚠️ |

---

## 📞 Contacts et Responsabilités

### Équipe de Gouvernance

| Rôle | Responsable | Contact |
|------|--------------|---------|
| **Model Owner** | Lead Actuary | actuary@company.com |
| **Data Science Lead** | Lead Data Scientist | ds-lead@company.com |
| **Engineering Lead** | ML Engineer | ml-eng@company.com |
| **Risk Manager** | Risk Officer | risk@company.com |
| **Compliance Officer** | Compliance | compliance@company.com |

### Escalation Technique

- **Support N1** : data-science-support@company.com
- **Support N2** : ml-engineering@company.com
- **Emergency** : oncall-ml@company.com

---

## 🔄 Amélioration Continue

### Feedback Loop
1. **Collecte** : Feedback utilisateurs, métriques monitoring
2. **Analyse** : Root cause des incidents
3. **Action** : Améliorations processus et modèles
4. **Validation** : Tests et review
5. **Déploiement** : Mise en production

### Roadmap Évolution
- **Q4 2026** : MLflow avancé, auto-retraining
- **Q1 2027** : Model monitoring temps réel
- **Q2 2027** : A/B testing framework
- **Q3 2027** : Auto-ML pipeline optimisé

---

*Document maintenu par l'équipe de gouvernance de modèle. Dernière mise à jour : 26 Août 2026*