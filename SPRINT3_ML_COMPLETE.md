# Sprint 3 : Machine Learning Pipeline - Récapitulatif d'implémentation

## ✅ Statut : TERMINÉ

Ce document récapitule toutes les fonctionnalités implémentées pour le Sprint 3 de PronoTurf, qui introduit le pipeline complet de Machine Learning pour les prédictions hippiques.

---

## 📦 Architecture ML

### Modules créés

| Module | Fichier | Description |
|--------|---------|-------------|
| FeatureEngineer | `app/ml/features.py` | Extraction et calcul des features pour le ML |
| HorseRacingModel | `app/ml/model.py` | Modèle Gradient Boosting pour prédictions |
| SHAPExplainer | `app/ml/explainer.py` | Calcul des SHAP values pour explicabilité |
| RacePredictionService | `app/ml/predictor.py` | Service principal de prédiction |
| ModelTrainer | `app/ml/training.py` | Gestion de l'entraînement du modèle |

### Structure du package ML

```
backend/app/ml/
├── __init__.py           # Exports des classes principales
├── features.py           # Feature engineering (45+ features)
├── model.py              # Modèle Gradient Boosting
├── explainer.py          # SHAP explainability
├── predictor.py          # Service de prédiction
└── training.py           # Entraînement du modèle
```

---

## 🧮 Feature Engineering

### 45+ Features extraites

#### Features du cheval (12)
- `horse_age` : Âge du cheval
- `horse_gender_*` : Genre (male/female/hongre) encodé one-hot
- `horse_recent_avg_position` : Position moyenne récente
- `horse_recent_win_rate` : Taux de victoires récent
- `horse_recent_place_rate` : Taux de placement récent
- `horse_recent_runs` : Nombre de courses récentes
- `horse_distance_win_rate` : Win rate sur distance similaire (±200m)
- `horse_distance_runs` : Nombre de courses sur distance similaire
- `horse_surface_win_rate` : Win rate sur surface similaire
- `horse_surface_runs` : Nombre de courses sur surface similaire

#### Features du jockey (4)
- `jockey_win_rate` : Taux de victoires (30 derniers jours)
- `jockey_place_rate` : Taux de placement (30 derniers jours)
- `jockey_recent_runs` : Nombre de courses récentes
- `jockey_horse_affinity` : Affinité jockey-cheval (win rate combiné)

#### Features de l'entraîneur (3)
- `trainer_win_rate` : Taux de victoires (30 derniers jours)
- `trainer_place_rate` : Taux de placement (30 derniers jours)
- `trainer_recent_runs` : Nombre de courses récentes

#### Features de la course (11)
- `course_distance` : Distance de la course en mètres
- `course_number_of_runners` : Nombre de partants
- `course_prize_money` : Allocation de la course
- `discipline_*` : Discipline encodée one-hot (plat, trot_monte, trot_attele, haies, steeple, cross)
- `surface_*` : Surface encodée one-hot (pelouse, piste, sable, fibre)

#### Features du partant (9)
- `numero_corde` : Numéro de départ
- `poids_porte` : Poids porté en kg
- `handicap_value` : Valeur de handicap
- `days_since_last_race` : Jours depuis dernière course
- `has_oeilleres` : Porte des œillères (0/1)
- `recent_avg_position` : Position moyenne récente du partant
- `recent_best_position` : Meilleure position récente
- `recent_form_length` : Nombre de courses dans la forme récente
- `has_won_recently` : A gagné récemment (0/1)
- `odds_pmu` : Cote PMU (optionnel)

#### Features de l'hippodrome (2)
- `hippodrome_affinity_win_rate` : Win rate du cheval sur cet hippodrome
- `hippodrome_affinity_runs` : Nombre de courses sur cet hippodrome

### Calcul intelligent des features

- **Performances récentes** : Analyse des 5 dernières courses
- **Distance similaire** : Recherche dans une fenêtre de ±200m
- **Affinités** : Calcul des combinaisons gagnantes jockey-cheval
- **Normalisation** : Cap sur les compteurs pour éviter l'overfitting

---

## 🤖 Modèle ML

### Algorithme : Gradient Boosting

**Hyperparamètres par défaut** :
- `n_estimators` : 200 arbres
- `learning_rate` : 0.1
- `max_depth` : 5
- `min_samples_split` : 20
- `min_samples_leaf` : 10
- `subsample` : 0.8

### Target

Le modèle prédit la **probabilité qu'un cheval termine dans le top 3**.

- **Label = 1** : Cheval arrivé 1er, 2ème ou 3ème
- **Label = 0** : Cheval hors du top 3

### Métriques de performance

- **Accuracy** : Taux de prédictions correctes
- **Precision** : Fiabilité des prédictions positives
- **Recall** : Capacité à détecter les top 3
- **F1-Score** : Harmonie entre precision et recall
- **ROC-AUC** : Performance globale du classifieur

---

## 🔍 Explicabilité SHAP

### SHAP values calculées

Pour chaque prédiction, le système calcule :

- **Base value** : Probabilité moyenne du modèle
- **SHAP values** : Contribution de chaque feature
- **Top 5 features positives** : Atouts du cheval
- **Top 5 features négatives** : Handicaps du cheval

### Interprétation automatique

Le système génère des explications en français :

```python
# Exemple d'explication générée
"Le taux de victoire du jockey (15.0%) favorise fortement ce partant"
"La position moyenne récente (6.2) défavorise légèrement ses chances"
"Le numéro de corde 1 favorise fortement ce partant"
```

### Résumé d'impact

Pour chaque prédiction, un résumé textuel est généré :

```
**Analyse de TORNADO (N°5)**

Les facteurs favorables l'emportent sur les défavorables (impact net: +0.234).

**Principaux atouts:**
1. Le taux de victoire du jockey (18.0%) favorise fortement ce partant
2. Le repos depuis la dernière course (14 jours) favorise ce partant
3. Le numéro de corde 5 favorise légèrement ses chances

**Principaux handicaps:**
1. L'âge du cheval (8 ans) défavorise légèrement ses chances
2. La position moyenne récente (5.4) défavorise ce partant
```

---

## 🎯 Service de Prédiction

### Fonctionnalités

Le `RacePredictionService` permet de :

1. **Prédire une course** : `predict_course(course_id)`
2. **Prédire une réunion** : `predict_reunion(reunion_id)`
3. **Prédire un programme quotidien** : `predict_daily_program(date)`
4. **Détecter les value bets** : Identifier les paris à valeur

### Détection de value bets

Un value bet est détecté lorsque :

```python
# Probabilité du modèle > Probabilité implicite de la cote
edge = model_probability - (1 / odds_pmu)

# Si edge > 10% (configurable)
if edge > 0.10:
    # C'est un value bet !
```

Niveaux de value :
- **High** : edge ≥ 30%
- **Medium** : 20% ≤ edge < 30%
- **Low** : 10% ≤ edge < 20%

### Recommandations générées

Pour chaque course, le service génère :

- **Gagnant** : Meilleure probabilité
- **Placé** : Top 3
- **Tiercé** : Top 3 dans l'ordre
- **Quarté** : Top 4
- **Quinté** : Top 5

---

## 📜 Scripts CLI

### 1. `train_initial_model.py`

Entraîne un nouveau modèle ML.

**Usage basique** :
```bash
python scripts/train_initial_model.py
```

**Options disponibles** :
- `--min-date YYYY-MM-DD` : Date minimale des courses
- `--max-date YYYY-MM-DD` : Date maximale des courses
- `--test-size 0.2` : Proportion pour validation
- `--n-estimators 200` : Nombre d'arbres
- `--learning-rate 0.1` : Taux d'apprentissage
- `--max-depth 5` : Profondeur des arbres
- `--include-odds` : Inclure les cotes PMU

**Exemples** :

```bash
# Entraîner avec une période spécifique
python scripts/train_initial_model.py \
    --min-date 2024-01-01 \
    --max-date 2024-12-31

# Modèle personnalisé
python scripts/train_initial_model.py \
    --n-estimators 300 \
    --learning-rate 0.05 \
    --max-depth 6
```

### 2. `test_predictions.py`

Teste les prédictions du modèle.

**Usage basique** :
```bash
# Prédire une course
python scripts/test_predictions.py --course-id 123

# Prédire aujourd'hui
python scripts/test_predictions.py --today

# Prédire une date
python scripts/test_predictions.py --date 2025-02-01
```

**Options disponibles** :
- `--course-id INT` : ID de la course
- `--today` : Courses du jour
- `--date YYYY-MM-DD` : Courses d'une date
- `--with-explanations` : Inclure SHAP (plus lent)
- `--output FILE.json` : Sauvegarder en JSON
- `--verbose` : Affichage détaillé

---

## ⏰ Tâches Celery

### Tâches implémentées

| Tâche | Fonction | Description |
|-------|----------|-------------|
| `generate_daily_predictions` | Génération quotidienne | Génère les pronostics pour toutes les courses du jour |
| `generate_prediction_for_course` | Génération unitaire | Génère le pronostic pour une course spécifique |
| `train_ml_model` | Entraînement | Entraîne un nouveau modèle avec les données |
| `update_model_performance` | Évaluation | Met à jour les métriques de performance |

### Schedule Celery Beat (configuré)

| Tâche | Schedule | Description |
|-------|----------|-------------|
| `generate-daily-predictions` | 7h00 | Génération pronostics quotidiens |
| `train-ml-model` | Dimanche 2h | Entraînement hebdomadaire |

### Utilisation manuelle

```python
# Via Python
from app.tasks.ml_tasks import generate_daily_predictions

# Lancer manuellement
generate_daily_predictions.delay()

# Avec date spécifique
generate_daily_predictions.delay(target_date="2025-02-01")
```

```bash
# Via Celery CLI
celery -A app.tasks.celery_app call app.tasks.ml_tasks.generate_daily_predictions
```

---

## 🌐 Endpoints API

### Endpoints créés

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/api/v1/pronostics/today` | Pronostics du jour |
| GET | `/api/v1/pronostics/date/{date}` | Pronostics d'une date |
| GET | `/api/v1/pronostics/course/{course_id}` | Pronostic d'une course |
| POST | `/api/v1/pronostics/generate/course/{course_id}` | Générer pronostic (async) |
| POST | `/api/v1/pronostics/generate/daily` | Générer programme quotidien (async) |
| GET | `/api/v1/pronostics/value-bets/today` | Value bets du jour |
| GET | `/api/v1/pronostics/stats/accuracy` | Statistiques de précision |

### Exemples d'utilisation

**Récupérer les pronostics du jour** :
```bash
curl http://localhost:8000/api/v1/pronostics/today
```

**Pronostic d'une course avec explications** :
```bash
curl "http://localhost:8000/api/v1/pronostics/course/123?include_explanations=true"
```

**Générer les pronostics du jour (async)** :
```bash
curl -X POST http://localhost:8000/api/v1/pronostics/generate/daily
```

**Value bets du jour** :
```bash
curl "http://localhost:8000/api/v1/pronostics/value-bets/today?min_edge=0.15"
```

---

## 🗄️ Modèles de données

### Nouveaux modèles créés

| Modèle | Table | Description |
|--------|-------|-------------|
| `Pronostic` | `pronostics` | Prédictions globales pour une course |
| `PartantPrediction` | `partant_predictions` | Prédictions individuelles par partant |
| `MLModel` | `ml_models` | Métadonnées des modèles entraînés |
| `TrainingLog` | `training_logs` | Logs des sessions d'entraînement |

### Table : pronostics

```sql
CREATE TABLE pronostics (
    pronostic_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    course_id INT UNSIGNED NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    gagnant_predicted JSON,
    place_predicted JSON,
    tierce_predicted JSON,
    quarte_predicted JSON,
    quinte_predicted JSON,
    confidence_score DECIMAL(5, 2),
    value_bet_detected BOOLEAN DEFAULT FALSE,
    shap_values JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE
);
```

### Table : partant_predictions

```sql
CREATE TABLE partant_predictions (
    prediction_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    pronostic_id INT UNSIGNED NOT NULL,
    partant_id INT UNSIGNED NOT NULL,
    predicted_position TINYINT UNSIGNED,
    win_probability DECIMAL(5, 4),
    place_probability DECIMAL(5, 4),
    confidence_score DECIMAL(5, 2),
    shap_contributions JSON,
    top_positive_features JSON,
    top_negative_features JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pronostic_id) REFERENCES pronostics(pronostic_id) ON DELETE CASCADE,
    FOREIGN KEY (partant_id) REFERENCES partants(partant_id) ON DELETE CASCADE
);
```

---

## 🚀 Workflow complet

### 1. Premier entraînement

```bash
# 1. S'assurer d'avoir des données de courses terminées
# (Via synchronisation PMU)

# 2. Entraîner le modèle initial
docker exec pronoturf-backend python scripts/train_initial_model.py \
    --min-date 2024-01-01 \
    --n-estimators 200

# 3. Vérifier les performances
# Objectif: Accuracy > 0.60, ROC-AUC > 0.70

# 4. Tester les prédictions
docker exec pronoturf-backend python scripts/test_predictions.py --today
```

### 2. Génération quotidienne automatique

```bash
# Les prédictions sont générées automatiquement à 7h00
# via Celery Beat

# Vérifier via l'API
curl http://localhost:8000/api/v1/pronostics/today
```

### 3. Génération manuelle

```bash
# Via script CLI
python scripts/test_predictions.py --today --output predictions.json

# Via API
curl -X POST http://localhost:8000/api/v1/pronostics/generate/daily

# Via Celery
celery -A app.tasks.celery_app call app.tasks.ml_tasks.generate_daily_predictions
```

### 4. Ré-entraînement hebdomadaire

Le modèle est automatiquement ré-entraîné chaque dimanche à 2h avec les nouvelles données.

**Entraînement manuel** :
```bash
docker exec pronoturf-backend python scripts/train_initial_model.py
```

---

## 📊 Format des prédictions

### Exemple de réponse API

```json
{
  "course_id": 123,
  "course_name": "R1C3 - Prix de Longchamp",
  "course_distance": 2400,
  "course_discipline": "plat",
  "number_of_runners": 16,
  "predictions": [
    {
      "partant_id": 456,
      "numero_corde": 5,
      "horse_name": "TORNADO",
      "horse_age": 4,
      "jockey_name": "C. Soumillon",
      "trainer_name": "A. Fabre",
      "odds_pmu": 3.5,
      "probability": 0.68,
      "confidence_level": "high",
      "explanation": {
        "base_value": 0.35,
        "top_positive_features": [
          {
            "feature": "jockey_win_rate",
            "feature_value": 0.18,
            "shap_value": 0.12,
            "impact": "positive",
            "explanation": "Le taux de victoire du jockey (18.0%) favorise fortement ce partant"
          }
        ],
        "prediction_impact_summary": "**Analyse de TORNADO (N°5)**\n\nLes facteurs favorables l'emportent..."
      }
    }
  ],
  "recommendations": {
    "gagnant": {
      "numero": 5,
      "horse_name": "TORNADO",
      "probability": 0.68
    },
    "place": [5, 12, 3],
    "tierce": [5, 12, 3],
    "quarte": [5, 12, 3, 8],
    "quinte": [5, 12, 3, 8, 14]
  },
  "value_bets": [
    {
      "numero_corde": 12,
      "horse_name": "ECLIPSE",
      "odds_pmu": 8.5,
      "model_probability": 0.25,
      "implied_probability": 0.12,
      "edge": 0.13,
      "edge_percentage": 108.3,
      "value_level": "medium"
    }
  ],
  "generated_at": "2025-02-01T14:30:00",
  "model_version": "v1.0_20250201"
}
```

---

## 📈 Métriques de performance attendues

### Objectifs de performance

Pour les courses hippiques (naturellement difficiles à prédire) :

| Métrique | Seuil minimum | Objectif | Excellent |
|----------|---------------|----------|-----------|
| **Accuracy** | 0.55 | 0.60 | 0.65+ |
| **Precision** | 0.60 | 0.65 | 0.70+ |
| **Recall** | 0.50 | 0.55 | 0.60+ |
| **ROC-AUC** | 0.65 | 0.70 | 0.75+ |

### Interprétation

- **0.50-0.55** : Modèle faible, nécessite plus de données
- **0.55-0.65** : Modèle acceptable, performances correctes
- **0.65-0.75** : Bon modèle, performances solides
- **0.75+** : Excellent modèle (rare dans le domaine hippique)

---

## 🔧 Configuration

### Variables d'environnement

Ajoutées dans `backend/app/core/config.py` :

```python
ML_MODELS_PATH: str = "/app/models"
ML_MODEL_PATH: str = "/app/models/horse_racing_model.pkl"
ML_MODEL_VERSION: str = "v1"
ML_RETRAIN_SCHEDULE: str = "0 2 * * 1"  # Dimanche 2h
```

### Dépendances

Toutes les dépendances ML sont déjà dans `requirements.txt` :

```
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
joblib==1.3.2
shap==0.43.0
```

---

## ✅ Checklist Sprint 3

- [x] Module de feature engineering (45+ features)
- [x] Modèle Gradient Boosting
- [x] Calcul des SHAP values
- [x] Service de prédiction complet
- [x] Module d'entraînement
- [x] Scripts CLI (train + test)
- [x] Tâches Celery ML
- [x] Modèles SQLAlchemy (Pronostic, PartantPrediction, MLModel, TrainingLog)
- [x] Endpoints API pronostics
- [x] Détection value bets
- [x] Documentation complète

---

## 🎯 Prochaines étapes (Sprint 4)

Le Sprint 3 ML est terminé ! Prochains objectifs :

### Sprint 4 : Frontend & Notifications

- [ ] Pages React pour les pronostics
- [ ] Dashboard Streamlit avancé
- [ ] Graphiques Plotly interactifs
- [ ] Notifications Telegram
- [ ] Notifications Email
- [ ] Système de favoris
- [ ] Gestion paris simulés
- [ ] Tracking bankroll

---

## 🎉 Résultat

**Sprint 3 : 100% complété !**

Le système PronoTurf dispose maintenant d'un pipeline ML complet :

✅ **Feature engineering** intelligent avec 45+ features
✅ **Modèle Gradient Boosting** performant
✅ **Explicabilité SHAP** pour comprendre les prédictions
✅ **Détection de value bets** automatique
✅ **API REST** complète pour les pronostics
✅ **Tâches Celery** pour génération automatique
✅ **Scripts CLI** pour entraînement et test
✅ **Documentation** exhaustive

**Le système est prêt à générer des pronostics quotidiens avec explications !** 🚀
