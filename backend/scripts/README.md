# Scripts PronoTurf

Ce répertoire contient les scripts CLI pour l'entraînement et le test du modèle ML.

## Scripts disponibles

### 1. `train_initial_model.py`

Entraîne un nouveau modèle de prédiction hippique à partir des données historiques.

**Usage basique:**
```bash
python scripts/train_initial_model.py
```

**Options:**
- `--min-date YYYY-MM-DD` : Date minimale des courses à utiliser
- `--max-date YYYY-MM-DD` : Date maximale des courses à utiliser
- `--test-size FLOAT` : Proportion des données pour la validation (défaut: 0.2)
- `--output PATH` : Chemin de sauvegarde du modèle (défaut: models/horse_racing_model.pkl)
- `--n-estimators INT` : Nombre d'arbres (défaut: 200)
- `--learning-rate FLOAT` : Taux d'apprentissage (défaut: 0.1)
- `--max-depth INT` : Profondeur maximale des arbres (défaut: 5)
- `--include-odds` : Inclure les cotes PMU dans les features

**Exemples:**

```bash
# Entraîner avec toutes les données disponibles
python scripts/train_initial_model.py

# Entraîner sur une période spécifique
python scripts/train_initial_model.py --min-date 2024-01-01 --max-date 2024-12-31

# Entraîner avec des paramètres personnalisés
python scripts/train_initial_model.py \
    --n-estimators 300 \
    --learning-rate 0.05 \
    --max-depth 6

# Entraîner en incluant les cotes PMU
python scripts/train_initial_model.py --include-odds
```

**Prérequis:**
- Base de données MySQL accessible
- Données de courses terminées dans la base
- Au moins 100 courses avec résultats pour l'entraînement

**Sortie:**
- Modèle sauvegardé dans `models/horse_racing_model.pkl`
- Métriques de performance en JSON dans `models/horse_racing_model_metrics.json`
- Logs d'entraînement affichés dans la console

---

### 2. `test_predictions.py`

Teste le modèle ML en générant des prédictions pour des courses.

**Usage basique:**
```bash
# Prédire une course spécifique
python scripts/test_predictions.py --course-id 123

# Prédire toutes les courses du jour
python scripts/test_predictions.py --today

# Prédire une date spécifique
python scripts/test_predictions.py --date 2025-02-01
```

**Options:**
- `--course-id INT` : ID de la course à prédire
- `--today` : Prédire toutes les courses du jour
- `--date YYYY-MM-DD` : Prédire toutes les courses d'une date
- `--model-path PATH` : Chemin vers le modèle (défaut: models/horse_racing_model.pkl)
- `--with-explanations` : Inclure les explications SHAP (plus lent)
- `--output PATH` : Sauvegarder les résultats en JSON
- `--verbose` : Affichage détaillé avec explications

**Exemples:**

```bash
# Prédire une course avec explications
python scripts/test_predictions.py --course-id 42 --with-explanations --verbose

# Prédire et sauvegarder en JSON
python scripts/test_predictions.py --today --output predictions_today.json

# Utiliser un modèle personnalisé
python scripts/test_predictions.py \
    --course-id 42 \
    --model-path custom_models/my_model.pkl
```

**Prérequis:**
- Modèle entraîné disponible
- Courses avec partants dans la base de données
- Données des chevaux, jockeys et entraîneurs

**Sortie:**
- Prédictions affichées dans la console
- Recommandations de paris (gagnant, placé, tiercé, quarté, quinté)
- Value bets détectés
- Explications SHAP (si `--with-explanations`)
- Fichier JSON (si `--output` spécifié)

---

## Workflow recommandé

### 1. Premier entraînement

```bash
# 1. Vérifier que des données sont disponibles
# (vous devez avoir des courses terminées avec résultats)

# 2. Entraîner le modèle initial
python scripts/train_initial_model.py \
    --min-date 2024-01-01 \
    --n-estimators 200 \
    --learning-rate 0.1

# 3. Vérifier les performances
# Les métriques sont affichées à la fin de l'entraînement
# Objectif: Accuracy > 0.60, ROC-AUC > 0.70
```

### 2. Test des prédictions

```bash
# Tester sur quelques courses
python scripts/test_predictions.py --course-id 1 --with-explanations --verbose

# Tester sur toutes les courses du jour
python scripts/test_predictions.py --today
```

### 3. Ré-entraînement périodique

```bash
# Ré-entraîner avec les nouvelles données chaque semaine/mois
python scripts/train_initial_model.py --min-date 2024-01-01
```

---

## Utilisation dans Docker

### Depuis le container backend

```bash
# Entrer dans le container
docker exec -it pronoturf-backend bash

# Entraîner le modèle
python scripts/train_initial_model.py

# Tester les prédictions
python scripts/test_predictions.py --today
```

### Depuis l'hôte (via docker exec)

```bash
# Entraîner
docker exec pronoturf-backend python scripts/train_initial_model.py

# Tester
docker exec pronoturf-backend python scripts/test_predictions.py --course-id 1
```

---

## Troubleshooting

### Erreur: "No finished races found in the database"

**Solution:** Vous devez avoir des courses terminées avec résultats.
```bash
# Synchroniser des données PMU
docker exec pronoturf-backend python -c "
from app.core.database import SessionLocal
from app.services.pmu_service import PMUService
from datetime import date, timedelta

db = SessionLocal()
service = PMUService(db)

# Synchroniser les 30 derniers jours
for i in range(30):
    d = date.today() - timedelta(days=i)
    service.sync_program_for_date(d)
db.close()
"
```

### Erreur: "Model not found"

**Solution:** Entraînez d'abord un modèle avec `train_initial_model.py`

### Performances faibles (accuracy < 0.55)

**Solutions:**
1. **Plus de données:** Entraîner sur une période plus longue
2. **Ajuster les hyperparamètres:**
   ```bash
   python scripts/train_initial_model.py \
       --n-estimators 300 \
       --learning-rate 0.05 \
       --max-depth 6
   ```
3. **Inclure les cotes:** `--include-odds`

### SHAP trop lent

**Solution:** Ne pas utiliser `--with-explanations` pour les prédictions en masse.
Utilisez-le seulement pour analyser des courses spécifiques.

---

## Intégration avec Celery

Une fois le modèle entraîné, les tâches Celery peuvent l'utiliser automatiquement:

```python
# Les tâches Celery utilisent le même service
from app.tasks.ml_tasks import generate_daily_predictions

# Générer les prédictions quotidiennes
generate_daily_predictions.delay()
```

---

## Métriques de performance

### Métriques clés

- **Accuracy:** Taux de prédictions correctes (objectif: > 0.60)
- **Precision:** Fiabilité des prédictions positives (objectif: > 0.65)
- **Recall:** Capacité à détecter les chevaux gagnants (objectif: > 0.55)
- **ROC-AUC:** Performance globale (objectif: > 0.70)

### Interprétation

- **Accuracy 0.50-0.55:** Modèle faible, besoin de plus de données
- **Accuracy 0.55-0.65:** Modèle acceptable, peut être amélioré
- **Accuracy 0.65-0.75:** Bon modèle, performances solides
- **Accuracy > 0.75:** Excellent modèle (rare dans les courses hippiques)

**Note:** Les courses hippiques sont intrinsèquement difficiles à prédire.
Une accuracy de 0.60-0.65 est déjà très bonne pour ce domaine.

---

## Support

Pour plus d'informations, consultez:
- Documentation principale: `/README.md`
- CDC: `/CDC.md`
- Sprint 3 docs: `/SPRINT3_ML.md` (à venir)
