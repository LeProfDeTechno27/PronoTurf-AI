"""
Machine Learning Package pour PronoTurf

Ce package contient tous les modules nécessaires pour:
- Feature engineering (extraction de features)
- Modèle de prédiction (Gradient Boosting)
- Explicabilité (SHAP values)
- Service de prédiction complet
- Entraînement du modèle
"""

from app.ml.features import FeatureEngineer
from app.ml.model import HorseRacingModel, TrainingDataPreparer
from app.ml.explainer import SHAPExplainer
from app.ml.predictor import RacePredictionService
from app.ml.training import ModelTrainer, train_initial_model

__all__ = [
    "FeatureEngineer",
    "HorseRacingModel",
    "TrainingDataPreparer",
    "SHAPExplainer",
    "RacePredictionService",
    "ModelTrainer",
    "train_initial_model",
]
