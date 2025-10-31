"""
Machine Learning Package pour PronoTurf

Ce package contient tous les modules n�cessaires pour:
- Feature engineering (extraction de features)
- Mod�le de pr�diction (Gradient Boosting)
- Explicabilit� (SHAP values)
- Service de pr�diction complet
- Entra�nement du mod�le
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
