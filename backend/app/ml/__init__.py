# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""Machine Learning package for PronoTurf.

This package groups all modules required for:
* feature engineering,
* prediction model training (Gradient Boosting),
* explainability tooling (SHAP values),
* the complete prediction service,
* initial model training helpers.
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