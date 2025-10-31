"""
Modèle de Machine Learning pour la prédiction hippique

Ce module contient la définition du modèle Gradient Boosting
et les méthodes pour l'entraînement et les prédictions.
"""

import logging
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sqlalchemy.orm import Session

from app.ml.features import FeatureEngineer

logger = logging.getLogger(__name__)


class HorseRacingModel:
    """
    Modèle ML pour la prédiction de résultats de courses hippiques

    Utilise Gradient Boosting pour prédire la probabilité qu'un cheval
    termine dans le top 3.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
    ):
        """
        Initialise le modèle

        Args:
            model_path: Chemin vers le modèle sauvegardé (optionnel)
            n_estimators: Nombre d'arbres
            learning_rate: Taux d'apprentissage
            max_depth: Profondeur maximale des arbres
            min_samples_split: Min échantillons pour split
            min_samples_leaf: Min échantillons par feuille
        """
        self.model_path = model_path
        self.feature_names = None
        self.model_version = None
        self.training_date = None
        self.performance_metrics = {}

        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=0.8,
                random_state=42,
                verbose=1
            )
            self.feature_engineer = None
            self.model_version = self._generate_version()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Entraîne le modèle sur les données fournies

        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement (1 si top 3, 0 sinon)
            X_val: Features de validation (optionnel)
            y_val: Labels de validation (optionnel)

        Returns:
            Dictionnaire contenant les métriques de performance
        """
        logger.info(f"Starting model training with {len(X_train)} samples")

        # Sauvegarder les noms de features
        self.feature_names = list(X_train.columns)

        # Entraîner le modèle
        self.model.fit(X_train, y_train)

        # Évaluer sur l'ensemble d'entraînement
        train_metrics = self._evaluate(X_train, y_train, "Train")

        # Évaluer sur l'ensemble de validation si fourni
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate(X_val, y_val, "Validation")

        # Sauvegarder les métriques
        self.performance_metrics = {
            "train": train_metrics,
            "validation": val_metrics,
            "training_date": datetime.now().isoformat(),
            "n_samples_train": len(X_train),
            "n_samples_val": len(X_val) if X_val is not None else 0,
            "n_features": len(self.feature_names),
        }

        self.training_date = datetime.now()

        logger.info(f"Training completed. Train accuracy: {train_metrics['accuracy']:.4f}")
        if val_metrics:
            logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")

        return self.performance_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédit les probabilités pour les données fournies

        Args:
            X: Features

        Returns:
            Array des probabilités (classe 1 = top 3)
        """
        # Vérifier que les features sont les bonnes
        if self.feature_names and list(X.columns) != self.feature_names:
            # Réordonner les colonnes
            X = X[self.feature_names]

        probas = self.model.predict_proba(X)
        return probas[:, 1]  # Probabilité de la classe 1 (top 3)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Prédit les classes pour les données fournies

        Args:
            X: Features
            threshold: Seuil de décision

        Returns:
            Array des prédictions binaires
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retourne l'importance des features

        Args:
            top_n: Nombre de top features à retourner

        Returns:
            DataFrame avec les importances des features
        """
        if not self.feature_names:
            raise ValueError("Model not trained yet")

        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feature_importance_df.head(top_n)

    def save_model(self, path: Path) -> None:
        """
        Sauvegarde le modèle sur le disque

        Args:
            path: Chemin du fichier de sauvegarde
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_version': self.model_version,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'performance_metrics': self.performance_metrics,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

        # Sauvegarder aussi les métriques en JSON
        metrics_path = path.parent / f"{path.stem}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)

    def load_model(self, path: Path) -> None:
        """
        Charge un modèle depuis le disque

        Args:
            path: Chemin du fichier du modèle
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_version = model_data['model_version']
        self.training_date = datetime.fromisoformat(model_data['training_date']) if model_data.get('training_date') else None
        self.performance_metrics = model_data.get('performance_metrics', {})

        logger.info(f"Model loaded from {path} (version {self.model_version})")

    def _evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        dataset_name: str
    ) -> Dict[str, float]:
        """
        Évalue le modèle sur un dataset

        Args:
            X: Features
            y: Labels
            dataset_name: Nom du dataset (pour logging)

        Returns:
            Dictionnaire des métriques
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0,
        }

        logger.info(f"{dataset_name} metrics: " +
                   f"Accuracy={metrics['accuracy']:.4f}, " +
                   f"Precision={metrics['precision']:.4f}, " +
                   f"Recall={metrics['recall']:.4f}, " +
                   f"F1={metrics['f1']:.4f}, " +
                   f"ROC-AUC={metrics['roc_auc']:.4f}")

        return metrics

    def _generate_version(self) -> str:
        """
        Génère un numéro de version pour le modèle

        Returns:
            Version string (ex: "v1.0_20250130")
        """
        date_str = datetime.now().strftime("%Y%m%d")
        return f"v1.0_{date_str}"


class TrainingDataPreparer:
    """
    Prépare les données d'entraînement à partir de la base de données
    """

    def __init__(self, db: Session):
        """
        Initialise le préparateur de données

        Args:
            db: Session de base de données
        """
        self.db = db
        self.feature_engineer = FeatureEngineer(db)

    def prepare_training_data(
        self,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        test_size: float = 0.2,
        include_odds: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """
        Prépare les données d'entraînement et de validation

        Args:
            min_date: Date minimale des courses (format YYYY-MM-DD)
            max_date: Date maximale des courses (format YYYY-MM-DD)
            test_size: Proportion des données pour la validation
            include_odds: Inclure les cotes PMU dans les features

        Returns:
            Tuple (X_train, y_train, X_val, y_val)
        """
        from app.models.partant import Partant
        from app.models.course import Course, CourseStatus
        from app.models.reunion import Reunion

        logger.info("Preparing training data...")

        # Construire la requête
        query = (
            self.db.query(Partant)
            .join(Course)
            .join(Reunion)
            .filter(
                Partant.final_position.isnot(None),  # Seulement les courses terminées
                Course.status == CourseStatus.FINISHED
            )
        )

        if min_date:
            query = query.filter(Reunion.reunion_date >= min_date)
        if max_date:
            query = query.filter(Reunion.reunion_date <= max_date)

        partants = query.all()

        if not partants:
            raise ValueError("No finished races found in the database")

        logger.info(f"Found {len(partants)} partants from finished races")

        # Extraire les features et labels
        X_list = []
        y_list = []

        for partant in partants:
            try:
                features = self.feature_engineer.extract_features_for_partant(
                    partant,
                    include_odds=include_odds
                )
                X_list.append(features)

                # Label: 1 si dans le top 3, 0 sinon
                label = 1 if partant.final_position <= 3 else 0
                y_list.append(label)

            except Exception as e:
                logger.warning(f"Error extracting features for partant {partant.partant_id}: {e}")
                continue

        if not X_list:
            raise ValueError("No features could be extracted")

        # Convertir en DataFrames
        X = pd.DataFrame(X_list)
        y = np.array(y_list)

        # Assurer que toutes les features sont présentes
        expected_features = self.feature_engineer.get_feature_names()
        for feature in expected_features:
            if feature not in X.columns:
                X[feature] = 0

        # Réordonner les colonnes
        X = X[expected_features]

        # Remplir les NaN
        X = X.fillna(0)

        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
        logger.info(f"Label distribution: {np.sum(y)} positives ({np.mean(y):.2%}), " +
                   f"{len(y) - np.sum(y)} negatives")

        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y  # Préserver la distribution des classes
        )

        logger.info(f"Train set: {len(X_train)} samples, Validation set: {len(X_val)} samples")

        return X_train, y_train, X_val, y_val
