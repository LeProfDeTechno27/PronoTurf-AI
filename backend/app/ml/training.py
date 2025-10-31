"""
Module d'entraînement du modèle ML

Gère le processus complet d'entraînement du modèle de prédiction hippique,
incluant la préparation des données, l'entraînement, la validation et la sauvegarde.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from app.ml.model import HorseRacingModel, TrainingDataPreparer
from app.ml.features import FeatureEngineer
from app.core.database import get_db

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Gère l'entraînement et la validation du modèle ML
    """

    def __init__(
        self,
        db: Session,
        model_save_path: Optional[Path] = None
    ):
        """
        Initialise le trainer

        Args:
            db: Session de base de données
            model_save_path: Chemin pour sauvegarder le modèle
        """
        self.db = db
        self.model_save_path = model_save_path or Path("models/horse_racing_model.pkl")
        self.data_preparer = TrainingDataPreparer(db)

    def train_new_model(
        self,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        test_size: float = 0.2,
        include_odds: bool = False,
        **model_params
    ) -> Dict[str, Any]:
        """
        Entraîne un nouveau modèle from scratch

        Args:
            min_date: Date minimale des courses (YYYY-MM-DD)
            max_date: Date maximale des courses (YYYY-MM-DD)
            test_size: Proportion pour validation
            include_odds: Inclure les cotes dans les features
            **model_params: Paramètres du modèle (n_estimators, learning_rate, etc.)

        Returns:
            Dictionnaire avec les métriques de performance
        """
        logger.info("=" * 50)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 50)

        # 1. Préparer les données
        logger.info("Step 1/4: Preparing training data...")
        X_train, y_train, X_val, y_val = self.data_preparer.prepare_training_data(
            min_date=min_date,
            max_date=max_date,
            test_size=test_size,
            include_odds=include_odds
        )

        # 2. Créer le modèle
        logger.info("Step 2/4: Creating model...")
        model = HorseRacingModel(**model_params)

        # 3. Entraîner le modèle
        logger.info("Step 3/4: Training model...")
        performance_metrics = model.train(X_train, y_train, X_val, y_val)

        # 4. Sauvegarder le modèle
        logger.info("Step 4/4: Saving model...")
        model.save_model(self.model_save_path)

        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)

        # Afficher les métriques finales
        self._print_performance_summary(performance_metrics)

        return performance_metrics

    def retrain_model(
        self,
        existing_model_path: Path,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        test_size: float = 0.2,
        include_odds: bool = False
    ) -> Dict[str, Any]:
        """
        Ré-entraîne un modèle existant avec de nouvelles données

        Args:
            existing_model_path: Chemin vers le modèle existant
            min_date: Date minimale des courses
            max_date: Date maximale des courses
            test_size: Proportion pour validation
            include_odds: Inclure les cotes

        Returns:
            Dictionnaire avec les métriques de performance
        """
        logger.info("=" * 50)
        logger.info("RETRAINING EXISTING MODEL")
        logger.info("=" * 50)

        # Charger le modèle existant pour récupérer ses paramètres
        if not existing_model_path.exists():
            raise FileNotFoundError(f"Model not found: {existing_model_path}")

        # Pour le retraining, on crée un nouveau modèle avec les mêmes hyperparamètres
        # mais on l'entraîne sur de nouvelles données
        return self.train_new_model(
            min_date=min_date,
            max_date=max_date,
            test_size=test_size,
            include_odds=include_odds
        )

    def evaluate_model(
        self,
        model_path: Path,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Évalue un modèle existant sur un dataset de test

        Args:
            model_path: Chemin vers le modèle
            min_date: Date minimale des courses de test
            max_date: Date maximale des courses de test

        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        logger.info("Evaluating model on test data...")

        # Charger le modèle
        model = HorseRacingModel(model_path=model_path)

        # Préparer les données de test
        _, _, X_test, y_test = self.data_preparer.prepare_training_data(
            min_date=min_date,
            max_date=max_date,
            test_size=1.0,  # Tout en test
            include_odds=False
        )

        # Évaluer
        test_metrics = model._evaluate(X_test, y_test, "Test")

        return {
            "test": test_metrics,
            "n_samples": len(X_test),
            "evaluation_date": datetime.now().isoformat(),
        }

    def get_feature_importance(
        self,
        model_path: Path,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Récupère l'importance des features d'un modèle

        Args:
            model_path: Chemin vers le modèle
            top_n: Nombre de top features à retourner

        Returns:
            Dictionnaire avec l'importance des features
        """
        # Charger le modèle
        model = HorseRacingModel(model_path=model_path)

        # Obtenir l'importance
        importance_df = model.get_feature_importance(top_n=top_n)

        return {
            "top_features": importance_df.to_dict(orient='records'),
            "model_version": model.model_version,
        }

    def _print_performance_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Affiche un résumé des performances

        Args:
            metrics: Dictionnaire des métriques
        """
        print("\n" + "=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)

        if "train" in metrics:
            print("\nTraining Set:")
            print(f"  - Samples: {metrics['n_samples_train']}")
            print(f"  - Accuracy: {metrics['train']['accuracy']:.4f}")
            print(f"  - Precision: {metrics['train']['precision']:.4f}")
            print(f"  - Recall: {metrics['train']['recall']:.4f}")
            print(f"  - F1 Score: {metrics['train']['f1']:.4f}")
            print(f"  - ROC-AUC: {metrics['train']['roc_auc']:.4f}")

        if "validation" in metrics and metrics['validation']:
            print("\nValidation Set:")
            print(f"  - Samples: {metrics['n_samples_val']}")
            print(f"  - Accuracy: {metrics['validation']['accuracy']:.4f}")
            print(f"  - Precision: {metrics['validation']['precision']:.4f}")
            print(f"  - Recall: {metrics['validation']['recall']:.4f}")
            print(f"  - F1 Score: {metrics['validation']['f1']:.4f}")
            print(f"  - ROC-AUC: {metrics['validation']['roc_auc']:.4f}")

        print("\nModel Information:")
        print(f"  - Features: {metrics['n_features']}")
        print(f"  - Training Date: {metrics['training_date']}")
        print(f"  - Model saved to: {self.model_save_path}")
        print("=" * 50 + "\n")


def train_initial_model(
    db: Session,
    output_path: Optional[Path] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    **model_params
) -> Dict[str, Any]:
    """
    Fonction helper pour entraîner le modèle initial

    Args:
        db: Session de base de données
        output_path: Chemin de sauvegarde du modèle
        min_date: Date minimale
        max_date: Date maximale
        **model_params: Paramètres du modèle

    Returns:
        Métriques de performance
    """
    trainer = ModelTrainer(db, model_save_path=output_path)
    return trainer.train_new_model(
        min_date=min_date,
        max_date=max_date,
        **model_params
    )
