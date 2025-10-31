#!/usr/bin/env python3
"""
Script CLI pour entraîner le modèle ML initial

Ce script prépare les données d'entraînement depuis la base de données
et entraîne un nouveau modèle de prédiction hippique.

Usage:
    python scripts/train_initial_model.py [OPTIONS]

Examples:
    # Entraîner avec toutes les données disponibles
    python scripts/train_initial_model.py

    # Entraîner avec une plage de dates spécifique
    python scripts/train_initial_model.py --min-date 2024-01-01 --max-date 2024-12-31

    # Personnaliser les paramètres du modèle
    python scripts/train_initial_model.py --n-estimators 300 --learning-rate 0.05
"""

import sys
import argparse
import logging
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import SessionLocal
from app.ml.training import ModelTrainer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Entraîne le modèle ML de prédiction hippique"
    )

    parser.add_argument(
        "--min-date",
        type=str,
        help="Date minimale des courses (YYYY-MM-DD)",
        default=None
    )

    parser.add_argument(
        "--max-date",
        type=str,
        help="Date maximale des courses (YYYY-MM-DD)",
        default=None
    )

    parser.add_argument(
        "--test-size",
        type=float,
        help="Proportion des données pour la validation (0.0-1.0)",
        default=0.2
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Chemin de sauvegarde du modèle",
        default="models/horse_racing_model.pkl"
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        help="Nombre d'arbres du Gradient Boosting",
        default=200
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Taux d'apprentissage",
        default=0.1
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        help="Profondeur maximale des arbres",
        default=5
    )

    parser.add_argument(
        "--include-odds",
        action="store_true",
        help="Inclure les cotes PMU dans les features",
        default=False
    )

    return parser.parse_args()


def main():
    """Fonction principale"""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("PronoTurf ML Model Training Script")
    logger.info("=" * 60)
    logger.info(f"Output path: {args.output}")
    logger.info(f"Date range: {args.min_date or 'None'} to {args.max_date or 'None'}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Include odds: {args.include_odds}")
    logger.info(f"Model params: n_estimators={args.n_estimators}, " +
               f"learning_rate={args.learning_rate}, max_depth={args.max_depth}")
    logger.info("=" * 60)

    # Créer une session de base de données
    db = SessionLocal()

    try:
        # Créer le trainer
        output_path = Path(args.output)
        trainer = ModelTrainer(db, model_save_path=output_path)

        # Paramètres du modèle
        model_params = {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
        }

        # Entraîner le modèle
        metrics = trainer.train_new_model(
            min_date=args.min_date,
            max_date=args.max_date,
            test_size=args.test_size,
            include_odds=args.include_odds,
            **model_params
        )

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {output_path.absolute()}")
        logger.info("\nYou can now use this model for predictions with:")
        logger.info(f"  - The prediction API endpoints")
        logger.info(f"  - The test_predictions.py script")
        logger.info(f"  - The Celery tasks for daily predictions")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
