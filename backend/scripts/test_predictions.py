#!/usr/bin/env python3
# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Script CLI pour tester les prédictions du modèle ML

Ce script permet de générer des prédictions pour des courses spécifiques
ou pour tout le programme d'une journée.

Usage:
    python scripts/test_predictions.py [OPTIONS]

Examples:
    # Prédire une course spécifique
    python scripts/test_predictions.py --course-id 123

    # Prédire toutes les courses du jour
    python scripts/test_predictions.py --today

    # Prédire toutes les courses d'une date spécifique
    python scripts/test_predictions.py --date 2025-02-01

    # Prédire avec explications SHAP
    python scripts/test_predictions.py --course-id 123 --with-explanations
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import date

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import SessionLocal
from app.ml.predictor import RacePredictionService

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Test les prédictions du modèle ML"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--course-id",
        type=int,
        help="ID de la course à prédire"
    )
    group.add_argument(
        "--today",
        action="store_true",
        help="Prédire toutes les courses du jour"
    )
    group.add_argument(
        "--date",
        type=str,
        help="Prédire toutes les courses d'une date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="Chemin vers le modèle ML",
        default="models/horse_racing_model.pkl"
    )

    parser.add_argument(
        "--with-explanations",
        action="store_true",
        help="Inclure les explications SHAP (plus lent)",
        default=False
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Fichier de sortie JSON (optionnel)",
        default=None
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affichage détaillé",
        default=False
    )

    return parser.parse_args()


def print_course_prediction(prediction: dict, verbose: bool = False):
    """Affiche une prédiction de course de manière formatée"""
    print("\n" + "=" * 80)
    print(f"COURSE: {prediction['course_name']}")
    print("=" * 80)
    print(f"Distance: {prediction['course_distance']}m")
    print(f"Discipline: {prediction['course_discipline']}")
    print(f"Nombre de partants: {prediction['number_of_runners']}")
    print(f"Modèle: {prediction['model_version']}")

    print("\n" + "-" * 80)
    print("PRÉDICTIONS (Top 10):")
    print("-" * 80)
    print(f"{'Pos':<4} {'N°':<4} {'Cheval':<30} {'Prob.':<8} {'Conf.':<10} {'Cote PMU':<10}")
    print("-" * 80)

    for i, pred in enumerate(prediction['predictions'][:10], 1):
        odds_str = f"{pred['odds_pmu']:.2f}" if pred['odds_pmu'] else "N/A"
        print(f"{i:<4} {pred['numero_corde']:<4} {pred['horse_name']:<30} "
              f"{pred['probability']:<8.2%} {pred['confidence_level']:<10} {odds_str:<10}")

    print("\n" + "-" * 80)
    print("RECOMMANDATIONS:")
    print("-" * 80)
    rec = prediction['recommendations']

    if rec['gagnant']:
        print(f"Gagnant: N°{rec['gagnant']['numero']} - {rec['gagnant']['horse_name']} "
              f"(prob: {rec['gagnant']['probability']:.2%})")

    if rec['place']:
        place_nums = [f"N°{p['numero']}" for p in rec['place']]
        print(f"Placé (top 3): {', '.join(place_nums)}")

    if rec['tierce']:
        print(f"Tiercé: {'-'.join(map(str, rec['tierce']))}")

    if rec['quarte']:
        print(f"Quarté: {'-'.join(map(str, rec['quarte']))}")

    if rec['quinte']:
        print(f"Quinté: {'-'.join(map(str, rec['quinte']))}")

    # Value bets
    if prediction['value_bets']:
        print("\n" + "-" * 80)
        print("VALUE BETS DÉTECTÉS:")
        print("-" * 80)
        print(f"{'N°':<4} {'Cheval':<30} {'Cote':<8} {'Prob.IA':<10} {'Edge':<10} {'Level':<10}")
        print("-" * 80)
        for vb in prediction['value_bets']:
            print(f"{vb['numero_corde']:<4} {vb['horse_name']:<30} "
                  f"{vb['odds_pmu']:<8.2f} {vb['model_probability']:<10.2%} "
                  f"{vb['edge']:<10.2%} {vb['value_level']:<10}")

    # Explications (si verbose et disponibles)
    if verbose and prediction['predictions'][0].get('explanation'):
        print("\n" + "-" * 80)
        print("EXPLICATION DU FAVORI:")
        print("-" * 80)
        fav = prediction['predictions'][0]
        expl = fav['explanation']

        if expl and 'prediction_impact_summary' in expl:
            print(expl['prediction_impact_summary'])

    print("=" * 80 + "\n")


def main():
    """Fonction principale"""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("PronoTurf Prediction Test Script")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Include explanations: {args.with_explanations}")
    logger.info("=" * 60)

    # Créer une session de base de données
    db = SessionLocal()

    try:
        # Créer le service de prédiction
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            logger.error("Please train a model first using: python scripts/train_initial_model.py")
            return 1

        predictor = RacePredictionService(db, model_path=model_path)

        # Générer les prédictions
        results = None

        if args.course_id:
            logger.info(f"Predicting course {args.course_id}...")
            results = predictor.predict_course(
                course_id=args.course_id,
                include_explanations=args.with_explanations
            )
            print_course_prediction(results, verbose=args.verbose)

        elif args.today:
            logger.info("Predicting today's races...")
            results = predictor.predict_daily_program(
                include_explanations=args.with_explanations
            )
            logger.info(f"Found {results['number_of_races']} races for today")
            for race in results['races']:
                print_course_prediction(race, verbose=args.verbose)

        elif args.date:
            logger.info(f"Predicting races for {args.date}...")
            target_date = date.fromisoformat(args.date)
            results = predictor.predict_daily_program(
                target_date=target_date,
                include_explanations=args.with_explanations
            )
            logger.info(f"Found {results['number_of_races']} races for {args.date}")
            for race in results['races']:
                print_course_prediction(race, verbose=args.verbose)

        # Sauvegarder en JSON si demandé
        if args.output and results:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path.absolute()}")

        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION TEST COMPLETED!")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return 1

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())