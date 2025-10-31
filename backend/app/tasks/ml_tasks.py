"""
Tâches Celery pour le Machine Learning et les prédictions
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.core.database import SessionLocal
from app.ml.predictor import RacePredictionService
from app.ml.training import ModelTrainer
from app.models.course import Course
from app.models.reunion import Reunion
from app.models.pronostic import Pronostic
from app.models.partant_prediction import PartantPrediction

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def generate_daily_predictions(self, target_date: Optional[str] = None):
    """
    Génère les prédictions pour toutes les courses du jour

    Args:
        target_date: Date cible au format YYYY-MM-DD (None = aujourd'hui)

    Returns:
        Dictionnaire avec le statut et les statistiques
    """
    db = SessionLocal()

    try:
        logger.info("Starting daily predictions generation")

        # Parser la date
        if target_date:
            prediction_date = date.fromisoformat(target_date)
        else:
            prediction_date = date.today()

        logger.info(f"Generating predictions for {prediction_date}")

        # Créer le service de prédiction
        predictor = RacePredictionService(db)

        # Générer les prédictions pour tout le programme
        result = predictor.predict_daily_program(
            target_date=prediction_date,
            include_explanations=True
        )

        # Sauvegarder les prédictions dans la base de données
        saved_count = _save_predictions_to_db(db, result['races'])

        logger.info(f"Daily predictions generated successfully: {saved_count} races")

        return {
            "status": "success",
            "date": prediction_date.isoformat(),
            "races_predicted": len(result['races']),
            "predictions_saved": saved_count,
            "generated_at": result['generated_at']
        }

    except Exception as e:
        logger.error(f"Error generating daily predictions: {e}", exc_info=True)
        # Retry avec backoff exponentiel
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))

    finally:
        db.close()


@celery_app.task(bind=True)
def train_ml_model(
    self,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 5
):
    """
    Entraîne le modèle ML avec les nouvelles données

    Args:
        min_date: Date minimale (YYYY-MM-DD)
        max_date: Date maximale (YYYY-MM-DD)
        n_estimators: Nombre d'arbres
        learning_rate: Taux d'apprentissage
        max_depth: Profondeur maximale

    Returns:
        Dictionnaire avec les métriques de performance
    """
    db = SessionLocal()

    try:
        logger.info("Starting ML model training")
        logger.info(f"Date range: {min_date} to {max_date}")

        # Créer le trainer
        output_path = Path("models/horse_racing_model.pkl")
        trainer = ModelTrainer(db, model_save_path=output_path)

        # Entraîner le modèle
        metrics = trainer.train_new_model(
            min_date=min_date,
            max_date=max_date,
            test_size=0.2,
            include_odds=False,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth
        )

        # Sauvegarder les métriques dans la base
        _save_training_metrics_to_db(db, metrics, output_path)

        logger.info("ML model training completed successfully")

        return {
            "status": "success",
            "metrics": metrics,
            "model_path": str(output_path.absolute())
        }

    except Exception as e:
        logger.error(f"Error training ML model: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }

    finally:
        db.close()


@celery_app.task
def update_model_performance(days_back: int = 7):
    """
    Met à jour les métriques de performance du modèle ML
    en évaluant ses prédictions sur les courses récentes

    Args:
        days_back: Nombre de jours en arrière pour l'évaluation

    Returns:
        Dictionnaire avec les métriques de performance
    """
    db = SessionLocal()

    try:
        logger.info(f"Updating model performance metrics (last {days_back} days)")

        # Récupérer les pronostics et résultats réels
        from datetime import timedelta

        cutoff_date = date.today() - timedelta(days=days_back)

        # Comparer prédictions vs résultats réels
        # TODO: Implémenter l'évaluation détaillée

        logger.info("Model performance updated successfully")

        return {
            "status": "success",
            "days_evaluated": days_back,
            "updated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error updating model performance: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }

    finally:
        db.close()


@celery_app.task(bind=True, max_retries=3)
def generate_prediction_for_course(self, course_id: int):
    """
    Génère une prédiction pour une course spécifique

    Args:
        course_id: ID de la course

    Returns:
        Dictionnaire avec la prédiction générée
    """
    db = SessionLocal()

    try:
        logger.info(f"Generating prediction for course {course_id}")

        # Créer le service de prédiction
        predictor = RacePredictionService(db)

        # Générer la prédiction
        result = predictor.predict_course(
            course_id=course_id,
            include_explanations=True,
            detect_value_bets=True
        )

        # Sauvegarder la prédiction
        saved = _save_predictions_to_db(db, [result])

        logger.info(f"Prediction for course {course_id} generated successfully")

        return {
            "status": "success",
            "course_id": course_id,
            "prediction_saved": saved > 0,
            "result": result
        }

    except Exception as e:
        logger.error(f"Error generating prediction for course {course_id}: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=30 * (2 ** self.request.retries))

    finally:
        db.close()


def _save_predictions_to_db(db: Session, race_predictions: list) -> int:
    """
    Sauvegarde les prédictions dans la base de données

    Args:
        db: Session de base de données
        race_predictions: Liste des prédictions de courses

    Returns:
        Nombre de prédictions sauvegardées
    """
    import json

    saved_count = 0

    for race_pred in race_predictions:
        try:
            # Créer le pronostic pour la course
            pronostic = Pronostic(
                course_id=race_pred['course_id'],
                model_version=race_pred.get('model_version', 'unknown'),
                confidence_score=race_pred['predictions'][0]['probability'] * 100 if race_pred['predictions'] else 0,
                value_bet_detected=len(race_pred.get('value_bets', [])) > 0,
                gagnant_predicted=json.dumps(race_pred['recommendations']['gagnant']),
                place_predicted=json.dumps(race_pred['recommendations']['place']),
                tierce_predicted=json.dumps(race_pred['recommendations']['tierce']),
                quarte_predicted=json.dumps(race_pred['recommendations']['quarte']),
                quinte_predicted=json.dumps(race_pred['recommendations']['quinte']),
            )

            db.add(pronostic)
            db.flush()  # Pour obtenir l'ID

            # Sauvegarder les prédictions individuelles des partants
            for pred in race_pred['predictions']:
                partant_pred = PartantPrediction(
                    pronostic_id=pronostic.pronostic_id,
                    partant_id=pred['partant_id'],
                    win_probability=pred['probability'],
                    confidence_level=pred['confidence_level'],
                    shap_values=json.dumps(pred.get('explanation', {}).get('shap_values', {})) if pred.get('explanation') else None,
                    shap_contributions=json.dumps(pred.get('explanation', {})) if pred.get('explanation') else None,
                    top_positive_features=json.dumps(pred.get('explanation', {}).get('top_positive_features', [])) if pred.get('explanation') else None,
                    top_negative_features=json.dumps(pred.get('explanation', {}).get('top_negative_features', [])) if pred.get('explanation') else None,
                )
                db.add(partant_pred)

            db.commit()
            saved_count += 1

        except Exception as e:
            logger.error(f"Error saving prediction for course {race_pred['course_id']}: {e}")
            db.rollback()
            continue

    return saved_count


def _save_training_metrics_to_db(db: Session, metrics: dict, model_path: Path):
    """
    Sauvegarde les métriques d'entraînement dans la base de données

    Args:
        db: Session de base de données
        metrics: Dictionnaire des métriques
        model_path: Chemin du modèle
    """
    import json
    from app.models.ml_model import MLModel

    try:
        ml_model = MLModel(
            model_name="horse_racing_gradient_boosting",
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            algorithm="GradientBoosting",
            file_path=str(model_path),
            performance_metrics=json.dumps(metrics),
            features_used=json.dumps(metrics.get('feature_names', [])),
            is_active=True
        )

        # Désactiver les anciens modèles
        db.query(MLModel).update({"is_active": False})

        db.add(ml_model)
        db.commit()

        logger.info(f"Training metrics saved to database for model version {ml_model.version}")

    except Exception as e:
        logger.error(f"Error saving training metrics to database: {e}")
        db.rollback()
