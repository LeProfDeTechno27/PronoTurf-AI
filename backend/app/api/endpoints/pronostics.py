"""
Endpoints API pour les pronostics hippiques

Ces endpoints permettent de générer et consulter les prédictions ML
pour les courses hippiques.
"""

import logging
from typing import Optional
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session, selectinload

from app.core.database import get_db
from app.ml.predictor import RacePredictionService
from app.models.pronostic import Pronostic
from app.models.course import Course
from app.models.partant import Partant
from app.models.partant_prediction import PartantPrediction
from app.tasks.ml_tasks import generate_prediction_for_course, generate_daily_predictions

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_value_level(edge: float) -> str:
    """Détermine le niveau de value bet basé sur l'edge calculé."""
    if edge >= 0.3:
        return "high"
    if edge >= 0.2:
        return "medium"
    return "low"


@router.get("/today")
async def get_today_pronostics(
    background_tasks: BackgroundTasks,
    force_regenerate: bool = Query(False, description="Force la régénération des pronostics"),
    db: Session = Depends(get_db)
):
    """
    Récupère les pronostics pour toutes les courses du jour

    Si les pronostics n'existent pas ou si force_regenerate=True,
    lance une tâche Celery pour les générer.
    """
    today = date.today()

    # Vérifier si des pronostics existent déjà pour aujourd'hui
    if not force_regenerate:
        from sqlalchemy import func
        from app.models.reunion import Reunion

        existing_pronostics = (
            db.query(Pronostic)
            .join(Course)
            .join(Reunion)
            .filter(
                Reunion.reunion_date == today,
                func.DATE(Pronostic.generated_at) == today
            )
            .count()
        )

        if existing_pronostics > 0:
            logger.info(f"Found {existing_pronostics} existing pronostics for today")
            # Retourner les pronostics existants
            pronostics = (
                db.query(Pronostic)
                .join(Course)
                .join(Reunion)
                .filter(
                    Reunion.reunion_date == today,
                    func.DATE(Pronostic.generated_at) == today
                )
                .all()
            )

            return {
                "date": today.isoformat(),
                "count": len(pronostics),
                "pronostics": [_format_pronostic(p) for p in pronostics],
                "status": "cached"
            }

    # Lancer la génération en arrière-plan
    logger.info("Triggering background task to generate today's pronostics")
    background_tasks.add_task(
        generate_daily_predictions.delay,
        target_date=today.isoformat()
    )

    return {
        "date": today.isoformat(),
        "status": "generating",
        "message": "Pronostics generation started. Check back in a few minutes."
    }


@router.get("/date/{target_date}")
async def get_pronostics_by_date(
    target_date: str,
    db: Session = Depends(get_db)
):
    """
    Récupère les pronostics pour une date spécifique (YYYY-MM-DD)
    """
    try:
        prediction_date = date.fromisoformat(target_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    from app.models.reunion import Reunion

    pronostics = (
        db.query(Pronostic)
        .join(Course)
        .join(Reunion)
        .filter(Reunion.reunion_date == prediction_date)
        .all()
    )

    if not pronostics:
        raise HTTPException(
            status_code=404,
            detail=f"No pronostics found for {target_date}"
        )

    return {
        "date": target_date,
        "count": len(pronostics),
        "pronostics": [_format_pronostic(p) for p in pronostics]
    }


@router.get("/course/{course_id}")
async def get_pronostic_for_course(
    course_id: int,
    background_tasks: BackgroundTasks,
    force_regenerate: bool = Query(False, description="Force la régénération du pronostic"),
    include_explanations: bool = Query(True, description="Inclure les explications SHAP"),
    db: Session = Depends(get_db)
):
    """
    Récupère ou génère le pronostic pour une course spécifique

    Args:
        course_id: ID de la course
        force_regenerate: Force la régénération
        include_explanations: Inclure les explications SHAP
    """
    # Vérifier que la course existe
    course = db.query(Course).filter(Course.course_id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

    # Chercher un pronostic existant
    if not force_regenerate:
        pronostic = (
            db.query(Pronostic)
            .filter(Pronostic.course_id == course_id)
            .order_by(Pronostic.generated_at.desc())
            .first()
        )

        if pronostic:
            logger.info(f"Found existing pronostic for course {course_id}")
            return {
                "course_id": course_id,
                "pronostic": _format_pronostic(pronostic, include_partants=include_explanations),
                "status": "cached"
            }

    # Générer le pronostic en temps réel
    logger.info(f"Generating pronostic for course {course_id}")

    try:
        predictor = RacePredictionService(db)
        result = predictor.predict_course(
            course_id=course_id,
            include_explanations=include_explanations,
            detect_value_bets=True
        )

        # Sauvegarder dans la base de données en arrière-plan
        background_tasks.add_task(
            generate_prediction_for_course.delay,
            course_id=course_id
        )

        return {
            "course_id": course_id,
            "pronostic": result,
            "status": "fresh"
        }

    except Exception as e:
        logger.error(f"Error generating pronostic for course {course_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating pronostic: {str(e)}"
        )


@router.post("/generate/course/{course_id}")
async def trigger_course_pronostic_generation(
    course_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Déclenche la génération d'un pronostic pour une course (tâche async)

    Cette route lance une tâche Celery et retourne immédiatement.
    Utilisez GET /pronostics/course/{course_id} pour récupérer le résultat.
    """
    # Vérifier que la course existe
    course = db.query(Course).filter(Course.course_id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")

    # Lancer la tâche Celery
    task = generate_prediction_for_course.delay(course_id)

    return {
        "course_id": course_id,
        "task_id": task.id,
        "status": "queued",
        "message": f"Pronostic generation queued for course {course_id}"
    }


@router.post("/generate/daily")
async def trigger_daily_pronostics_generation(
    background_tasks: BackgroundTasks,
    target_date: Optional[str] = Query(None, description="Date cible (YYYY-MM-DD), None = today")
):
    """
    Déclenche la génération de tous les pronostics du jour (tâche async)

    Cette route lance une tâche Celery et retourne immédiatement.
    """
    # Lancer la tâche Celery
    task = generate_daily_predictions.delay(target_date=target_date)

    return {
        "task_id": task.id,
        "status": "queued",
        "date": target_date or date.today().isoformat(),
        "message": "Daily pronostics generation queued"
    }


@router.get("/value-bets/today")
async def get_today_value_bets(
    min_edge: float = Query(0.1, description="Edge minimum (0.0-1.0)", ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    """
    Récupère tous les value bets détectés pour aujourd'hui

    Args:
        min_edge: Edge minimum pour filtrer les value bets (défaut: 0.1 = 10%)
    """
    from app.models.reunion import Reunion

    today = date.today()

    # Récupérer les pronostics du jour avec leurs prédictions de partants
    pronostics = (
        db.query(Pronostic)
        .join(Course)
        .join(Reunion)
        .filter(Reunion.reunion_date == today)
        .options(
            selectinload(Pronostic.course)
            .selectinload(Course.reunion)
            .selectinload(Reunion.hippodrome),
            selectinload(Pronostic.partant_predictions)
            .selectinload(PartantPrediction.partant)
            .selectinload(Partant.horse),
            selectinload(Pronostic.partant_predictions)
            .selectinload(PartantPrediction.partant)
            .selectinload(Partant.jockey),
            selectinload(Pronostic.partant_predictions)
            .selectinload(PartantPrediction.partant)
            .selectinload(Partant.trainer),
        )
        .all()
    )

    courses_with_value_bets = []
    total_value_bets = 0

    for pronostic in pronostics:
        course = pronostic.course
        if not course:
            continue

        course_value_bets = []
        for partant_prediction in pronostic.partant_predictions or []:
            if partant_prediction.win_probability is None:
                continue

            partant = partant_prediction.partant
            if not partant or not partant.odds_pmu:
                continue

            odds = float(partant.odds_pmu)
            if odds <= 1.0:
                continue

            model_probability = float(partant_prediction.win_probability)
            implied_probability = 1.0 / odds
            edge = model_probability - implied_probability

            if edge < min_edge:
                continue

            edge_percentage = edge / implied_probability if implied_probability > 0 else None

            course_value_bets.append({
                "partant_id": partant.partant_id,
                "numero_corde": partant.numero_corde,
                "horse_name": partant.horse.name if partant.horse else None,
                "jockey_name": partant.jockey_name,
                "trainer_name": partant.trainer_name,
                "odds_pmu": odds,
                "model_probability": model_probability,
                "implied_probability": implied_probability,
                "edge": edge,
                "edge_percentage": edge_percentage,
                "value_level": _get_value_level(edge),
                "confidence_level": partant_prediction.confidence_level,
            })

        if not course_value_bets:
            continue

        course_value_bets.sort(key=lambda x: x["edge"], reverse=True)
        total_value_bets += len(course_value_bets)

        reunion = course.reunion
        hippodrome_name = None
        reunion_number = None
        if reunion:
            reunion_number = reunion.reunion_number
            if reunion.hippodrome:
                hippodrome_name = reunion.hippodrome.name

        courses_with_value_bets.append({
            "course_id": course.course_id,
            "course_name": course.full_name if hasattr(course, "full_name") else course.course_name,
            "reunion_id": course.reunion_id,
            "reunion_number": reunion_number,
            "hippodrome": hippodrome_name,
            "scheduled_time": course.scheduled_time.isoformat() if course.scheduled_time else None,
            "value_bets": course_value_bets,
        })

    courses_with_value_bets.sort(
        key=lambda item: (
            item.get("reunion_number") or 0,
            item.get("scheduled_time") or ""
        )
    )

    return {
        "date": today.isoformat(),
        "min_edge": min_edge,
        "total_courses": len(courses_with_value_bets),
        "count": total_value_bets,
        "courses": courses_with_value_bets
    }


@router.get("/stats/accuracy")
async def get_model_accuracy_stats(
    days_back: int = Query(7, description="Nombre de jours en arrière", ge=1, le=90),
    db: Session = Depends(get_db)
):
    """
    Récupère les statistiques de précision du modèle

    Compare les prédictions du modèle avec les résultats réels
    sur les N derniers jours.
    """
    from datetime import timedelta
    from app.models.reunion import Reunion
    from app.models.partant import Partant
    from app.models.partant_prediction import PartantPrediction

    cutoff_date = date.today() - timedelta(days=days_back)

    # Récupérer les prédictions avec résultats réels
    predictions_with_results = (
        db.query(PartantPrediction, Partant)
        .join(Partant, PartantPrediction.partant_id == Partant.partant_id)
        .join(Pronostic, PartantPrediction.pronostic_id == Pronostic.pronostic_id)
        .join(Course, Pronostic.course_id == Course.course_id)
        .join(Reunion, Course.reunion_id == Reunion.reunion_id)
        .filter(
            Reunion.reunion_date >= cutoff_date,
            Partant.final_position.isnot(None)  # Seulement les courses terminées
        )
        .all()
    )

    if not predictions_with_results:
        return {
            "days_evaluated": days_back,
            "total_predictions": 0,
            "message": "No predictions with results found in this period"
        }

    # Calculer les métriques
    total = len(predictions_with_results)
    correct_top3 = sum(
        1 for pred, partant in predictions_with_results
        if partant.final_position <= 3 and pred.win_probability >= 0.3
    )

    # Statistiques basiques
    stats = {
        "days_evaluated": days_back,
        "total_predictions": total,
        "correct_top3_predictions": correct_top3,
        "accuracy": correct_top3 / total if total > 0 else 0,
        "period_start": cutoff_date.isoformat(),
        "period_end": date.today().isoformat()
    }

    return stats


def _format_pronostic(pronostic: Pronostic, include_partants: bool = False) -> dict:
    """
    Formate un pronostic pour l'API

    Args:
        pronostic: Instance de Pronostic
        include_partants: Inclure les prédictions détaillées des partants

    Returns:
        Dictionnaire formaté
    """
    import json

    result = {
        "pronostic_id": pronostic.pronostic_id,
        "course_id": pronostic.course_id,
        "model_version": pronostic.model_version,
        "generated_at": pronostic.generated_at.isoformat(),
        "confidence_score": float(pronostic.confidence_score) if pronostic.confidence_score else None,
        "value_bet_detected": pronostic.value_bet_detected,
        "recommendations": {
            "gagnant": json.loads(pronostic.gagnant_predicted) if pronostic.gagnant_predicted else None,
            "place": json.loads(pronostic.place_predicted) if pronostic.place_predicted else None,
            "tierce": json.loads(pronostic.tierce_predicted) if pronostic.tierce_predicted else None,
            "quarte": json.loads(pronostic.quarte_predicted) if pronostic.quarte_predicted else None,
            "quinte": json.loads(pronostic.quinte_predicted) if pronostic.quinte_predicted else None,
        }
    }

    if include_partants and pronostic.partant_predictions:
        result["partant_predictions"] = [
            {
                "partant_id": pp.partant_id,
                "win_probability": float(pp.win_probability) if pp.win_probability else None,
                "confidence_level": pp.confidence_level,
                "top_positive_features": json.loads(pp.top_positive_features) if pp.top_positive_features else [],
                "top_negative_features": json.loads(pp.top_negative_features) if pp.top_negative_features else [],
            }
            for pp in sorted(pronostic.partant_predictions, key=lambda x: x.win_probability or 0, reverse=True)
        ]

    return result
