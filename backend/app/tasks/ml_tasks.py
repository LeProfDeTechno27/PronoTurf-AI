"""Tâches Celery pour le Machine Learning et les prédictions."""

import json
import logging
from collections import Counter
from datetime import date, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.core.database import SessionLocal
from app.ml.predictor import RacePredictionService
from app.ml.training import ModelTrainer
from app.models.course import Course, CourseStatus
from app.models.reunion import Reunion
from app.models.pronostic import Pronostic
from app.models.partant_prediction import PartantPrediction

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
except Exception:  # pragma: no cover - defensive import guard
    accuracy_score = precision_score = recall_score = f1_score = roc_auc_score = log_loss = None
    average_precision_score = precision_recall_curve = None
    roc_curve = None
    confusion_matrix = None

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


def _safe_average(values: List[float]) -> Optional[float]:
    """Retourne la moyenne d'une liste ou ``None`` si elle est vide."""

    return sum(values) / len(values) if values else None


def _build_calibration_table(
    scores: List[float],
    truths: List[int],
    *,
    bins: int = 5,
) -> List[Dict[str, object]]:
    """Construit un tableau de calibration par quantiles.

    L'objectif est d'exposer à la fois le volume de prédictions par tranche et
    l'écart éventuel entre probabilité moyenne et fréquence observée.
    """

    if not scores or not truths or len(scores) != len(truths):
        return []

    combined = sorted(zip(scores, truths), key=lambda item: item[0])
    bucket_size = max(1, len(combined) // bins)

    calibration_rows: List[Dict[str, object]] = []

    for idx in range(bins):
        start = idx * bucket_size
        end = (idx + 1) * bucket_size if idx < bins - 1 else len(combined)

        if start >= len(combined):
            break

        bucket = combined[start:end]
        bucket_scores = [item[0] for item in bucket]
        bucket_truths = [item[1] for item in bucket]

        calibration_rows.append(
            {
                "bin": idx + 1,
                "count": len(bucket),
                "min_probability": min(bucket_scores),
                "max_probability": max(bucket_scores),
                "average_probability": _safe_average(bucket_scores),
                "empirical_rate": _safe_average(bucket_truths),
            }
        )

    return calibration_rows


def _describe_calibration_quality(
    calibration_rows: List[Dict[str, object]]
) -> Dict[str, Optional[float]]:
    """Synthétise les écarts de calibration observés sur les quantiles."""

    if not calibration_rows:
        return {
            "expected_calibration_error": None,
            "maximum_calibration_gap": None,
            "weighted_bias": None,
            "bins": [],
        }

    total = sum(int(row.get("count", 0) or 0) for row in calibration_rows)
    if not total:
        return {
            "expected_calibration_error": None,
            "maximum_calibration_gap": None,
            "weighted_bias": None,
            "bins": [],
        }

    expected_error = 0.0
    weighted_bias = 0.0
    max_gap = 0.0
    enriched_bins: List[Dict[str, object]] = []

    for row in calibration_rows:
        count = int(row.get("count", 0) or 0)
        weight = count / total if total else 0.0
        average_probability = row.get("average_probability")
        empirical_rate = row.get("empirical_rate")

        # Le « gap » correspond à la différence entre probabilité estimée et fréquence
        # observée : une valeur positive indique que le modèle est trop conservateur,
        # une valeur négative qu'il est trop confiant.
        gap: Optional[float] = None
        if average_probability is not None and empirical_rate is not None:
            gap = empirical_rate - average_probability
            expected_error += weight * abs(gap)
            weighted_bias += weight * gap
            max_gap = max(max_gap, abs(gap))

        enriched_bins.append(
            {
                **row,
                "weight": weight,
                "calibration_gap": gap,
            }
        )

    return {
        "expected_calibration_error": expected_error,
        "maximum_calibration_gap": max_gap,
        "weighted_bias": weighted_bias,
        "bins": enriched_bins,
    }


def _build_gain_curve(
    scores: List[float],
    truths: List[int],
    *,
    steps: int = 5,
) -> List[Dict[str, Optional[float]]]:
    """Construit une courbe de gain cumulative sur plusieurs paliers.

    L'objectif est de mesurer la capacité du modèle à concentrer rapidement
    les bons partants (top 3) lorsqu'on ne retient que les meilleures
    probabilités. Chaque ligne représente la performance cumulée après avoir
    couvert ``coverage`` pourcent des partants.
    """

    if not scores or not truths or len(scores) != len(truths):
        return []

    combined = sorted(zip(scores, truths), key=lambda item: item[0], reverse=True)
    total = len(combined)
    total_positive = sum(truths)

    gain_curve: List[Dict[str, Optional[float]]] = []
    cumulative_hits = 0

    for step in range(1, steps + 1):
        cutoff = max(1, ceil(total * (step / steps)))
        selection = combined[:cutoff]
        cumulative_hits = sum(truth for _, truth in selection)

        coverage = cutoff / total
        cumulative_hit_rate = cumulative_hits / cutoff if cutoff else None
        capture_rate: Optional[float] = None
        if total_positive:
            capture_rate = cumulative_hits / total_positive

        gain_curve.append(
            {
                "step": step,
                "coverage": coverage,
                "observations": cutoff,
                "cumulative_hit_rate": cumulative_hit_rate,
                "capture_rate": capture_rate,
            }
        )

    return gain_curve


def _build_precision_recall_curve(
    scores: List[float],
    truths: List[int],
    *,
    sample_points: int = 8,
) -> List[Dict[str, Optional[float]]]:
    """Construit une table compacte de la courbe précision-rappel."""

    if (
        not scores
        or not truths
        or len(scores) != len(truths)
        or precision_recall_curve is None
    ):
        return []

    precision, recall, thresholds = precision_recall_curve(truths, scores)

    if len(thresholds) == 0:
        return []

    curve: List[Dict[str, Optional[float]]] = []

    for idx, threshold in enumerate(list(thresholds)):
        current_precision = float(precision[idx + 1])
        current_recall = float(recall[idx + 1])
        denom = current_precision + current_recall
        f1_score_value = (2 * current_precision * current_recall / denom) if denom else 0.0
        curve.append(
            {
                "threshold": float(threshold),
                "precision": current_precision,
                "recall": current_recall,
                "f1": f1_score_value,
            }
        )

    # Ajoute le point terminal (tous positifs) pour compléter la courbe.
    end_precision = float(precision[-1])
    end_recall = float(recall[-1])
    denom = end_precision + end_recall
    curve.append(
        {
            "threshold": 0.0,
            "precision": end_precision,
            "recall": end_recall,
            "f1": (2 * end_precision * end_recall / denom) if denom else 0.0,
        }
    )

    if len(curve) > sample_points:
        step = max(1, len(curve) // sample_points)
        reduced = [curve[idx] for idx in range(0, len(curve), step)]
        if reduced[-1] != curve[-1]:
            reduced.append(curve[-1])
        curve = reduced[:sample_points]

    return curve


def _build_roc_curve(
    scores: List[float],
    truths: List[int],
    *,
    sample_points: int = 12,
) -> List[Dict[str, Optional[float]]]:
    """Échantillonne la courbe ROC pour suivre le compromis rappel/spécificité."""

    if (
        not scores
        or not truths
        or len(scores) != len(truths)
        or len(set(truths)) < 2
        or roc_curve is None
    ):
        return []

    false_positive_rate, true_positive_rate, thresholds = roc_curve(truths, scores)

    if len(thresholds) == 0:
        return []

    total_points = len(thresholds)
    step = max(1, total_points // sample_points)
    sampled_indices = list(range(0, total_points, step))
    if sampled_indices[-1] != total_points - 1:
        sampled_indices.append(total_points - 1)

    roc_points: List[Dict[str, Optional[float]]] = []

    for idx in sampled_indices:
        threshold_value = thresholds[idx]
        # Le premier seuil retourné par scikit-learn est ``inf`` : on le remplace
        # par ``None`` pour indiquer qu'aucune coupure n'est appliquée.
        if threshold_value == float("inf"):
            threshold: Optional[float] = None
        else:
            threshold = float(threshold_value)

        fpr_value = float(false_positive_rate[idx])
        tpr_value = float(true_positive_rate[idx])

        roc_points.append(
            {
                "threshold": threshold,
                "false_positive_rate": fpr_value,
                "true_positive_rate": tpr_value,
                # Youden J pour identifier le meilleur seuil (TPR - FPR).
                "youden_j": tpr_value - fpr_value,
                "specificity": 1.0 - fpr_value,
            }
        )

    return roc_points


def _build_lift_table(
    scores: List[float],
    truths: List[int],
    *,
    buckets: int = 5,
) -> Dict[str, object]:
    """Construit un tableau de *lift* pour comparer chaque tranche au taux global."""

    if not scores or not truths or len(scores) != len(truths):
        return {"baseline_rate": None, "buckets": []}

    combined = sorted(zip(scores, truths), key=lambda item: item[0], reverse=True)
    total = len(combined)
    total_positive = sum(truths)
    baseline_rate = total_positive / total if total else 0.0

    bucket_size = max(1, total // buckets)
    buckets_rows: List[Dict[str, Optional[float]]] = []
    cumulative_positive = 0

    for idx in range(buckets):
        start = idx * bucket_size
        end = (idx + 1) * bucket_size if idx < buckets - 1 else total

        if start >= total:
            break

        bucket = combined[start:end]
        bucket_total = len(bucket)
        bucket_positive = sum(truth for _, truth in bucket)
        hit_rate: Optional[float] = None
        if bucket_total:
            hit_rate = bucket_positive / bucket_total

        cumulative_positive += bucket_positive

        buckets_rows.append(
            {
                "bucket": idx + 1,
                "from_fraction": start / total,
                "to_fraction": end / total,
                "observations": bucket_total,
                "hit_rate": hit_rate,
                "lift": (hit_rate / baseline_rate) if hit_rate is not None and baseline_rate > 0 else None,
                "cumulative_capture": (
                    cumulative_positive / total_positive if total_positive else None
                ),
                "cumulative_coverage": end / total,
            }
        )

    return {"baseline_rate": baseline_rate if total else None, "buckets": buckets_rows}


def _compute_ks_analysis(
    scores: List[float],
    truths: List[int],
    *,
    sample_points: int = 20,
) -> Dict[str, object]:
    """Mesure la séparation des distributions via le test KS discret.

    Le calcul reporte à la fois la statistique (distance maximale entre les
    distributions cumulées des positifs et négatifs) et une version compacte de
    la courbe pour visualiser rapidement les écarts. Cette vue complète la
    calibration : un modèle bien calibré mais incapable de séparer les classes
    sera pénalisé par une statistique KS faible.
    """

    if not scores or not truths or len(scores) != len(truths):
        return {"ks_statistic": None, "ks_threshold": None, "curve": []}

    total_positive = sum(truths)
    total_negative = len(truths) - total_positive

    if total_positive == 0 or total_negative == 0:
        # Dans ces cas extrêmes, la statistique KS est peu informative : on
        # retourne des valeurs nulles tout en conservant la structure attendue.
        return {"ks_statistic": None, "ks_threshold": None, "curve": []}

    combined = sorted(zip(scores, truths), key=lambda item: item[0], reverse=True)

    ks_statistic = 0.0
    ks_threshold: Optional[float] = None
    curve: List[Dict[str, float]] = []

    positives_seen = 0
    negatives_seen = 0
    step = max(1, len(combined) // sample_points)

    for index, (score, truth) in enumerate(combined, start=1):
        if truth:
            positives_seen += 1
        else:
            negatives_seen += 1

        true_positive_rate = positives_seen / total_positive
        false_positive_rate = negatives_seen / total_negative
        distance = abs(true_positive_rate - false_positive_rate)

        if distance >= ks_statistic:
            ks_statistic = distance
            ks_threshold = score

        if index % step == 0 or index == len(combined):
            curve.append(
                {
                    "fraction": index / len(combined),
                    "threshold": score,
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                    "distance": distance,
                }
            )

    return {
        "ks_statistic": ks_statistic,
        "ks_threshold": ks_threshold,
        "curve": curve,
    }


def _evaluate_threshold_grid(
    scores: List[float],
    truths: List[int],
    thresholds: List[float],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Calcule la sensibilité des métriques pour plusieurs seuils."""

    if not scores or not truths:
        return {}

    evaluation: Dict[str, Dict[str, Optional[float]]] = {}

    for threshold in sorted(set(thresholds)):
        predicted = [1 if score >= threshold else 0 for score in scores]

        accuracy = precision = recall = f1 = None
        if accuracy_score:
            accuracy = accuracy_score(truths, predicted)
            precision = precision_score(truths, predicted, zero_division=0)
            recall = recall_score(truths, predicted, zero_division=0)
            f1 = f1_score(truths, predicted, zero_division=0)

        evaluation[f"{threshold:.2f}"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positive_rate": sum(predicted) / len(predicted) if predicted else None,
        }

    return evaluation


def _summarise_group_performance(
    truths: List[int],
    predicted: List[int],
    scores: List[float],
) -> Dict[str, Optional[float]]:
    """Assemble un petit tableau de bord de métriques pour un sous-ensemble donné."""

    summary: Dict[str, Optional[float]] = {
        "samples": len(truths),
        "positive_rate": sum(predicted) / len(predicted) if predicted else None,
        "average_probability": _safe_average(scores),
    }

    if not truths or not predicted:
        summary.update({
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        })
        return summary

    if accuracy_score:
        summary.update(
            {
                "accuracy": accuracy_score(truths, predicted),
                "precision": precision_score(truths, predicted, zero_division=0),
                "recall": recall_score(truths, predicted, zero_division=0),
                "f1": f1_score(truths, predicted, zero_division=0),
            }
        )
    else:  # pragma: no cover - dépend de l'environnement
        summary.update({
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        })

    return summary


def _coerce_metrics(payload: Optional[object]) -> Dict[str, object]:
    """Convertit un champ JSON éventuel en dictionnaire python."""

    if not payload:
        return {}

    if isinstance(payload, dict):
        return dict(payload)

    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"raw": payload}

    return {"raw": payload}


@celery_app.task
def update_model_performance(days_back: int = 7, probability_threshold: float = 0.3):
    """Évalue les prédictions récentes et met à jour les métriques du modèle.

    Args:
        days_back: Nombre de jours d'historique à prendre en compte.
        probability_threshold: Seuil utilisé pour convertir une probabilité en label.

    Returns:
        Un dictionnaire détaillant les métriques calculées.
    """

    db = SessionLocal()

    try:
        logger.info(
            "Updating model performance metrics (last %s days, threshold=%s)",
            days_back,
            probability_threshold,
        )

        from app.models.partant import Partant
        from app.models.reunion import Reunion
        from app.models.ml_model import MLModel

        cutoff_date = date.today() - timedelta(days=days_back)
        cutoff_datetime = datetime.combine(cutoff_date, datetime.min.time())

        predictions_with_results: List[Tuple[PartantPrediction, Partant, Pronostic, Course]] = (
            db.query(PartantPrediction, Partant, Pronostic, Course)
            .join(Partant, PartantPrediction.partant_id == Partant.partant_id)
            .join(Pronostic, PartantPrediction.pronostic_id == Pronostic.pronostic_id)
            .join(Course, Pronostic.course_id == Course.course_id)
            .join(Reunion, Course.reunion_id == Reunion.reunion_id)
            .filter(
                Reunion.reunion_date >= cutoff_date,
                Partant.final_position.isnot(None),
                Partant.disqualified.isnot(True),
                PartantPrediction.win_probability.isnot(None),
                Course.status == CourseStatus.FINISHED,
                Pronostic.generated_at >= cutoff_datetime,
            )
            .all()
        )

        if not predictions_with_results:
            logger.warning("No predictions with results found for evaluation window")
            return {
                "status": "no_data",
                "days_evaluated": days_back,
                "cutoff_date": cutoff_date.isoformat(),
                "evaluated_samples": 0,
                "message": "No predictions with associated race results in the given window",
            }

        y_true: List[int] = []
        y_scores: List[float] = []
        y_pred: List[int] = []
        confidence_counter: Counter[str] = Counter()
        confidence_breakdown: Dict[str, Dict[str, List[float]]] = {}
        model_versions: Counter[str] = Counter()
        course_stats: Dict[int, Dict[str, object]] = {}

        # Parcourt chaque pronostic couplé à un résultat officiel pour préparer les listes
        # nécessaires aux métriques (labels réels, scores, version du modèle, etc.).
        for prediction, partant, pronostic, course in predictions_with_results:
            probability = float(prediction.win_probability)
            probability = max(0.0, min(probability, 1.0))
            is_top3 = 1 if partant.final_position and partant.final_position <= 3 else 0
            predicted_label = 1 if probability >= probability_threshold else 0

            y_true.append(is_top3)
            y_scores.append(probability)
            y_pred.append(predicted_label)

            confidence_counter[prediction.confidence_level or "unknown"] += 1
            level_bucket = confidence_breakdown.setdefault(
                prediction.confidence_level or "unknown",
                {"truths": [], "predictions": [], "scores": []},
            )
            level_bucket["truths"].append(is_top3)
            level_bucket["predictions"].append(predicted_label)
            level_bucket["scores"].append(probability)
            model_versions[pronostic.model_version or "unknown"] += 1

            course_entry = course_stats.setdefault(
                course.course_id,
                {
                    "predictions": [],
                    "value_bet_detected": bool(pronostic.value_bet_detected),
                },
            )

            course_entry["predictions"].append(
                {
                    "probability": probability,
                    "final_position": partant.final_position,
                    "is_top3": bool(is_top3),
                }
            )

        evaluation_timestamp = datetime.now().isoformat()

        accuracy = precision = recall = f1 = roc_auc = logloss = None
        if accuracy_score:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if len(set(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, y_scores)
            clipped_scores = [min(max(score, 1e-6), 1 - 1e-6) for score in y_scores]
            try:
                logloss = log_loss(y_true, clipped_scores)
            except ValueError:
                logloss = None

        brier_score = sum((score - truth) ** 2 for score, truth in zip(y_scores, y_true)) / len(y_true)

        cm = [
            [0, 0],
            [0, 0],
        ]
        if confusion_matrix:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

        positives = sum(y_pred)
        negatives = len(y_pred) - positives

        avg_positive_prob = _safe_average([score for score, label in zip(y_scores, y_pred) if label == 1])
        avg_negative_prob = _safe_average([score for score, label in zip(y_scores, y_pred) if label == 0])

        course_count = len(course_stats)
        top1_correct = 0
        top3_course_hits = 0
        winner_probabilities: List[float] = []
        top3_probabilities: List[float] = []

        for data in course_stats.values():
            predictions: List[Dict[str, object]] = data["predictions"]  # type: ignore[assignment]
            sorted_predictions = sorted(predictions, key=lambda item: item["probability"], reverse=True)
            if not sorted_predictions:
                continue

            winner_entry = next((item for item in sorted_predictions if item["final_position"] == 1), None)
            if winner_entry:
                winner_probabilities.append(float(winner_entry["probability"]))

            top1 = sorted_predictions[0]
            if top1.get("final_position") == 1:
                top1_correct += 1

            top3_predictions = sorted_predictions[:3]
            if any(item.get("final_position") and int(item["final_position"]) <= 3 for item in top3_predictions):
                top3_course_hits += 1

            for item in sorted_predictions:
                if item.get("final_position") and int(item["final_position"]) <= 3:
                    top3_probabilities.append(float(item["probability"]))

        calibration_table = _build_calibration_table(y_scores, y_true, bins=5)
        # Résume l'ampleur des écarts de calibration pour suivre un indicateur
        # synthétique (ECE, biais signé, écart maximal) en plus du tableau brut.
        calibration_diagnostics = _describe_calibration_quality(calibration_table)
        threshold_grid = _evaluate_threshold_grid(
            y_scores,
            y_true,
            thresholds=[0.2, probability_threshold, 0.4, 0.5],
        )

        # Fournit une vision cumulative du gain : en ne conservant que les
        # meilleures probabilités, quelle part des arrivées dans les 3 est
        # capturée ? Cette courbe complète la calibration en évaluant la
        # puissance de tri du modèle.
        gain_curve = _build_gain_curve(
            y_scores,
            y_true,
            steps=5,
        )

        # Construit un tableau de lift : chaque tranche de probabilité est
        # comparée au taux de réussite moyen pour visualiser rapidement la
        # surperformance (ou sous-performance) des segments prioritaires.
        lift_analysis = _build_lift_table(
            y_scores,
            y_true,
            buckets=5,
        )

        # Trace la courbe précision-rappel pour suivre la capacité du modèle à
        # maintenir une précision élevée lorsque l'on pousse le rappel. Utile
        # pour les opérateurs qui doivent choisir un compromis précision/rappel
        # selon leur tolérance au risque.
        precision_recall_table = _build_precision_recall_curve(
            y_scores,
            y_true,
            sample_points=8,
        )
        average_precision = (
            float(average_precision_score(y_true, y_scores))
            if average_precision_score is not None
            else None
        )

        # Échantillonne la courbe ROC pour exposer la progression du rappel au
        # fur et à mesure que l'on accepte davantage de faux positifs. Cette
        # vue complète la précision-rappel en fournissant la spécificité.
        roc_curve_points = _build_roc_curve(
            y_scores,
            y_true,
            sample_points=12,
        )

        # Mesure la séparation effective entre gagnants et perdants via une
        # statistique de Kolmogorov-Smirnov. Utile pour identifier un seuil
        # discriminant même si les métriques globales semblent correctes.
        ks_analysis = _compute_ks_analysis(
            y_scores,
            y_true,
            sample_points=10,
        )

        # Consolide un tableau de bord par niveau de confiance afin d'inspecter
        # rapidement la fiabilité réelle de chaque segment (utile pour piloter
        # alertes ou limites d'enjeux par exemple).
        confidence_level_metrics = {
            level: _summarise_group_performance(
                data["truths"],
                data["predictions"],
                data["scores"],
            )
            for level, data in sorted(confidence_breakdown.items())
        }

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "log_loss": logloss,
            "brier_score": brier_score,
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1],
            },
            "positive_prediction_rate": positives / len(y_pred) if y_pred else 0.0,
            "average_positive_probability": avg_positive_prob,
            "average_negative_probability": avg_negative_prob,
            "top1_accuracy": top1_correct / course_count if course_count else None,
            "course_top3_hit_rate": top3_course_hits / course_count if course_count else None,
            "average_winner_probability": _safe_average(winner_probabilities),
            "average_top3_probability": _safe_average(top3_probabilities),
            "calibration_table": calibration_table,
            "calibration_diagnostics": calibration_diagnostics,
            "threshold_sensitivity": threshold_grid,
            "gain_curve": gain_curve,
            "lift_analysis": lift_analysis,
            "average_precision": average_precision,
            "precision_recall_curve": precision_recall_table,
            "roc_curve": roc_curve_points,
            "ks_analysis": ks_analysis,
            "confidence_level_metrics": confidence_level_metrics,
        }

        confidence_distribution = {
            level: confidence_counter[level]
            for level in sorted(confidence_counter.keys())
        }

        evaluation_summary = {
            "timestamp": evaluation_timestamp,
            "days_back": days_back,
            "probability_threshold": probability_threshold,
            "samples": len(y_true),
            "courses": course_count,
            "metrics": metrics,
            "confidence_distribution": confidence_distribution,
            "model_version_breakdown": dict(model_versions),
            "confidence_level_metrics": confidence_level_metrics,
            "calibration_diagnostics": calibration_diagnostics,
            "lift_analysis": lift_analysis,
            "precision_recall_curve": precision_recall_table,
            "roc_curve": roc_curve_points,
        }

        active_model = (
            db.query(MLModel)
            .filter(MLModel.is_active.is_(True))
            .order_by(MLModel.training_date.desc())
            .first()
        )

        model_updated = False
        if active_model:
            active_model.accuracy = metrics["accuracy"]
            active_model.precision_score = metrics["precision"]
            active_model.recall_score = metrics["recall"]
            active_model.f1_score = metrics["f1"]
            active_model.roc_auc = metrics["roc_auc"]

            existing_metrics = _coerce_metrics(active_model.performance_metrics)
            history: List[Dict[str, object]] = existing_metrics.get("evaluation_history", [])  # type: ignore[assignment]
            history.append(evaluation_summary)
            existing_metrics["last_evaluation"] = evaluation_summary
            existing_metrics["evaluation_history"] = history[-20:]
            active_model.performance_metrics = existing_metrics

            db.commit()
            model_updated = True

        logger.info("Model performance updated successfully (%s samples)", len(y_true))

        return {
            "status": "success",
            "days_evaluated": days_back,
            "cutoff_date": cutoff_date.isoformat(),
            "evaluated_samples": len(y_true),
            "courses_evaluated": course_count,
            "probability_threshold": probability_threshold,
            "metrics": metrics,
            "confidence_distribution": confidence_distribution,
            "confidence_level_metrics": confidence_level_metrics,
            "calibration_diagnostics": calibration_diagnostics,
            "lift_analysis": lift_analysis,
            "model_version_breakdown": dict(model_versions),
            "value_bet_courses": sum(1 for data in course_stats.values() if data["value_bet_detected"]),
            "evaluation_timestamp": evaluation_timestamp,
            "model_updated": model_updated,
        }

    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"Error updating model performance: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
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
