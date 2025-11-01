"""Tâches Celery pour le Machine Learning et les prédictions."""

import json
import logging
from collections import Counter
from datetime import date, datetime, time, timedelta
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.core.database import SessionLocal
from app.ml.predictor import RacePredictionService
from app.ml.training import ModelTrainer
from app.models.course import Course, CourseStatus, StartType
from app.models.hippodrome import TrackType
from app.models.horse import Gender
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


def _summarise_threshold_recommendations(
    grid: Dict[str, Dict[str, Optional[float]]]
) -> Dict[str, object]:
    """Identifie les seuils opérationnels les plus intéressants."""

    if not grid:
        return {
            "grid": [],
            "best_f1": None,
            "maximize_precision": None,
            "maximize_recall": None,
        }

    # On convertit la grille en liste triée pour exposer les seuils de manière lisible.
    ordered_grid: List[Dict[str, Optional[float]]] = []
    for threshold_label, metrics in grid.items():
        try:
            threshold_value = float(threshold_label)
        except (TypeError, ValueError):
            # On ignore les libellés non numériques afin de ne pas casser la vue.
            continue

        ordered_grid.append(
            {
                "threshold": threshold_value,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "positive_rate": metrics.get("positive_rate"),
            }
        )

    ordered_grid.sort(key=lambda row: row["threshold"])

    def _select_best(metric: str) -> Optional[Dict[str, Optional[float]]]:
        """Retourne la ligne optimisant ``metric`` (avec tie-break sur le seuil)."""

        best_row: Optional[Dict[str, Optional[float]]] = None
        best_value = float("-inf")

        for row in ordered_grid:
            value = row.get(metric)
            if value is None:
                continue

            if (
                value > best_value + 1e-12
                or (
                    best_row is not None
                    and abs(value - best_value) <= 1e-12
                    and row["threshold"] < best_row["threshold"]
                )
            ):
                best_value = float(value)
                best_row = row

        return best_row

    best_f1_row = _select_best("f1")
    best_precision_row = _select_best("precision")
    best_recall_row = _select_best("recall")

    # Pour le meilleur F1, on expose l'ensemble des métriques associées afin
    # d'aider l'opérateur à comprendre le compromis proposé.
    best_f1_payload: Optional[Dict[str, Optional[float]]] = None
    if best_f1_row is not None:
        best_f1_payload = {
            "threshold": best_f1_row["threshold"],
            "f1": best_f1_row.get("f1"),
            "precision": best_f1_row.get("precision"),
            "recall": best_f1_row.get("recall"),
            "positive_rate": best_f1_row.get("positive_rate"),
        }

    def _row_to_summary(row: Optional[Dict[str, Optional[float]]], metric: str) -> Optional[Dict[str, Optional[float]]]:
        if row is None:
            return None

        return {
            "threshold": row["threshold"],
            metric: row.get(metric),
            "positive_rate": row.get("positive_rate"),
        }

    return {
        "grid": ordered_grid,
        "best_f1": best_f1_payload,
        "maximize_precision": _row_to_summary(best_precision_row, "precision"),
        "maximize_recall": _row_to_summary(best_recall_row, "recall"),
    }


def _analyse_odds_alignment(
    samples: List[Dict[str, object]]
) -> Dict[str, object]:
    """Quantifie l'alignement entre les probabilités projetées et les cotes publiques."""

    def _empty_payload(priced_count: int = 0) -> Dict[str, object]:
        """Construit une réponse neutre lorsque les données sont insuffisantes."""

        return {
            "priced_samples": priced_count,
            "usable_samples": 0,
            "pearson_correlation": None,
            "mean_probability_gap": None,
            "mean_absolute_error": None,
            "root_mean_squared_error": None,
            "average_predicted_probability": None,
            "average_implied_probability": None,
            "average_overround": None,
            "median_overround": None,
            "courses_with_overlay": 0,
            "course_overrounds": [],
        }

    if not samples:
        return _empty_payload(0)

    priced_samples = [
        sample
        for sample in samples
        if sample.get("odds") is not None and float(sample.get("odds", 0.0)) > 0.0
    ]

    if not priced_samples:
        return _empty_payload(0)

    predicted_probabilities: List[float] = []
    implied_probabilities: List[float] = []
    course_implied_map: Dict[object, List[float]] = {}

    for sample in priced_samples:
        probability = float(sample.get("probability", 0.0))
        odds = float(sample.get("odds", 0.0))

        if odds <= 0.0:
            # Les cotes nulles ou négatives ne peuvent pas être converties en probabilité implicite.
            continue

        implied_probability = 1.0 / odds

        predicted_probabilities.append(probability)
        implied_probabilities.append(implied_probability)

        course_id = sample.get("course_id")
        course_implied_map.setdefault(course_id, []).append(implied_probability)

    usable_samples = len(predicted_probabilities)
    if usable_samples == 0:
        # Si toutes les entrées avaient des cotes invalides, on reste sur un retour neutre.
        return _empty_payload(len(priced_samples))

    # Moyennes et écarts moyens pour visualiser l'écart général au marché.
    mean_gap = sum(
        probability - implied
        for probability, implied in zip(predicted_probabilities, implied_probabilities)
    ) / usable_samples

    mean_absolute_error = sum(
        abs(probability - implied)
        for probability, implied in zip(predicted_probabilities, implied_probabilities)
    ) / usable_samples

    root_mean_squared_error = sqrt(
        sum(
            (probability - implied) ** 2
            for probability, implied in zip(predicted_probabilities, implied_probabilities)
        )
        / usable_samples
    )

    average_predicted_probability = sum(predicted_probabilities) / usable_samples
    average_implied_probability = sum(implied_probabilities) / usable_samples

    # Calcul de la corrélation de Pearson pour mesurer la cohérence du classement proposé
    # par rapport aux cotes publiées.
    mean_predicted = average_predicted_probability
    mean_implied = average_implied_probability

    numerator = sum(
        (probability - mean_predicted) * (implied - mean_implied)
        for probability, implied in zip(predicted_probabilities, implied_probabilities)
    )
    denominator_predicted = sum(
        (probability - mean_predicted) ** 2 for probability in predicted_probabilities
    )
    denominator_implied = sum(
        (implied - mean_implied) ** 2 for implied in implied_probabilities
    )

    if denominator_predicted <= 0.0 or denominator_implied <= 0.0:
        pearson_correlation = None
    else:
        pearson_correlation = numerator / sqrt(denominator_predicted * denominator_implied)

    course_overrounds: List[Dict[str, object]] = []
    overround_values: List[float] = []

    for course_id, implied_values in course_implied_map.items():
        total_implied = sum(implied_values)
        overround = total_implied - 1.0
        overround_values.append(overround)
        course_overrounds.append(
            {
                "course_id": course_id,
                "runner_count": len(implied_values),
                "implied_probability_sum": total_implied,
                "overround": overround,
            }
        )

    def _course_sort_key(entry: Dict[str, object]) -> Tuple[int, object]:
        course_identifier = entry.get("course_id")
        return (1, 0) if course_identifier is None else (0, course_identifier)

    course_overrounds.sort(key=_course_sort_key)

    courses_with_overlay = sum(1 for value in overround_values if value < 0.0)

    average_overround = (
        sum(overround_values) / len(overround_values)
        if overround_values
        else None
    )

    median_overround: Optional[float]
    if not overround_values:
        median_overround = None
    else:
        sorted_overrounds = sorted(overround_values)
        mid = len(sorted_overrounds) // 2
        if len(sorted_overrounds) % 2 == 1:
            median_overround = sorted_overrounds[mid]
        else:
            median_overround = (
                sorted_overrounds[mid - 1] + sorted_overrounds[mid]
            ) / 2.0

    return {
        "priced_samples": len(priced_samples),
        "usable_samples": usable_samples,
        "pearson_correlation": pearson_correlation,
        "mean_probability_gap": mean_gap,
        "mean_absolute_error": mean_absolute_error,
        "root_mean_squared_error": root_mean_squared_error,
        "average_predicted_probability": average_predicted_probability,
        "average_implied_probability": average_implied_probability,
        "average_overround": average_overround,
        "median_overround": median_overround,
        "courses_with_overlay": courses_with_overlay,
        "course_overrounds": course_overrounds,
    }


def _summarise_betting_value(
    samples: List[Dict[str, object]],
    threshold: float,
) -> Dict[str, object]:
    """Estime la rentabilité théorique et réalisée des paris."""

    if not samples:
        return {
            "priced_samples": 0,
            "bets_considered": 0,
            "realized_roi": None,
            "expected_value_per_bet": None,
            "average_edge": None,
            "average_predicted_probability": None,
            "average_implied_probability": None,
            "actual_win_rate": None,
            "best_value_candidates": [],
        }

    priced_samples = [
        sample
        for sample in samples
        if sample.get("odds") is not None and float(sample.get("odds")) > 1.0
    ]

    if not priced_samples:
        return {
            "priced_samples": 0,
            "bets_considered": 0,
            "realized_roi": None,
            "expected_value_per_bet": None,
            "average_edge": None,
            "average_predicted_probability": None,
            "average_implied_probability": None,
            "actual_win_rate": None,
            "best_value_candidates": [],
        }

    bets = [
        sample
        for sample in priced_samples
        if float(sample.get("probability", 0.0)) >= threshold
    ]

    if not bets:
        return {
            "priced_samples": len(priced_samples),
            "bets_considered": 0,
            "realized_roi": None,
            "expected_value_per_bet": None,
            "average_edge": None,
            "average_predicted_probability": None,
            "average_implied_probability": None,
            "actual_win_rate": None,
            "best_value_candidates": [],
        }

    realized_return = 0.0
    expected_values: List[float] = []
    edges: List[float] = []
    predicted_probs: List[float] = []
    implied_probs: List[float] = []

    for bet in bets:
        probability = float(bet.get("probability", 0.0))
        odds = float(bet.get("odds", 0.0))
        implied = 1.0 / odds if odds > 0 else 0.0

        predicted_probs.append(probability)
        implied_probs.append(implied)
        edges.append(probability - implied)

        expected_gain = probability * (odds - 1.0) - (1.0 - probability)
        expected_values.append(expected_gain)

        if bet.get("is_winner"):
            realized_return += odds - 1.0
        else:
            realized_return -= 1.0

    bets_considered = len(bets)
    realized_roi = realized_return / bets_considered if bets_considered else None

    best_value_candidates = sorted(
        (
            {
                "course_id": bet.get("course_id"),
                "partant_id": bet.get("partant_id"),
                "horse_name": bet.get("horse_name"),
                "probability": float(bet.get("probability", 0.0)),
                "odds": float(bet.get("odds", 0.0)),
                "edge": float(edge),
                "won": bool(bet.get("is_winner")),
                "final_position": bet.get("final_position"),
            }
            for bet, edge in zip(bets, edges)
        ),
        key=lambda candidate: candidate["edge"],
        reverse=True,
    )[:3]

    return {
        "priced_samples": len(priced_samples),
        "bets_considered": bets_considered,
        "realized_roi": realized_roi,
        "expected_value_per_bet": sum(expected_values) / bets_considered,
        "average_edge": sum(edges) / bets_considered,
        "average_predicted_probability": sum(predicted_probs) / bets_considered,
        "average_implied_probability": sum(implied_probs) / bets_considered,
        "actual_win_rate": sum(1 for bet in bets if bet.get("is_winner")) / bets_considered,
        "best_value_candidates": best_value_candidates,
    }


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


def _summarise_daily_performance(
    daily_breakdown: Dict[str, Dict[str, object]]
) -> List[Dict[str, Optional[float]]]:
    """Agrège les performances jour par jour pour suivre les dérives temporelles."""

    if not daily_breakdown:
        return []

    timeline: List[Dict[str, Optional[float]]] = []

    for day in sorted(daily_breakdown.keys()):
        payload = daily_breakdown[day]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        value_bet_courses: Set[int] = set(payload.get("value_bet_courses", set()))

        sample_count = len(truths)
        positive_rate = (
            sum(predicted) / sample_count if sample_count else None
        )
        observed_positive_rate = (
            sum(truths) / sample_count if sample_count else None
        )

        accuracy = precision = recall = f1 = None
        if accuracy_score and sample_count:
            accuracy = accuracy_score(truths, predicted)
            precision = precision_score(truths, predicted, zero_division=0)
            recall = recall_score(truths, predicted, zero_division=0)
            f1 = f1_score(truths, predicted, zero_division=0)

        timeline.append(
            {
                "day": day,
                "samples": sample_count,
                "courses": len(courses),
                "value_bet_courses": len(value_bet_courses),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "positive_rate": positive_rate,
                "observed_positive_rate": observed_positive_rate,
                "average_probability": _safe_average(scores),
            }
        )

    return timeline


def _summarise_segment_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Construit un panorama de métriques par segment métier (discipline, surface, etc.)."""

    if not breakdown:
        return {}

    segment_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)

        segment_metrics[segment] = summary

    return segment_metrics


def _summarise_model_version_performance(
    breakdown: Dict[str, Dict[str, object]],
    total_samples: int,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Dresse un état des lieux détaillé des performances par version de modèle."""

    if not breakdown:
        return {}

    version_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for version in sorted(breakdown.keys()):
        payload = breakdown[version]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        confidence_levels: Counter[str] = Counter(payload.get("confidence_levels", {}))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["confidence_distribution"] = dict(confidence_levels)

        version_metrics[version] = summary

    return version_metrics


def _summarise_actor_performance(
    breakdown: Dict[str, Dict[str, Any]],
    *,
    top_n: int = 5,
    min_samples: int = 3,
) -> List[Dict[str, Any]]:
    """Construit un classement pour les jockeys/entraîneurs suivis."""

    if not breakdown:
        return []

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    leaderboard: List[Dict[str, Any]] = []

    for identifier, payload in breakdown.items():
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        label = payload.get("label") or identifier
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "identifier": identifier,
                "label": label,
                "samples": len(truths),
                "courses": len(courses),
                "horses": len(horses),
                "observed_positive_rate": (sum(truths) / len(truths)) if truths else None,
                "share": (len(truths) / total_samples) if total_samples else None,
            }
        )

        leaderboard.append(summary)

    leaderboard.sort(
        key=lambda item: (
            -item["samples"],
            -((item.get("f1") or 0.0)),
            -((item.get("precision") or 0.0)),
            item.get("label"),
        )
    )

    threshold = min_samples if total_samples >= min_samples else 1
    filtered = [item for item in leaderboard if item["samples"] >= threshold]

    return filtered[:top_n] if filtered else leaderboard[:top_n]


def _summarise_hippodrome_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> List[Dict[str, Any]]:
    """Mesure la fiabilité du modèle hippodrome par hippodrome."""

    if not breakdown:
        return []

    leaderboard: List[Dict[str, Any]] = []

    for identifier, payload in breakdown.items():
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        horses: Set[int] = set(payload.get("horses", set()))
        label = payload.get("label") or identifier

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "identifier": identifier,
                "label": label,
                "samples": len(truths),
                "courses": len(courses),
                "reunions": len(reunions),
                "horses": len(horses),
                "observed_positive_rate": (sum(truths) / len(truths)) if truths else None,
            }
        )

        leaderboard.append(summary)

    leaderboard.sort(
        key=lambda item: (
            -item["samples"],
            -((item.get("f1") or 0.0)),
            -((item.get("precision") or 0.0)),
            item.get("label"),
        )
    )

    return leaderboard


def _normalise_track_type_label(track_type: Optional[object]) -> Tuple[str, str]:
    """Normalise le type de piste pour alimenter les tableaux de bord."""

    if track_type is None:
        return "unknown", "Type de piste inconnu"

    if isinstance(track_type, TrackType):
        normalized = track_type.value
    else:
        normalized = str(track_type).strip().lower()

    mapping = {
        "plat": ("flat", "Piste plate"),
        "trot": ("trot", "Piste de trot"),
        "obstacles": ("obstacles", "Piste d'obstacles"),
        "mixte": ("mixed", "Piste mixte"),
    }

    if not normalized:
        return "unknown", "Type de piste inconnu"

    if normalized in mapping:
        return mapping[normalized]

    return normalized, normalized.capitalize()


def _summarise_track_type_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Construit une vue de performance agrégée par type de piste."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    track_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        hippodromes: Set[int] = set(payload.get("hippodromes", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["hippodromes"] = len(hippodromes)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)

        track_metrics[segment] = summary

    return track_metrics


def _categorise_course_distance(distance: Optional[int]) -> str:
    """Classe les distances officielles en familles d'effort comparables."""

    if not distance:
        return "unknown"

    if distance < 1600:
        return "short_distance"

    if distance <= 2400:
        return "middle_distance"

    return "long_distance"


def _categorise_day_part(scheduled_time: Optional[object]) -> str:
    """Regroupe l'horaire officiel en grandes plages (matin, après-midi, soirée)."""

    if not isinstance(scheduled_time, time):
        return "unknown"

    total_minutes = scheduled_time.hour * 60 + scheduled_time.minute

    if total_minutes < 12 * 60:
        return "morning"

    if total_minutes < 18 * 60:
        return "afternoon"

    return "evening"


def _categorise_field_size(field_size: Optional[int]) -> str:
    """Regroupe les tailles de champs en segments homogènes pour l'analyse."""

    if not field_size:
        return "unknown"

    if field_size <= 8:
        return "small_field"

    if field_size <= 12:
        return "medium_field"

    return "large_field"


def _categorise_prize_money(prize_money: Optional[object]) -> str:
    """Classe l'allocation financière afin de suivre l'impact du niveau de dotation."""

    if prize_money is None:
        return "unknown"

    try:
        value = float(prize_money)
    except (TypeError, ValueError):  # pragma: no cover - robustesse sur types inattendus
        return "unknown"

    if value < 10000:
        return "low_prize"

    if value < 30000:
        return "medium_prize"

    if value < 70000:
        return "high_prize"

    return "premium_prize"


def _categorise_handicap_value(handicap_value: Optional[object]) -> Tuple[str, str]:
    """Regroupe les valeurs de handicap pour analyser l'équité donnée à chaque partant."""

    if handicap_value is None:
        return "unknown", "Handicap inconnu"

    try:
        value = float(handicap_value)
    except (TypeError, ValueError):  # pragma: no cover - sécurise les imports de données exotiques
        return "unknown", "Handicap inconnu"

    if value <= 10:
        return "light_handicap", "Handicap léger (≤10)"

    if value <= 20:
        return "medium_handicap", "Handicap moyen (11-20)"

    if value <= 30:
        return "competitive_handicap", "Handicap relevé (21-30)"

    return "high_handicap", "Handicap très élevé (>30)"


def _categorise_carried_weight(
    carried_weight: Optional[object],
) -> Tuple[str, str, Optional[float]]:
    """Classe le poids porté afin de suivre l'impact de la charge sur la performance."""

    if carried_weight is None:
        return "unknown", "Poids inconnu", None

    try:
        value = float(carried_weight)
    except (TypeError, ValueError):  # pragma: no cover - sécurise les valeurs inattendues
        return "unknown", "Poids inconnu", None

    if value < 54.0:
        return "very_light", "Très léger (<54 kg)", value

    if value < 57.0:
        return "light", "Léger (54-57 kg)", value

    if value < 60.0:
        return "medium", "Moyen (57-60 kg)", value

    return "heavy", "Lourd (≥60 kg)", value


def _summarise_weight_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble une vue agrégée par classe de poids porté."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    weight_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        weights = [
            float(value)
            for value in payload.get("weights", [])
            if value is not None
        ]
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = sum(truths) / len(truths) if truths else None
        summary["average_weight"] = _safe_average(weights)
        summary["min_weight"] = min(weights) if weights else None
        summary["max_weight"] = max(weights) if weights else None
        summary["courses"] = len(courses)
        summary["horses"] = len(horses)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)

        weight_metrics[segment] = summary

    return weight_metrics


def _categorise_odds_band(odds: Optional[object]) -> Tuple[str, str]:
    """Regroupe les partants par profil de cote pour suivre la robustesse du modèle."""

    if odds is None:
        return "unpriced", "Non coté"

    try:
        value = float(odds)
    except (TypeError, ValueError):  # pragma: no cover - sécurité sur valeurs exotiques
        return "unpriced", "Non coté"

    if value <= 4.0:
        return "favorite", "Favori (≤4/1)"

    if value <= 8.0:
        return "challenger", "Challenger (4-8/1)"

    if value <= 15.0:
        return "outsider", "Outsider (8-15/1)"

    return "long_shot", "Très longue cote (>15/1)"


def _summarise_odds_band_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble un tableau de bord par segment de cote PMU."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    odds_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        odds_values = [
            float(value)
            for value in payload.get("odds", [])
            if value is not None
        ]
        implied_probabilities = [
            float(value)
            for value in payload.get("implied_probabilities", [])
            if value is not None
        ]
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["average_odds"] = _safe_average(odds_values)
        summary["min_odds"] = min(odds_values) if odds_values else None
        summary["max_odds"] = max(odds_values) if odds_values else None
        summary["average_implied_probability"] = _safe_average(implied_probabilities)
        summary["courses"] = len(courses)
        summary["horses"] = len(horses)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)

        odds_metrics[segment] = summary

    return odds_metrics


def _format_minutes_as_time(value: Optional[float]) -> Optional[str]:
    """Convertit un nombre de minutes depuis minuit en format ``HH:MM`` lisible."""

    if value is None:
        return None

    minutes = int(round(value))
    hour = (minutes // 60) % 24
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _resolve_horse_age(
    partant: object,
    course: Course,
    pronostic: Pronostic,
) -> Optional[int]:
    """Calcule l'âge du cheval au moment de la course ou ``None`` si inconnu."""

    horse = getattr(partant, "horse", None)
    if horse is None:
        return None

    explicit_age = getattr(partant, "horse_age", None)
    if explicit_age is not None:
        try:
            value = int(explicit_age)
        except (TypeError, ValueError):  # pragma: no cover - garde-fou sur données erratiques
            return None
        return value if value >= 0 else None

    birth_year = getattr(horse, "birth_year", None)
    if birth_year:
        try:
            reference_year: Optional[int] = None
            reunion = getattr(course, "reunion", None)
            if reunion and getattr(reunion, "reunion_date", None):
                reference_year = reunion.reunion_date.year
            elif getattr(pronostic, "generated_at", None):
                reference_year = pronostic.generated_at.year
            else:
                reference_year = date.today().year

            age = int(reference_year) - int(birth_year)
        except (TypeError, ValueError):  # pragma: no cover - tolérance sur types inattendus
            return None

        return age if age >= 0 else None

    computed_age = getattr(horse, "age", None)
    if computed_age is None:
        return None

    try:
        value = int(computed_age)
    except (TypeError, ValueError):  # pragma: no cover - dernières sécurités
        return None

    return value if value >= 0 else None


def _categorise_horse_age(age: Optional[int]) -> str:
    """Regroupe les âges des chevaux pour analyser la maturité qui performe."""

    if age is None or age <= 0:
        return "unknown"

    if age <= 3:
        return "juvenile"

    if age <= 5:
        return "prime"

    if age <= 8:
        return "experienced"

    return "senior"


def _categorise_draw_position(
    draw: Optional[int],
    field_size: Optional[int],
) -> str:
    """Regroupe les numéros de corde pour comparer inside/middle/outside."""

    if not draw or draw <= 0:
        return "unknown"

    if field_size and field_size > 0:
        inside_boundary = max(1, ceil(field_size / 3))
        outside_boundary = max(inside_boundary, field_size - inside_boundary + 1)

        if draw <= inside_boundary:
            return "inside"

        if draw >= outside_boundary:
            return "outside"

        return "middle"

    if draw <= 4:
        return "inside"

    if draw >= 9:
        return "outside"

    return "middle"


def _categorise_start_type(start_type: Optional[object]) -> str:
    """Regroupe les modes de départ en familles exploitables côté monitoring."""

    if not start_type:
        return "unknown"

    # Les valeurs issues de SQLAlchemy sont déjà des chaînes ``str`` (l'enum
    # hérite de ``str``). On tolère cependant un objet ``StartType`` pour
    # conserver une fonction purement utilitaire.
    if isinstance(start_type, StartType):
        value = start_type.value
    else:
        value = str(start_type)

    label = value.lower()

    # On conserve une distinction explicite entre les départs mécanisés
    # (« stalle », « autostart ») et les départs manuels qui sont regroupés
    # sous un même segment pour disposer d'assez d'échantillons.
    if label in {"stalle", "autostart"}:
        return label

    if label in {"volte", "elastique", "corde"}:
        return "manual_start"

    return label or "unknown"


def _normalise_race_category_label(category: Optional[object]) -> Tuple[str, str]:
    """Normalise une catégorie de course en identifiant stable et libellé lisible."""

    if not category:
        return "unknown", "Catégorie inconnue"

    cleaned = str(category).strip()
    if not cleaned:
        return "unknown", "Catégorie inconnue"

    identifier = cleaned.lower().replace(" ", "_")
    return identifier, cleaned


def _normalise_race_class_label(race_class: Optional[object]) -> Tuple[str, str]:
    """Normalise une classe officielle en conservant une étiquette business."""

    if not race_class:
        return "unknown", "Classe inconnue"

    cleaned = str(race_class).strip()
    if not cleaned:
        return "unknown", "Classe inconnue"

    identifier = f"class_{cleaned.lower().replace(' ', '_')}"
    display = f"Classe {cleaned.upper()}"
    return identifier, display


def _categorise_rest_period(rest_days: Optional[int]) -> str:
    """Segmente les jours de repos pour analyser l'effet de la fraîcheur."""

    if rest_days is None:
        return "unknown"

    if rest_days < 14:
        return "very_fresh"

    if rest_days < 30:
        return "fresh"

    if rest_days < 90:
        return "normal_cycle"

    return "extended_break"


def _summarise_distance_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Mesure la stabilité prédictive en fonction de la distance disputée."""

    if not breakdown:
        return {}

    distance_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        distances = [int(value) for value in payload.get("distances", []) if value]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["average_distance"] = _safe_average(distances)
        summary["min_distance"] = min(distances) if distances else None
        summary["max_distance"] = max(distances) if distances else None

        distance_metrics[segment] = summary

    return distance_metrics


def _resolve_day_part_label(segment: str) -> str:
    """Associe un segment technique à un libellé métier lisible."""

    mapping = {
        "morning": "Matin",
        "afternoon": "Après-midi",
        "evening": "Soir",
        "unknown": "Horaire inconnu",
    }
    return mapping.get(segment, segment.replace("_", " ").title())


def _summarise_day_part_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Analyse l'efficacité du modèle selon la plage horaire des réunions."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    day_part_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        minutes = [int(value) for value in payload.get("minutes", []) if value is not None]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = _resolve_day_part_label(segment)

        average_minutes = _safe_average(minutes) if minutes else None
        summary["average_post_time"] = _format_minutes_as_time(average_minutes)
        summary["earliest_post_time"] = _format_minutes_as_time(min(minutes)) if minutes else None
        summary["latest_post_time"] = _format_minutes_as_time(max(minutes)) if minutes else None

        day_part_metrics[segment] = summary

    return day_part_metrics


def _categorise_race_order(
    course_number: Optional[object],
) -> Tuple[str, str, Optional[int]]:
    """Classe la course selon sa position dans la réunion pour repérer les biais."""

    try:
        number = int(course_number) if course_number is not None else None
    except (TypeError, ValueError):  # pragma: no cover - résilience sur entrées inattendues
        number = None

    if number is None or number <= 0:
        return "unknown", "Ordre inconnu", None

    if number <= 3:
        return "early_card", "Début de réunion (courses 1-3)", number

    if number <= 6:
        return "mid_card", "Milieu de réunion (courses 4-6)", number

    return "late_card", "Fin de réunion (courses 7+)", number


def _summarise_race_order_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble une vue des performances selon l'ordre des courses dans la réunion."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())

    order_priority = {"early_card": 0, "mid_card": 1, "late_card": 2, "unknown": 3}

    race_order_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment, payload in sorted(
        breakdown.items(),
        key=lambda item: (order_priority.get(item[0], 99), item[0]),
    ):
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        course_numbers = [int(value) for value in payload.get("course_numbers", []) if value]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)

        summary["average_course_number"] = _safe_average(course_numbers)
        summary["min_course_number"] = min(course_numbers) if course_numbers else None
        summary["max_course_number"] = max(course_numbers) if course_numbers else None

        race_order_metrics[segment] = summary

    return race_order_metrics


def _categorise_month(
    race_date: Optional[object],
) -> Tuple[str, str, Optional[int], Optional[int]]:
    """Retourne le mois (AAAA-MM) auquel rattacher la réunion suivie."""

    if isinstance(race_date, datetime):
        race_date = race_date.date()

    if not isinstance(race_date, date):
        return "unknown", "Mois inconnu", None, None

    month_names = {
        1: "Janvier",
        2: "Février",
        3: "Mars",
        4: "Avril",
        5: "Mai",
        6: "Juin",
        7: "Juillet",
        8: "Août",
        9: "Septembre",
        10: "Octobre",
        11: "Novembre",
        12: "Décembre",
    }

    month_index = race_date.month
    year = race_date.year
    segment = f"{year:04d}-{month_index:02d}"
    label = f"{month_names.get(month_index, race_date.strftime('%B'))} {year}"

    return segment, label, month_index, year


def _summarise_month_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble une vue des performances agrégées par mois calendaire."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())

    def _sort_key(item: Tuple[str, Dict[str, object]]) -> Tuple[int, int, str]:
        payload = item[1]
        year = payload.get("year")
        month_index = payload.get("month_index")
        return (
            int(year) if isinstance(year, int) else 9999,
            int(month_index) if isinstance(month_index, int) else 13,
            item[0],
        )

    month_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment, payload in sorted(breakdown.items(), key=_sort_key):
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        dates: Set[str] = set(payload.get("dates", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)
        summary["month_index"] = payload.get("month_index")
        summary["year"] = payload.get("year")
        summary["first_date"] = min(dates) if dates else None
        summary["last_date"] = max(dates) if dates else None

        month_metrics[segment] = summary

    return month_metrics


def _categorise_weekday(
    race_date: Optional[object],
) -> Tuple[str, str, Optional[int]]:
    """Classe la réunion selon le jour de la semaine pour surveiller les dérives."""

    if isinstance(race_date, datetime):
        race_date = race_date.date()

    if not isinstance(race_date, date):
        return "unknown", "Jour inconnu", None

    weekday_index = race_date.weekday()
    mapping = {
        0: ("monday", "Lundi"),
        1: ("tuesday", "Mardi"),
        2: ("wednesday", "Mercredi"),
        3: ("thursday", "Jeudi"),
        4: ("friday", "Vendredi"),
        5: ("saturday", "Samedi"),
        6: ("sunday", "Dimanche"),
    }

    segment, label = mapping.get(
        weekday_index,
        (f"weekday_{weekday_index}", race_date.strftime("%A")),
    )
    return segment, label, weekday_index


def _summarise_weekday_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble une vue synthétique des performances par jour de la semaine."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    weekday_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment, payload in sorted(
        breakdown.items(),
        key=lambda item: (item[1].get("weekday_index", 7), item[0]),
    ):
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        dates: Set[str] = set(payload.get("dates", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)
        summary["weekday_index"] = payload.get("weekday_index")
        summary["first_date"] = min(dates) if dates else None
        summary["last_date"] = max(dates) if dates else None

        weekday_metrics[segment] = summary

    return weekday_metrics


def _summarise_field_size_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Évalue la précision du modèle selon la taille des pelotons rencontrés."""

    if not breakdown:
        return {}

    field_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        field_sizes = [int(size) for size in payload.get("field_sizes", []) if size]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["average_field_size"] = _safe_average(field_sizes)
        summary["min_field_size"] = min(field_sizes) if field_sizes else None
        summary["max_field_size"] = max(field_sizes) if field_sizes else None

        field_metrics[segment] = summary

    return field_metrics


def _summarise_prize_money_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Évalue la robustesse du modèle selon la dotation financière des courses."""

    if not breakdown:
        return {}

    prize_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        prize_amounts = [
            float(value)
            for value in payload.get("prize_amounts", [])
            if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["average_prize_eur"] = _safe_average(prize_amounts)
        summary["min_prize_eur"] = min(prize_amounts) if prize_amounts else None
        summary["max_prize_eur"] = max(prize_amounts) if prize_amounts else None

        prize_metrics[segment] = summary

    return prize_metrics


def _summarise_handicap_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Synthétise les performances selon la valeur de handicap imposée."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    handicap_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))
        handicap_values = [
            float(value)
            for value in payload.get("handicaps", [])
            if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["horses"] = len(horses)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)
        summary["average_handicap_value"] = _safe_average(handicap_values)
        summary["min_handicap_value"] = min(handicap_values) if handicap_values else None
        summary["max_handicap_value"] = max(handicap_values) if handicap_values else None

        handicap_metrics[segment] = summary

    return handicap_metrics


def _categorise_confidence_score(
    confidence_score: Optional[object],
) -> Tuple[str, str, Optional[float]]:
    """Normalise un score de confiance en pourcentage et retourne le segment associé."""

    if confidence_score is None:
        return "unknown", "Confiance inconnue", None

    try:
        value = float(confidence_score)
    except (TypeError, ValueError):  # pragma: no cover - garde-fou contre valeurs inattendues
        return "unknown", "Confiance inconnue", None

    # Certains pipelines stockent un ratio [0, 1], d'autres déjà un pourcentage.
    # On contraint la valeur sur 0-100 pour homogénéiser les analyses.
    value = max(0.0, min(value, 100.0)) if value > 1.0 else max(0.0, min(value * 100.0, 100.0))

    if value < 30.0:
        return "very_low", "Confiance très faible (<30%)", value

    if value < 50.0:
        return "low", "Confiance faible (30-50%)", value

    if value < 70.0:
        return "medium", "Confiance moyenne (50-70%)", value

    if value < 85.0:
        return "high", "Confiance élevée (70-85%)", value

    return "very_high", "Confiance très élevée (≥85%)", value


def _summarise_confidence_score_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble un tableau de bord par tranche de confiance pronostic."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    confidence_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        confidence_scores = [
            float(value)
            for value in payload.get("confidence_scores", [])
            if value is not None
        ]
        courses: Set[int] = set(payload.get("courses", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = sum(truths) / len(truths) if truths else None
        summary["average_confidence"] = _safe_average(confidence_scores)
        summary["min_confidence"] = min(confidence_scores) if confidence_scores else None
        summary["max_confidence"] = max(confidence_scores) if confidence_scores else None
        summary["courses"] = len(courses)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)

        confidence_metrics[segment] = summary

    return confidence_metrics


def _categorise_horse_gender(gender: Optional[object]) -> Tuple[str, str]:
    """Normalise le genre du cheval pour faciliter l'agrégation."""

    if gender is None:
        return "unknown", "Genre inconnu"

    if isinstance(gender, Gender):
        normalized = gender.value
    else:
        normalized = str(gender).strip().lower()

    if not normalized:
        return "unknown", "Genre inconnu"

    normalized = (
        normalized.replace("â", "a")
        .replace("à", "a")
        .replace("é", "e")
        .replace("è", "e")
    )

    mapping = {
        "male": ("male", "Mâle"),
        "femelle": ("female", "Femelle"),
        "female": ("female", "Femelle"),
        "hongre": ("hongre", "Hongre"),
    }

    if normalized in mapping:
        return mapping[normalized]

    return "unknown", "Genre inconnu"


def _summarise_horse_gender_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Mesure la qualité prédictive selon le genre déclaré des chevaux."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    gender_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["horses"] = len(horses)
        summary["label"] = payload.get("label", segment.title())
        summary["share"] = (
            summary["samples"] / total_samples if total_samples else None
        )

        gender_metrics[segment] = summary

    return gender_metrics


def _summarise_horse_age_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Mesure la stabilité des prédictions selon la maturité des chevaux."""

    if not breakdown:
        return {}

    age_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))
        ages = [int(value) for value in payload.get("ages", []) if value is not None]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["horses"] = len(horses)
        summary["average_age"] = _safe_average(ages)
        summary["min_age"] = min(ages) if ages else None
        summary["max_age"] = max(ages) if ages else None

        age_metrics[segment] = summary

    return age_metrics


def _summarise_draw_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compare la précision selon la position dans les stalles de départ."""

    if not breakdown:
        return {}

    draw_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        draws = [int(draw) for draw in payload.get("draws", []) if draw]
        field_sizes = [int(size) for size in payload.get("field_sizes", []) if size]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["average_draw"] = _safe_average(draws)
        summary["min_draw"] = min(draws) if draws else None
        summary["max_draw"] = max(draws) if draws else None
        summary["average_field_size"] = _safe_average(field_sizes)
        summary["min_field_size"] = min(field_sizes) if field_sizes else None
        summary["max_field_size"] = max(field_sizes) if field_sizes else None

        draw_metrics[segment] = summary

    return draw_metrics


def _summarise_race_profile_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Analyse les performances selon les catégories ou classes officielles."""

    if not breakdown:
        return {}

    profile_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        label = payload.get("label") or segment

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["label"] = label

        profile_metrics[segment] = summary

    return profile_metrics


def _summarise_start_type_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compare la stabilité du modèle en fonction des procédures de départ."""

    if not breakdown:
        return {}

    start_type_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)

        start_type_metrics[segment] = summary

    return start_type_metrics


def _summarise_rest_period_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Mesure la qualité prédictive selon le nombre de jours de repos."""

    if not breakdown:
        return {}

    rest_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        rest_days = [int(value) for value in payload.get("rest_days", []) if value is not None]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["average_rest_days"] = _safe_average(rest_days)
        summary["min_rest_days"] = min(rest_days) if rest_days else None
        summary["max_rest_days"] = max(rest_days) if rest_days else None

        rest_metrics[segment] = summary

    return rest_metrics


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
        confidence_score_breakdown: Dict[str, Dict[str, object]] = {}
        discipline_breakdown: Dict[str, Dict[str, object]] = {}
        distance_breakdown: Dict[str, Dict[str, object]] = {}
        surface_breakdown: Dict[str, Dict[str, object]] = {}
        prize_money_breakdown: Dict[str, Dict[str, object]] = {}
        handicap_breakdown: Dict[str, Dict[str, object]] = {}
        weight_breakdown: Dict[str, Dict[str, object]] = {}
        odds_band_breakdown: Dict[str, Dict[str, object]] = {}
        horse_age_breakdown: Dict[str, Dict[str, object]] = {}
        horse_gender_breakdown: Dict[str, Dict[str, object]] = {}
        day_part_breakdown: Dict[str, Dict[str, object]] = {}
        weekday_breakdown: Dict[str, Dict[str, object]] = {}
        month_breakdown: Dict[str, Dict[str, object]] = {}
        race_order_breakdown: Dict[str, Dict[str, object]] = {}
        track_type_breakdown: Dict[str, Dict[str, object]] = {}
        race_category_breakdown: Dict[str, Dict[str, object]] = {}
        race_class_breakdown: Dict[str, Dict[str, object]] = {}
        value_bet_breakdown: Dict[str, Dict[str, object]] = {}
        field_size_breakdown: Dict[str, Dict[str, object]] = {}
        draw_breakdown: Dict[str, Dict[str, object]] = {}
        start_type_breakdown: Dict[str, Dict[str, object]] = {}
        rest_period_breakdown: Dict[str, Dict[str, object]] = {}
        jockey_breakdown: Dict[str, Dict[str, object]] = {}
        trainer_breakdown: Dict[str, Dict[str, object]] = {}
        hippodrome_breakdown: Dict[str, Dict[str, object]] = {}
        # Prépare une vision par version du modèle afin d'identifier rapidement
        # les régressions potentielles lorsqu'une version minoritaire décroche.
        model_versions: Counter[str] = Counter()
        model_version_breakdown: Dict[str, Dict[str, object]] = {}
        course_stats: Dict[int, Dict[str, object]] = {}
        daily_breakdown: Dict[str, Dict[str, object]] = {}
        betting_samples: List[Dict[str, object]] = []

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

            confidence_segment, confidence_label, confidence_value = _categorise_confidence_score(
                getattr(pronostic, "confidence_score", None)
            )
            confidence_bucket = confidence_score_breakdown.setdefault(
                confidence_segment,
                {
                    "label": confidence_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "confidence_scores": [],
                    "courses": set(),
                },
            )
            confidence_bucket["label"] = confidence_label
            confidence_bucket["truths"].append(is_top3)
            confidence_bucket["predictions"].append(predicted_label)
            confidence_bucket["scores"].append(probability)
            if confidence_value is not None:
                confidence_bucket.setdefault("confidence_scores", []).append(confidence_value)
            confidence_bucket.setdefault("courses", set()).add(course.course_id)
            version_label = pronostic.model_version or "unknown"
            model_versions[version_label] += 1
            version_bucket = model_version_breakdown.setdefault(
                version_label,
                {
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "confidence_levels": Counter(),
                },
            )
            version_bucket["truths"].append(is_top3)
            version_bucket["predictions"].append(predicted_label)
            version_bucket["scores"].append(probability)
            version_bucket.setdefault("courses", set()).add(course.course_id)
            version_bucket.setdefault("confidence_levels", Counter())[prediction.confidence_level or "unknown"] += 1

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
            # Stocke le nombre de partants observés afin de catégoriser ensuite
            # les courses par taille de peloton (utile pour repérer les champs
            # où le modèle excelle ou se dégrade).
            course_entry["field_size"] = (
                getattr(course, "number_of_runners", None)
                or len(course_entry["predictions"])
            )

            # Segmente l'horaire officiel de départ pour identifier si le modèle
            # se comporte différemment entre les réunions matinales, l'après-midi
            # et les nocturnes. Les opérateurs peuvent ainsi ajuster leur
            # stratégie d'engagement selon le moment de la journée.
            day_part_segment = _categorise_day_part(
                getattr(course, "scheduled_time", None)
            )
            day_part_bucket = day_part_breakdown.setdefault(
                day_part_segment,
                {
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "minutes": [],
                },
            )
            day_part_bucket["truths"].append(is_top3)
            day_part_bucket["predictions"].append(predicted_label)
            day_part_bucket["scores"].append(probability)
            day_part_bucket.setdefault("courses", set()).add(course.course_id)

            scheduled_time_value = getattr(course, "scheduled_time", None)
            if isinstance(scheduled_time_value, time):
                minutes_value = scheduled_time_value.hour * 60 + scheduled_time_value.minute
                day_part_bucket.setdefault("minutes", []).append(minutes_value)

            # Ventile également les performances par jour de la semaine afin de
            # repérer rapidement si certains créneaux (week-ends, réunions
            # nocturnes) introduisent un biais de précision. On conserve les
            # identifiants de course/réunion et les dates ISO pour alimenter le
            # tableau de bord dédié.
            reunion_obj = getattr(course, "reunion", None)
            race_date = getattr(reunion_obj, "reunion_date", None)
            weekday_segment, weekday_label, weekday_index = _categorise_weekday(race_date)
            weekday_bucket = weekday_breakdown.setdefault(
                weekday_segment,
                {
                    "label": weekday_label,
                    "weekday_index": weekday_index,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "dates": set(),
                },
            )
            weekday_bucket["truths"].append(is_top3)
            weekday_bucket["predictions"].append(predicted_label)
            weekday_bucket["scores"].append(probability)
            weekday_bucket.setdefault("courses", set()).add(course.course_id)
            weekday_bucket.setdefault("reunions", set()).add(
                getattr(reunion_obj, "reunion_id", getattr(course, "reunion_id", None))
            )
            if isinstance(race_date, date):
                weekday_bucket.setdefault("dates", set()).add(race_date.isoformat())

            # Alimente également un suivi mensuel pour détecter d'éventuelles
            # variations saisonnières (meeting d'hiver/été) dans la précision
            # du modèle. Les identifiants de course et réunion sont conservés
            # afin d'établir des tableaux de bord détaillés.
            month_segment, month_label, month_index, month_year = _categorise_month(
                race_date
            )
            month_bucket = month_breakdown.setdefault(
                month_segment,
                {
                    "label": month_label,
                    "month_index": month_index,
                    "year": month_year,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "dates": set(),
                },
            )
            month_bucket["truths"].append(is_top3)
            month_bucket["predictions"].append(predicted_label)
            month_bucket["scores"].append(probability)
            month_bucket.setdefault("courses", set()).add(course.course_id)
            month_bucket.setdefault("reunions", set()).add(
                getattr(reunion_obj, "reunion_id", getattr(course, "reunion_id", None))
            )
            if isinstance(race_date, date):
                month_bucket.setdefault("dates", set()).add(race_date.isoformat())

            # Segmente les performances selon la position de la course dans la
            # réunion (début/milieu/fin) pour détecter d'éventuels écarts en fin
            # de programme lorsque la piste ou la concurrence évoluent.
            race_order_segment, race_order_label, course_number = _categorise_race_order(
                getattr(course, "course_number", None)
            )
            race_order_bucket = race_order_breakdown.setdefault(
                race_order_segment,
                {
                    "label": race_order_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "course_numbers": [],
                },
            )
            race_order_bucket["truths"].append(is_top3)
            race_order_bucket["predictions"].append(predicted_label)
            race_order_bucket["scores"].append(probability)
            race_order_bucket.setdefault("courses", set()).add(course.course_id)
            race_order_bucket.setdefault("reunions", set()).add(
                getattr(reunion_obj, "reunion_id", getattr(course, "reunion_id", None))
            )
            if course_number is not None:
                race_order_bucket.setdefault("course_numbers", []).append(course_number)

            betting_samples.append(
                {
                    "probability": probability,
                    "odds": float(partant.odds_pmu) if partant.odds_pmu is not None else None,
                    "is_winner": bool(partant.final_position == 1),
                    "course_id": course.course_id,
                    "partant_id": partant.partant_id,
                    "horse_name": partant.horse.name if getattr(partant, "horse", None) else None,
                    "final_position": partant.final_position,
                }
            )

            # Classe les partants selon leur profil de cote (favori, challenger,
            # outsider, etc.) afin de vérifier si le modèle reste fiable lorsque
            # l'on s'éloigne des chevaux les plus attendus par le marché.
            odds_segment, odds_label = _categorise_odds_band(getattr(partant, "odds_pmu", None))
            odds_bucket = odds_band_breakdown.setdefault(
                odds_segment,
                {
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "odds": [],
                    "implied_probabilities": [],
                    "courses": set(),
                    "horses": set(),
                    "label": odds_label,
                },
            )
            odds_bucket["truths"].append(is_top3)
            odds_bucket["predictions"].append(predicted_label)
            odds_bucket["scores"].append(probability)
            odds_bucket.setdefault("courses", set()).add(course.course_id)
            if getattr(partant, "horse_id", None) is not None:
                odds_bucket.setdefault("horses", set()).add(partant.horse_id)

            raw_odds_value = getattr(partant, "odds_pmu", None)
            try:
                odds_value = float(raw_odds_value) if raw_odds_value is not None else None
            except (TypeError, ValueError):  # pragma: no cover - robustesse en entrée
                odds_value = None

            if odds_value and odds_value > 0:
                odds_bucket.setdefault("odds", []).append(odds_value)
                odds_bucket.setdefault("implied_probabilities", []).append(1.0 / odds_value)

            # Cartographie les performances par attribut métier pour identifier
            # rapidement les segments qui décrochent (discipline, surface,
            # appétit value bet).
            discipline_label = (
                course.discipline.value
                if getattr(course, "discipline", None)
                else "unknown"
            )
            discipline_bucket = discipline_breakdown.setdefault(
                discipline_label,
                {"truths": [], "predictions": [], "scores": [], "courses": set()},
            )
            discipline_bucket["truths"].append(is_top3)
            discipline_bucket["predictions"].append(predicted_label)
            discipline_bucket["scores"].append(probability)
            discipline_bucket.setdefault("courses", set()).add(course.course_id)

            # Ventile également les performances selon la distance officielle afin
            # de vérifier que le modèle reste stable entre sprint, classique et tenue.
            distance_segment = _categorise_course_distance(
                getattr(course, "distance", None)
            )
            distance_bucket = distance_breakdown.setdefault(
                distance_segment,
                {
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "distances": [],
                },
            )
            distance_bucket["truths"].append(is_top3)
            distance_bucket["predictions"].append(predicted_label)
            distance_bucket["scores"].append(probability)
            distance_bucket.setdefault("courses", set()).add(course.course_id)
            course_distance = getattr(course, "distance", None)
            if course_distance:
                distance_bucket.setdefault("distances", []).append(int(course_distance))

            surface_label = (
                course.surface_type.value
                if getattr(course, "surface_type", None)
                else "unknown"
            )
            surface_bucket = surface_breakdown.setdefault(
                surface_label,
                {"truths": [], "predictions": [], "scores": [], "courses": set()},
            )
            surface_bucket["truths"].append(is_top3)
            surface_bucket["predictions"].append(predicted_label)
            surface_bucket["scores"].append(probability)
            surface_bucket.setdefault("courses", set()).add(course.course_id)

            prize_segment = _categorise_prize_money(
                getattr(course, "prize_money", None)
            )
            prize_bucket = prize_money_breakdown.setdefault(
                prize_segment,
                {
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "prize_amounts": [],
                },
            )
            prize_bucket["truths"].append(is_top3)
            prize_bucket["predictions"].append(predicted_label)
            prize_bucket["scores"].append(probability)
            courses_seen = prize_bucket.setdefault("courses", set())
            is_new_prize_course = course.course_id not in courses_seen
            courses_seen.add(course.course_id)
            prize_value = getattr(course, "prize_money", None)
            if prize_value is not None and is_new_prize_course:
                prize_bucket.setdefault("prize_amounts", []).append(float(prize_value))

            handicap_segment, handicap_label = _categorise_handicap_value(
                getattr(partant, "handicap_value", None)
            )
            handicap_bucket = handicap_breakdown.setdefault(
                handicap_segment,
                {
                    "label": handicap_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "handicaps": [],
                },
            )
            handicap_bucket["label"] = handicap_label
            handicap_bucket["truths"].append(is_top3)
            handicap_bucket["predictions"].append(predicted_label)
            handicap_bucket["scores"].append(probability)
            handicap_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                handicap_bucket.setdefault("horses", set()).add(partant.horse_id)
            raw_handicap = getattr(partant, "handicap_value", None)
            try:
                handicap_value = float(raw_handicap) if raw_handicap is not None else None
            except (TypeError, ValueError):  # pragma: no cover - robustesse face aux données incohérentes
                handicap_value = None
            if handicap_value is not None:
                handicap_bucket.setdefault("handicaps", []).append(handicap_value)

            weight_segment, weight_label, weight_value = _categorise_carried_weight(
                getattr(partant, "poids_porte", None)
            )
            weight_bucket = weight_breakdown.setdefault(
                weight_segment,
                {
                    "label": weight_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "weights": [],
                },
            )
            weight_bucket["label"] = weight_label
            weight_bucket["truths"].append(is_top3)
            weight_bucket["predictions"].append(predicted_label)
            weight_bucket["scores"].append(probability)
            weight_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                weight_bucket.setdefault("horses", set()).add(partant.horse_id)
            if weight_value is not None:
                weight_bucket.setdefault("weights", []).append(weight_value)

            horse_age = _resolve_horse_age(partant, course, pronostic)
            age_segment = _categorise_horse_age(horse_age)
            age_bucket = horse_age_breakdown.setdefault(
                age_segment,
                {
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "ages": [],
                },
            )
            age_bucket["truths"].append(is_top3)
            age_bucket["predictions"].append(predicted_label)
            age_bucket["scores"].append(probability)
            age_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                age_bucket.setdefault("horses", set()).add(partant.horse_id)
            if horse_age is not None:
                age_bucket.setdefault("ages", []).append(int(horse_age))

            horse_entity = getattr(partant, "horse", None)
            gender_key, gender_label = _categorise_horse_gender(
                getattr(horse_entity, "gender", None)
            )
            gender_bucket = horse_gender_breakdown.setdefault(
                gender_key,
                {
                    "label": gender_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                },
            )
            gender_bucket["label"] = gender_label
            gender_bucket["truths"].append(is_top3)
            gender_bucket["predictions"].append(predicted_label)
            gender_bucket["scores"].append(probability)
            gender_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                gender_bucket.setdefault("horses", set()).add(partant.horse_id)

            category_key, category_label = _normalise_race_category_label(
                getattr(course, "race_category", None)
            )
            category_bucket = race_category_breakdown.setdefault(
                category_key,
                {
                    "label": category_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                },
            )
            category_bucket["label"] = category_label
            category_bucket["truths"].append(is_top3)
            category_bucket["predictions"].append(predicted_label)
            category_bucket["scores"].append(probability)
            category_bucket.setdefault("courses", set()).add(course.course_id)

            class_key, class_label = _normalise_race_class_label(
                getattr(course, "race_class", None)
            )
            class_bucket = race_class_breakdown.setdefault(
                class_key,
                {
                    "label": class_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                },
            )
            class_bucket["label"] = class_label
            class_bucket["truths"].append(is_top3)
            class_bucket["predictions"].append(predicted_label)
            class_bucket["scores"].append(probability)
            class_bucket.setdefault("courses", set()).add(course.course_id)

            value_bet_label = "value_bet" if pronostic.value_bet_detected else "standard"
            value_bet_bucket = value_bet_breakdown.setdefault(
                value_bet_label,
                {"truths": [], "predictions": [], "scores": [], "courses": set()},
            )
            value_bet_bucket["truths"].append(is_top3)
            value_bet_bucket["predictions"].append(predicted_label)
            value_bet_bucket["scores"].append(probability)
            value_bet_bucket.setdefault("courses", set()).add(course.course_id)

            field_size = course_entry.get("field_size")
            field_segment = _categorise_field_size(
                int(field_size) if field_size else None
            )
            field_bucket = field_size_breakdown.setdefault(
                field_segment,
                {
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "field_sizes": [],
                },
            )
            field_bucket["truths"].append(is_top3)
            field_bucket["predictions"].append(predicted_label)
            field_bucket["scores"].append(probability)
            courses_seen: Set[int] = field_bucket.setdefault("courses", set())
            is_new_course = course.course_id not in courses_seen
            courses_seen.add(course.course_id)
            if field_size and is_new_course:
                field_bucket.setdefault("field_sizes", []).append(int(field_size))

            draw_value = getattr(partant, "numero_corde", None)
            draw_segment = _categorise_draw_position(
                int(draw_value) if draw_value is not None else None,
                int(field_size) if field_size else None,
            )
            draw_bucket = draw_breakdown.setdefault(
                draw_segment,
                {
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "draws": [],
                    "field_sizes": [],
                },
            )
            draw_bucket["truths"].append(is_top3)
            draw_bucket["predictions"].append(predicted_label)
            draw_bucket["scores"].append(probability)
            draw_bucket.setdefault("courses", set()).add(course.course_id)
            if draw_value is not None:
                draw_bucket.setdefault("draws", []).append(int(draw_value))
            if field_size:
                draw_bucket.setdefault("field_sizes", []).append(int(field_size))

            # Suivi spécifique des modes de départ (stalle, autostart, volte...)
            # afin d'identifier si le modèle décroche sur un protocole précis.
            start_segment = _categorise_start_type(
                getattr(course, "start_type", None)
            )
            start_bucket = start_type_breakdown.setdefault(
                start_segment,
                {
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                },
            )
            start_bucket["truths"].append(is_top3)
            start_bucket["predictions"].append(predicted_label)
            start_bucket["scores"].append(probability)
            start_bucket.setdefault("courses", set()).add(course.course_id)

            rest_segment = _categorise_rest_period(
                getattr(partant, "days_since_last_race", None)
            )
            rest_bucket = rest_period_breakdown.setdefault(
                rest_segment,
                {
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "rest_days": [],
                },
            )
            rest_bucket["truths"].append(is_top3)
            rest_bucket["predictions"].append(predicted_label)
            rest_bucket["scores"].append(probability)
            rest_bucket.setdefault("courses", set()).add(course.course_id)
            rest_days_value = getattr(partant, "days_since_last_race", None)
            if rest_days_value is not None:
                rest_bucket.setdefault("rest_days", []).append(int(rest_days_value))

            jockey_identifier = str(partant.jockey_id) if partant.jockey_id else "unknown"
            jockey_label = (
                partant.jockey.full_name
                if getattr(partant, "jockey", None) and getattr(partant.jockey, "full_name", None)
                else (f"Jockey #{partant.jockey_id}" if partant.jockey_id else "Jockey inconnu")
            )
            jockey_bucket = jockey_breakdown.setdefault(
                jockey_identifier,
                {
                    "label": jockey_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                },
            )
            jockey_bucket["label"] = jockey_label
            jockey_bucket["truths"].append(is_top3)
            jockey_bucket["predictions"].append(predicted_label)
            jockey_bucket["scores"].append(probability)
            jockey_bucket.setdefault("courses", set()).add(course.course_id)
            jockey_bucket.setdefault("horses", set()).add(partant.horse_id)

            trainer_identifier = str(partant.trainer_id) if partant.trainer_id else "unknown"
            trainer_label = (
                partant.trainer.full_name
                if getattr(partant, "trainer", None) and getattr(partant.trainer, "full_name", None)
                else (f"Entraîneur #{partant.trainer_id}" if partant.trainer_id else "Entraîneur inconnu")
            )
            trainer_bucket = trainer_breakdown.setdefault(
                trainer_identifier,
                {
                    "label": trainer_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                },
            )
            trainer_bucket["label"] = trainer_label
            trainer_bucket["truths"].append(is_top3)
            trainer_bucket["predictions"].append(predicted_label)
            trainer_bucket["scores"].append(probability)
            trainer_bucket.setdefault("courses", set()).add(course.course_id)
            trainer_bucket.setdefault("horses", set()).add(partant.horse_id)

            # Enfin, on garde une vue géographique afin de repérer les hippodromes
            # où le modèle excelle ou se dégrade. Cette information aide à prioriser
            # les analyses locales (qualité des données, biais spécifiques, météo...).
            reunion_entity = getattr(course, "reunion", None)
            hippodrome_entity = (
                getattr(reunion_entity, "hippodrome", None) if reunion_entity else None
            )
            hippodrome_id = None
            venue_key = "unknown"
            venue_label = "Hippodrome inconnu"
            track_type_key = "unknown"
            track_type_label = "Type de piste inconnu"

            if hippodrome_entity is not None:
                hippodrome_id = getattr(hippodrome_entity, "hippodrome_id", None)
                hippodrome_code = getattr(hippodrome_entity, "code", None)
                venue_key = hippodrome_code or (
                    f"hippodrome_{hippodrome_id}" if hippodrome_id else "unknown"
                )
                venue_label = getattr(hippodrome_entity, "name", None) or venue_key
                track_type_key, track_type_label = _normalise_track_type_label(
                    getattr(hippodrome_entity, "track_type", None)
                )
            elif reunion_entity is not None:
                hippodrome_id = getattr(reunion_entity, "hippodrome_id", None)
                venue_key = f"hippodrome_{hippodrome_id}" if hippodrome_id else "unknown"
                venue_label = (
                    f"Hippodrome #{hippodrome_id}" if hippodrome_id else "Hippodrome inconnu"
                )

            venue_bucket = hippodrome_breakdown.setdefault(
                venue_key,
                {
                    "label": venue_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "horses": set(),
                },
            )
            venue_bucket["label"] = venue_label
            venue_bucket["truths"].append(is_top3)
            venue_bucket["predictions"].append(predicted_label)
            venue_bucket["scores"].append(probability)
            venue_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                venue_bucket.setdefault("horses", set()).add(partant.horse_id)
            if reunion_entity is not None and getattr(reunion_entity, "reunion_id", None):
                venue_bucket.setdefault("reunions", set()).add(reunion_entity.reunion_id)

            track_type_bucket = track_type_breakdown.setdefault(
                track_type_key,
                {
                    "label": track_type_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "hippodromes": set(),
                },
            )
            track_type_bucket["label"] = track_type_label
            track_type_bucket["truths"].append(is_top3)
            track_type_bucket["predictions"].append(predicted_label)
            track_type_bucket["scores"].append(probability)
            track_type_bucket.setdefault("courses", set()).add(course.course_id)
            if reunion_entity is not None and getattr(reunion_entity, "reunion_id", None):
                track_type_bucket.setdefault("reunions", set()).add(
                    reunion_entity.reunion_id
                )
            if hippodrome_id is not None:
                track_type_bucket.setdefault("hippodromes", set()).add(hippodrome_id)

            # On conserve également une vue chronologique afin d'identifier les
            # journées où le modèle surperforme ou décroche brutalement.
            generation_day = (
                (pronostic.generated_at.date() if pronostic.generated_at else None)
                or (
                    course.reunion.reunion_date
                    if hasattr(course, "reunion") and course.reunion
                    else None
                )
                or cutoff_date
            )
            day_key = generation_day.isoformat()
            day_bucket = daily_breakdown.setdefault(
                day_key,
                {
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "value_bet_courses": set(),
                },
            )
            day_bucket["truths"].append(is_top3)
            day_bucket["predictions"].append(predicted_label)
            day_bucket["scores"].append(probability)
            day_bucket["courses"].add(course.course_id)
            if pronostic.value_bet_detected:
                day_bucket["value_bet_courses"].add(course.course_id)

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
        # La grille multi-seuils étant calculée, on extrait directement les
        # recommandations (meilleur F1, précision ou rappel maximal) pour
        # éviter aux opérateurs de parcourir manuellement toutes les lignes.
        threshold_recommendations = _summarise_threshold_recommendations(
            threshold_grid
        )

        # Analyse la valeur financière potentielle des paris générés par le
        # modèle en confrontant les probabilités projetées aux cotes publiques.
        betting_value_analysis = _summarise_betting_value(
            betting_samples,
            probability_threshold,
        )

        # Mesure l'alignement global avec le marché (corrélation et surcote)
        # pour contextualiser les écarts du modèle par rapport aux bookmakers.
        odds_alignment = _analyse_odds_alignment(betting_samples)

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

        confidence_score_performance = _summarise_confidence_score_performance(
            confidence_score_breakdown
        )

        daily_performance = _summarise_daily_performance(daily_breakdown)
        day_part_performance = _summarise_day_part_performance(day_part_breakdown)
        month_performance = _summarise_month_performance(month_breakdown)
        weekday_performance = _summarise_weekday_performance(weekday_breakdown)
        race_order_performance = _summarise_race_order_performance(race_order_breakdown)
        discipline_performance = _summarise_segment_performance(discipline_breakdown)
        distance_performance = _summarise_distance_performance(distance_breakdown)
        surface_performance = _summarise_segment_performance(surface_breakdown)
        prize_money_performance = _summarise_prize_money_performance(
            prize_money_breakdown
        )
        handicap_performance = _summarise_handicap_performance(handicap_breakdown)
        weight_performance = _summarise_weight_performance(weight_breakdown)
        odds_band_performance = _summarise_odds_band_performance(odds_band_breakdown)
        horse_age_performance = _summarise_horse_age_performance(
            horse_age_breakdown
        )
        horse_gender_performance = _summarise_horse_gender_performance(
            horse_gender_breakdown
        )
        value_bet_performance = _summarise_segment_performance(value_bet_breakdown)
        field_size_performance = _summarise_field_size_performance(field_size_breakdown)
        draw_performance = _summarise_draw_performance(draw_breakdown)
        race_category_performance = _summarise_race_profile_performance(
            race_category_breakdown
        )
        race_class_performance = _summarise_race_profile_performance(
            race_class_breakdown
        )
        start_type_performance = _summarise_start_type_performance(start_type_breakdown)
        rest_period_performance = _summarise_rest_period_performance(
            rest_period_breakdown
        )
        model_version_performance = _summarise_model_version_performance(
            model_version_breakdown,
            len(y_true),
        )
        jockey_performance = _summarise_actor_performance(jockey_breakdown)
        trainer_performance = _summarise_actor_performance(trainer_breakdown)
        hippodrome_performance = _summarise_hippodrome_performance(hippodrome_breakdown)
        track_type_performance = _summarise_track_type_performance(track_type_breakdown)

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
            "threshold_recommendations": threshold_recommendations,
            "betting_value_analysis": betting_value_analysis,
            "odds_alignment": odds_alignment,
            "gain_curve": gain_curve,
            "lift_analysis": lift_analysis,
            "average_precision": average_precision,
            "precision_recall_curve": precision_recall_table,
            "roc_curve": roc_curve_points,
            "ks_analysis": ks_analysis,
            "confidence_level_metrics": confidence_level_metrics,
            "confidence_score_performance": confidence_score_performance,
            "daily_performance": daily_performance,
            "day_part_performance": day_part_performance,
            "month_performance": month_performance,
            "weekday_performance": weekday_performance,
            "race_order_performance": race_order_performance,
            "discipline_performance": discipline_performance,
            "distance_performance": distance_performance,
            "surface_performance": surface_performance,
            "prize_money_performance": prize_money_performance,
            "handicap_performance": handicap_performance,
            "weight_performance": weight_performance,
            "odds_band_performance": odds_band_performance,
            "horse_age_performance": horse_age_performance,
            "horse_gender_performance": horse_gender_performance,
            "race_category_performance": race_category_performance,
            "race_class_performance": race_class_performance,
            "value_bet_performance": value_bet_performance,
            "field_size_performance": field_size_performance,
            "draw_performance": draw_performance,
            "start_type_performance": start_type_performance,
            "rest_period_performance": rest_period_performance,
            "model_version_performance": model_version_performance,
            "jockey_performance": jockey_performance,
            "trainer_performance": trainer_performance,
            "hippodrome_performance": hippodrome_performance,
            "track_type_performance": track_type_performance,
            "odds_band_performance": odds_band_performance,
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
            "confidence_score_performance": confidence_score_performance,
            "calibration_diagnostics": calibration_diagnostics,
            "threshold_recommendations": threshold_recommendations,
            "betting_value_analysis": betting_value_analysis,
            "odds_alignment": odds_alignment,
            "lift_analysis": lift_analysis,
            "precision_recall_curve": precision_recall_table,
            "roc_curve": roc_curve_points,
            "daily_performance": daily_performance,
            "day_part_performance": day_part_performance,
            "month_performance": month_performance,
            "weekday_performance": weekday_performance,
            "race_order_performance": race_order_performance,
            "discipline_performance": discipline_performance,
            "distance_performance": distance_performance,
            "surface_performance": surface_performance,
            "prize_money_performance": prize_money_performance,
            "handicap_performance": handicap_performance,
            "weight_performance": weight_performance,
            "horse_age_performance": horse_age_performance,
            "horse_gender_performance": horse_gender_performance,
            "race_category_performance": race_category_performance,
            "race_class_performance": race_class_performance,
            "value_bet_performance": value_bet_performance,
            "field_size_performance": field_size_performance,
            "draw_performance": draw_performance,
            "start_type_performance": start_type_performance,
            "rest_period_performance": rest_period_performance,
            "model_version_performance": model_version_performance,
            "jockey_performance": jockey_performance,
            "trainer_performance": trainer_performance,
            "hippodrome_performance": hippodrome_performance,
            "track_type_performance": track_type_performance,
            "odds_band_performance": odds_band_performance,
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
            "confidence_score_performance": confidence_score_performance,
            "calibration_diagnostics": calibration_diagnostics,
            "threshold_recommendations": threshold_recommendations,
            "betting_value_analysis": betting_value_analysis,
            "odds_alignment": odds_alignment,
            "lift_analysis": lift_analysis,
            "daily_performance": daily_performance,
            "day_part_performance": day_part_performance,
            "month_performance": month_performance,
            "weekday_performance": weekday_performance,
            "race_order_performance": race_order_performance,
            "distance_performance": distance_performance,
            "draw_performance": draw_performance,
            "prize_money_performance": prize_money_performance,
            "handicap_performance": handicap_performance,
            "weight_performance": weight_performance,
            "horse_age_performance": horse_age_performance,
            "horse_gender_performance": horse_gender_performance,
            "track_type_performance": track_type_performance,
            "odds_band_performance": odds_band_performance,
            "model_version_breakdown": dict(model_versions),
            "model_version_performance": model_version_performance,
            "rest_period_performance": rest_period_performance,
            "jockey_performance": jockey_performance,
            "trainer_performance": trainer_performance,
            "hippodrome_performance": hippodrome_performance,
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
