"""Tâches Celery pour le Machine Learning et les prédictions."""

import json
import logging
from collections import Counter
from datetime import date, datetime, timedelta
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.core.database import SessionLocal
from app.ml.predictor import RacePredictionService
from app.ml.training import ModelTrainer
from app.models.course import Course, CourseStatus, StartType
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


def _categorise_course_distance(distance: Optional[int]) -> str:
    """Classe les distances officielles en familles d'effort comparables."""

    if not distance:
        return "unknown"

    if distance < 1600:
        return "short_distance"

    if distance <= 2400:
        return "middle_distance"

    return "long_distance"


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
        discipline_breakdown: Dict[str, Dict[str, object]] = {}
        distance_breakdown: Dict[str, Dict[str, object]] = {}
        surface_breakdown: Dict[str, Dict[str, object]] = {}
        prize_money_breakdown: Dict[str, Dict[str, object]] = {}
        race_category_breakdown: Dict[str, Dict[str, object]] = {}
        race_class_breakdown: Dict[str, Dict[str, object]] = {}
        value_bet_breakdown: Dict[str, Dict[str, object]] = {}
        field_size_breakdown: Dict[str, Dict[str, object]] = {}
        draw_breakdown: Dict[str, Dict[str, object]] = {}
        start_type_breakdown: Dict[str, Dict[str, object]] = {}
        rest_period_breakdown: Dict[str, Dict[str, object]] = {}
        jockey_breakdown: Dict[str, Dict[str, object]] = {}
        trainer_breakdown: Dict[str, Dict[str, object]] = {}
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

        daily_performance = _summarise_daily_performance(daily_breakdown)
        discipline_performance = _summarise_segment_performance(discipline_breakdown)
        distance_performance = _summarise_distance_performance(distance_breakdown)
        surface_performance = _summarise_segment_performance(surface_breakdown)
        prize_money_performance = _summarise_prize_money_performance(
            prize_money_breakdown
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
            "daily_performance": daily_performance,
            "discipline_performance": discipline_performance,
            "distance_performance": distance_performance,
            "surface_performance": surface_performance,
            "prize_money_performance": prize_money_performance,
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
            "threshold_recommendations": threshold_recommendations,
            "betting_value_analysis": betting_value_analysis,
            "odds_alignment": odds_alignment,
            "lift_analysis": lift_analysis,
            "precision_recall_curve": precision_recall_table,
            "roc_curve": roc_curve_points,
            "daily_performance": daily_performance,
            "discipline_performance": discipline_performance,
            "distance_performance": distance_performance,
            "surface_performance": surface_performance,
            "prize_money_performance": prize_money_performance,
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
            "threshold_recommendations": threshold_recommendations,
            "betting_value_analysis": betting_value_analysis,
            "odds_alignment": odds_alignment,
            "lift_analysis": lift_analysis,
            "daily_performance": daily_performance,
            "distance_performance": distance_performance,
            "draw_performance": draw_performance,
            "prize_money_performance": prize_money_performance,
            "model_version_breakdown": dict(model_versions),
            "model_version_performance": model_version_performance,
            "rest_period_performance": rest_period_performance,
            "jockey_performance": jockey_performance,
            "trainer_performance": trainer_performance,
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
