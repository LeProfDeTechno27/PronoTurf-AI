"""Tâches Celery pour le Machine Learning et les prédictions."""

import json
import logging
from collections import Counter
from datetime import date, datetime, time, timedelta
from math import ceil, sqrt, log2
from statistics import median
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


def _compute_matthews_correlation(
    true_negative: int,
    false_positive: int,
    false_negative: int,
    true_positive: int,
) -> Optional[float]:
    """Calcule le coefficient de corrélation de Matthews en gérant les cas dégénérés.

    Cet indicateur fournit une mesure équilibrée de la qualité de la classification
    binaire en tenant compte simultanément de la précision sur les classes positives
    et négatives. Lorsque l'un des termes du dénominateur est nul (aucun positif ou
    négatif prédit/observé), le score n'est pas défini et nous retournons ``None``
    pour éviter une division par zéro.
    """

    denominator = sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )

    if denominator == 0:
        return None

    numerator = (true_positive * true_negative) - (false_positive * false_negative)
    return numerator / denominator


def _compute_binary_classification_insights(
    true_negative: int,
    false_positive: int,
    false_negative: int,
    true_positive: int,
) -> Dict[str, Optional[float]]:
    """Dérive des métriques complémentaires pour équilibrer le diagnostic binaire.

    En plus du rappel (déjà exposé via ``recall``), les opérateurs ont besoin d'une
    visibilité sur la capacité du modèle à filtrer correctement les faux positifs
    et à confirmer les prédictions négatives. Cette fonction calcule donc la
    spécificité (taux de vrais négatifs), le taux de faux positifs, la valeur
    prédictive négative ainsi que la balanced accuracy qui combine rappel et
    spécificité. Tous les calculs gèrent explicitement les cas dégénérés pour
    éviter les divisions par zéro.
    """

    total_negatives = true_negative + false_positive
    total_predicted_negative = true_negative + false_negative
    total_actual_positive = true_positive + false_negative

    specificity = (
        true_negative / total_negatives if total_negatives > 0 else None
    )
    false_positive_rate = (
        false_positive / total_negatives if total_negatives > 0 else None
    )
    negative_predictive_value = (
        true_negative / total_predicted_negative
        if total_predicted_negative > 0
        else None
    )
    sensitivity = (
        true_positive / total_actual_positive if total_actual_positive > 0 else None
    )

    balanced_accuracy = (
        (specificity + sensitivity) / 2
        if specificity is not None and sensitivity is not None
        else None
    )

    return {
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "negative_predictive_value": negative_predictive_value,
        "balanced_accuracy": balanced_accuracy,
    }


def _compute_percentile(sorted_values: List[float], percentile: float) -> Optional[float]:
    """Calcule un percentile (0-1) par interpolation linéaire."""

    if not sorted_values:
        return None

    # On borne explicitement la valeur demandée pour éviter les dépassements.
    percentile = max(0.0, min(1.0, percentile))

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = percentile * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index

    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    return lower_value + (upper_value - lower_value) * weight


def _summarise_probability_distribution(
    truths: List[int], scores: List[float]
) -> Dict[str, object]:
    """Analyse la distribution des probabilités prédites et leur séparation.

    L'objectif est d'offrir un diagnostic rapide de la calibration globale :
    - quelle est la dispersion des probabilités émises par le modèle ?
    - les gagnants observés reçoivent-ils des scores nettement supérieurs aux
      perdants ?
    - quelle marge existe-t-il entre les médianes positives et négatives ?

    Ces éléments complètent les métriques agrégées (précision, rappel, Brier) en
    mettant en évidence d'éventuels recouvrements entre gagnants/perdants malgré
    des taux globaux stables.
    """

    # Conversion défensive : certaines valeurs peuvent provenir de ``Decimal``.
    cleaned_scores = [float(score) for score in scores if score is not None]
    positives = [
        float(score)
        for score, truth in zip(scores, truths)
        if score is not None and int(truth) == 1
    ]
    negatives = [
        float(score)
        for score, truth in zip(scores, truths)
        if score is not None and int(truth) == 0
    ]

    def _build_stats(values: List[float]) -> Dict[str, object]:
        """Construit les statistiques descriptives pour une liste de scores."""

        if not values:
            return {
                "count": 0,
                "average": None,
                "median": None,
                "p10": None,
                "p90": None,
                "min": None,
                "max": None,
                "std": None,
            }

        ordered = sorted(values)
        mean_value = _safe_average(ordered)
        std_value: Optional[float] = None
        if mean_value is not None:
            if len(ordered) == 1:
                std_value = 0.0
            else:
                variance = sum((value - mean_value) ** 2 for value in ordered) / len(ordered)
                std_value = sqrt(variance)

        return {
            "count": len(ordered),
            "average": mean_value,
            "median": float(median(ordered)),
            "p10": _compute_percentile(ordered, 0.10),
            "p90": _compute_percentile(ordered, 0.90),
            "min": float(ordered[0]),
            "max": float(ordered[-1]),
            "std": std_value,
        }

    overall_stats = _build_stats(cleaned_scores)
    positive_stats = _build_stats(positives)
    negative_stats = _build_stats(negatives)

    average_gap: Optional[float] = None
    if positive_stats["average"] is not None and negative_stats["average"] is not None:
        average_gap = positive_stats["average"] - negative_stats["average"]

    median_gap: Optional[float] = None
    if positive_stats["median"] is not None and negative_stats["median"] is not None:
        median_gap = positive_stats["median"] - negative_stats["median"]

    return {
        "overall": overall_stats,
        "positives": positive_stats,
        "negatives": negative_stats,
        "average_gap": average_gap,
        "median_gap": median_gap,
    }


def _compute_normalised_dcg(
    ranked_entries: List[Tuple[Dict[str, object], Optional[int]]],
    cutoff: int,
) -> Optional[float]:
    """Calcule un NDCG@k binaire basé sur la position finale des partants.

    Le score retourne ``1.0`` lorsque les chevaux réellement placés (≤ 3) sont
    correctement classés dans les ``cutoff`` premiers pronostics, ``0`` quand ils
    sont relégués en bas de liste. S'il n'existe aucune pertinence dans les
    données (aucun podium identifié), la fonction renvoie ``1.0`` par convention
    afin de ne pas pénaliser une réunion sans signal exploitable.
    """

    if not ranked_entries:
        return None

    effective_cutoff = max(1, cutoff)

    # Transforme les positions finales en pertinence (1 si podium, 0 sinon).
    relevances = [
        1 if final_position is not None and final_position <= 3 else 0
        for _, final_position in ranked_entries
    ]

    predicted_relevances = relevances[:effective_cutoff]
    ideal_relevances = sorted(relevances, reverse=True)[:effective_cutoff]

    def _dcg(values: List[int]) -> float:
        """Calcule le Discounted Cumulative Gain d'une liste binaire."""

        return sum(
            relevance / log2(index + 2)
            for index, relevance in enumerate(values)
            if relevance
        )

    ideal_dcg = _dcg(ideal_relevances)
    if ideal_dcg == 0:
        return 1.0

    return _dcg(predicted_relevances) / ideal_dcg


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


def _decompose_brier_score(
    calibration_rows: List[Dict[str, object]],
    *,
    base_rate: Optional[float],
    brier_score: Optional[float],
) -> Dict[str, Optional[float]]:
    """Décompose la Brier score en composantes de Murphy pour guider les actions.

    La décomposition distingue trois éléments :

    - ``reliability`` mesure l'écart moyen entre probabilité prédite et fréquence
      observée sur chaque quantile. Plus il est faible, meilleure est la
      calibration.
    - ``resolution`` capture la capacité du modèle à séparer des groupes avec des
      fréquences observées différentes. Plus il est élevé, plus l'information
      apportée par les probabilités est discriminante.
    - ``uncertainty`` correspond à la variance intrinsèque du jeu de données
      (proportion de victoires). Il sert de référence pour calculer un score de
      compétence (« *skill score* »).

    Les opérateurs peuvent ainsi identifier si une dérive de la Brier score
    provient d'un manque de calibration (reliability), d'une perte de
    différenciation (resolution) ou simplement d'une variation du taux de
    gagnants (uncertainty). Lorsque le taux de gagnants est extrême (0 % ou 100
    %), la composante d'incertitude annule toute interprétation et nous
    neutralisons le *skill score*.
    """

    if not calibration_rows or base_rate is None or brier_score is None:
        return {
            "reliability": None,
            "resolution": None,
            "uncertainty": None,
            "skill_score": None,
            "bins": [],
        }

    total = sum(int(row.get("count", 0) or 0) for row in calibration_rows)
    if not total:
        return {
            "reliability": None,
            "resolution": None,
            "uncertainty": base_rate * (1 - base_rate),
            "skill_score": None,
            "bins": [],
        }

    reliability = 0.0
    resolution = 0.0
    decomposition_rows: List[Dict[str, object]] = []

    for row in calibration_rows:
        count = int(row.get("count", 0) or 0)
        if count <= 0:
            continue

        average_probability = row.get("average_probability")
        empirical_rate = row.get("empirical_rate")
        if average_probability is None or empirical_rate is None:
            continue

        weight = count / total
        reliability_contrib = weight * (average_probability - empirical_rate) ** 2
        resolution_contrib = weight * (empirical_rate - base_rate) ** 2

        reliability += reliability_contrib
        resolution += resolution_contrib

        decomposition_rows.append(
            {
                "bin": row.get("bin"),
                "count": count,
                "weight": weight,
                "average_probability": average_probability,
                "empirical_rate": empirical_rate,
                "reliability_contribution": reliability_contrib,
                "resolution_contribution": resolution_contrib,
            }
        )

    uncertainty = base_rate * (1 - base_rate)
    skill_score: Optional[float]
    if uncertainty and uncertainty > 0:
        skill_score = 1 - (brier_score / uncertainty)
    else:
        skill_score = None

    return {
        "reliability": reliability if decomposition_rows else None,
        "resolution": resolution if decomposition_rows else None,
        "uncertainty": uncertainty,
        "skill_score": skill_score,
        "bins": decomposition_rows,
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


def _compute_probability_edge(
    predicted_probability: Optional[object],
    market_odds: Optional[object],
) -> Optional[float]:
    """Calcule l'écart entre la probabilité du modèle et l'implicite du marché."""

    if predicted_probability is None:
        return None

    try:
        model_probability = float(predicted_probability)
    except (TypeError, ValueError):
        return None

    if model_probability < 0.0 or model_probability > 1.0:
        model_probability = max(0.0, min(model_probability, 1.0))

    try:
        odds_value = float(market_odds) if market_odds is not None else None
    except (TypeError, ValueError):
        odds_value = None

    if odds_value is None or odds_value <= 1.0:
        return None

    implied_probability = 1.0 / odds_value
    return model_probability - implied_probability


def _categorise_probability_edge(
    edge_value: Optional[float],
) -> Tuple[str, str]:
    """Regroupe l'écart modèle/marché en segments interprétables."""

    if edge_value is None:
        return "unknown_edge", "Écart de probabilité inconnu"

    if edge_value >= 0.18:
        return "strong_positive_edge", "Écart très favorable (≥ 18 pts)"

    if edge_value >= 0.08:
        return "positive_edge", "Écart favorable (8-18 pts)"

    if edge_value >= 0.03:
        return "slight_positive_edge", "Écart légèrement favorable (3-8 pts)"

    if edge_value <= -0.10:
        return "strong_negative_edge", "Écart très défavorable (≤ -10 pts)"

    if edge_value <= -0.04:
        return "negative_edge", "Écart défavorable (-10 à -4 pts)"

    return "neutral_edge", "Écart neutre (-4 à +3 pts)"


def _summarise_probability_edge_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Analyse la réussite par tranche d'écart avec les cotes publiques."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    edge_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    order_priority = {
        "strong_positive_edge": 0,
        "positive_edge": 1,
        "slight_positive_edge": 2,
        "neutral_edge": 3,
        "negative_edge": 4,
        "strong_negative_edge": 5,
        "unknown_edge": 6,
    }

    for segment, payload in sorted(
        breakdown.items(),
        key=lambda item: (order_priority.get(item[0], 99), item[0]),
    ):
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))
        edges = [float(value) for value in payload.get("edges", []) if value is not None]
        implied_probabilities = [
            float(value)
            for value in payload.get("implied_probabilities", [])
            if value is not None
        ]
        odds_values = [
            float(value) for value in payload.get("odds", []) if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["courses"] = len(courses)
        summary["horses"] = len(horses)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )

        if edges:
            ordered = sorted(edges)
            midpoint = len(ordered) // 2
            if len(ordered) % 2 == 0:
                median_edge = (ordered[midpoint - 1] + ordered[midpoint]) / 2
            else:
                median_edge = ordered[midpoint]

            summary.update(
                {
                    "average_edge": _safe_average(edges),
                    "median_edge": median_edge,
                    "min_edge": ordered[0],
                    "max_edge": ordered[-1],
                }
            )
        else:
            summary.update(
                {
                    "average_edge": None,
                    "median_edge": None,
                    "min_edge": None,
                    "max_edge": None,
                }
            )

        summary["average_implied_probability"] = _safe_average(implied_probabilities)
        summary["average_odds"] = _safe_average(odds_values)

        edge_metrics[segment] = summary

    return edge_metrics


def _categorise_probability_error(
    absolute_error: Optional[object],
) -> Tuple[str, str, Optional[float]]:
    """Classe l'écart absolu probabilité / réalité en fourchettes parlantes."""

    if absolute_error is None:
        return "error_unknown", "Erreur inconnue", None

    try:
        value = float(absolute_error)
    except (TypeError, ValueError):  # pragma: no cover - robustesse sur données inattendues
        return "error_unknown", "Erreur inconnue", None

    # Les écarts théoriques sont bornés dans [0, 1]. On contraint donc la valeur
    # afin d'éviter qu'une dérive d'arrondi ne vienne polluer la catégorisation.
    value = max(0.0, min(value, 1.0))

    if value <= 0.10:
        return "error_0_10", "Ultra précis (≤ 10 pts)", value

    if value <= 0.20:
        return "error_10_20", "Fiable (10-20 pts)", value

    if value <= 0.35:
        return "error_20_35", "Approximation (20-35 pts)", value

    if value <= 0.50:
        return "error_35_50", "Fragile (35-50 pts)", value

    return "error_50_plus", "À surveiller (> 50 pts)", value


def _summarise_probability_error_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Analyse la précision réelle selon l'écart absolu des probabilités."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    error_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        errors = [
            float(value)
            for value in payload.get("errors", [])
            if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["courses"] = len(courses)
        summary["observed_positive_rate"] = sum(truths) / len(truths) if truths else None

        if errors:
            ordered = sorted(errors)
            midpoint = len(ordered) // 2
            if len(ordered) % 2 == 0:
                median_error = (ordered[midpoint - 1] + ordered[midpoint]) / 2
            else:
                median_error = ordered[midpoint]

            summary.update(
                {
                    "average_absolute_error": _safe_average(errors),
                    "median_absolute_error": median_error,
                    "min_absolute_error": ordered[0],
                    "max_absolute_error": ordered[-1],
                }
            )
        else:
            summary.update(
                {
                    "average_absolute_error": None,
                    "median_absolute_error": None,
                    "min_absolute_error": None,
                    "max_absolute_error": None,
                }
            )

        error_metrics[segment] = summary

    return error_metrics


def _categorise_prediction_outcome(
    predicted_label: int,
    actual_label: int,
) -> Tuple[str, str]:
    """Retourne une clé et un libellé pour chaque couple prédiction / réalité."""

    if actual_label == 1 and predicted_label == 1:
        return "true_positive", "Succès (positif confirmé)"

    if actual_label == 0 and predicted_label == 0:
        return "true_negative", "Succès (négatif confirmé)"

    if actual_label == 0 and predicted_label == 1:
        return "false_positive", "Faux positif (à filtrer)"

    return "false_negative", "Faux négatif (à investiguer)"


def _summarise_prediction_outcome_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Dresse un tableau de bord par type d'issue (TP/FP/TN/FN)."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    outcome_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = [
            float(value)
            for value in payload.get("scores", [])
            if value is not None
        ]
        courses: Set[int] = set(payload.get("courses", set()))
        pronostics: Set[int] = set(payload.get("pronostics", set()))
        model_versions: Set[str] = set(payload.get("model_versions", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "label": payload.get("label", segment),
                "share": (summary["samples"] / total_samples) if total_samples else None,
                "courses": len(courses),
                "pronostics": len(pronostics),
                "model_version_count": len(model_versions),
                "model_versions": sorted(model_versions),
                "observed_positive_rate": sum(truths) / len(truths) if truths else None,
                "prediction_rate": sum(predicted) / len(predicted) if predicted else None,
            }
        )

        average_probability = _safe_average(scores)
        observed_rate = summary["observed_positive_rate"]
        summary["average_probability"] = average_probability

        if average_probability is not None and observed_rate is not None:
            summary["average_calibration_gap"] = average_probability - observed_rate
        else:
            summary["average_calibration_gap"] = None

        if truths:
            summary["accuracy_within_segment"] = sum(
                1 for truth, prediction in zip(truths, predicted) if truth == prediction
            ) / len(truths)
        else:
            summary["accuracy_within_segment"] = None

        outcome_metrics[segment] = summary

    return outcome_metrics


def _categorise_probability_margin(
    primary_probability: Optional[object],
    secondary_probability: Optional[object],
) -> Tuple[str, str, Optional[float]]:
    """Classe l'écart entre les deux meilleurs pronostics d'une course."""

    if primary_probability is None:
        return "margin_unknown", "Marge inconnue", None

    try:
        top_probability = float(primary_probability)
    except (TypeError, ValueError):  # pragma: no cover - robustesse face à des valeurs corrompues
        return "margin_unknown", "Marge inconnue", None

    # Lorsque la deuxième cote est absente (course mono-partant ou données incomplètes),
    # on conserve un segment dédié afin de ne pas mélanger ces cas aux marges habituelles.
    if secondary_probability is None:
        return "margin_singleton", "Sans challenger déclaré", top_probability

    try:
        runner_up_probability = float(secondary_probability)
    except (TypeError, ValueError):  # pragma: no cover - sécurité supplémentaire
        return "margin_unknown", "Marge inconnue", None

    margin = max(0.0, min(top_probability - runner_up_probability, 1.0))
    # Arrondi technique pour éviter qu'un flottant représenté comme 0.0500000000001
    # ne bascule dans la tranche supérieure alors qu'il devrait rester "≤ 5 pts".
    margin = round(margin, 10)

    if margin <= 0.05:
        return "margin_tight", "Très serré (≤ 5 pts)", margin

    if margin <= 0.10:
        return "margin_close", "Serré (5-10 pts)", margin

    if margin <= 0.20:
        return "margin_buffered", "Dégagé (10-20 pts)", margin

    return "margin_clear", "Très confortable (> 20 pts)", margin


def _summarise_probability_margin_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Analyse la fiabilité des pronostics selon la marge sur le dauphin."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    margin_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = [float(value) for value in payload.get("scores", []) if value is not None]
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))
        margins = [
            float(value)
            for value in payload.get("margins", [])
            if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "label": payload.get("label", segment),
                "share": (summary["samples"] / total_samples) if total_samples else None,
                "courses": len(courses),
                "horses": len(horses),
                "observed_positive_rate": sum(truths) / len(truths) if truths else None,
                "average_probability": _safe_average(scores),
            }
        )

        if margins:
            ordered_margins = sorted(margins)
            midpoint = len(ordered_margins) // 2
            if len(ordered_margins) % 2 == 0:
                median_margin = (
                    ordered_margins[midpoint - 1] + ordered_margins[midpoint]
                ) / 2
            else:
                median_margin = ordered_margins[midpoint]

            summary.update(
                {
                    "average_margin": _safe_average(margins),
                    "median_margin": median_margin,
                    "min_margin": ordered_margins[0],
                    "max_margin": ordered_margins[-1],
                }
            )
        else:
            summary.update(
                {
                    "average_margin": None,
                    "median_margin": None,
                    "min_margin": None,
                    "max_margin": None,
                }
            )

        margin_metrics[segment] = summary

    return margin_metrics


def _categorise_favourite_alignment(
    model_entry: Optional[Dict[str, object]],
    pmu_entry: Optional[Dict[str, object]],
) -> Tuple[str, str]:
    """Identifie si le favori modèle est aligné avec le favori PMU."""

    if pmu_entry is None:
        return "pmu_missing", "Favori PMU indisponible"

    if model_entry is None:
        return "model_missing", "Favori modèle indisponible"

    try:
        model_horse = model_entry.get("horse_id")
        pmu_horse = pmu_entry.get("horse_id")
    except AttributeError:  # pragma: no cover - sécurité sur structures inattendues
        return "pmu_missing", "Favori PMU indisponible"

    if model_horse is not None and pmu_horse is not None and model_horse == pmu_horse:
        return "aligned", "Favori modèle aligné sur les cotes PMU"

    return "divergent", "Favori modèle différent du PMU"


def _summarise_favourite_alignment_performance(
    breakdown: Dict[str, Dict[str, object]],
    total_courses: int,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Synthétise le comportement du favori modèle vs le favori marché."""

    if not breakdown:
        return {}

    alignment_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        courses: Set[int] = set(payload.get("courses", set()))
        course_count = len(courses)
        pmu_courses = int(payload.get("pmu_courses", 0))

        model_truths = [int(value) for value in payload.get("model_truths", [])]
        model_predictions = [int(value) for value in payload.get("model_predictions", [])]
        model_scores = [float(value) for value in payload.get("model_scores", [])]

        summary = _summarise_group_performance(model_truths, model_predictions, model_scores)
        summary.update(
            {
                "label": payload.get("label", segment),
                "courses": course_count,
                "share": (course_count / total_courses) if total_courses else None,
                "model_win_rate": (
                    payload.get("model_wins", 0) / course_count if course_count else None
                ),
                "pmu_win_rate": (
                    payload.get("pmu_wins", 0) / pmu_courses if pmu_courses else None
                ),
                "aligned_winner_rate": (
                    payload.get("aligned_wins", 0) / course_count if course_count else None
                ),
                "average_model_probability": _safe_average(model_scores),
                "average_pmu_probability": _safe_average(
                    [float(value) for value in payload.get("pmu_scores", [])]
                ),
                "average_pmu_odds": _safe_average(
                    [float(value) for value in payload.get("pmu_odds", [])]
                ),
                "average_probability_gap": _safe_average(
                    [float(value) for value in payload.get("probability_gaps", [])]
                ),
                "average_pmu_rank_in_model": _safe_average(
                    [float(value) for value in payload.get("pmu_ranks", [])]
                ),
            }
        )

        pmu_truths = [int(value) for value in payload.get("pmu_truths", [])]
        summary["pmu_positive_rate"] = (
            sum(pmu_truths) / len(pmu_truths) if pmu_truths else None
        )

        alignment_metrics[segment] = summary

    return alignment_metrics


def _summarise_winner_rankings(
    winner_ranks: List[int],
    total_courses: int,
) -> Dict[str, Optional[float]]:
    """Condense la position des vainqueurs dans le classement du modèle.

    Cette synthèse fournit un complément indispensable aux indicateurs « Top 1 »
    et « Top 3 » déjà exposés :

    * ``mean_rank`` et ``median_rank`` révèlent la place typique du gagnant dans
      le classement probabiliste ;
    * ``mean_reciprocal_rank`` (MRR) mesure la qualité moyenne de ranking sur
      l'ensemble des courses, en valorisant fortement les vainqueurs trouvés en
      tête de liste ;
    * ``share_top1`` et ``share_top3`` suivent la fréquence des gagnants situés
      dans les toutes premières positions, normalisées par le volume total de
      courses évaluées afin de suivre aisément l'évolution d'une exécution à
      l'autre ;
    * ``distribution`` conserve un histogramme ordonné des rangs observés pour
      faciliter l'analyse terrain.
    """

    if not winner_ranks:
        return {
            "evaluated_courses": total_courses,
            "mean_rank": None,
            "median_rank": None,
            "mean_reciprocal_rank": None,
            "share_top1": None,
            "share_top3": None,
            "distribution": {},
        }

    distribution = Counter(winner_ranks)
    evaluated = len(winner_ranks)
    mean_rank = sum(winner_ranks) / evaluated
    median_rank = float(median(winner_ranks))
    mean_reciprocal_rank = sum(1.0 / rank for rank in winner_ranks) / evaluated

    top1_share = (distribution.get(1, 0) / total_courses) if total_courses else None
    top3_share = (
        sum(count for rank, count in distribution.items() if rank <= 3) / total_courses
        if total_courses
        else None
    )

    return {
        "evaluated_courses": total_courses,
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "mean_reciprocal_rank": mean_reciprocal_rank,
        "share_top1": top1_share,
        "share_top3": top3_share,
        "distribution": {
            f"rank_{rank}": count for rank, count in sorted(distribution.items())
        },
    }


def _summarise_topn_performance(
    topn_tracking: Dict[int, Dict[str, object]],
    total_courses: int,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Agrège les statistiques de réussite pour les ``Top N`` du modèle.

    Cette synthèse complète les indicateurs globaux (Top 1 / Top 3) en conservant
    un historique par taille de panier. Chaque entrée rapporte :

    * la proportion de courses couvertes par le segment (``coverage_rate``) ;
    * la fréquence à laquelle un gagnant ou un placé (Top 3) est capturé ;
    * la probabilité moyenne/médiane proposée sur les chevaux retenus ;
    * la meilleure position finale moyenne/médiane observée dans le segment ;
    * les volumes exacts de courses et de sélections contribuant au calcul.
    """

    summary: Dict[str, Dict[str, Optional[float]]] = {}

    for top_n, payload in sorted(topn_tracking.items()):
        courses = int(payload.get("courses", 0))
        probabilities = [
            float(value)
            for value in payload.get("probabilities", [])
            if value is not None
        ]
        best_finishes = [
            float(value)
            for value in payload.get("best_finishes", [])
            if value is not None
        ]
        winner_hits = int(payload.get("winner_hits", 0))
        place_hits = int(payload.get("place_hits", 0))

        summary[f"top{top_n}"] = {
            "label": f"Top {top_n}",
            "courses_covered": courses,
            "coverage_rate": (courses / total_courses) if total_courses else None,
            "samples": len(probabilities),
            "winner_hits": winner_hits,
            "winner_hit_rate": (winner_hits / courses) if courses else None,
            "top3_hits": place_hits,
            "top3_hit_rate": (place_hits / courses) if courses else None,
            "average_probability": _safe_average(probabilities),
            "median_probability": float(median(probabilities)) if probabilities else None,
            "best_finish_average": _safe_average(best_finishes),
            "best_finish_median": float(median(best_finishes)) if best_finishes else None,
        }

    return summary


def _compute_spearman_correlation(
    predicted_probabilities: List[float],
    finish_positions: List[int],
) -> Optional[float]:
    """Calcule la corrélation de Spearman entre le ranking prévu et l'arrivée."""

    n = len(predicted_probabilities)
    if n < 2 or n != len(finish_positions):
        return None

    # Classement du modèle : probabilité décroissante (0 -> favori).
    predicted_ranks = [0] * n
    for rank, index in enumerate(
        sorted(range(n), key=lambda idx: (-predicted_probabilities[idx], idx)), start=1
    ):
        predicted_ranks[index] = rank

    # Classement réel : position d'arrivée croissante (1 -> vainqueur).
    actual_ranks = [0] * n
    for rank, index in enumerate(
        sorted(range(n), key=lambda idx: (finish_positions[idx], idx)), start=1
    ):
        actual_ranks[index] = rank

    denominator = n * (n**2 - 1)
    if denominator == 0:
        return None

    diff_squared = sum(
        (predicted_ranks[idx] - actual_ranks[idx]) ** 2 for idx in range(n)
    )
    return 1 - (6 * diff_squared) / denominator


def _summarise_rank_correlation_performance(
    ranking_samples: Dict[int, Dict[str, object]]
) -> Dict[str, object]:
    """Synthétise la corrélation rang/pronostic par course et globalement."""

    course_details: Dict[str, Dict[str, object]] = {}
    spearman_scores: List[float] = []

    for course_id, payload in ranking_samples.items():
        probabilities = [
            float(value)
            for value in payload.get("probabilities", [])
            if value is not None
        ]
        finish_positions = [
            int(value)
            for value in payload.get("finish_positions", [])
            if value is not None
        ]

        correlation = _compute_spearman_correlation(probabilities, finish_positions)
        if correlation is None:
            continue

        key = payload.get("key") or f"course_{course_id}"
        label = payload.get("label") or key.replace("_", " ")

        course_details[key] = {
            "course_id": course_id,
            "label": label,
            "runner_count": len(probabilities),
            "spearman": correlation,
        }
        spearman_scores.append(correlation)

    evaluated_courses = len(spearman_scores)
    tracked_courses = len(ranking_samples)

    return {
        "tracked_courses": tracked_courses,
        "evaluated_courses": evaluated_courses,
        "courses_missing_results": tracked_courses - evaluated_courses,
        "average_spearman": _safe_average(spearman_scores),
        "median_spearman": float(median(spearman_scores)) if spearman_scores else None,
        "best_spearman": max(spearman_scores) if spearman_scores else None,
        "worst_spearman": min(spearman_scores) if spearman_scores else None,
        "course_details": course_details,
    }


def _summarise_rank_error_metrics(
    rank_error_tracking: Dict[int, Dict[str, object]]
) -> Dict[str, object]:
    """Mesure l'écart de classement entre le modèle et l'arrivée officielle."""

    tracked_courses = len(rank_error_tracking)
    total_samples = 0
    total_perfect = 0
    absolute_errors_all: List[float] = []
    squared_errors_all: List[float] = []
    signed_errors_all: List[float] = []
    course_details: Dict[str, Dict[str, object]] = {}

    for course_id, payload in rank_error_tracking.items():
        absolute_errors = [
            float(value)
            for value in payload.get("absolute_errors", [])
            if value is not None
        ]
        if not absolute_errors:
            continue

        squared_errors = [
            float(value)
            for value in payload.get("squared_errors", [])
            if value is not None
        ]
        if not squared_errors:
            squared_errors = [error**2 for error in absolute_errors]

        signed_errors = [
            float(value)
            for value in payload.get("signed_errors", [])
            if value is not None
        ]

        samples = len(absolute_errors)
        perfect_predictions = int(payload.get("perfect_predictions", 0))
        runner_count = int(payload.get("runner_count", samples))

        total_samples += samples
        total_perfect += perfect_predictions
        absolute_errors_all.extend(absolute_errors)
        squared_errors_all.extend(squared_errors)
        signed_errors_all.extend(signed_errors)

        key = payload.get("key") or f"course_{course_id}"
        label = payload.get("label") or key.replace("_", " ")

        course_details[key] = {
            "course_id": course_id,
            "label": label,
            "samples": samples,
            "runner_count": runner_count,
            "mean_absolute_error": sum(absolute_errors) / samples,
            "median_absolute_error": float(median(absolute_errors)),
            "max_absolute_error": max(absolute_errors),
            "rmse": sqrt(sum(squared_errors) / samples),
            "perfect_predictions": perfect_predictions,
            "perfect_share": (perfect_predictions / samples) if samples else None,
            "average_bias": (
                sum(signed_errors) / samples if signed_errors else None
            ),
            "median_bias": (
                float(median(signed_errors)) if signed_errors else None
            ),
        }

    evaluated_courses = len(course_details)

    if total_samples == 0:
        return {
            "tracked_courses": tracked_courses,
            "evaluated_courses": evaluated_courses,
            "courses_missing_results": tracked_courses - evaluated_courses,
            "samples": 0,
            "mean_absolute_error": None,
            "median_absolute_error": None,
            "rmse": None,
            "max_absolute_error": None,
            "perfect_predictions": 0,
            "perfect_share": None,
            "average_bias": None,
            "median_bias": None,
            "course_details": course_details,
        }

    return {
        "tracked_courses": tracked_courses,
        "evaluated_courses": evaluated_courses,
        "courses_missing_results": tracked_courses - evaluated_courses,
        "samples": total_samples,
        "mean_absolute_error": sum(absolute_errors_all) / total_samples,
        "median_absolute_error": float(median(absolute_errors_all)),
        "rmse": sqrt(sum(squared_errors_all) / total_samples)
        if squared_errors_all
        else None,
        "max_absolute_error": max(absolute_errors_all) if absolute_errors_all else None,
        "perfect_predictions": total_perfect,
        "perfect_share": total_perfect / total_samples,
        "average_bias": (
            sum(signed_errors_all) / total_samples if signed_errors_all else None
        ),
        "median_bias": (
            float(median(signed_errors_all)) if signed_errors_all else None
        ),
        "course_details": course_details,
    }


def _categorise_experience_level(
    career_starts: Optional[object],
    *,
    prefix: str,
    display_name: str,
) -> Tuple[str, str, Optional[int]]:
    """Catégorise le volume de courses disputées pour un type d'acteur donné."""

    if career_starts is None:
        return f"{prefix}_experience_unknown", f"{display_name} - expérience inconnue", None

    try:
        value = int(career_starts)
    except (TypeError, ValueError):  # pragma: no cover - robustesse vis-à-vis de valeurs corrompues
        return f"{prefix}_experience_unknown", f"{display_name} - expérience inconnue", None

    value = max(0, value)

    if value <= 150:
        return (
            f"{prefix}_experience_rookie",
            f"{display_name} débutant (≤ 150 courses)",
            value,
        )

    if value <= 400:
        return (
            f"{prefix}_experience_progressing",
            f"{display_name} en progression (151-400 courses)",
            value,
        )

    if value <= 800:
        return (
            f"{prefix}_experience_confirmed",
            f"{display_name} confirmé (401-800 courses)",
            value,
        )

    if value <= 1200:
        return (
            f"{prefix}_experience_expert",
            f"{display_name} expert (801-1 200 courses)",
            value,
        )

    return (
        f"{prefix}_experience_veteran",
        f"{display_name} vétéran (> 1 200 courses)",
        value,
    )


def _categorise_jockey_experience(
    career_starts: Optional[object],
) -> Tuple[str, str, Optional[int]]:
    """Découpe l'expérience d'un jockey en bandes exploitables pour le monitoring."""

    return _categorise_experience_level(
        career_starts,
        prefix="jockey",
        display_name="Jockey",
    )


def _categorise_trainer_experience(
    career_starts: Optional[object],
) -> Tuple[str, str, Optional[int]]:
    """Découpe l'expérience d'un entraîneur en bandes exploitables pour le monitoring."""

    return _categorise_experience_level(
        career_starts,
        prefix="trainer",
        display_name="Entraîneur",
    )


def _summarise_experience_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Synthétise la performance par niveau d'expérience (jockeys/entraîneurs)."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    experience_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        actors: Set[int] = set(payload.get("actors", set()))
        starts = [
            int(value)
            for value in payload.get("career_starts", [])
            if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "label": payload.get("label", segment),
                "share": (len(truths) / total_samples) if total_samples else None,
                "courses": len(courses),
                "actors": len(actors),
                "observed_positive_rate": sum(truths) / len(truths) if truths else None,
                "average_career_starts": _safe_average(starts),
                "min_career_starts": min(starts) if starts else None,
                "max_career_starts": max(starts) if starts else None,
            }
        )

        experience_metrics[segment] = summary

    return experience_metrics


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


def _describe_prediction_confidence_level(level: str) -> str:
    """Retourne un libellé lisible pour un niveau de confiance brut."""

    mapping = {
        "high": "Confiance élevée",
        "medium": "Confiance moyenne",
        "low": "Confiance faible",
        "unknown": "Confiance inconnue",
    }
    return mapping.get(level, f"Confiance {level}")


def _summarise_prediction_confidence_performance(
    breakdown: Dict[str, Dict[str, object]],
    total_samples: int,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Dresse un panorama complet des performances par niveau de confiance."""

    if not breakdown:
        return {}

    confidence_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for level in sorted(breakdown.keys()):
        payload = breakdown[level]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        pronostics: Set[int] = set(payload.get("pronostics", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        sample_count = len(truths)
        summary.update(
            {
                "label": payload.get("label")
                or _describe_prediction_confidence_level(level),
                "share": (sample_count / total_samples) if total_samples else None,
                "observed_positive_rate": (
                    sum(truths) / sample_count if sample_count else None
                ),
                "courses": len(courses),
                "pronostics": len(pronostics),
            }
        )

        confidence_metrics[level] = summary

    return confidence_metrics


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


def _summarise_discipline_surface_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Analyse les combinaisons discipline/surface pour détecter les biais croisés."""

    if not breakdown:
        return {}

    combined_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for identifier in sorted(breakdown.keys()):
        payload = breakdown[identifier]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        distances = [
            float(value)
            for value in payload.get("distances", [])
            if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "label": payload.get("label") or identifier,
                "discipline": payload.get("discipline"),
                "surface": payload.get("surface"),
                "courses": len(courses),
                "reunions": len(reunions),
                "average_distance": _safe_average(distances),
                "min_distance": min(distances) if distances else None,
                "max_distance": max(distances) if distances else None,
                "observed_positive_rate": (sum(truths) / len(truths)) if truths else None,
            }
        )

        combined_metrics[identifier] = summary

    return combined_metrics


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


def _summarise_jockey_trainer_performance(
    breakdown: Dict[str, Dict[str, Any]],
    *,
    top_n: int = 8,
    min_samples: int = 2,
) -> List[Dict[str, Any]]:
    """Analyse les binômes jockey/entraîneur pour détecter les synergies fortes."""

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
        jockeys: Set[int] = set(payload.get("jockeys", set()))
        trainers: Set[int] = set(payload.get("trainers", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "identifier": identifier,
                "label": label,
                "jockey_label": payload.get("jockey_label"),
                "trainer_label": payload.get("trainer_label"),
                "samples": len(truths),
                "courses": len(courses),
                "horses": len(horses),
                "jockeys": len(jockeys),
                "trainers": len(trainers),
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


def _summarise_owner_trainer_performance(
    breakdown: Dict[str, Dict[str, Any]],
    *,
    top_n: int = 8,
    min_samples: int = 2,
) -> List[Dict[str, Any]]:
    """Identifie les tandems propriétaire/entraîneur les plus performants."""

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
        owners: Set[str] = set(payload.get("owners", set()))
        trainers: Set[int] = set(payload.get("trainers", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "identifier": identifier,
                "label": label,
                "owner_label": payload.get("owner_label"),
                "trainer_label": payload.get("trainer_label"),
                "samples": len(truths),
                "courses": len(courses),
                "horses": len(horses),
                "owners": len(owners),
                "trainers": len(trainers),
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


def _summarise_owner_jockey_performance(
    breakdown: Dict[str, Dict[str, Any]],
    *,
    top_n: int = 8,
    min_samples: int = 2,
) -> List[Dict[str, Any]]:
    """Classe les binômes propriétaire/jockey qui convertissent le plus souvent."""

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
        owners: Set[str] = set(payload.get("owners", set()))
        jockeys: Set[int] = set(payload.get("jockeys", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "identifier": identifier,
                "label": label,
                "owner_label": payload.get("owner_label"),
                "jockey_label": payload.get("jockey_label"),
                "samples": len(truths),
                "courses": len(courses),
                "horses": len(horses),
                "owners": len(owners),
                "jockeys": len(jockeys),
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


def _normalise_city_label(city: Optional[object]) -> Tuple[str, str]:
    """Normalise une ville d'hippodrome pour stabiliser les tableaux de bord."""

    if city is None:
        return "unknown", "Ville inconnue"

    raw_value = str(city).strip()

    if not raw_value:
        return "unknown", "Ville inconnue"

    # On génère un slug simple afin de garantir des clés déterministes.
    slug = "".join(char if char.isalnum() else "_" for char in raw_value)
    slug = slug.strip("_").lower() or "unknown"

    # Pour l'affichage on capitalise chaque mot afin de conserver les accents.
    label = " ".join(word.capitalize() for word in raw_value.split()) or raw_value

    return slug, label


def _normalise_country_label(country: Optional[object]) -> Tuple[str, str]:
    """Normalise un champ ``country`` pour garantir des regroupements cohérents."""

    if country is None:
        return "unknown", "Pays inconnu"

    raw_value = str(country).strip()

    if not raw_value:
        return "unknown", "Pays inconnu"

    iso_mapping = {
        "FR": "France",
        "FRA": "France",
        "BE": "Belgique",
        "BEL": "Belgique",
        "CH": "Suisse",
        "CHE": "Suisse",
        "GB": "Royaume-Uni",
        "UK": "Royaume-Uni",
        "GBR": "Royaume-Uni",
        "IE": "Irlande",
        "IRL": "Irlande",
        "ES": "Espagne",
        "ESP": "Espagne",
        "IT": "Italie",
        "ITA": "Italie",
    }

    upper_value = raw_value.upper()
    label = iso_mapping.get(upper_value, raw_value)

    slug_source = upper_value if len(upper_value) <= 3 else raw_value.lower()
    slug = "".join(char if char.isalnum() else "_" for char in slug_source).strip("_")

    if not slug:
        slug = "unknown"

    return slug.lower(), label


def _summarise_country_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble un tableau de bord des performances agrégées par pays."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    country_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        hippodromes: Set[int] = set(payload.get("hippodromes", set()))
        cities: Set[str] = set(payload.get("cities", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (
            summary["samples"] / total_samples if total_samples else None
        )
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["hippodromes"] = len(hippodromes)
        if cities:
            summary["cities"] = sorted(cities)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )

        country_metrics[segment] = summary

    return country_metrics


def _summarise_city_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Construit une synthèse des performances du modèle par ville."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    city_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        hippodromes: Set[int] = set(payload.get("hippodromes", set()))
        countries: Set[str] = set(payload.get("countries", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["hippodromes"] = len(hippodromes)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        if countries:
            summary["countries"] = sorted(countries)

        city_metrics[segment] = summary

    return city_metrics


def _normalise_api_source_label(source: Optional[object]) -> Tuple[str, str]:
    """Nettoie la source API d'une réunion pour stabiliser les regroupements."""

    if source is None:
        return "unknown", "Source API inconnue"

    raw_value = str(source).strip()
    if not raw_value:
        return "unknown", "Source API inconnue"

    slug = "".join(char if char.isalnum() else "_" for char in raw_value).strip("_")
    slug = slug.lower() or "unknown"

    return slug, raw_value


def _summarise_api_source_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble les performances agrégées par source d'alimentation API."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    api_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = [int(value) for value in payload.get("truths", [])]
        predicted = [int(value) for value in payload.get("predictions", [])]
        scores = [float(score) for score in payload.get("scores", [])]
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        hippodromes: Set[int] = set(payload.get("hippodromes", set()))
        pronostics: Set[int] = set(payload.get("pronostics", set()))
        model_versions: Set[str] = set(payload.get("model_versions", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        if hippodromes:
            summary["hippodromes"] = len(hippodromes)
        if pronostics:
            summary["pronostics"] = len(pronostics)
        if model_versions:
            summary["model_versions"] = sorted(model_versions)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )

        api_metrics[segment] = summary

    return api_metrics


def _normalise_owner_label(owner: Optional[object]) -> Tuple[str, str]:
    """Normalise un propriétaire pour stabiliser les ventilations dédiées."""

    if owner is None:
        return "unknown", "Propriétaire inconnu"

    raw_value = str(owner).strip()

    if not raw_value:
        return "unknown", "Propriétaire inconnu"

    slug = "".join(char if char.isalnum() else "_" for char in raw_value)
    slug = slug.strip("_").lower() or "unknown"

    return slug, raw_value


def _summarise_owner_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble un tableau de bord des performances agrégées par propriétaire."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    owner_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))
        trainers: Set[int] = set(payload.get("trainers", set()))
        jockeys: Set[int] = set(payload.get("jockeys", set()))
        hippodromes: Set[int] = set(payload.get("hippodromes", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["courses"] = len(courses)
        summary["horses"] = len(horses)
        if trainers:
            summary["trainers"] = len(trainers)
        if jockeys:
            summary["jockeys"] = len(jockeys)
        if hippodromes:
            summary["hippodromes"] = len(hippodromes)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )

        owner_metrics[segment] = summary

    return owner_metrics


def _categorise_prediction_rank(rank: Optional[int]) -> Tuple[str, str]:
    """Convertit un rang prédit en couple (clé technique, libellé lisible)."""

    if rank is None or rank <= 0:
        return "unknown", "Rang inconnu"
    if rank == 1:
        return "rank_1", "Sélection prioritaire (rang 1)"
    if rank == 2:
        return "rank_2", "Deuxième choix (rang 2)"
    if rank == 3:
        return "rank_3", "Troisième choix (rang 3)"
    if rank <= 5:
        return "rank_4_5", "Sélections intermédiaires (rangs 4-5)"
    return "rank_6_plus", "Sélections élargies (rang 6 et +)"


def _summarise_prediction_rank_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Synthétise les performances selon la hiérarchie des partants sélectionnés par le modèle."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    rank_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = [int(value) for value in payload.get("truths", [])]
        predicted = [int(value) for value in payload.get("predictions", [])]
        scores = [float(score) for score in payload.get("scores", [])]
        final_positions = [int(pos) for pos in payload.get("final_positions", [])]
        ranks = [int(value) for value in payload.get("ranks", [])]
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["courses"] = len(courses)
        if horses:
            summary["horses"] = len(horses)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["average_rank"] = _safe_average(ranks)

        if final_positions:
            summary["average_final_position"] = _safe_average(final_positions)
            summary["best_final_position"] = min(final_positions)
            summary["worst_final_position"] = max(final_positions)
        else:
            summary["average_final_position"] = None
            summary["best_final_position"] = None
            summary["worst_final_position"] = None

        rank_metrics[segment] = summary

    return rank_metrics


def _categorise_final_position(final_position: Optional[object]) -> Tuple[str, str]:
    """Range la position d'arrivée observée dans des segments lisibles."""

    if final_position is None:
        return "unknown", "Position inconnue"

    try:
        position_value = int(final_position)
    except (TypeError, ValueError):  # pragma: no cover - tolérance pour formats inattendus
        return "unknown", "Position inconnue"

    if position_value <= 0:
        return "unknown", "Position inconnue"

    if position_value == 1:
        return "winner", "Vainqueur"

    if position_value == 2:
        return "runner_up", "Deuxième place"

    if position_value == 3:
        return "third_place", "Troisième place"

    if position_value <= 6:
        return "top6", "Places 4 à 6"

    if position_value <= 10:
        return "top10", "Places 7 à 10"

    return "beyond_top10", "Au-delà de la 10e place"


def _summarise_final_position_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Synthétise les performances réelles selon la position d'arrivée enregistrée."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    position_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = [int(value) for value in payload.get("truths", [])]
        predicted = [int(value) for value in payload.get("predictions", [])]
        scores = [float(score) for score in payload.get("scores", [])]
        raw_positions = [
            int(position)
            for position in payload.get("positions", [])
            if position is not None
        ]
        position_values = [float(value) for value in raw_positions]
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        sample_count = len(truths)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (sample_count / total_samples) if total_samples else None
        summary["courses"] = len(courses)
        if horses:
            summary["horses"] = len(horses)
        summary["observed_positive_rate"] = (
            sum(truths) / sample_count if sample_count else None
        )
        summary["average_final_position"] = (
            _safe_average(position_values) if position_values else None
        )
        summary["best_final_position"] = min(raw_positions) if raw_positions else None
        summary["worst_final_position"] = max(raw_positions) if raw_positions else None
        summary["top3_rate"] = (
            sum(1 for value in raw_positions if value <= 3) / len(raw_positions)
            if raw_positions
            else None
        )

        position_metrics[segment] = summary

    return position_metrics


def _categorise_publication_lead_time(
    generated_at: Optional[datetime],
    race_date: Optional[date],
    scheduled_time: Optional[time],
) -> Tuple[str, str, Optional[float]]:
    """Détermine la fenêtre de publication en heures avant le départ officiel."""

    if not generated_at or not race_date:
        return "unknown", "Publication à délai inconnu", None

    if not isinstance(generated_at, datetime):
        return "unknown", "Publication à délai inconnu", None

    race_time = scheduled_time or time(12, 0)
    race_datetime = datetime.combine(race_date, race_time)
    lead_delta = race_datetime - generated_at
    lead_hours = lead_delta.total_seconds() / 3600

    if lead_hours < 0:
        return "post_race", "Publication postérieure à la course", lead_hours
    if lead_hours < 2:
        return "less_2h", "Publication < 2h avant départ", lead_hours
    if lead_hours < 6:
        return "between_2h_6h", "Publication 2-6h avant départ", lead_hours
    if lead_hours < 12:
        return "between_6h_12h", "Publication 6-12h avant départ", lead_hours
    if lead_hours < 24:
        return "between_12h_24h", "Publication 12-24h avant départ", lead_hours
    if lead_hours < 48:
        return "between_24h_48h", "Publication 24-48h avant départ", lead_hours
    return "beyond_48h", "Publication ≥ 48h avant départ", lead_hours


def _summarise_lead_time_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Observe la précision du modèle en fonction du délai de publication."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    lead_time_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        pronostics: Set[int] = set(payload.get("pronostics", set()))
        lead_times = [float(value) for value in payload.get("lead_times", [])]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["courses"] = len(courses)
        summary["pronostics"] = len(pronostics)

        if lead_times:
            ordered = sorted(lead_times)
            midpoint = len(ordered) // 2
            if len(ordered) % 2 == 0:
                median_value = (ordered[midpoint - 1] + ordered[midpoint]) / 2
            else:
                median_value = ordered[midpoint]

            summary.update(
                {
                    "average_lead_hours": _safe_average(lead_times),
                    "median_lead_hours": median_value,
                    "min_lead_hours": ordered[0],
                    "max_lead_hours": ordered[-1],
                }
            )
        else:
            summary.update(
                {
                    "average_lead_hours": None,
                    "median_lead_hours": None,
                    "min_lead_hours": None,
                    "max_lead_hours": None,
                }
            )

        lead_time_metrics[segment] = summary

    return lead_time_metrics


def _normalise_nationality_label(
    nationality: Optional[object]
) -> Tuple[str, str]:
    """Standardise une nationalité pour harmoniser les tableaux de bord."""

    if nationality is None:
        return "unknown", "Nationalité inconnue"

    raw_value = str(nationality).strip()

    if not raw_value:
        return "unknown", "Nationalité inconnue"

    iso_mapping = {
        "FR": "France",
        "FRA": "France",
        "BE": "Belgique",
        "BEL": "Belgique",
        "CH": "Suisse",
        "CHE": "Suisse",
        "IT": "Italie",
        "ITA": "Italie",
        "DE": "Allemagne",
        "DEU": "Allemagne",
        "ES": "Espagne",
        "ESP": "Espagne",
        "GB": "Royaume-Uni",
        "UK": "Royaume-Uni",
        "GBR": "Royaume-Uni",
        "IE": "Irlande",
        "IRL": "Irlande",
    }

    upper_value = raw_value.upper()
    label = iso_mapping.get(upper_value)

    if label is None:
        # Pour un libellé déjà explicite ("France", "États-Unis"...), on le met
        # simplement en casse titre pour un rendu homogène.
        label = raw_value.title()

    slug_source = label if label else raw_value
    slug = "".join(char if char.isalnum() else "_" for char in slug_source.lower()).strip("_")

    if not slug:
        slug = "unknown"

    return slug, label or raw_value


def _summarise_nationality_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Agrège les performances selon la nationalité des jockeys/entraîneurs."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    nationality_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        horses: Set[int] = set(payload.get("horses", set()))
        actors: Set[int] = set(payload.get("actors", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (
            summary["samples"] / total_samples if total_samples else None
        )
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["horses"] = len(horses)
        summary["actors"] = len(actors)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )

        nationality_metrics[segment] = summary

    return nationality_metrics


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


def _categorise_track_length(
    track_length: Optional[object],
) -> Tuple[str, str, Optional[int]]:
    """Classe la longueur des pistes pour identifier les profils de virages."""

    if track_length is None:
        return "unknown", "Longueur de piste inconnue", None

    try:
        value = int(track_length)
    except (TypeError, ValueError):  # pragma: no cover - nettoyage défensif
        return "unknown", "Longueur de piste inconnue", None

    if value <= 1400:
        return "compact_loop", "Piste compacte (≤ 1 400 m)", value

    if value <= 1700:
        return "standard_loop", "Piste standard (1 401-1 700 m)", value

    return "extended_loop", "Piste longue (> 1 700 m)", value


def _summarise_track_length_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble les métriques d'efficacité selon la longueur de la piste."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    ordering = {
        "compact_loop": 0,
        "standard_loop": 1,
        "extended_loop": 2,
        "unknown": 3,
    }
    length_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment, payload in sorted(
        breakdown.items(), key=lambda item: (ordering.get(item[0], 99), item[0])
    ):
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        hippodromes: Set[int] = set(payload.get("hippodromes", set()))
        lengths = [
            int(value)
            for value in payload.get("track_lengths", [])
            if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["hippodromes"] = len(hippodromes)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["average_track_length"] = _safe_average(lengths)
        summary["min_track_length"] = min(lengths) if lengths else None
        summary["max_track_length"] = max(lengths) if lengths else None

        length_metrics[segment] = summary

    return length_metrics


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


def _categorise_prize_per_runner(
    prize_money: Optional[object],
    field_size: Optional[object],
) -> Tuple[str, str, Optional[float], Optional[int]]:
    """Calcule la dotation moyenne par partant et la segmente pour le reporting."""

    if prize_money is None or field_size in (None, 0):
        return "unknown", "Dotation/partant inconnue", None, None

    try:
        prize_value = float(prize_money)
        field_value = int(field_size)
    except (TypeError, ValueError):  # pragma: no cover - entrées incohérentes
        return "unknown", "Dotation/partant inconnue", None, None

    if field_value <= 0:
        return "unknown", "Dotation/partant inconnue", None, None

    per_runner = prize_value / float(field_value)

    if per_runner < 600:
        segment = "per_runner_low"
        label = "< 600 € par partant"
    elif per_runner < 1200:
        segment = "per_runner_mid"
        label = "600-1200 € par partant"
    elif per_runner < 2000:
        segment = "per_runner_high"
        label = "1200-2000 € par partant"
    else:
        segment = "per_runner_premium"
        label = "> 2000 € par partant"

    return segment, label, per_runner, field_value


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


def _categorise_equipment_profile(
    equipment: Optional[object],
) -> Tuple[str, str, Optional[int], Optional[bool]]:
    """Normalise les informations d'équipement pour un partant donné."""

    if equipment is None:
        return "unknown", "Équipement inconnu", None, None

    items: List[str] = []

    if isinstance(equipment, dict):
        raw_items = equipment.get("items")
        if isinstance(raw_items, list):
            items = [str(item).strip() for item in raw_items if str(item).strip()]
        elif isinstance(raw_items, str):
            items = [raw_items.strip()]
    elif isinstance(equipment, list):
        items = [str(item).strip() for item in equipment if str(item).strip()]
    elif isinstance(equipment, str):  # pragma: no cover - garde-fou supplémentaire
        items = [equipment.strip()]

    normalised = [item.lower() for item in items if item]
    has_blinkers = any("oeill" in item or "œill" in item for item in normalised)

    if not items:
        return "no_equipment", "Aucun équipement déclaré", 0, False

    if has_blinkers:
        return "blinkers", "Œillères déclarées", len(items), True

    if len(items) == 1:
        return "single_gear", "Équipement isolé", 1, False

    return "multi_gear", "Équipement combiné (≥2)", len(items), False


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


def _categorise_reunion_number(
    reunion_number: Optional[object],
) -> Tuple[str, str, Optional[int]]:
    """Classe la réunion selon son numéro officiel (R1, R2, etc.)."""

    try:
        number = int(reunion_number) if reunion_number is not None else None
    except (TypeError, ValueError):  # pragma: no cover - résilience face aux entrées libres
        number = None

    if number is None or number <= 0:
        return "unknown", "Réunion inconnue", None

    if number <= 2:
        return "morning_cards", "Réunions R1-R2 (matinales)", number

    if number <= 5:
        return "day_cards", "Réunions R3-R5 (journée)", number

    return "evening_cards", "Réunions R6+ (soir)", number


def _summarise_reunion_number_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble une vue des performances par numéro de réunion (R1, R2, ...)."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())

    order_priority = {
        "morning_cards": 0,
        "day_cards": 1,
        "evening_cards": 2,
        "unknown": 3,
    }

    reunion_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment, payload in sorted(
        breakdown.items(),
        key=lambda item: (order_priority.get(item[0], 99), item[0]),
    ):
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        reunion_numbers = [
            int(value)
            for value in payload.get("reunion_numbers", [])
            if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)

        summary["average_reunion_number"] = _safe_average(reunion_numbers)
        summary["min_reunion_number"] = min(reunion_numbers) if reunion_numbers else None
        summary["max_reunion_number"] = max(reunion_numbers) if reunion_numbers else None

        reunion_metrics[segment] = summary

    return reunion_metrics


def _categorise_value_bet_flag(
    value_bet_detected: Optional[object],
) -> Tuple[str, str, Optional[bool]]:
    """Normalise le statut « value bet » d'un pronostic pour la segmentation."""

    if isinstance(value_bet_detected, bool):
        flag = value_bet_detected
    elif isinstance(value_bet_detected, str):
        normalised = value_bet_detected.strip().lower()
        if normalised in {"true", "1", "yes", "oui", "on"}:
            flag = True
        elif normalised in {"false", "0", "no", "non", "off"}:
            flag = False
        else:
            flag = None
    else:
        flag = None

    if flag is True:
        return "value_bet_detected", "Pronostic value bet détecté", True
    if flag is False:
        return "standard_pronostic", "Pronostic standard", False
    return "unknown_status", "Statut de value bet inconnu", None


def _summarise_value_bet_flag_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Mesure la précision selon que le pronostic est tagué value bet ou non."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    total_pronostics = sum(len(payload.get("pronostics", set())) for payload in breakdown.values())

    order_priority = {
        "value_bet_detected": 0,
        "standard_pronostic": 1,
        "unknown_status": 2,
    }

    flag_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment, payload in sorted(
        breakdown.items(),
        key=lambda item: (order_priority.get(item[0], 99), item[0]),
    ):
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        pronostics: Set[int] = set(payload.get("pronostics", set()))
        flags = [bool(flag) for flag in payload.get("flags", [])]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["pronostics"] = len(pronostics)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["pronostic_share"] = (
            len(pronostics) / total_pronostics if total_pronostics else None
        )
        summary["label"] = payload.get("label", segment)
        summary["value_bet_flag"] = payload.get("flag")
        summary["value_bet_flag_rate"] = (
            sum(flags) / len(flags) if flags else None
        )

        flag_metrics[segment] = summary

    return flag_metrics


def _categorise_year(
    race_date: Optional[object],
) -> Tuple[str, str, Optional[int]]:
    """Retourne l'année calendaire associée à la réunion analysée."""

    if isinstance(race_date, datetime):
        race_date = race_date.date()

    if not isinstance(race_date, date):
        return "unknown", "Année inconnue", None

    year_value = race_date.year
    segment = f"{year_value:04d}"
    label = f"Année {year_value}"

    return segment, label, year_value


def _summarise_year_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble une vue des performances agrégées par année civile."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())

    def _sort_key(item: Tuple[str, Dict[str, object]]) -> Tuple[int, str]:
        payload = item[1]
        year_value = payload.get("year")
        return (
            int(year_value) if isinstance(year_value, int) else 9999,
            item[0],
        )

    year_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment, payload in sorted(breakdown.items(), key=_sort_key):
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        dates: Set[str] = set(payload.get("dates", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = sum(truths) / len(truths) if truths else None
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)
        summary["year"] = payload.get("year")
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["first_date"] = min(dates) if dates else None
        summary["last_date"] = max(dates) if dates else None

        year_metrics[segment] = summary

    return year_metrics


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


def _categorise_quarter(
    race_date: Optional[object],
) -> Tuple[str, str, Optional[int], Optional[int]]:
    """Identifie le trimestre civil auquel rattacher la réunion évaluée."""

    if isinstance(race_date, datetime):
        race_date = race_date.date()

    if not isinstance(race_date, date):
        return "unknown", "Trimestre inconnu", None, None

    quarter_index = ((race_date.month - 1) // 3) + 1
    year = race_date.year
    segment = f"{year:04d}-Q{quarter_index}"
    label = f"T{quarter_index} {year}"

    return segment, label, quarter_index, year


def _categorise_season(
    race_date: Optional[object],
) -> Tuple[str, str, Optional[int], Optional[int]]:
    """Associe la réunion à une saison météorologique normalisée."""

    if isinstance(race_date, datetime):
        race_date = race_date.date()

    if not isinstance(race_date, date):
        return "unknown", "Saison inconnue", None, None

    month = race_date.month
    year = race_date.year

    # Regroupe les mois selon les saisons météorologiques classiques en
    # conservant un ordre stable pour le tri et l'affichage.
    if month in (12, 1, 2):
        return f"{year:04d}-winter", f"Hiver {year}", 1, year

    if month in (3, 4, 5):
        return f"{year:04d}-spring", f"Printemps {year}", 2, year

    if month in (6, 7, 8):
        return f"{year:04d}-summer", f"Été {year}", 3, year

    return f"{year:04d}-autumn", f"Automne {year}", 4, year


def _summarise_quarter_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble une vue des performances agrégées par trimestre civil."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())

    def _sort_key(item: Tuple[str, Dict[str, object]]) -> Tuple[int, int, str]:
        payload = item[1]
        year = payload.get("year")
        quarter_index = payload.get("quarter_index")
        return (
            int(year) if isinstance(year, int) else 9999,
            int(quarter_index) if isinstance(quarter_index, int) else 5,
            item[0],
        )

    quarter_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment, payload in sorted(breakdown.items(), key=_sort_key):
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        dates: Set[str] = set(payload.get("dates", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = sum(truths) / len(truths) if truths else None
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)
        summary["quarter_index"] = payload.get("quarter_index")
        summary["year"] = payload.get("year")
        summary["dates"] = sorted(dates)

        quarter_metrics[segment] = summary

    return quarter_metrics


def _summarise_season_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Condense les performances agrégées par saison météorologique."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())

    def _sort_key(item: Tuple[str, Dict[str, object]]) -> Tuple[int, int, str]:
        payload = item[1]
        year = payload.get("year")
        season_index = payload.get("season_index")
        return (
            int(year) if isinstance(year, int) else 9999,
            int(season_index) if isinstance(season_index, int) else 9,
            item[0],
        )

    season_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment, payload in sorted(breakdown.items(), key=_sort_key):
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        dates: Set[str] = set(payload.get("dates", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "label": payload.get("label", segment),
                "season_index": payload.get("season_index"),
                "year": payload.get("year"),
                "courses": len(courses),
                "reunions": len(reunions),
                "share": (len(truths) / total_samples) if total_samples else None,
                "observed_positive_rate": sum(truths) / len(truths) if truths else None,
                "dates": sorted(dates),
            }
        )

        season_metrics[segment] = summary

    return season_metrics


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


def _summarise_prize_per_runner_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Analyse la stabilité du modèle selon la dotation moyenne par partant."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    per_runner_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        per_runner_values = [
            float(value)
            for value in payload.get("per_runner_values", [])
            if value is not None
        ]
        field_sizes = [
            int(value)
            for value in payload.get("field_sizes", [])
            if value not in (None, "unknown")
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["label"] = payload.get("label", segment)
        summary["average_prize_per_runner_eur"] = _safe_average(per_runner_values)
        summary["min_prize_per_runner_eur"] = (
            min(per_runner_values) if per_runner_values else None
        )
        summary["max_prize_per_runner_eur"] = (
            max(per_runner_values) if per_runner_values else None
        )
        summary["average_field_size"] = _safe_average(field_sizes)
        summary["min_field_size"] = min(field_sizes) if field_sizes else None
        summary["max_field_size"] = max(field_sizes) if field_sizes else None

        per_runner_metrics[segment] = summary

    return per_runner_metrics


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


def _summarise_equipment_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Mesure l'impact des configurations de matériel sur les performances."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    equipment_metrics: Dict[str, Dict[str, Optional[float]]] = {}

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

        item_counts = [
            int(value)
            for value in payload.get("item_counts", [])
            if isinstance(value, (int, float))
        ]
        summary["average_equipment_items"] = _safe_average(item_counts)
        summary["min_equipment_items"] = min(item_counts) if item_counts else None
        summary["max_equipment_items"] = max(item_counts) if item_counts else None

        blinkers_flags = [
            bool(flag)
            for flag in payload.get("blinkers_flags", [])
            if isinstance(flag, bool)
        ]
        summary["blinkers_rate"] = (
            sum(1 for flag in blinkers_flags if flag) / len(blinkers_flags)
            if blinkers_flags
            else None
        )

        equipment_metrics[segment] = summary

    return equipment_metrics


def _categorise_recent_form(
    recent_positions: Optional[List[int]],
) -> Tuple[str, str]:
    """Classe la forme récente d'un cheval selon ses dernières arrivées."""

    if not recent_positions:
        return "unknown", "Forme inconnue"

    # Écarte toute valeur non entière qui pourrait remonter d'une source externe.
    clean_positions = [
        int(position)
        for position in recent_positions
        if isinstance(position, (int, float))
    ]

    if not clean_positions:
        return "unknown", "Forme inconnue"

    best_finish = min(clean_positions)
    average_finish = sum(clean_positions) / len(clean_positions)

    if best_finish == 1:
        return "recent_winner", "Gagnant récent"

    if average_finish <= 3.0:
        return "strong_form", "Forme solide (moyenne ≤3)"

    if average_finish <= 5.0:
        return "steady_form", "Forme régulière (moyenne 3-5)"

    if average_finish <= 8.0:
        return "inconsistent_form", "Forme irrégulière (moyenne 5-8)"

    return "poor_form", "Forme en difficulté (>8)"


def _summarise_recent_form_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Expose les performances selon les segments de forme récente."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    form_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))

        average_positions = [
            float(value)
            for value in payload.get("average_positions", [])
            if value is not None
        ]
        best_positions = [
            int(value)
            for value in payload.get("best_positions", [])
            if value is not None
        ]
        worst_positions = [
            int(value)
            for value in payload.get("worst_positions", [])
            if value is not None
        ]
        starts_counts = [
            int(value)
            for value in payload.get("starts_counts", [])
            if value is not None
        ]
        win_flags = [
            bool(value)
            for value in payload.get("win_flags", [])
            if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["horses"] = len(horses)
        summary["share"] = (
            summary["samples"] / total_samples if total_samples else None
        )
        summary["label"] = payload.get("label", segment)
        summary["average_recent_position"] = _safe_average(average_positions)
        summary["best_recent_position"] = (
            min(best_positions) if best_positions else None
        )
        summary["worst_recent_position"] = (
            max(worst_positions) if worst_positions else None
        )
        summary["average_recent_history_size"] = _safe_average(
            [float(value) for value in starts_counts]
        )
        summary["recent_win_rate"] = (
            sum(1 for flag in win_flags if flag) / len(win_flags)
            if win_flags
            else None
        )

        form_metrics[segment] = summary

    return form_metrics


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


def _categorise_win_probability(
    probability: Optional[object],
) -> Tuple[str, str, Optional[float]]:
    """Regroupe une probabilité de victoire en bandes homogènes pour analyse."""

    if probability is None:
        return "unknown", "Probabilité inconnue", None

    try:
        value = float(probability)
    except (TypeError, ValueError):  # pragma: no cover - garde-fou contre saisies invalides
        return "unknown", "Probabilité inconnue", None

    # Les probabilités sont bornées dans [0, 1] mais on protège contre les arrondis exotiques.
    value = max(0.0, min(value, 1.0))

    bands: List[Tuple[float, str, str]] = [
        (0.20, "under_20", "Probabilité < 20%"),
        (0.30, "between_20_30", "Probabilité 20-30%"),
        (0.40, "between_30_40", "Probabilité 30-40%"),
        (0.50, "between_40_50", "Probabilité 40-50%"),
        (0.60, "between_50_60", "Probabilité 50-60%"),
        (0.70, "between_60_70", "Probabilité 60-70%"),
        (0.80, "between_70_80", "Probabilité 70-80%"),
        (0.90, "between_80_90", "Probabilité 80-90%"),
    ]

    for upper_bound, segment_key, label in bands:
        if value < upper_bound:
            return segment_key, label, value

    return "at_least_90", "Probabilité ≥ 90%", value


def _summarise_win_probability_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Analyse la calibration réelle par bande de probabilité de victoire brute."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    probability_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        probabilities = [
            float(value)
            for value in payload.get("probabilities", [])
            if value is not None
        ]
        courses: Set[int] = set(payload.get("courses", set()))
        pronostics: Set[int] = set(payload.get("pronostics", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["courses"] = len(courses)
        summary["pronostics"] = len(pronostics)
        summary["average_probability"] = _safe_average(probabilities)
        summary["min_probability"] = min(probabilities) if probabilities else None
        summary["max_probability"] = max(probabilities) if probabilities else None
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )

        if summary["average_probability"] is not None and summary["observed_positive_rate"] is not None:
            summary["average_calibration_gap"] = (
                summary["average_probability"] - summary["observed_positive_rate"]
            )
        else:
            summary["average_calibration_gap"] = None

        probability_metrics[segment] = summary

    return probability_metrics


def _categorise_place_probability(
    probability: Optional[object],
) -> Tuple[str, str, Optional[float]]:
    """Regroupe une probabilité de place (top 3) en bandes métiers lisibles."""

    if probability is None:
        return "unknown", "Probabilité de place inconnue", None

    try:
        value = float(probability)
    except (TypeError, ValueError):  # pragma: no cover - résilience en entrée
        return "unknown", "Probabilité de place inconnue", None

    value = max(0.0, min(value, 1.0))

    bands: List[Tuple[float, str, str]] = [
        (0.30, "under_30", "Probabilité de place < 30%"),
        (0.40, "between_30_40", "Probabilité de place 30-40%"),
        (0.50, "between_40_50", "Probabilité de place 40-50%"),
        (0.60, "between_50_60", "Probabilité de place 50-60%"),
        (0.70, "between_60_70", "Probabilité de place 60-70%"),
        (0.80, "between_70_80", "Probabilité de place 70-80%"),
    ]

    for upper_bound, segment_key, label in bands:
        if value < upper_bound:
            return segment_key, label, value

    return "at_least_80", "Probabilité de place ≥ 80%", value


def _summarise_place_probability_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Analyse la calibration des probabilités de place par bande homogène."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    place_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        stored_probabilities = [
            float(value)
            for value in payload.get("place_probabilities", [])
            if value is not None
        ]

        if stored_probabilities:
            probability_sample = stored_probabilities
        else:
            probability_sample = [
                float(value)
                for value in scores
                if value is not None
            ]

        courses: Set[int] = set(payload.get("courses", set()))
        pronostics: Set[int] = set(payload.get("pronostics", set()))

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["label"] = payload.get("label", segment)
        summary["share"] = (len(truths) / total_samples) if total_samples else None
        summary["courses"] = len(courses)
        summary["pronostics"] = len(pronostics)
        summary["average_probability"] = _safe_average(probability_sample)
        summary["min_probability"] = min(probability_sample) if probability_sample else None
        summary["max_probability"] = max(probability_sample) if probability_sample else None
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )

        if summary["average_probability"] is not None and summary["observed_positive_rate"] is not None:
            summary["average_calibration_gap"] = (
                summary["average_probability"] - summary["observed_positive_rate"]
            )
        else:
            summary["average_calibration_gap"] = None

        place_metrics[segment] = summary

    return place_metrics


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


def _normalise_coat_color_label(
    raw_color: Optional[object],
) -> Tuple[str, str]:
    """Uniformise la robe pour agréger les performances par couleur."""

    if raw_color is None:
        return "unknown", "Robe inconnue"

    color_text = str(raw_color).strip()
    if not color_text:
        return "unknown", "Robe inconnue"

    normalized = (
        color_text.lower()
        .replace("œ", "oe")
        .replace("ç", "c")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("û", "u")
    )
    normalized = normalized.replace("-", " ").replace("_", " ")

    synonyms: List[Tuple[str, Tuple[str, ...], str]] = [
        ("alezan", ("alezan", "alezane", "alezans", "chestnut"), "Alezan"),
        ("bai", ("bai", "bais", "bay", "bay brown"), "Bai"),
        (
            "gris",
            ("gris", "grise", "grey", "grise pommele", "gris pommele", "roan"),
            "Gris",
        ),
        ("noir", ("noir", "noire", "black"), "Noir"),
        ("rouan", ("rouan", "rouane"), "Rouan"),
    ]

    for key, keywords, label in synonyms:
        if any(keyword in normalized for keyword in keywords):
            return key, label

    slug = "_".join(part for part in normalized.split() if part)
    if not slug:
        return "unknown", "Robe inconnue"

    return slug, color_text.title()


def _summarise_horse_coat_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compare la stabilité prédictive selon la robe officielle des chevaux."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    coat_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))
        raw_inputs = [
            str(value).strip()
            for value in payload.get("raw_inputs", [])
            if isinstance(value, str) and value.strip()
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "label": payload.get("label", segment.title()),
                "share": (
                    summary["samples"] / total_samples if total_samples else None
                ),
                "courses": len(courses),
                "horses": len(horses),
                "observed_positive_rate": (
                    sum(truths) / len(truths) if truths else None
                ),
                "input_examples": sorted(set(raw_inputs))[:3] if raw_inputs else [],
            }
        )

        coat_metrics[segment] = summary

    return coat_metrics


def _normalise_horse_breed_label(
    raw_breed: Optional[object],
) -> Tuple[str, str]:
    """Uniformise la race déclarée pour agréger les performances par lignée."""

    if raw_breed is None:
        return "unknown", "Race inconnue"

    breed_text = str(raw_breed).strip()
    if not breed_text:
        return "unknown", "Race inconnue"

    normalized = (
        breed_text.lower()
        .replace("œ", "oe")
        .replace("ê", "e")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ç", "c")
        .replace("à", "a")
        .replace("â", "a")
        .replace("-", " ")
        .replace("_", " ")
    )

    synonyms: List[Tuple[str, Tuple[str, ...], str]] = [
        ("pur_sang", ("pur sang", "pursang", "thoroughbred"), "Pur-sang"),
        (
            "anglo_arabe",
            ("anglo arabe", "anglo-arabe", "angloarab"),
            "Anglo-arabe",
        ),
        (
            "trotteur_francais",
            (
                "trotteur francais",
                "trotteur français",
                "trotteur",
            ),
            "Trotteur français",
        ),
        ("aqps", ("aqps", "autre que pur sang"), "AQPS"),
    ]

    for slug, keywords, label in synonyms:
        if any(keyword in normalized for keyword in keywords):
            return slug, label

    slug = "_".join(part for part in normalized.split() if part)
    if not slug:
        return "unknown", "Race inconnue"

    return slug, breed_text.title()


def _summarise_horse_breed_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Dresse un panorama des performances selon la race déclarée."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    breed_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))
        raw_inputs = [
            str(value).strip()
            for value in payload.get("raw_inputs", [])
            if isinstance(value, str) and value.strip()
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "label": payload.get("label", segment.title()),
                "share": (
                    summary["samples"] / total_samples if total_samples else None
                ),
                "courses": len(courses),
                "horses": len(horses),
                "observed_positive_rate": (
                    sum(truths) / len(truths) if truths else None
                ),
                "input_examples": sorted(set(raw_inputs))[:3] if raw_inputs else [],
            }
        )

        breed_metrics[segment] = summary

    return breed_metrics


def _normalise_sire_label(raw_sire: Optional[object]) -> Tuple[str, str]:
    """Uniformise le nom du père pour agréger les performances par lignée."""

    if raw_sire is None:
        return "unknown", "Père inconnu"

    sire_text = str(raw_sire).strip()
    if not sire_text:
        return "unknown", "Père inconnu"

    normalized = (
        sire_text.lower()
        .replace("œ", "oe")
        .replace("ê", "e")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ë", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("û", "u")
    )

    slug = "".join(char if char.isalnum() else "_" for char in normalized)
    slug = "_".join(part for part in slug.split("_") if part)

    if not slug:
        return "unknown", sire_text

    return slug, sire_text


def _normalise_dam_label(raw_dam: Optional[object]) -> Tuple[str, str]:
    """Uniformise le nom de la mère pour suivre les lignées maternelles."""

    if raw_dam is None:
        return "unknown", "Mère inconnue"

    dam_text = str(raw_dam).strip()
    if not dam_text:
        return "unknown", "Mère inconnue"

    normalized = (
        dam_text.lower()
        .replace("œ", "oe")
        .replace("ê", "e")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ë", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("û", "u")
    )

    slug = "".join(char if char.isalnum() else "_" for char in normalized)
    slug = "_".join(part for part in slug.split("_") if part)

    if not slug:
        return "unknown", dam_text

    return slug, dam_text


def _summarise_sire_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble un tableau de bord des performances par père déclaré."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    sire_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))
        trainers: Set[int] = set(payload.get("trainers", set()))
        raw_inputs = [
            str(value).strip()
            for value in payload.get("raw_inputs", [])
            if isinstance(value, str) and value.strip()
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "label": payload.get("label", segment.title()),
                "share": (
                    summary["samples"] / total_samples if total_samples else None
                ),
                "courses": len(courses),
                "horses": len(horses),
                "observed_positive_rate": (
                    sum(truths) / len(truths) if truths else None
                ),
                "input_examples": sorted(set(raw_inputs))[:3] if raw_inputs else [],
            }
        )

        if trainers:
            summary["trainers"] = len(trainers)

        sire_metrics[segment] = summary

    return sire_metrics


def _summarise_dam_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Assemble un tableau de bord des performances par mère déclarée."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    dam_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        horses: Set[int] = set(payload.get("horses", set()))
        raw_inputs = [
            str(value).strip()
            for value in payload.get("raw_inputs", [])
            if isinstance(value, str) and value.strip()
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "label": payload.get("label", segment.title()),
                "share": (
                    summary["samples"] / total_samples if total_samples else None
                ),
                "courses": len(courses),
                "horses": len(horses),
                "observed_positive_rate": (
                    sum(truths) / len(truths) if truths else None
                ),
                "input_examples": sorted(set(raw_inputs))[:3] if raw_inputs else [],
            }
        )

        dam_metrics[segment] = summary

    return dam_metrics


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


def _categorise_draw_parity(draw_position: Optional[int]) -> Tuple[str, str]:
    """Classe la corde selon une parité exploitable par le tableau de bord."""

    if draw_position is None:
        return "unknown", "Numéro de corde inconnu"

    return ("even", "Numéro pair") if draw_position % 2 == 0 else ("odd", "Numéro impair")


def _summarise_draw_parity_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Synthétise la réussite du modèle selon la parité du numéro de corde."""

    if not breakdown:
        return {}

    parity_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        draws = [int(value) for value in payload.get("draws", []) if value is not None]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["label"] = payload.get("label") or segment
        summary["average_draw"] = _safe_average(draws)
        summary["min_draw"] = min(draws) if draws else None
        summary["max_draw"] = max(draws) if draws else None

        parity_metrics[segment] = summary

    return parity_metrics


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


def _compute_start_delay_minutes(
    scheduled: Optional[time],
    actual: Optional[time],
) -> Optional[int]:
    """Calcule le décalage en minutes entre l'horaire prévu et l'heure réelle."""

    if not scheduled or not actual:
        return None

    scheduled_dt = datetime.combine(date.today(), scheduled)
    actual_dt = datetime.combine(date.today(), actual)
    delta_minutes = int((actual_dt - scheduled_dt).total_seconds() // 60)

    return delta_minutes


def _categorise_start_delay(
    scheduled: Optional[time],
    actual: Optional[time],
) -> Tuple[str, str, Optional[int]]:
    """Regroupe les courses selon le respect des horaires de départ officiels."""

    delay_minutes = _compute_start_delay_minutes(scheduled, actual)

    if delay_minutes is None:
        return "unknown", "Horaire réel inconnu", None

    if delay_minutes <= -3:
        return "ahead_of_schedule", "Départ anticipé (≥3 min d'avance)", delay_minutes

    if abs(delay_minutes) <= 5:
        return "on_time", "Départ ponctuel (±5 min)", delay_minutes

    if delay_minutes <= 15:
        return "slight_delay", "Retard modéré (6-15 min)", delay_minutes

    return "heavy_delay", "Retard conséquent (>15 min)", delay_minutes


def _summarise_start_delay_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Synthétise la stabilité du modèle selon la ponctualité des départs."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    delay_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        delay_values = [
            float(value)
            for value in payload.get("delays", [])
            if isinstance(value, (int, float))
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "label": payload.get("label", segment),
                "share": (len(truths) / total_samples) if total_samples else None,
                "courses": len(courses),
                "reunions": len(reunions),
                "observed_positive_rate": (sum(truths) / len(truths)) if truths else None,
                "average_delay_minutes": _safe_average(delay_values),
                "min_delay_minutes": min(delay_values) if delay_values else None,
                "max_delay_minutes": max(delay_values) if delay_values else None,
            }
        )

        delay_metrics[segment] = summary

    return delay_metrics


def _categorise_weather_profile(
    weather_payload: Optional[object],
) -> Tuple[str, str, Optional[float]]:
    """Normalise les conditions météo en segments exploitables pour l'analyse."""

    if not weather_payload:
        return "unknown", "Conditions inconnues", None

    condition_value: Optional[str] = None
    temperature_value: Optional[float] = None

    if isinstance(weather_payload, dict):
        for key in ("condition", "weather", "meteo", "label", "libelle", "state"):
            raw = weather_payload.get(key)
            if raw:
                condition_value = str(raw)
                break
        for key in ("temperature", "temp", "temperature_c", "temp_c", "temperatureC"):
            raw_temp = weather_payload.get(key)
            if raw_temp is not None:
                try:
                    temperature_value = float(raw_temp)
                except (TypeError, ValueError):  # pragma: no cover - tolérance JSON
                    temperature_value = None
                break
    elif isinstance(weather_payload, str):
        condition_value = weather_payload
    else:
        condition_value = str(weather_payload)

    if not condition_value:
        return "unknown", "Conditions inconnues", temperature_value

    normalised = condition_value.strip().lower()

    keyword_map = [
        ("clear", ("sun", "soleil", "clair", "clear", "ensoleill"), "Conditions claires"),
        ("rain", ("rain", "pluie", "averse", "pluv"), "Pluie / Averses"),
        ("storm", ("orage", "storm", "thunder"), "Orageux"),
        ("snow", ("neige", "snow", "blizzard"), "Neige / Verglas"),
        ("fog", ("brouillard", "fog", "mist"), "Brouillard"),
        ("wind", ("vent", "wind"), "Vent soutenu"),
        ("cloud", ("nuage", "cloud", "overcast", "gris"), "Couvert / Nuageux"),
    ]

    for segment, keywords, label in keyword_map:
        if any(keyword in normalised for keyword in keywords):
            return segment, label, temperature_value

    label = condition_value.strip().capitalize()
    return "other", label or "Conditions variées", temperature_value


def _summarise_weather_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Mesure l'influence des conditions météo sur la précision du modèle."""

    if not breakdown:
        return {}

    weather_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment in sorted(breakdown.keys()):
        payload = breakdown[segment]
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        temperatures = [
            float(value)
            for value in payload.get("temperatures", [])
            if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary["observed_positive_rate"] = (
            sum(truths) / len(truths) if truths else None
        )
        summary["courses"] = len(courses)
        summary["reunions"] = len(reunions)
        summary["label"] = payload.get("label")
        summary["average_temperature"] = _safe_average(temperatures)
        summary["min_temperature"] = min(temperatures) if temperatures else None
        summary["max_temperature"] = max(temperatures) if temperatures else None

        weather_metrics[segment] = summary

    return weather_metrics


def _categorise_temperature_band(
    temperature: Optional[object],
) -> Tuple[str, str, Optional[float]]:
    """Projette une température en bandes lisibles pour le monitoring ML."""

    if temperature is None:
        return "temperature_unknown", "Température inconnue", None

    try:
        value = float(temperature)
    except (TypeError, ValueError):  # pragma: no cover - résilience saisies atypiques
        return "temperature_unknown", "Température inconnue", None

    if value <= 0.0:
        return "freezing", "Gel / ≤0°C", value

    if value <= 7.0:
        return "very_cold", "Très froid (1-7°C)", value

    if value <= 14.0:
        return "cool", "Frais (8-14°C)", value

    if value <= 22.0:
        return "mild", "Tempéré (15-22°C)", value

    if value <= 28.0:
        return "warm", "Chaud (23-28°C)", value

    return "hot", "Caniculaire (>28°C)", value


def _summarise_temperature_band_performance(
    breakdown: Dict[str, Dict[str, object]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Mesure la précision du modèle en fonction de la température ambiante."""

    if not breakdown:
        return {}

    total_samples = sum(len(payload.get("truths", [])) for payload in breakdown.values())
    order_priority = {
        "freezing": 0,
        "very_cold": 1,
        "cool": 2,
        "mild": 3,
        "warm": 4,
        "hot": 5,
        "temperature_unknown": 6,
    }

    temperature_metrics: Dict[str, Dict[str, Optional[float]]] = {}

    for segment, payload in sorted(
        breakdown.items(), key=lambda item: (order_priority.get(item[0], 99), item[0])
    ):
        truths = list(payload.get("truths", []))
        predicted = list(payload.get("predictions", []))
        scores = list(payload.get("scores", []))
        courses: Set[int] = set(payload.get("courses", set()))
        reunions: Set[int] = set(payload.get("reunions", set()))
        temperatures = [
            float(value)
            for value in payload.get("temperatures", [])
            if value is not None
        ]

        summary = _summarise_group_performance(truths, predicted, scores)
        summary.update(
            {
                "label": payload.get("label", segment),
                "share": (len(truths) / total_samples) if total_samples else None,
                "courses": len(courses),
                "reunions": len(reunions),
                "observed_positive_rate": sum(truths) / len(truths) if truths else None,
                "average_temperature": _safe_average(temperatures),
                "min_temperature": min(temperatures) if temperatures else None,
                "max_temperature": max(temperatures) if temperatures else None,
            }
        )

        if temperatures:
            ordered = sorted(temperatures)
            midpoint = len(ordered) // 2
            if len(ordered) % 2 == 0:
                median_temperature = (ordered[midpoint - 1] + ordered[midpoint]) / 2
            else:
                median_temperature = ordered[midpoint]
            summary["median_temperature"] = median_temperature
        else:
            summary["median_temperature"] = None

        temperature_metrics[segment] = summary

    return temperature_metrics


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
        win_probability_breakdown: Dict[str, Dict[str, object]] = {}
        place_probability_breakdown: Dict[str, Dict[str, object]] = {}
        discipline_breakdown: Dict[str, Dict[str, object]] = {}
        discipline_surface_breakdown: Dict[str, Dict[str, object]] = {}
        distance_breakdown: Dict[str, Dict[str, object]] = {}
        surface_breakdown: Dict[str, Dict[str, object]] = {}
        prize_money_breakdown: Dict[str, Dict[str, object]] = {}
        prize_per_runner_breakdown: Dict[str, Dict[str, object]] = {}
        handicap_breakdown: Dict[str, Dict[str, object]] = {}
        weight_breakdown: Dict[str, Dict[str, object]] = {}
        odds_band_breakdown: Dict[str, Dict[str, object]] = {}
        probability_edge_breakdown: Dict[str, Dict[str, object]] = {}
        probability_error_breakdown: Dict[str, Dict[str, object]] = {}
        probability_margin_breakdown: Dict[str, Dict[str, object]] = {}
        favourite_alignment_breakdown: Dict[str, Dict[str, object]] = {}
        horse_age_breakdown: Dict[str, Dict[str, object]] = {}
        horse_gender_breakdown: Dict[str, Dict[str, object]] = {}
        horse_coat_breakdown: Dict[str, Dict[str, object]] = {}
        horse_breed_breakdown: Dict[str, Dict[str, object]] = {}
        horse_sire_breakdown: Dict[str, Dict[str, object]] = {}
        horse_dam_breakdown: Dict[str, Dict[str, object]] = {}
        owner_breakdown: Dict[str, Dict[str, object]] = {}
        owner_trainer_breakdown: Dict[str, Dict[str, object]] = {}
        owner_jockey_breakdown: Dict[str, Dict[str, object]] = {}
        recent_form_breakdown: Dict[str, Dict[str, object]] = {}
        prediction_outcome_breakdown: Dict[str, Dict[str, object]] = {}
        equipment_breakdown: Dict[str, Dict[str, object]] = {}
        weather_breakdown: Dict[str, Dict[str, object]] = {}
        temperature_band_breakdown: Dict[str, Dict[str, object]] = {}
        day_part_breakdown: Dict[str, Dict[str, object]] = {}
        weekday_breakdown: Dict[str, Dict[str, object]] = {}
        reunion_number_breakdown: Dict[str, Dict[str, object]] = {}
        year_breakdown: Dict[str, Dict[str, object]] = {}
        month_breakdown: Dict[str, Dict[str, object]] = {}
        # Agrégation saisonnière pour offrir une lecture "hiver / printemps /"
        # "été / automne" complémentaire aux vues mensuelles et trimestrielles.
        season_breakdown: Dict[str, Dict[str, object]] = {}
        quarter_breakdown: Dict[str, Dict[str, object]] = {}
        lead_time_breakdown: Dict[str, Dict[str, object]] = {}
        race_order_breakdown: Dict[str, Dict[str, object]] = {}
        track_type_breakdown: Dict[str, Dict[str, object]] = {}
        track_length_breakdown: Dict[str, Dict[str, object]] = {}
        race_category_breakdown: Dict[str, Dict[str, object]] = {}
        race_class_breakdown: Dict[str, Dict[str, object]] = {}
        value_bet_breakdown: Dict[str, Dict[str, object]] = {}
        value_bet_flag_breakdown: Dict[str, Dict[str, object]] = {}
        field_size_breakdown: Dict[str, Dict[str, object]] = {}
        draw_breakdown: Dict[str, Dict[str, object]] = {}
        draw_parity_breakdown: Dict[str, Dict[str, object]] = {}
        start_type_breakdown: Dict[str, Dict[str, object]] = {}
        start_delay_breakdown: Dict[str, Dict[str, object]] = {}
        rest_period_breakdown: Dict[str, Dict[str, object]] = {}
        jockey_breakdown: Dict[str, Dict[str, object]] = {}
        trainer_breakdown: Dict[str, Dict[str, object]] = {}
        jockey_trainer_breakdown: Dict[str, Dict[str, object]] = {}
        jockey_experience_breakdown: Dict[str, Dict[str, object]] = {}
        trainer_experience_breakdown: Dict[str, Dict[str, object]] = {}
        jockey_nationality_breakdown: Dict[str, Dict[str, object]] = {}
        trainer_nationality_breakdown: Dict[str, Dict[str, object]] = {}
        hippodrome_breakdown: Dict[str, Dict[str, object]] = {}
        country_breakdown: Dict[str, Dict[str, object]] = {}
        city_breakdown: Dict[str, Dict[str, object]] = {}
        api_source_breakdown: Dict[str, Dict[str, object]] = {}
        # Prépare une vision par version du modèle afin d'identifier rapidement
        # les régressions potentielles lorsqu'une version minoritaire décroche.
        model_versions: Counter[str] = Counter()
        model_version_breakdown: Dict[str, Dict[str, object]] = {}
        course_stats: Dict[int, Dict[str, object]] = {}
        prediction_rank_breakdown: Dict[str, Dict[str, object]] = {}
        topn_tracking: Dict[int, Dict[str, object]] = {}
        ndcg_at_3_scores: List[float] = []
        ndcg_at_5_scores: List[float] = []
        final_position_breakdown: Dict[str, Dict[str, object]] = {}
        daily_breakdown: Dict[str, Dict[str, object]] = {}
        betting_samples: List[Dict[str, object]] = []
        rank_correlation_tracking: Dict[int, Dict[str, object]] = {}
        # Suit la précision du classement en termes d'écart absolu/signé entre
        # le rang prédit et la position réelle des partants.
        rank_error_tracking: Dict[int, Dict[str, object]] = {}

        # Parcourt chaque pronostic couplé à un résultat officiel pour préparer les listes
        # nécessaires aux métriques (labels réels, scores, version du modèle, etc.).
        for prediction, partant, pronostic, course in predictions_with_results:
            probability = float(prediction.win_probability)
            probability = max(0.0, min(probability, 1.0))
            is_top3 = 1 if partant.final_position and partant.final_position <= 3 else 0
            predicted_label = 1 if probability >= probability_threshold else 0

            correlation_bucket = rank_correlation_tracking.setdefault(
                course.course_id,
                {
                    "key": f"course_{course.course_id}",
                    "label": getattr(course, "course_name", None)
                    or f"Course {getattr(course, 'course_number', '?')}",
                    "probabilities": [],
                    "finish_positions": [],
                },
            )
            if getattr(course, "course_name", None):
                correlation_bucket["label"] = str(course.course_name)
            if getattr(partant, "final_position", None) is not None:
                correlation_bucket.setdefault("probabilities", []).append(probability)
                correlation_bucket.setdefault("finish_positions", []).append(
                    int(partant.final_position)
                )

            # Mesure l'écart absolu entre la probabilité annoncée et l'issue réelle
            # afin de piloter un tableau de bord de précision par bandes d'erreur.
            absolute_error = abs(probability - is_top3)
            (
                error_segment,
                error_label,
                normalised_error,
            ) = _categorise_probability_error(absolute_error)
            error_bucket = probability_error_breakdown.setdefault(
                error_segment,
                {
                    "label": error_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "errors": [],
                    "courses": set(),
                },
            )
            error_bucket["label"] = error_label
            error_bucket["truths"].append(is_top3)
            error_bucket["predictions"].append(predicted_label)
            error_bucket["scores"].append(probability)
            error_bucket.setdefault("courses", set()).add(course.course_id)
            stored_error = normalised_error if normalised_error is not None else absolute_error
            error_bucket.setdefault("errors", []).append(stored_error)

            # Ventile immédiatement l'échantillon selon l'issue de classification
            # (vrai positif, faux négatif, etc.) pour produire un tableau de bord
            # pédagogique sur les erreurs du modèle.
            (
                outcome_key,
                outcome_label,
            ) = _categorise_prediction_outcome(predicted_label, is_top3)
            outcome_bucket = prediction_outcome_breakdown.setdefault(
                outcome_key,
                {
                    "label": outcome_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "pronostics": set(),
                    "model_versions": set(),
                },
            )
            outcome_bucket["label"] = outcome_label
            outcome_bucket["truths"].append(is_top3)
            outcome_bucket["predictions"].append(predicted_label)
            outcome_bucket["scores"].append(probability)
            outcome_bucket.setdefault("courses", set()).add(course.course_id)
            outcome_bucket.setdefault("pronostics", set()).add(pronostic.pronostic_id)
            outcome_bucket.setdefault("model_versions", set()).add(
                pronostic.model_version or "unknown"
            )

            reunion_entity = getattr(course, "reunion", None)

            # Regroupe immédiatement les échantillons par source API afin de
            # détecter les éventuelles dérives liées à une alimentation
            # spécifique (Turfinfo, PMU, Aspiturf...).
            api_key, api_label = _normalise_api_source_label(
                getattr(reunion_entity, "api_source", None)
            )
            api_bucket = api_source_breakdown.setdefault(
                api_key,
                {
                    "label": api_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "hippodromes": set(),
                    "pronostics": set(),
                    "model_versions": set(),
                },
            )
            api_bucket["label"] = api_label
            api_bucket["truths"].append(is_top3)
            api_bucket["predictions"].append(predicted_label)
            api_bucket["scores"].append(probability)
            api_bucket.setdefault("courses", set()).add(course.course_id)
            api_bucket.setdefault("pronostics", set()).add(pronostic.pronostic_id)
            if pronostic.model_version:
                api_bucket.setdefault("model_versions", set()).add(pronostic.model_version)
            if reunion_entity is not None and getattr(reunion_entity, "reunion_id", None):
                api_bucket.setdefault("reunions", set()).add(reunion_entity.reunion_id)

            y_true.append(is_top3)
            y_scores.append(probability)
            y_pred.append(predicted_label)

            confidence_counter[prediction.confidence_level or "unknown"] += 1
            confidence_level = prediction.confidence_level or "unknown"
            level_bucket = confidence_breakdown.setdefault(
                confidence_level,
                {
                    "label": _describe_prediction_confidence_level(confidence_level),
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "pronostics": set(),
                },
            )
            level_bucket["label"] = _describe_prediction_confidence_level(confidence_level)
            level_bucket["truths"].append(is_top3)
            level_bucket["predictions"].append(predicted_label)
            level_bucket["scores"].append(probability)
            level_bucket.setdefault("courses", set()).add(course.course_id)
            level_bucket.setdefault("pronostics", set()).add(pronostic.pronostic_id)

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

            probability_segment, probability_label, normalized_probability = _categorise_win_probability(
                probability
            )
            probability_bucket = win_probability_breakdown.setdefault(
                probability_segment,
                {
                    "label": probability_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "probabilities": [],
                    "courses": set(),
                    "pronostics": set(),
                },
            )
            probability_bucket["label"] = probability_label
            probability_bucket["truths"].append(is_top3)
            probability_bucket["predictions"].append(predicted_label)
            probability_bucket["scores"].append(probability)
            probability_bucket.setdefault("courses", set()).add(course.course_id)
            probability_bucket.setdefault("pronostics", set()).add(pronostic.pronostic_id)
            probability_bucket.setdefault("probabilities", []).append(
                normalized_probability if normalized_probability is not None else probability
            )
            place_segment, place_label, normalized_place_probability = _categorise_place_probability(
                getattr(prediction, "place_probability", None)
            )
            place_bucket = place_probability_breakdown.setdefault(
                place_segment,
                {
                    "label": place_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "place_probabilities": [],
                    "courses": set(),
                    "pronostics": set(),
                },
            )
            place_bucket["label"] = place_label
            place_bucket["truths"].append(is_top3)
            place_bucket["predictions"].append(predicted_label)
            score_value = (
                normalized_place_probability
                if normalized_place_probability is not None
                else probability
            )
            place_bucket.setdefault("scores", []).append(score_value)
            place_bucket.setdefault("courses", set()).add(course.course_id)
            place_bucket.setdefault("pronostics", set()).add(pronostic.pronostic_id)
            if normalized_place_probability is not None:
                place_bucket.setdefault("place_probabilities", []).append(
                    normalized_place_probability
                )
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
                    "label": getattr(course, "course_name", None)
                    or f"Course {getattr(course, 'course_number', course.course_id)}",
                },
            )

            course_entry["label"] = (
                getattr(course, "course_name", None)
                or course_entry.get("label")
                or f"Course {getattr(course, 'course_number', course.course_id)}"
            )

            course_entry["predictions"].append(
                {
                    "probability": probability,
                    "final_position": partant.final_position,
                    "is_top3": bool(is_top3),
                    "truth": int(is_top3),
                    "predicted_label": int(predicted_label),
                    "horse_id": partant.horse_id,
                    "odds": float(partant.odds_pmu)
                    if partant.odds_pmu is not None
                    else None,
                }
            )
            # Stocke le nombre de partants observés afin de catégoriser ensuite
            # les courses par taille de peloton (utile pour repérer les champs
            # où le modèle excelle ou se dégrade).
            course_entry["field_size"] = (
                getattr(course, "number_of_runners", None)
                or len(course_entry["predictions"])
            )

            # Ventile les observations selon la position d'arrivée réelle pour
            # identifier les segments (gagnant, podium, au-delà) où le modèle
            # réussit ou échoue le plus fréquemment.
            if getattr(partant, "final_position", None) is not None:
                final_segment, final_label = _categorise_final_position(
                    partant.final_position
                )
                final_bucket = final_position_breakdown.setdefault(
                    final_segment,
                    {
                        "label": final_label,
                        "truths": [],
                        "predictions": [],
                        "scores": [],
                        "positions": [],
                        "courses": set(),
                        "horses": set(),
                    },
                )
                final_bucket["label"] = final_label
                final_bucket.setdefault("truths", []).append(is_top3)
                final_bucket.setdefault("predictions", []).append(predicted_label)
                final_bucket.setdefault("scores", []).append(probability)
                final_bucket.setdefault("positions", []).append(
                    int(partant.final_position)
                )
                final_bucket.setdefault("courses", set()).add(course.course_id)
                if getattr(partant, "horse_id", None) is not None:
                    final_bucket.setdefault("horses", set()).add(partant.horse_id)

            # Segmente les performances selon que le pronostic a été marqué
            # comme value bet. Cela permet de suivre si les pronostics mis en
            # avant conservent un avantage réel sur les pronostics standards.
            flag_segment, flag_label, flag_value = _categorise_value_bet_flag(
                getattr(pronostic, "value_bet_detected", None)
            )
            flag_bucket = value_bet_flag_breakdown.setdefault(
                flag_segment,
                {
                    "label": flag_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "pronostics": set(),
                    "flags": [],
                },
            )
            flag_bucket["label"] = flag_label
            flag_bucket["truths"].append(is_top3)
            flag_bucket["predictions"].append(predicted_label)
            flag_bucket["scores"].append(probability)
            flag_bucket.setdefault("courses", set()).add(course.course_id)
            flag_bucket.setdefault("pronostics", set()).add(pronostic.pronostic_id)
            if flag_value is not None:
                flag_bucket.setdefault("flags", []).append(flag_value)
                flag_bucket["flag"] = flag_value

            reunion_obj = getattr(course, "reunion", None)

            # Évalue le délai entre la génération du pronostic et l'heure
            # officielle de départ afin de vérifier si les publications
            # tardives/anticipées conservent une précision homogène.
            lead_segment, lead_label, lead_hours = _categorise_publication_lead_time(
                getattr(pronostic, "generated_at", None),
                getattr(reunion_obj, "reunion_date", None),
                getattr(course, "scheduled_time", None),
            )
            lead_bucket = lead_time_breakdown.setdefault(
                lead_segment,
                {
                    "label": lead_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "pronostics": set(),
                    "lead_times": [],
                },
            )
            lead_bucket["label"] = lead_label
            lead_bucket["truths"].append(is_top3)
            lead_bucket["predictions"].append(predicted_label)
            lead_bucket["scores"].append(probability)
            lead_bucket.setdefault("courses", set()).add(course.course_id)
            lead_bucket.setdefault("pronostics", set()).add(pronostic.pronostic_id)
            if lead_hours is not None:
                lead_bucket.setdefault("lead_times", []).append(lead_hours)

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

            # En parallèle, on garde une vue annuelle afin d'observer la
            # trajectoire globale du modèle sur plusieurs saisons sans se
            # limiter aux découpages mensuels/ trimestriels. Les identifiants
            # de course et de réunion sont conservés pour faciliter les
            # investigations en cas de dérive sur une année spécifique.
            year_segment, year_label, year_value = _categorise_year(race_date)
            year_bucket = year_breakdown.setdefault(
                year_segment,
                {
                    "label": year_label,
                    "year": year_value,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "dates": set(),
                },
            )
            year_bucket["truths"].append(is_top3)
            year_bucket["predictions"].append(predicted_label)
            year_bucket["scores"].append(probability)
            year_bucket.setdefault("courses", set()).add(course.course_id)
            year_bucket.setdefault("reunions", set()).add(
                getattr(reunion_obj, "reunion_id", getattr(course, "reunion_id", None))
            )
            if isinstance(race_date, date):
                year_bucket.setdefault("dates", set()).add(race_date.isoformat())

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

            # Aggrège également par trimestre civil afin de disposer d'une
            # vision synthétique des meetings saisonniers (hiver, printemps,
            # été, automne). Cela permet aux analystes de repérer rapidement
            # les périodes où le modèle décroche, sans attendre la fin d'un
            # mois complet.
            quarter_segment, quarter_label, quarter_index, quarter_year = _categorise_quarter(
                race_date
            )
            quarter_bucket = quarter_breakdown.setdefault(
                quarter_segment,
                {
                    "label": quarter_label,
                    "quarter_index": quarter_index,
                    "year": quarter_year,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "dates": set(),
                },
            )
            quarter_bucket["label"] = quarter_label
            quarter_bucket["truths"].append(is_top3)
            quarter_bucket["predictions"].append(predicted_label)
            quarter_bucket["scores"].append(probability)
            quarter_bucket.setdefault("courses", set()).add(course.course_id)
            quarter_bucket.setdefault("reunions", set()).add(
                getattr(reunion_obj, "reunion_id", getattr(course, "reunion_id", None))
            )
            if isinstance(race_date, date):
                quarter_bucket.setdefault("dates", set()).add(race_date.isoformat())

            # Catégorise également les observations par saison météorologique
            # (printemps, été, automne, hiver) pour repérer les tendances
            # multi-mois qui pourraient échapper aux coupes mensuelles.
            (
                season_segment,
                season_label,
                season_index,
                season_year,
            ) = _categorise_season(race_date)
            season_bucket = season_breakdown.setdefault(
                season_segment,
                {
                    "label": season_label,
                    "season_index": season_index,
                    "year": season_year,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "dates": set(),
                },
            )
            season_bucket["label"] = season_label
            season_bucket.setdefault("truths", []).append(is_top3)
            season_bucket.setdefault("predictions", []).append(predicted_label)
            season_bucket.setdefault("scores", []).append(probability)
            season_bucket.setdefault("courses", set()).add(course.course_id)
            season_bucket.setdefault("reunions", set()).add(
                getattr(reunion_obj, "reunion_id", getattr(course, "reunion_id", None))
            )
            if isinstance(race_date, date):
                season_bucket.setdefault("dates", set()).add(race_date.isoformat())

            # Regroupe les performances par numéro de réunion (R1, R2, etc.)
            # afin de vérifier si les matinales, les réunions de journée ou les
            # nocturnes présentent des profils de précision distincts.
            reunion_segment, reunion_label, reunion_number = _categorise_reunion_number(
                getattr(reunion_obj, "reunion_number", getattr(course, "reunion_number", None))
            )
            reunion_bucket = reunion_number_breakdown.setdefault(
                reunion_segment,
                {
                    "label": reunion_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "reunion_numbers": [],
                },
            )
            reunion_bucket["label"] = reunion_label
            reunion_bucket["truths"].append(is_top3)
            reunion_bucket["predictions"].append(predicted_label)
            reunion_bucket["scores"].append(probability)
            reunion_bucket.setdefault("courses", set()).add(course.course_id)
            reunion_bucket.setdefault("reunions", set()).add(
                getattr(reunion_obj, "reunion_id", getattr(course, "reunion_id", None))
            )
            if reunion_number is not None:
                reunion_bucket.setdefault("reunion_numbers", []).append(reunion_number)

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

            weather_segment, weather_label, weather_temperature = _categorise_weather_profile(
                getattr(reunion_obj, "weather_conditions", None)
            )
            weather_bucket = weather_breakdown.setdefault(
                weather_segment,
                {
                    "label": weather_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "temperatures": [],
                },
            )
            weather_bucket["label"] = weather_label
            weather_bucket["truths"].append(is_top3)
            weather_bucket["predictions"].append(predicted_label)
            weather_bucket["scores"].append(probability)
            weather_bucket.setdefault("courses", set()).add(course.course_id)
            weather_bucket.setdefault("reunions", set()).add(
                getattr(reunion_obj, "reunion_id", getattr(course, "reunion_id", None))
            )
            if weather_temperature is not None:
                weather_bucket.setdefault("temperatures", []).append(weather_temperature)

            temperature_segment, temperature_label, normalised_temperature = _categorise_temperature_band(
                weather_temperature
            )
            temperature_bucket = temperature_band_breakdown.setdefault(
                temperature_segment,
                {
                    "label": temperature_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "temperatures": [],
                },
            )
            temperature_bucket["label"] = temperature_label
            temperature_bucket["truths"].append(is_top3)
            temperature_bucket["predictions"].append(predicted_label)
            temperature_bucket["scores"].append(probability)
            temperature_bucket.setdefault("courses", set()).add(course.course_id)
            temperature_bucket.setdefault("reunions", set()).add(
                getattr(reunion_obj, "reunion_id", getattr(course, "reunion_id", None))
            )
            if normalised_temperature is not None:
                temperature_bucket.setdefault("temperatures", []).append(
                    normalised_temperature
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

            edge_value = _compute_probability_edge(probability, odds_value)
            implied_probability = (1.0 / odds_value) if odds_value and odds_value > 0 else None
            edge_segment, edge_label = _categorise_probability_edge(edge_value)
            edge_bucket = probability_edge_breakdown.setdefault(
                edge_segment,
                {
                    "label": edge_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "edges": [],
                    "implied_probabilities": [],
                    "odds": [],
                    "courses": set(),
                    "horses": set(),
                },
            )
            edge_bucket["label"] = edge_label
            edge_bucket["truths"].append(is_top3)
            edge_bucket["predictions"].append(predicted_label)
            edge_bucket["scores"].append(probability)
            edge_bucket.setdefault("courses", set()).add(course.course_id)
            if getattr(partant, "horse_id", None) is not None:
                edge_bucket.setdefault("horses", set()).add(partant.horse_id)
            if edge_value is not None:
                edge_bucket.setdefault("edges", []).append(edge_value)
            if implied_probability is not None:
                edge_bucket.setdefault("implied_probabilities", []).append(implied_probability)
            if odds_value is not None:
                edge_bucket.setdefault("odds", []).append(odds_value)

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

            discipline_surface_key = f"{discipline_label}__{surface_label}"
            discipline_surface_bucket = discipline_surface_breakdown.setdefault(
                discipline_surface_key,
                {
                    "discipline": discipline_label,
                    "surface": surface_label,
                    "label": f"{discipline_label.replace('_', ' ').title()} · {surface_label.replace('_', ' ').title()}",
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "distances": [],
                },
            )
            discipline_surface_bucket["truths"].append(is_top3)
            discipline_surface_bucket["predictions"].append(predicted_label)
            discipline_surface_bucket["scores"].append(probability)
            discipline_surface_bucket.setdefault("courses", set()).add(course.course_id)
            if reunion_entity is not None and getattr(reunion_entity, "reunion_id", None):
                discipline_surface_bucket.setdefault("reunions", set()).add(
                    reunion_entity.reunion_id
                )

            course_distance = getattr(course, "distance", None)
            if course_distance:
                distance_bucket.setdefault("distances", []).append(int(course_distance))
                discipline_surface_bucket.setdefault("distances", []).append(
                    int(course_distance)
                )

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

            per_runner_segment, per_runner_label, per_runner_value, per_runner_field = _categorise_prize_per_runner(
                getattr(course, "prize_money", None),
                getattr(course, "number_of_runners", None),
            )
            per_runner_bucket = prize_per_runner_breakdown.setdefault(
                per_runner_segment,
                {
                    "label": per_runner_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "per_runner_values": [],
                    "field_sizes": [],
                },
            )
            per_runner_bucket["label"] = per_runner_label
            per_runner_bucket["truths"].append(is_top3)
            per_runner_bucket["predictions"].append(predicted_label)
            per_runner_bucket["scores"].append(probability)
            per_runner_courses = per_runner_bucket.setdefault("courses", set())
            is_new_per_runner_course = course.course_id not in per_runner_courses
            per_runner_courses.add(course.course_id)
            if per_runner_value is not None and is_new_per_runner_course:
                per_runner_bucket.setdefault("per_runner_values", []).append(per_runner_value)
            if per_runner_field is not None and is_new_per_runner_course:
                per_runner_bucket.setdefault("field_sizes", []).append(per_runner_field)

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

            equipment_segment, equipment_label, equipment_count, has_blinkers = _categorise_equipment_profile(
                getattr(partant, "equipment", None)
            )
            equipment_bucket = equipment_breakdown.setdefault(
                equipment_segment,
                {
                    "label": equipment_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "item_counts": [],
                    "blinkers_flags": [],
                },
            )
            equipment_bucket["label"] = equipment_label
            equipment_bucket["truths"].append(is_top3)
            equipment_bucket["predictions"].append(predicted_label)
            equipment_bucket["scores"].append(probability)
            equipment_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                equipment_bucket.setdefault("horses", set()).add(partant.horse_id)
            if equipment_count is not None:
                equipment_bucket.setdefault("item_counts", []).append(equipment_count)
            if has_blinkers is not None:
                equipment_bucket.setdefault("blinkers_flags", []).append(has_blinkers)

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
            sire_key, sire_label = _normalise_sire_label(
                getattr(horse_entity, "sire", None) if horse_entity else None
            )
            sire_bucket = horse_sire_breakdown.setdefault(
                sire_key,
                {
                    "label": sire_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "trainers": set(),
                    "raw_inputs": [],
                },
            )
            sire_bucket["label"] = sire_label
            sire_bucket["truths"].append(is_top3)
            sire_bucket["predictions"].append(predicted_label)
            sire_bucket["scores"].append(probability)
            sire_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                sire_bucket.setdefault("horses", set()).add(partant.horse_id)
            if getattr(partant, "trainer_id", None):
                sire_bucket.setdefault("trainers", set()).add(partant.trainer_id)
            raw_sire_name = getattr(horse_entity, "sire", None)
            if isinstance(raw_sire_name, str) and raw_sire_name.strip():
                sire_bucket.setdefault("raw_inputs", []).append(raw_sire_name.strip())

            dam_key, dam_label = _normalise_dam_label(
                getattr(horse_entity, "dam", None) if horse_entity else None
            )
            dam_bucket = horse_dam_breakdown.setdefault(
                dam_key,
                {
                    "label": dam_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "raw_inputs": [],
                },
            )
            dam_bucket["label"] = dam_label
            dam_bucket["truths"].append(is_top3)
            dam_bucket["predictions"].append(predicted_label)
            dam_bucket["scores"].append(probability)
            dam_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                dam_bucket.setdefault("horses", set()).add(partant.horse_id)
            raw_dam_name = getattr(horse_entity, "dam", None)
            if isinstance(raw_dam_name, str) and raw_dam_name.strip():
                dam_bucket.setdefault("raw_inputs", []).append(raw_dam_name.strip())

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

            coat_key, coat_label = _normalise_coat_color_label(
                getattr(horse_entity, "coat_color", None) if horse_entity else None
            )
            coat_bucket = horse_coat_breakdown.setdefault(
                coat_key,
                {
                    "label": coat_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "raw_inputs": [],
                },
            )
            coat_bucket["label"] = coat_label
            coat_bucket["truths"].append(is_top3)
            coat_bucket["predictions"].append(predicted_label)
            coat_bucket["scores"].append(probability)
            coat_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                coat_bucket.setdefault("horses", set()).add(partant.horse_id)
            if getattr(horse_entity, "coat_color", None):
                coat_bucket.setdefault("raw_inputs", []).append(
                    str(getattr(horse_entity, "coat_color"))
                )

            breed_key, breed_label = _normalise_horse_breed_label(
                getattr(horse_entity, "breed", None) if horse_entity else None
            )
            breed_bucket = horse_breed_breakdown.setdefault(
                breed_key,
                {
                    "label": breed_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "raw_inputs": [],
                },
            )
            breed_bucket["label"] = breed_label
            breed_bucket["truths"].append(is_top3)
            breed_bucket["predictions"].append(predicted_label)
            breed_bucket["scores"].append(probability)
            breed_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                breed_bucket.setdefault("horses", set()).add(partant.horse_id)
            if getattr(horse_entity, "breed", None):
                breed_bucket.setdefault("raw_inputs", []).append(
                    str(getattr(horse_entity, "breed"))
                )

            owner_key, owner_label = _normalise_owner_label(
                getattr(horse_entity, "owner", None) if horse_entity else None
            )
            owner_bucket = owner_breakdown.setdefault(
                owner_key,
                {
                    "label": owner_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "trainers": set(),
                    "jockeys": set(),
                    "hippodromes": set(),
                },
            )
            owner_bucket["label"] = owner_label
            owner_bucket["truths"].append(is_top3)
            owner_bucket["predictions"].append(predicted_label)
            owner_bucket["scores"].append(probability)
            owner_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                owner_bucket.setdefault("horses", set()).add(partant.horse_id)
            if partant.trainer_id:
                owner_bucket.setdefault("trainers", set()).add(partant.trainer_id)
            if partant.jockey_id:
                owner_bucket.setdefault("jockeys", set()).add(partant.jockey_id)
            reunion_for_owner = getattr(course, "reunion", None)
            owner_hippodrome_id = None
            if reunion_for_owner is not None:
                owner_hippodrome_id = getattr(reunion_for_owner, "hippodrome_id", None)
            if owner_hippodrome_id is not None:
                owner_bucket.setdefault("hippodromes", set()).add(int(owner_hippodrome_id))

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

            parity_segment, parity_label = _categorise_draw_parity(
                int(draw_value) if draw_value is not None else None
            )
            parity_bucket = draw_parity_breakdown.setdefault(
                parity_segment,
                {
                    "label": parity_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "draws": [],
                },
            )
            parity_bucket["label"] = parity_label
            parity_bucket["truths"].append(is_top3)
            parity_bucket["predictions"].append(predicted_label)
            parity_bucket["scores"].append(probability)
            parity_bucket.setdefault("courses", set()).add(course.course_id)
            if draw_value is not None:
                parity_bucket.setdefault("draws", []).append(int(draw_value))

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

            delay_segment, delay_label, delay_minutes = _categorise_start_delay(
                getattr(course, "scheduled_time", None),
                getattr(course, "actual_start_time", None),
            )
            delay_bucket = start_delay_breakdown.setdefault(
                delay_segment,
                {
                    "label": delay_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "delays": [],
                },
            )
            delay_bucket["label"] = delay_label
            delay_bucket["truths"].append(is_top3)
            delay_bucket["predictions"].append(predicted_label)
            delay_bucket["scores"].append(probability)
            delay_bucket.setdefault("courses", set()).add(course.course_id)
            if getattr(reunion_entity, "reunion_id", None) is not None:
                delay_bucket.setdefault("reunions", set()).add(reunion_entity.reunion_id)
            if delay_minutes is not None:
                delay_bucket.setdefault("delays", []).append(delay_minutes)

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

            recent_form_list = getattr(partant, "recent_form_list", None)
            recent_segment, recent_label = _categorise_recent_form(recent_form_list)
            recent_form_bucket = recent_form_breakdown.setdefault(
                recent_segment,
                {
                    "label": recent_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "average_positions": [],
                    "best_positions": [],
                    "worst_positions": [],
                    "starts_counts": [],
                    "win_flags": [],
                },
            )
            recent_form_bucket["label"] = recent_label
            recent_form_bucket["truths"].append(is_top3)
            recent_form_bucket["predictions"].append(predicted_label)
            recent_form_bucket["scores"].append(probability)
            recent_form_bucket.setdefault("courses", set()).add(course.course_id)
            if getattr(partant, "horse_id", None):
                recent_form_bucket.setdefault("horses", set()).add(partant.horse_id)

            if recent_form_list:
                average_form = getattr(partant, "average_recent_position", None)
                if average_form is not None:
                    recent_form_bucket.setdefault("average_positions", []).append(
                        float(average_form)
                    )
                recent_form_bucket.setdefault("best_positions", []).append(
                    int(min(recent_form_list))
                )
                recent_form_bucket.setdefault("worst_positions", []).append(
                    int(max(recent_form_list))
                )
                recent_form_bucket.setdefault("starts_counts", []).append(
                    len(recent_form_list)
                )
                recent_form_bucket.setdefault("win_flags", []).append(
                    bool(getattr(partant, "has_won_recently", False))
                )

            reunion_entity = getattr(course, "reunion", None)

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

            (
                jockey_exp_segment,
                jockey_exp_label,
                normalised_jockey_starts,
            ) = _categorise_jockey_experience(
                getattr(getattr(partant, "jockey", None), "career_starts", None)
            )
            jockey_experience_bucket = jockey_experience_breakdown.setdefault(
                jockey_exp_segment,
                {
                    "label": jockey_exp_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "actors": set(),
                    "career_starts": [],
                },
            )
            jockey_experience_bucket["label"] = jockey_exp_label
            jockey_experience_bucket["truths"].append(is_top3)
            jockey_experience_bucket["predictions"].append(predicted_label)
            jockey_experience_bucket["scores"].append(probability)
            jockey_experience_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.jockey_id:
                jockey_experience_bucket.setdefault("actors", set()).add(partant.jockey_id)
            if normalised_jockey_starts is not None:
                jockey_experience_bucket.setdefault("career_starts", []).append(
                    int(normalised_jockey_starts)
                )

            # Agrège également les performances par nationalité du jockey afin
            # de repérer d'éventuels biais liés aux profils internationaux.
            jockey_nat_key, jockey_nat_label = _normalise_nationality_label(
                getattr(getattr(partant, "jockey", None), "nationality", None)
            )
            jockey_nat_bucket = jockey_nationality_breakdown.setdefault(
                jockey_nat_key,
                {
                    "label": jockey_nat_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "horses": set(),
                    "actors": set(),
                },
            )
            jockey_nat_bucket["label"] = jockey_nat_label
            jockey_nat_bucket["truths"].append(is_top3)
            jockey_nat_bucket["predictions"].append(predicted_label)
            jockey_nat_bucket["scores"].append(probability)
            jockey_nat_bucket.setdefault("courses", set()).add(course.course_id)
            if reunion_entity is not None and getattr(reunion_entity, "reunion_id", None):
                jockey_nat_bucket.setdefault("reunions", set()).add(reunion_entity.reunion_id)
            if getattr(partant, "horse_id", None):
                jockey_nat_bucket.setdefault("horses", set()).add(partant.horse_id)
            if partant.jockey_id:
                jockey_nat_bucket.setdefault("actors", set()).add(partant.jockey_id)

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

            (
                trainer_exp_segment,
                trainer_exp_label,
                normalised_trainer_starts,
            ) = _categorise_trainer_experience(
                getattr(getattr(partant, "trainer", None), "career_starts", None)
            )
            trainer_experience_bucket = trainer_experience_breakdown.setdefault(
                trainer_exp_segment,
                {
                    "label": trainer_exp_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "actors": set(),
                    "career_starts": [],
                },
            )
            trainer_experience_bucket["label"] = trainer_exp_label
            trainer_experience_bucket["truths"].append(is_top3)
            trainer_experience_bucket["predictions"].append(predicted_label)
            trainer_experience_bucket["scores"].append(probability)
            trainer_experience_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.trainer_id:
                trainer_experience_bucket.setdefault("actors", set()).add(partant.trainer_id)
            if normalised_trainer_starts is not None:
                trainer_experience_bucket.setdefault("career_starts", []).append(
                    int(normalised_trainer_starts)
                )

            owner_trainer_identifier = f"{owner_key}__{trainer_identifier}"
            owner_trainer_label = f"{owner_label} × {trainer_label}"
            owner_trainer_bucket = owner_trainer_breakdown.setdefault(
                owner_trainer_identifier,
                {
                    "label": owner_trainer_label,
                    "owner_label": owner_label,
                    "trainer_label": trainer_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "owners": set(),
                    "trainers": set(),
                },
            )
            owner_trainer_bucket["label"] = owner_trainer_label
            owner_trainer_bucket["owner_label"] = owner_label
            owner_trainer_bucket["trainer_label"] = trainer_label
            owner_trainer_bucket["truths"].append(is_top3)
            owner_trainer_bucket["predictions"].append(predicted_label)
            owner_trainer_bucket["scores"].append(probability)
            owner_trainer_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                owner_trainer_bucket.setdefault("horses", set()).add(partant.horse_id)
            owner_trainer_bucket.setdefault("owners", set()).add(owner_key)
            if partant.trainer_id:
                owner_trainer_bucket.setdefault("trainers", set()).add(partant.trainer_id)

            owner_jockey_identifier = f"{owner_key}__{jockey_identifier}"
            owner_jockey_label = f"{owner_label} × {jockey_label}"
            owner_jockey_bucket = owner_jockey_breakdown.setdefault(
                owner_jockey_identifier,
                {
                    "label": owner_jockey_label,
                    "owner_label": owner_label,
                    "jockey_label": jockey_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "owners": set(),
                    "jockeys": set(),
                },
            )
            owner_jockey_bucket["label"] = owner_jockey_label
            owner_jockey_bucket["owner_label"] = owner_label
            owner_jockey_bucket["jockey_label"] = jockey_label
            owner_jockey_bucket["truths"].append(is_top3)
            owner_jockey_bucket["predictions"].append(predicted_label)
            owner_jockey_bucket["scores"].append(probability)
            owner_jockey_bucket.setdefault("courses", set()).add(course.course_id)
            if partant.horse_id:
                owner_jockey_bucket.setdefault("horses", set()).add(partant.horse_id)
            owner_jockey_bucket.setdefault("owners", set()).add(owner_key)
            if partant.jockey_id:
                owner_jockey_bucket.setdefault("jockeys", set()).add(partant.jockey_id)

            combo_identifier = f"{jockey_identifier}__{trainer_identifier}"
            combo_label = f"{jockey_label} × {trainer_label}"
            combo_bucket = jockey_trainer_breakdown.setdefault(
                combo_identifier,
                {
                    "label": combo_label,
                    "jockey_label": jockey_label,
                    "trainer_label": trainer_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "horses": set(),
                    "jockeys": set(),
                    "trainers": set(),
                },
            )
            combo_bucket["label"] = combo_label
            combo_bucket["jockey_label"] = jockey_label
            combo_bucket["trainer_label"] = trainer_label
            combo_bucket["truths"].append(is_top3)
            combo_bucket["predictions"].append(predicted_label)
            combo_bucket["scores"].append(probability)
            combo_bucket.setdefault("courses", set()).add(course.course_id)
            combo_bucket.setdefault("horses", set()).add(partant.horse_id)
            if partant.jockey_id is not None:
                combo_bucket.setdefault("jockeys", set()).add(partant.jockey_id)
            if partant.trainer_id is not None:
                combo_bucket.setdefault("trainers", set()).add(partant.trainer_id)

            trainer_nat_key, trainer_nat_label = _normalise_nationality_label(
                getattr(getattr(partant, "trainer", None), "nationality", None)
            )
            trainer_nat_bucket = trainer_nationality_breakdown.setdefault(
                trainer_nat_key,
                {
                    "label": trainer_nat_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "horses": set(),
                    "actors": set(),
                },
            )
            trainer_nat_bucket["label"] = trainer_nat_label
            trainer_nat_bucket["truths"].append(is_top3)
            trainer_nat_bucket["predictions"].append(predicted_label)
            trainer_nat_bucket["scores"].append(probability)
            trainer_nat_bucket.setdefault("courses", set()).add(course.course_id)
            if reunion_entity is not None and getattr(reunion_entity, "reunion_id", None):
                trainer_nat_bucket.setdefault("reunions", set()).add(reunion_entity.reunion_id)
            if getattr(partant, "horse_id", None):
                trainer_nat_bucket.setdefault("horses", set()).add(partant.horse_id)
            if partant.trainer_id:
                trainer_nat_bucket.setdefault("actors", set()).add(partant.trainer_id)

            # Enfin, on garde une vue géographique afin de repérer les hippodromes
            # où le modèle excelle ou se dégrade. Cette information aide à prioriser
            # les analyses locales (qualité des données, biais spécifiques, météo...).
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

            if hippodrome_id is not None:
                api_bucket.setdefault("hippodromes", set()).add(int(hippodrome_id))

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

            country_key, country_label = _normalise_country_label(
                getattr(hippodrome_entity, "country", None)
            )
            country_bucket = country_breakdown.setdefault(
                country_key,
                {
                    "label": country_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "hippodromes": set(),
                    "cities": set(),
                },
            )
            country_bucket["label"] = country_label
            country_bucket["truths"].append(is_top3)
            country_bucket["predictions"].append(predicted_label)
            country_bucket["scores"].append(probability)
            country_bucket.setdefault("courses", set()).add(course.course_id)
            if reunion_entity is not None and getattr(reunion_entity, "reunion_id", None):
                country_bucket.setdefault("reunions", set()).add(reunion_entity.reunion_id)
            if hippodrome_id is not None:
                country_bucket.setdefault("hippodromes", set()).add(int(hippodrome_id))
            city_value = (
                getattr(hippodrome_entity, "city", None)
                if hippodrome_entity is not None
                else None
            )
            if city_value:
                country_bucket.setdefault("cities", set()).add(str(city_value))

            city_key, city_label = _normalise_city_label(city_value)
            city_bucket = city_breakdown.setdefault(
                city_key,
                {
                    "label": city_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "hippodromes": set(),
                    "countries": set(),
                },
            )
            city_bucket["label"] = city_label
            city_bucket["truths"].append(is_top3)
            city_bucket["predictions"].append(predicted_label)
            city_bucket["scores"].append(probability)
            city_bucket.setdefault("courses", set()).add(course.course_id)
            if reunion_entity is not None and getattr(reunion_entity, "reunion_id", None):
                city_bucket.setdefault("reunions", set()).add(reunion_entity.reunion_id)
            if hippodrome_id is not None:
                city_bucket.setdefault("hippodromes", set()).add(int(hippodrome_id))
            if country_label:
                city_bucket.setdefault("countries", set()).add(country_label)

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

            track_length_segment, track_length_label, track_length_value = _categorise_track_length(
                getattr(hippodrome_entity, "track_length", None)
            )
            track_length_bucket = track_length_breakdown.setdefault(
                track_length_segment,
                {
                    "label": track_length_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "courses": set(),
                    "reunions": set(),
                    "hippodromes": set(),
                    "track_lengths": [],
                },
            )
            track_length_bucket["label"] = track_length_label
            track_length_bucket["truths"].append(is_top3)
            track_length_bucket["predictions"].append(predicted_label)
            track_length_bucket["scores"].append(probability)
            track_length_bucket.setdefault("courses", set()).add(course.course_id)
            if reunion_entity is not None and getattr(reunion_entity, "reunion_id", None):
                track_length_bucket.setdefault("reunions", set()).add(
                    reunion_entity.reunion_id
                )
            if hippodrome_id is not None:
                track_length_bucket.setdefault("hippodromes", set()).add(hippodrome_id)
            if track_length_value is not None:
                track_length_bucket.setdefault("track_lengths", []).append(track_length_value)

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

        for course_id, course_payload in course_stats.items():
            predictions_payload = list(course_payload.get("predictions", []))
            if not predictions_payload:
                continue

            sorted_predictions = sorted(
                predictions_payload,
                key=lambda item: float(item.get("probability", 0.0)),
                reverse=True,
            )

            # Calcule l'écart entre les deux meilleures probabilités pour suivre
            # la stabilité du favori face à son dauphin. Ce tableau de bord
            # permet de distinguer les courses remportées par un favori solide
            # de celles où le modèle hésite entre plusieurs partants.
            primary_probability = sorted_predictions[0].get("probability")
            secondary_probability = (
                sorted_predictions[1].get("probability")
                if len(sorted_predictions) > 1
                else None
            )
            (
                margin_segment,
                margin_label,
                normalised_margin,
            ) = _categorise_probability_margin(primary_probability, secondary_probability)
            margin_bucket = probability_margin_breakdown.setdefault(
                margin_segment,
                {
                    "label": margin_label,
                    "truths": [],
                    "predictions": [],
                    "scores": [],
                    "margins": [],
                    "courses": set(),
                    "horses": set(),
                },
            )
            margin_bucket["label"] = margin_label
            margin_bucket["truths"].append(int(sorted_predictions[0].get("truth", 0)))
            margin_bucket["predictions"].append(
                int(sorted_predictions[0].get("predicted_label", 0))
            )
            margin_bucket["scores"].append(float(primary_probability or 0.0))
            margin_bucket.setdefault("courses", set()).add(int(course_id))

            top_horse_id = sorted_predictions[0].get("horse_id")
            if top_horse_id is not None:
                try:
                    margin_bucket.setdefault("horses", set()).add(int(top_horse_id))
                except (TypeError, ValueError):  # pragma: no cover - sécurité défensive
                    pass

            if normalised_margin is not None:
                margin_bucket.setdefault("margins", []).append(normalised_margin)

            pmu_candidates = [
                sample
                for sample in predictions_payload
                if sample.get("odds") is not None
            ]
            pmu_favourite: Optional[Dict[str, object]] = None
            if pmu_candidates:
                pmu_favourite = min(
                    pmu_candidates,
                    key=lambda item: float(item.get("odds", float("inf"))),
                )

            top_entry: Optional[Dict[str, object]] = sorted_predictions[0] if sorted_predictions else None
            pmu_rank: Optional[int] = None
            if (
                pmu_favourite is not None
                and pmu_favourite.get("horse_id") is not None
                and sorted_predictions
            ):
                for rank_index, sample in enumerate(sorted_predictions, start=1):
                    if sample.get("horse_id") == pmu_favourite.get("horse_id"):
                        pmu_rank = rank_index
                        break

            alignment_segment, alignment_label = _categorise_favourite_alignment(
                top_entry,
                pmu_favourite,
            )
            alignment_bucket = favourite_alignment_breakdown.setdefault(
                alignment_segment,
                {
                    "label": alignment_label,
                    "courses": set(),
                    "model_truths": [],
                    "model_predictions": [],
                    "model_scores": [],
                    "model_wins": 0,
                    "pmu_truths": [],
                    "pmu_scores": [],
                    "pmu_odds": [],
                    "pmu_wins": 0,
                    "pmu_courses": 0,
                    "probability_gaps": [],
                    "pmu_ranks": [],
                    "aligned_wins": 0,
                },
            )
            alignment_bucket["label"] = alignment_label
            alignment_bucket.setdefault("courses", set()).add(int(course_id))

            if top_entry is not None:
                alignment_bucket.setdefault("model_truths", []).append(
                    int(top_entry.get("truth", 0))
                )
                alignment_bucket.setdefault("model_predictions", []).append(
                    int(top_entry.get("predicted_label", 0))
                )
                alignment_bucket.setdefault("model_scores", []).append(
                    float(top_entry.get("probability", 0.0))
                )
                if top_entry.get("final_position") == 1:
                    alignment_bucket["model_wins"] = alignment_bucket.get("model_wins", 0) + 1

            if pmu_favourite is not None:
                alignment_bucket["pmu_courses"] = alignment_bucket.get("pmu_courses", 0) + 1
                alignment_bucket.setdefault("pmu_truths", []).append(
                    int(pmu_favourite.get("truth", 0))
                )
                alignment_bucket.setdefault("pmu_scores", []).append(
                    float(pmu_favourite.get("probability", 0.0))
                )
                odds_value = pmu_favourite.get("odds")
                if odds_value is not None:
                    alignment_bucket.setdefault("pmu_odds", []).append(float(odds_value))
                if pmu_favourite.get("final_position") == 1:
                    alignment_bucket["pmu_wins"] = alignment_bucket.get("pmu_wins", 0) + 1
                if top_entry is not None:
                    alignment_bucket.setdefault("probability_gaps", []).append(
                        float(top_entry.get("probability", 0.0))
                        - float(pmu_favourite.get("probability", 0.0))
                    )
                    if (
                        top_entry.get("horse_id") == pmu_favourite.get("horse_id")
                        and top_entry.get("final_position") == 1
                    ):
                        alignment_bucket["aligned_wins"] = alignment_bucket.get(
                            "aligned_wins", 0
                        ) + 1
                if pmu_rank is not None:
                    alignment_bucket.setdefault("pmu_ranks", []).append(float(pmu_rank))

            for rank, sample in enumerate(sorted_predictions, start=1):
                segment_key, segment_label = _categorise_prediction_rank(rank)
                bucket = prediction_rank_breakdown.setdefault(
                    segment_key,
                    {
                        "label": segment_label,
                        "truths": [],
                        "predictions": [],
                        "scores": [],
                        "final_positions": [],
                        "ranks": [],
                        "courses": set(),
                        "horses": set(),
                    },
                )

                bucket["label"] = segment_label
                bucket.setdefault("truths", []).append(int(sample.get("truth", 0)))
                bucket.setdefault("predictions", []).append(
                    int(sample.get("predicted_label", 0))
                )
                bucket.setdefault("scores", []).append(
                    float(sample.get("probability", 0.0))
                )
                bucket.setdefault("ranks", []).append(rank)
                bucket.setdefault("courses", set()).add(int(course_id))

                final_position = sample.get("final_position")
                if final_position is not None:
                    bucket.setdefault("final_positions", []).append(int(final_position))

                horse_id = sample.get("horse_id")
                if horse_id:
                    bucket.setdefault("horses", set()).add(int(horse_id))

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

        brier_score: Optional[float] = None
        positive_rate: Optional[float] = None
        if y_true:
            brier_score = (
                sum((score - truth) ** 2 for score, truth in zip(y_scores, y_true))
                / len(y_true)
            )
            positive_rate = sum(y_true) / len(y_true)

        cm = [
            [0, 0],
            [0, 0],
        ]
        if confusion_matrix:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

        # Convertit la matrice de confusion en compte explicite (TN, FP, FN, TP)
        # pour calculer un coefficient de corrélation équilibré.
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        matthews_correlation = _compute_matthews_correlation(tn, fp, fn, tp)
        classification_insights = _compute_binary_classification_insights(tn, fp, fn, tp)

        positives = sum(y_pred)
        negatives = len(y_pred) - positives

        avg_positive_prob = _safe_average([score for score, label in zip(y_scores, y_pred) if label == 1])
        avg_negative_prob = _safe_average([score for score, label in zip(y_scores, y_pred) if label == 0])

        # Diagnostic complémentaire : on analyse la distribution des scores
        # probabilistes afin d'identifier un éventuel recouvrement entre gagnants
        # et perdants malgré des métriques globales satisfaisantes.
        probability_distribution_metrics = _summarise_probability_distribution(
            y_true,
            y_scores,
        )

        course_count = len(course_stats)
        favourite_alignment_performance = _summarise_favourite_alignment_performance(
            favourite_alignment_breakdown,
            course_count,
        )
        top1_correct = 0
        top3_course_hits = 0
        winner_probabilities: List[float] = []
        top3_probabilities: List[float] = []
        winner_ranks: List[int] = []

        for course_id, data in course_stats.items():
            predictions = data["predictions"]  # type: ignore[assignment]
            sorted_predictions = sorted(
                predictions, key=lambda item: item["probability"], reverse=True
            )
            if not sorted_predictions:
                continue

            ranked_entries: List[Tuple[Dict[str, object], Optional[int]]] = []
            for item in sorted_predictions:
                final_position_raw = item.get("final_position")
                try:
                    final_position_value = (
                        int(final_position_raw)
                        if final_position_raw is not None
                        else None
                    )
                except (TypeError, ValueError):  # pragma: no cover - sécurité sur données corrompues
                    final_position_value = None

                ranked_entries.append((item, final_position_value))

            error_bucket = rank_error_tracking.setdefault(
                course_id,
                {
                    "key": f"course_{course_id}",
                    "label": data.get("label"),
                    "absolute_errors": [],
                    "squared_errors": [],
                    "signed_errors": [],
                    "samples": 0,
                    "perfect_predictions": 0,
                },
            )
            error_bucket["label"] = data.get("label") or error_bucket.get("label")
            error_bucket["runner_count"] = len(ranked_entries)

            for predicted_rank, (_, final_position_value) in enumerate(
                ranked_entries, start=1
            ):
                if final_position_value is None:
                    continue

                absolute_error = abs(predicted_rank - final_position_value)
                signed_error = predicted_rank - final_position_value

                error_bucket.setdefault("absolute_errors", []).append(
                    float(absolute_error)
                )
                error_bucket.setdefault("squared_errors", []).append(
                    float(absolute_error**2)
                )
                error_bucket.setdefault("signed_errors", []).append(float(signed_error))
                error_bucket["samples"] = error_bucket.get("samples", 0) + 1
                if absolute_error == 0:
                    error_bucket["perfect_predictions"] = (
                        error_bucket.get("perfect_predictions", 0) + 1
                    )

            # Conserve des indicateurs par panier Top N (Top 1 → Top 5) afin de
            # suivre la précision cumulative de la sélection du modèle.
            top_limit = min(len(ranked_entries), 5)
            for top_n in range(1, top_limit + 1):
                bucket = topn_tracking.setdefault(
                    top_n,
                    {
                        "courses": 0,
                        "winner_hits": 0,
                        "place_hits": 0,
                        "probabilities": [],
                        "best_finishes": [],
                    },
                )
                bucket["courses"] += 1
                top_subset = ranked_entries[:top_n]
                bucket["probabilities"].extend(
                    float(entry_data[0]["probability"]) for entry_data in top_subset
                )
                finish_positions = [
                    pos for _, pos in top_subset if pos is not None
                ]
                if finish_positions:
                    bucket["best_finishes"].append(min(finish_positions))
                if any(pos == 1 for _, pos in top_subset if pos is not None):
                    bucket["winner_hits"] += 1
                if any(pos is not None and pos <= 3 for _, pos in top_subset):
                    bucket["place_hits"] += 1

            ndcg_at_3_value = _compute_normalised_dcg(ranked_entries, 3)
            if ndcg_at_3_value is not None:
                ndcg_at_3_scores.append(ndcg_at_3_value)

            ndcg_at_5_value = _compute_normalised_dcg(ranked_entries, 5)
            if ndcg_at_5_value is not None:
                ndcg_at_5_scores.append(ndcg_at_5_value)

            winner_rank: Optional[int] = None
            for index, (item, final_position_value) in enumerate(ranked_entries, start=1):
                if final_position_value == 1 and winner_rank is None:
                    winner_rank = index
                    winner_probabilities.append(float(item["probability"]))

                if final_position_value is not None and final_position_value <= 3:
                    top3_probabilities.append(float(item["probability"]))

            if winner_rank is not None:
                winner_ranks.append(winner_rank)

            top1_position = ranked_entries[0][1]
            if top1_position == 1:
                top1_correct += 1

            top3_predictions = ranked_entries[:3]
            if any(
                final_position_value is not None and final_position_value <= 3
                for _, final_position_value in top3_predictions
            ):
                top3_course_hits += 1

        winner_rank_metrics = _summarise_winner_rankings(winner_ranks, course_count)
        topn_performance = _summarise_topn_performance(topn_tracking, course_count)
        rank_error_metrics = _summarise_rank_error_metrics(rank_error_tracking)

        calibration_table = _build_calibration_table(y_scores, y_true, bins=5)
        # Résume l'ampleur des écarts de calibration pour suivre un indicateur
        # synthétique (ECE, biais signé, écart maximal) en plus du tableau brut.
        calibration_diagnostics = _describe_calibration_quality(calibration_table)
        brier_decomposition = _decompose_brier_score(
            calibration_table,
            base_rate=positive_rate,
            brier_score=brier_score,
        )
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
        confidence_level_metrics = _summarise_prediction_confidence_performance(
            confidence_breakdown,
            len(y_true),
        )

        confidence_score_performance = _summarise_confidence_score_performance(
            confidence_score_breakdown
        )
        win_probability_performance = _summarise_win_probability_performance(
            win_probability_breakdown
        )
        place_probability_performance = _summarise_place_probability_performance(
            place_probability_breakdown
        )
        probability_edge_performance = _summarise_probability_edge_performance(
            probability_edge_breakdown
        )
        probability_error_performance = _summarise_probability_error_performance(
            probability_error_breakdown
        )
        prediction_outcome_performance = _summarise_prediction_outcome_performance(
            prediction_outcome_breakdown
        )
        probability_margin_performance = _summarise_probability_margin_performance(
            probability_margin_breakdown
        )
        rank_correlation_performance = _summarise_rank_correlation_performance(
            rank_correlation_tracking
        )

        daily_performance = _summarise_daily_performance(daily_breakdown)
        day_part_performance = _summarise_day_part_performance(day_part_breakdown)
        lead_time_performance = _summarise_lead_time_performance(lead_time_breakdown)
        year_performance = _summarise_year_performance(year_breakdown)
        month_performance = _summarise_month_performance(month_breakdown)
        season_performance = _summarise_season_performance(season_breakdown)
        quarter_performance = _summarise_quarter_performance(quarter_breakdown)
        weekday_performance = _summarise_weekday_performance(weekday_breakdown)
        race_order_performance = _summarise_race_order_performance(race_order_breakdown)
        reunion_number_performance = _summarise_reunion_number_performance(
            reunion_number_breakdown
        )
        discipline_performance = _summarise_segment_performance(discipline_breakdown)
        distance_performance = _summarise_distance_performance(distance_breakdown)
        surface_performance = _summarise_segment_performance(surface_breakdown)
        discipline_surface_performance = _summarise_discipline_surface_performance(
            discipline_surface_breakdown
        )
        weather_performance = _summarise_weather_performance(weather_breakdown)
        temperature_band_performance = _summarise_temperature_band_performance(
            temperature_band_breakdown
        )
        prize_money_performance = _summarise_prize_money_performance(
            prize_money_breakdown
        )
        prize_per_runner_performance = _summarise_prize_per_runner_performance(
            prize_per_runner_breakdown
        )
        handicap_performance = _summarise_handicap_performance(handicap_breakdown)
        weight_performance = _summarise_weight_performance(weight_breakdown)
        equipment_performance = _summarise_equipment_performance(equipment_breakdown)
        odds_band_performance = _summarise_odds_band_performance(odds_band_breakdown)
        horse_age_performance = _summarise_horse_age_performance(
            horse_age_breakdown
        )
        horse_gender_performance = _summarise_horse_gender_performance(
            horse_gender_breakdown
        )
        horse_coat_performance = _summarise_horse_coat_performance(
            horse_coat_breakdown
        )
        horse_breed_performance = _summarise_horse_breed_performance(
            horse_breed_breakdown
        )
        horse_sire_performance = _summarise_sire_performance(
            horse_sire_breakdown
        )
        horse_dam_performance = _summarise_dam_performance(
            horse_dam_breakdown
        )
        owner_performance = _summarise_owner_performance(owner_breakdown)
        owner_trainer_performance = _summarise_owner_trainer_performance(
            owner_trainer_breakdown
        )
        owner_jockey_performance = _summarise_owner_jockey_performance(
            owner_jockey_breakdown
        )
        recent_form_performance = _summarise_recent_form_performance(
            recent_form_breakdown
        )
        value_bet_performance = _summarise_segment_performance(value_bet_breakdown)
        value_bet_flag_performance = _summarise_value_bet_flag_performance(
            value_bet_flag_breakdown
        )
        field_size_performance = _summarise_field_size_performance(field_size_breakdown)
        draw_performance = _summarise_draw_performance(draw_breakdown)
        draw_parity_performance = _summarise_draw_parity_performance(
            draw_parity_breakdown
        )
        race_category_performance = _summarise_race_profile_performance(
            race_category_breakdown
        )
        race_class_performance = _summarise_race_profile_performance(
            race_class_breakdown
        )
        start_delay_performance = _summarise_start_delay_performance(start_delay_breakdown)
        start_type_performance = _summarise_start_type_performance(start_type_breakdown)
        rest_period_performance = _summarise_rest_period_performance(
            rest_period_breakdown
        )
        model_version_performance = _summarise_model_version_performance(
            model_version_breakdown,
            len(y_true),
        )
        prediction_rank_performance = _summarise_prediction_rank_performance(
            prediction_rank_breakdown
        )
        final_position_performance = _summarise_final_position_performance(
            final_position_breakdown
        )
        jockey_performance = _summarise_actor_performance(jockey_breakdown)
        trainer_performance = _summarise_actor_performance(trainer_breakdown)
        jockey_trainer_performance = _summarise_jockey_trainer_performance(
            jockey_trainer_breakdown
        )
        jockey_experience_performance = _summarise_experience_performance(
            jockey_experience_breakdown
        )
        trainer_experience_performance = _summarise_experience_performance(
            trainer_experience_breakdown
        )
        jockey_nationality_performance = _summarise_nationality_performance(
            jockey_nationality_breakdown
        )
        trainer_nationality_performance = _summarise_nationality_performance(
            trainer_nationality_breakdown
        )
        hippodrome_performance = _summarise_hippodrome_performance(hippodrome_breakdown)
        track_type_performance = _summarise_track_type_performance(track_type_breakdown)
        track_length_performance = _summarise_track_length_performance(
            track_length_breakdown
        )
        country_performance = _summarise_country_performance(country_breakdown)
        city_performance = _summarise_city_performance(city_breakdown)
        api_source_performance = _summarise_api_source_performance(api_source_breakdown)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "log_loss": logloss,
            "brier_score": brier_score,
            "brier_decomposition": brier_decomposition,
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1],
            },
            "matthews_correlation": matthews_correlation,
            "specificity": classification_insights["specificity"],
            "false_positive_rate": classification_insights["false_positive_rate"],
            "negative_predictive_value": classification_insights[
                "negative_predictive_value"
            ],
            "balanced_accuracy": classification_insights["balanced_accuracy"],
            "positive_prediction_rate": positives / len(y_pred) if y_pred else 0.0,
            "average_positive_probability": avg_positive_prob,
            "average_negative_probability": avg_negative_prob,
            "probability_distribution_metrics": probability_distribution_metrics,
            "top1_accuracy": top1_correct / course_count if course_count else None,
            "course_top3_hit_rate": top3_course_hits / course_count if course_count else None,
            "ndcg_at_3": _safe_average(ndcg_at_3_scores),
            "ndcg_at_5": _safe_average(ndcg_at_5_scores),
            "winner_rank_metrics": winner_rank_metrics,
            "topn_performance": topn_performance,
            "rank_error_metrics": rank_error_metrics,
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
            "win_probability_performance": win_probability_performance,
            "place_probability_performance": place_probability_performance,
            "probability_edge_performance": probability_edge_performance,
            "probability_error_performance": probability_error_performance,
            "probability_margin_performance": probability_margin_performance,
            "favourite_alignment_performance": favourite_alignment_performance,
            "rank_correlation_performance": rank_correlation_performance,
            "rank_error_metrics": rank_error_metrics,
            "prediction_outcome_performance": prediction_outcome_performance,
            "daily_performance": daily_performance,
            "day_part_performance": day_part_performance,
            "lead_time_performance": lead_time_performance,
            "year_performance": year_performance,
            "month_performance": month_performance,
            "season_performance": season_performance,
            "quarter_performance": quarter_performance,
            "weekday_performance": weekday_performance,
            "race_order_performance": race_order_performance,
            "reunion_number_performance": reunion_number_performance,
            "discipline_performance": discipline_performance,
            "distance_performance": distance_performance,
            "surface_performance": surface_performance,
            "discipline_surface_performance": discipline_surface_performance,
            "weather_performance": weather_performance,
            "temperature_band_performance": temperature_band_performance,
            "prize_money_performance": prize_money_performance,
            "prize_per_runner_performance": prize_per_runner_performance,
            "handicap_performance": handicap_performance,
            "weight_performance": weight_performance,
            "equipment_performance": equipment_performance,
            "odds_band_performance": odds_band_performance,
            "horse_age_performance": horse_age_performance,
            "horse_gender_performance": horse_gender_performance,
            "horse_coat_performance": horse_coat_performance,
            "horse_breed_performance": horse_breed_performance,
            "horse_sire_performance": horse_sire_performance,
            "horse_dam_performance": horse_dam_performance,
            "owner_performance": owner_performance,
            "owner_trainer_performance": owner_trainer_performance,
            "owner_jockey_performance": owner_jockey_performance,
            "recent_form_performance": recent_form_performance,
            "race_category_performance": race_category_performance,
            "race_class_performance": race_class_performance,
            "value_bet_performance": value_bet_performance,
            "value_bet_flag_performance": value_bet_flag_performance,
            "field_size_performance": field_size_performance,
            "draw_performance": draw_performance,
            "draw_parity_performance": draw_parity_performance,
            "start_delay_performance": start_delay_performance,
            "start_type_performance": start_type_performance,
            "rest_period_performance": rest_period_performance,
            "model_version_performance": model_version_performance,
            "prediction_rank_performance": prediction_rank_performance,
            "final_position_performance": final_position_performance,
            "jockey_performance": jockey_performance,
            "trainer_performance": trainer_performance,
            "jockey_trainer_performance": jockey_trainer_performance,
            "jockey_experience_performance": jockey_experience_performance,
            "trainer_experience_performance": trainer_experience_performance,
            "jockey_nationality_performance": jockey_nationality_performance,
            "trainer_nationality_performance": trainer_nationality_performance,
            "hippodrome_performance": hippodrome_performance,
            "track_type_performance": track_type_performance,
            "track_length_performance": track_length_performance,
            "country_performance": country_performance,
            "city_performance": city_performance,
            "api_source_performance": api_source_performance,
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
            "winner_rank_metrics": winner_rank_metrics,
            "topn_performance": topn_performance,
            "confidence_level_metrics": confidence_level_metrics,
            "confidence_score_performance": confidence_score_performance,
            "probability_distribution_metrics": probability_distribution_metrics,
            "win_probability_performance": win_probability_performance,
            "place_probability_performance": place_probability_performance,
            "probability_edge_performance": probability_edge_performance,
            "probability_error_performance": probability_error_performance,
            "probability_margin_performance": probability_margin_performance,
            "favourite_alignment_performance": favourite_alignment_performance,
            "rank_correlation_performance": rank_correlation_performance,
            "rank_error_metrics": rank_error_metrics,
            "prediction_outcome_performance": prediction_outcome_performance,
            "calibration_diagnostics": calibration_diagnostics,
            "threshold_recommendations": threshold_recommendations,
            "betting_value_analysis": betting_value_analysis,
            "odds_alignment": odds_alignment,
            "lift_analysis": lift_analysis,
            "precision_recall_curve": precision_recall_table,
            "roc_curve": roc_curve_points,
            "daily_performance": daily_performance,
            "day_part_performance": day_part_performance,
            "lead_time_performance": lead_time_performance,
            "year_performance": year_performance,
            "month_performance": month_performance,
            "season_performance": season_performance,
            "quarter_performance": quarter_performance,
            "weekday_performance": weekday_performance,
            "race_order_performance": race_order_performance,
            "reunion_number_performance": reunion_number_performance,
            "discipline_performance": discipline_performance,
            "distance_performance": distance_performance,
            "surface_performance": surface_performance,
            "discipline_surface_performance": discipline_surface_performance,
            "weather_performance": weather_performance,
            "temperature_band_performance": temperature_band_performance,
            "prize_money_performance": prize_money_performance,
            "prize_per_runner_performance": prize_per_runner_performance,
            "handicap_performance": handicap_performance,
            "weight_performance": weight_performance,
            "equipment_performance": equipment_performance,
            "horse_age_performance": horse_age_performance,
            "horse_gender_performance": horse_gender_performance,
            "horse_coat_performance": horse_coat_performance,
            "horse_breed_performance": horse_breed_performance,
            "horse_sire_performance": horse_sire_performance,
            "horse_dam_performance": horse_dam_performance,
            "owner_performance": owner_performance,
            "owner_trainer_performance": owner_trainer_performance,
            "owner_jockey_performance": owner_jockey_performance,
            "recent_form_performance": recent_form_performance,
            "race_category_performance": race_category_performance,
            "race_class_performance": race_class_performance,
            "value_bet_performance": value_bet_performance,
            "value_bet_flag_performance": value_bet_flag_performance,
            "field_size_performance": field_size_performance,
            "draw_performance": draw_performance,
            "draw_parity_performance": draw_parity_performance,
            "start_delay_performance": start_delay_performance,
            "start_type_performance": start_type_performance,
            "rest_period_performance": rest_period_performance,
            "model_version_performance": model_version_performance,
            "prediction_rank_performance": prediction_rank_performance,
            "final_position_performance": final_position_performance,
            "jockey_performance": jockey_performance,
            "trainer_performance": trainer_performance,
            "jockey_trainer_performance": jockey_trainer_performance,
            "jockey_experience_performance": jockey_experience_performance,
            "trainer_experience_performance": trainer_experience_performance,
            "jockey_nationality_performance": jockey_nationality_performance,
            "trainer_nationality_performance": trainer_nationality_performance,
            "hippodrome_performance": hippodrome_performance,
            "track_type_performance": track_type_performance,
            "track_length_performance": track_length_performance,
            "country_performance": country_performance,
            "city_performance": city_performance,
            "api_source_performance": api_source_performance,
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
            "probability_distribution_metrics": probability_distribution_metrics,
            "topn_performance": topn_performance,
            "win_probability_performance": win_probability_performance,
            "place_probability_performance": place_probability_performance,
            "probability_edge_performance": probability_edge_performance,
            "probability_error_performance": probability_error_performance,
            "probability_margin_performance": probability_margin_performance,
            "favourite_alignment_performance": favourite_alignment_performance,
            "rank_correlation_performance": rank_correlation_performance,
            "rank_error_metrics": rank_error_metrics,
            "prediction_outcome_performance": prediction_outcome_performance,
            "calibration_diagnostics": calibration_diagnostics,
            "threshold_recommendations": threshold_recommendations,
            "betting_value_analysis": betting_value_analysis,
            "odds_alignment": odds_alignment,
            "lift_analysis": lift_analysis,
            "daily_performance": daily_performance,
            "day_part_performance": day_part_performance,
            "lead_time_performance": lead_time_performance,
            "year_performance": year_performance,
            "month_performance": month_performance,
            "season_performance": season_performance,
            "quarter_performance": quarter_performance,
            "weekday_performance": weekday_performance,
            "race_order_performance": race_order_performance,
            "reunion_number_performance": reunion_number_performance,
            "distance_performance": distance_performance,
            "discipline_surface_performance": discipline_surface_performance,
            "draw_performance": draw_performance,
            "draw_parity_performance": draw_parity_performance,
            "start_delay_performance": start_delay_performance,
            "weather_performance": weather_performance,
            "temperature_band_performance": temperature_band_performance,
            "prize_money_performance": prize_money_performance,
            "prize_per_runner_performance": prize_per_runner_performance,
            "handicap_performance": handicap_performance,
            "weight_performance": weight_performance,
            "equipment_performance": equipment_performance,
            "horse_age_performance": horse_age_performance,
            "horse_gender_performance": horse_gender_performance,
            "horse_coat_performance": horse_coat_performance,
            "horse_breed_performance": horse_breed_performance,
            "horse_sire_performance": horse_sire_performance,
            "horse_dam_performance": horse_dam_performance,
            "owner_performance": owner_performance,
            "owner_trainer_performance": owner_trainer_performance,
            "owner_jockey_performance": owner_jockey_performance,
            "recent_form_performance": recent_form_performance,
            "track_type_performance": track_type_performance,
            "track_length_performance": track_length_performance,
            "city_performance": city_performance,
            "odds_band_performance": odds_band_performance,
            "value_bet_flag_performance": value_bet_flag_performance,
            "model_version_breakdown": dict(model_versions),
            "model_version_performance": model_version_performance,
            "rest_period_performance": rest_period_performance,
            "jockey_performance": jockey_performance,
            "trainer_performance": trainer_performance,
            "jockey_trainer_performance": jockey_trainer_performance,
            "jockey_experience_performance": jockey_experience_performance,
            "trainer_experience_performance": trainer_experience_performance,
            "jockey_nationality_performance": jockey_nationality_performance,
            "trainer_nationality_performance": trainer_nationality_performance,
            "hippodrome_performance": hippodrome_performance,
            "country_performance": country_performance,
            "prediction_rank_performance": prediction_rank_performance,
            "final_position_performance": final_position_performance,
            "api_source_performance": api_source_performance,
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
