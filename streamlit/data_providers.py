"""Utility helpers to provide analytical datasets for the Streamlit dashboard.

The real PronoTurf backend will eventually expose dedicated API endpoints for
historical bets, predictions and bankroll tracking.  The dashboard is already
structured to consume tabular datasets and transform them into interactive
visualisations.  While the APIs are being finalised we offer a deterministic
set of synthetic records so that the dashboard can demonstrate the full
experience without relying on a running backend instance.

The module exposes cached functions returning ``pandas.DataFrame`` objects.
Each function is heavily documented so that it can later be swapped with real
HTTP calls (or SQL queries) without changing the rest of the UI code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Dataclass helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticCourse:
    """Represents the meta-data associated with a simulated race.

    Attributes:
        course_id:  Stable identifier used to build relational joins between
            bets and predictions.
        hippodrome: Name of the race track.
        discipline: Discipline of the race (plat, trot, obstacles, mixte).
        track_type: Synthetic classification used for aggregated analytics.
        weather:  High level weather category observed during the race.
        temperature:  Average temperature in Celsius.
    """

    course_id: int
    hippodrome: str
    discipline: str
    track_type: str
    weather: str
    temperature: float


@dataclass(frozen=True)
class MonitoringDescriptor:
    """Describe the configuration of a synthetic monitoring metric."""

    metric: str
    direction: str
    target: float
    base_value: float
    comment: str


@dataclass(frozen=True)
class FeatureDescriptor:
    """Capture information required to build feature contribution rows."""

    feature: str
    importance: float
    impact: float
    category: str
    description: str


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def _build_synthetic_courses() -> Dict[int, SyntheticCourse]:
    """Generate a deterministic mapping of course identifiers to meta-data."""

    base_courses: Tuple[SyntheticCourse, ...] = (
        SyntheticCourse(101, "ParisLongchamp", "Plat", "Pelouse", "Ensoleillé", 22.0),
        SyntheticCourse(102, "Vincennes", "Trot", "Piste", "Couvert", 18.0),
        SyntheticCourse(103, "Chantilly", "Plat", "Fibre", "Pluie fine", 16.5),
        SyntheticCourse(104, "Enghien", "Trot", "Piste", "Ensoleillé", 24.0),
        SyntheticCourse(105, "Auteuil", "Obstacles", "Herbe", "Nuageux", 14.0),
        SyntheticCourse(106, "Deauville", "Plat", "Fibre", "Ensoleillé", 21.0),
        SyntheticCourse(107, "Caen", "Trot", "Piste", "Pluie légère", 13.0),
        SyntheticCourse(108, "Cagnes-sur-Mer", "Plat", "Fibre", "Ensoleillé", 19.5),
    )

    return {course.course_id: course for course in base_courses}


def _iterate_bet_rows() -> Iterable[Dict[str, object]]:
    """Yield a reproducible set of bet records for the dashboard.

    The values follow a deterministic pattern that mimics realistic bankroll
    fluctuations:  a mix of wins/losses, multiple betting strategies and a
    healthy number of value bets.  The pattern is intentionally simple so that
    developers can reason about the outputs while still stressing the
    visualisations (60+ rows covering two months of activity).
    """

    courses = _build_synthetic_courses()
    start_day = date.today() - timedelta(days=59)

    strategy_cycle = ("Kelly", "Flat", "Value", "Hedging")
    stake_cycle = (12.0, 15.0, 18.0, 20.0)
    odds_cycle = (3.2, 2.4, 4.6, 5.1, 6.5)

    for idx in range(60):
        course_id = 101 + (idx % len(courses))
        course_meta = courses[course_id]
        event_date = start_day + timedelta(days=idx)

        strategy = strategy_cycle[idx % len(strategy_cycle)]
        stake = stake_cycle[idx % len(stake_cycle)]
        implied_probability = 1 / odds_cycle[idx % len(odds_cycle)]

        # We deterministically decide whether the bet was a win.  The pattern
        # cycles so that each strategy experiences wins and losses.
        win_flag = (idx % 5) in (0, 1, 4)
        odds = odds_cycle[idx % len(odds_cycle)]
        profit = (odds - 1) * stake if win_flag else -stake

        # Confidence score is influenced by the implied probability and the
        # index to create a visible spread on scatter plots.
        confidence = round(100 * (0.45 + (idx % 7) * 0.05 - implied_probability / 2), 1)

        yield {
            "date": event_date,
            "course_id": course_id,
            "hippodrome": course_meta.hippodrome,
            "discipline": course_meta.discipline,
            "track_type": course_meta.track_type,
            "weather": course_meta.weather,
            "temperature": course_meta.temperature,
            "strategy": strategy,
            "stake": stake,
            "odds": round(odds, 2),
            "implied_probability": round(implied_probability, 4),
            "confidence_score": max(min(confidence, 99.0), 20.0),
            "result": "won" if win_flag else "lost",
            "profit": round(profit, 2),
            "kelly_fraction": round(0.05 + (idx % 4) * 0.03, 2),
            "is_value_bet": odds * implied_probability < 0.85,
            "predicted_rank": 1 + (idx % 5),
            "actual_rank": 1 + ((idx + (2 if win_flag else 4)) % 8),
            "course_label": f"R{(idx % 6) + 1}C{(idx % 10) + 1}",
        }


def _iterate_prediction_rows() -> Iterable[Dict[str, object]]:
    """Yield the top-3 predictions for each synthetic course.

    The predictions mirror the betting history but also add additional races
    where no bet was placed so that the dashboard can display richer context.
    """

    courses = _build_synthetic_courses()
    start_day = date.today() - timedelta(days=29)

    for idx in range(30):
        course_id = 101 + (idx % len(courses))
        course_meta = courses[course_id]
        event_date = start_day + timedelta(days=idx)

        base_probability = 0.32 + (idx % 3) * 0.08

        for rank in range(1, 4):
            probability = max(min(base_probability - (rank - 1) * 0.06, 0.85), 0.12)
            yield {
                "date": event_date,
                "course_id": course_id,
                "hippodrome": course_meta.hippodrome,
                "discipline": course_meta.discipline,
                "track_type": course_meta.track_type,
                "horse_name": f"Horse {course_id}-{rank}",
                "predicted_rank": rank,
                "win_probability": round(probability, 3),
                "value_score": round(probability * (1.8 + rank * 0.3), 3),
                "course_label": f"R{(idx % 6) + 1}C{(idx % 10) + 1}",
            }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_bet_history() -> pd.DataFrame:
    """Return the synthetic bet history as a DataFrame.

    The index is not set on purpose so that Streamlit can apply its own
    default indexing when displaying the table.  Consumers may freely sort or
    filter the data using pandas operations.
    """

    records = list(_iterate_bet_rows())
    frame = pd.DataFrame.from_records(records)
    frame["date"] = pd.to_datetime(frame["date"])
    frame.sort_values("date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def load_predictions() -> pd.DataFrame:
    """Return the synthetic prediction leaderboard for recent races."""

    records = list(_iterate_prediction_rows())
    frame = pd.DataFrame.from_records(records)
    frame["date"] = pd.to_datetime(frame["date"])
    frame.sort_values(["date", "course_id", "predicted_rank"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def list_available_filters() -> Dict[str, List[str]]:
    """Expose the available categorical filters for the dashboard UI."""

    courses = _build_synthetic_courses()
    hippodromes = sorted({course.hippodrome for course in courses.values()})
    disciplines = sorted({course.discipline for course in courses.values()})
    track_types = sorted({course.track_type for course in courses.values()})

    return {
        "hippodromes": hippodromes,
        "disciplines": disciplines,
        "track_types": track_types,
    }


def _iterate_monitoring_rows() -> Iterable[Dict[str, object]]:
    """Yield synthetic monitoring snapshots for the quality dashboard."""

    base_date = date.today() - timedelta(days=42)
    sprints = ("Sprint 6", "Sprint 7", "Sprint 8")

    descriptors: Tuple[MonitoringDescriptor, ...] = (
        MonitoringDescriptor(
            metric="Brier score",
            direction="lower",
            target=0.23,
            base_value=0.245,
            comment="Calibration des probabilités",
        ),
        MonitoringDescriptor(
            metric="Log loss",
            direction="lower",
            target=0.68,
            base_value=0.73,
            comment="Qualité probabiliste globale",
        ),
        MonitoringDescriptor(
            metric="Précision top 1",
            direction="higher",
            target=0.38,
            base_value=0.34,
            comment="Taux de gagnant principal",
        ),
        MonitoringDescriptor(
            metric="Rappel top 3",
            direction="higher",
            target=0.72,
            base_value=0.66,
            comment="Couverture podium",
        ),
        MonitoringDescriptor(
            metric="Cohen kappa",
            direction="higher",
            target=0.28,
            base_value=0.22,
            comment="Accord modèle vs réalité",
        ),
    )

    for metric_index, descriptor in enumerate(descriptors):
        for sprint_index, sprint in enumerate(sprints):
            snapshot_date = base_date + timedelta(days=sprint_index * 14 + metric_index * 2)

            if descriptor.direction == "lower":
                value = descriptor.base_value - 0.01 * sprint_index
                status = "Conforme" if value <= descriptor.target else "À surveiller"
            else:
                value = descriptor.base_value + 0.02 * sprint_index
                status = "Conforme" if value >= descriptor.target else "À renforcer"

            yield {
                "metric": descriptor.metric,
                "direction": descriptor.direction,
                "target": round(descriptor.target, 3),
                "value": round(value, 3),
                "status": status,
                "sprint": sprint,
                "comment": descriptor.comment,
                "snapshot_date": snapshot_date,
            }


def _build_feature_rows() -> List[Dict[str, object]]:
    """Return a deterministic list of feature contribution insights."""

    descriptors: Tuple[FeatureDescriptor, ...] = (
        FeatureDescriptor(
            "rating_model",
            0.19,
            0.28,
            "Performance cheval",
            "Score interne combinant historiques et forme",
        ),
        FeatureDescriptor(
            "recent_form_index",
            0.16,
            0.24,
            "Performance cheval",
            "Indice de régularité sur 5 courses",
        ),
        FeatureDescriptor(
            "trainer_win_rate",
            0.14,
            0.18,
            "Entraîneur",
            "Taux de victoire entraîneur 12 mois",
        ),
        FeatureDescriptor(
            "jockey_win_rate",
            0.12,
            0.17,
            "Jockey",
            "Taux de victoire jockey 6 mois",
        ),
        FeatureDescriptor(
            "odds_public",
            0.10,
            -0.08,
            "Marché",
            "Cote publique normalisée",
        ),
        FeatureDescriptor(
            "draw_number",
            0.08,
            0.05,
            "Course",
            "Numéro de corde normalisé",
        ),
        FeatureDescriptor(
            "rest_days",
            0.07,
            -0.04,
            "Course",
            "Jours depuis la dernière course",
        ),
        FeatureDescriptor(
            "track_condition",
            0.06,
            0.09,
            "Course",
            "Indice synthétique d'état de piste",
        ),
        FeatureDescriptor(
            "pace_projection",
            0.05,
            0.07,
            "Course",
            "Projection d'allure (rapide, moyen, lent)",
        ),
        FeatureDescriptor(
            "value_edge",
            0.03,
            0.05,
            "Marché",
            "Différence probabilité modèle vs marché",
        ),
    )

    return [
        {
            "feature": descriptor.feature,
            "importance": round(descriptor.importance, 3),
            "avg_shap": round(descriptor.impact, 3),
            "category": descriptor.category,
            "description": descriptor.description,
        }
        for descriptor in descriptors
    ]


def load_monitoring_snapshots() -> pd.DataFrame:
    """Expose the monitoring snapshots used by the Streamlit monitoring tab."""

    records = list(_iterate_monitoring_rows())
    frame = pd.DataFrame.from_records(records)
    frame.sort_values(["metric", "snapshot_date"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def load_feature_contributions() -> pd.DataFrame:
    """Return feature contribution metrics mirroring SHAP aggregations."""

    frame = pd.DataFrame.from_records(_build_feature_rows())
    frame.sort_values("importance", ascending=False, inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame

