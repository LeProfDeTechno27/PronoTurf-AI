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


@dataclass(frozen=True)
class DataQualityDescriptor:
    """Describe a synthetic data-quality alert surfaced by monitoring."""

    check: str
    severity: str
    status: str
    impacted_rows: int
    recommendation: str
    days_since_seen: int


@dataclass(frozen=True)
class DriftDescriptor:
    """Represent a monitored feature in the drift surveillance dataset."""

    feature: str
    drift_score: float
    p_value: float
    status: str
    reference_mean: float
    current_mean: float
    comment: str


@dataclass(frozen=True)
class OperationalMilestone:
    """Capture the lifecycle of an operational milestone for the roadmap."""

    workstream: str
    milestone: str
    owner: str
    status: str
    start_date: date
    due_date: date
    confidence: int
    impact: str
    comment: str


@dataclass(frozen=True)
class OperationalRisk:
    """Describe a potential delivery risk surfaced during the sprint."""

    risk: str
    severity: str
    owner: str
    status: str
    mitigation: str
    next_review: date
    trend: str


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


def _build_operational_milestones() -> Tuple[OperationalMilestone, ...]:
    """Return a deterministic set of operational milestones for the roadmap."""

    today = date.today()

    return (
        OperationalMilestone(
            workstream="Monitoring",  # Completion of monitoring dashboards
            milestone="Stabilisation pipeline PSI",
            owner="L. Garnier",
            status="Livré",
            start_date=today - timedelta(days=21),
            due_date=today - timedelta(days=2),
            confidence=98,
            impact="Qualité modèle",
            comment="Pipeline PSI validé en préproduction et basculé en run.",
        ),
        OperationalMilestone(
            workstream="Data Quality",
            milestone="Runbook alerting data",
            owner="S. Dupont",
            status="En cours",
            start_date=today - timedelta(days=5),
            due_date=today + timedelta(days=4),
            confidence=85,
            impact="Fiabilisation",
            comment="Documentation et escalade PagerDuty en cours de relecture.",
        ),
        OperationalMilestone(
            workstream="MLOps",
            milestone="Automatisation retrain mensuel",
            owner="H. Martin",
            status="À risque",
            start_date=today - timedelta(days=3),
            due_date=today + timedelta(days=9),
            confidence=70,
            impact="Scalabilité",
            comment="Validation sécurité des credentials GitOps encore bloquante.",
        ),
        OperationalMilestone(
            workstream="Produit",
            milestone="Go-live dashboard V1",
            owner="P. Leroy",
            status="Livré",
            start_date=today - timedelta(days=10),
            due_date=today,
            confidence=100,
            impact="Stakeholders",
            comment="Conduite du changement finalisée, ateliers utilisateurs faits.",
        ),
    )


def _build_operational_risks() -> Tuple[OperationalRisk, ...]:
    """Create a small register of operational risks to surface in the UI."""

    today = date.today()

    return (
        OperationalRisk(
            risk="Latence API pronostics",
            severity="Critique",
            owner="SRE Team",
            status="Mitigation en cours",
            mitigation="Canary deploy du cache Redis et ajout observabilité APM.",
            next_review=today + timedelta(days=2),
            trend="En amélioration",
        ),
        OperationalRisk(
            risk="Rafraîchissement données courses étrangères",
            severity="Élevée",
            owner="DataOps",
            status="Surveillance",
            mitigation="Scripts d'ingestion externalisés vers Airflow nocturne.",
            next_review=today + timedelta(days=5),
            trend="Stable",
        ),
        OperationalRisk(
            risk="Rotation secrets d'entraînement",
            severity="Modérée",
            owner="Security",
            status="Planifié",
            mitigation="Procédure Vault automatisée à valider par le RSSI.",
            next_review=today + timedelta(days=8),
            trend="À suivre",
        ),
    )


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


def _build_data_quality_rows() -> List[Dict[str, object]]:
    """Construct deterministic data quality alerts for the monitoring UI."""

    descriptors: Tuple[DataQualityDescriptor, ...] = (
        DataQualityDescriptor(
            check="Données courses incomplètes",
            severity="Critique",
            status="Investigation en cours",
            impacted_rows=8,
            recommendation="Relancer l'ingestion PMU et combler les écarts manuels",
            days_since_seen=1,
        ),
        DataQualityDescriptor(
            check="Temps intermédiaires manquants",
            severity="Majeure",
            status="Correctif planifié",
            impacted_rows=24,
            recommendation="Aligner le mapping API pour capturer les temps partiels",
            days_since_seen=3,
        ),
        DataQualityDescriptor(
            check="Chevaux dupliqués",
            severity="Majeure",
            status="Résolu",
            impacted_rows=12,
            recommendation="Nettoyage terminé, suivi renforcé sur 7 jours",
            days_since_seen=6,
        ),
        DataQualityDescriptor(
            check="Cotes manquantes",
            severity="Mineure",
            status="Monitoring",
            impacted_rows=15,
            recommendation="Valeurs imputées via médiane marché",
            days_since_seen=9,
        ),
    )

    today = date.today()
    records: List[Dict[str, object]] = []

    for descriptor in descriptors:
        records.append(
            {
                "check": descriptor.check,
                "severity": descriptor.severity,
                "status": descriptor.status,
                "impacted_rows": descriptor.impacted_rows,
                "recommendation": descriptor.recommendation,
                "last_seen": today - timedelta(days=descriptor.days_since_seen),
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame["last_seen"] = pd.to_datetime(frame["last_seen"])

    severity_order = {"Critique": 0, "Majeure": 1, "Mineure": 2}
    frame["_severity_order"] = frame["severity"].map(severity_order)
    frame.sort_values(["_severity_order", "last_seen"], inplace=True)
    frame.drop(columns="_severity_order", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def load_data_quality_checks() -> pd.DataFrame:
    """Expose the synthetic data-quality alerts consumed by the dashboard."""

    return _build_data_quality_rows()


def _build_data_drift_rows() -> List[Dict[str, object]]:
    """Create deterministic drift diagnostics for monitored features."""

    descriptors: Tuple[DriftDescriptor, ...] = (
        DriftDescriptor(
            feature="win_probability",
            drift_score=0.28,
            p_value=0.03,
            status="À investiguer",
            reference_mean=0.31,
            current_mean=0.27,
            comment="Probabilités plus conservatrices sur les 2 dernières semaines",
        ),
        DriftDescriptor(
            feature="value_edge",
            drift_score=0.18,
            p_value=0.07,
            status="À surveiller",
            reference_mean=0.05,
            current_mean=0.03,
            comment="Écart modèle/marché réduit sur les réunions nocturnes",
        ),
        DriftDescriptor(
            feature="pace_projection",
            drift_score=0.11,
            p_value=0.21,
            status="Stable",
            reference_mean=1.48,
            current_mean=1.51,
            comment="Distribution similaire entre les catégories de rythme",
        ),
        DriftDescriptor(
            feature="track_condition",
            drift_score=0.09,
            p_value=0.26,
            status="Stable",
            reference_mean=0.64,
            current_mean=0.66,
            comment="Mix météo conforme à la période de référence",
        ),
    )

    window_end = date.today()
    window_start = window_end - timedelta(days=14)

    records: List[Dict[str, object]] = []
    for descriptor in descriptors:
        records.append(
            {
                "feature": descriptor.feature,
                "drift_score": round(descriptor.drift_score, 2),
                "p_value": round(descriptor.p_value, 3),
                "status": descriptor.status,
                "reference_mean": round(descriptor.reference_mean, 3),
                "current_mean": round(descriptor.current_mean, 3),
                "comment": descriptor.comment,
                "window_start": window_start,
                "window_end": window_end,
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame["window_start"] = pd.to_datetime(frame["window_start"])
    frame["window_end"] = pd.to_datetime(frame["window_end"])
    frame.sort_values("drift_score", ascending=False, inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def load_data_drift_metrics() -> pd.DataFrame:
    """Expose deterministic data drift diagnostics to Streamlit."""

    return _build_data_drift_rows()



def load_operational_milestones() -> pd.DataFrame:
    """Expose operational milestones to drive the Pilotage tab."""

    rows = [
        {
            "workstream": milestone.workstream,
            "milestone": milestone.milestone,
            "owner": milestone.owner,
            "status": milestone.status,
            "start_date": pd.Timestamp(milestone.start_date),
            "due_date": pd.Timestamp(milestone.due_date),
            "confidence": milestone.confidence,
            "impact": milestone.impact,
            "comment": milestone.comment,
        }
        for milestone in _build_operational_milestones()
    ]

    frame = pd.DataFrame.from_records(rows)
    frame.sort_values("due_date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def load_operational_risks() -> pd.DataFrame:
    """Expose the operational risk register for the dashboard."""

    rows = [
        {
            "risk": risk.risk,
            "severity": risk.severity,
            "owner": risk.owner,
            "status": risk.status,
            "mitigation": risk.mitigation,
            "next_review": pd.Timestamp(risk.next_review),
            "trend": risk.trend,
        }
        for risk in _build_operational_risks()
    ]

    frame = pd.DataFrame.from_records(rows)
    severity_order = {"Critique": 0, "Élevée": 1, "Modérée": 2}
    frame["_severity_order"] = frame["severity"].map(severity_order)
    frame.sort_values(["_severity_order", "next_review"], inplace=True)
    frame.drop(columns="_severity_order", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame
