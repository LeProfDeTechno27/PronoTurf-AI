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

