from __future__ import annotations

import importlib
import os
import unicodedata
from calendar import monthrange
from datetime import date, timedelta
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("CORS_ORIGINS", "[\"http://localhost:3000\"]")

analytics_module = importlib.import_module("app.api.endpoints.analytics")


test_app = FastAPI()
test_app.include_router(analytics_module.router, prefix="/api/v1/analytics")

pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


class StubAspiturfClient:
    """Client Aspiturf minimal pour les tests des endpoints analytics."""

    def __init__(
        self,
        rows: Iterable[Dict[str, Any]],
        partants: Dict[Tuple[date, str, int], List[Dict[str, Any]]],
    ) -> None:
        self._rows = [dict(row) for row in rows]
        self._partants = {
            (course_date, hippo.upper(), course_number): [dict(item) for item in items]
            for (course_date, hippo, course_number), items in partants.items()
        }

    async def __aenter__(self) -> "StubAspiturfClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None

    async def get_races(
        self,
        *,
        horse_id: Optional[str] = None,
        jockey_id: Optional[str] = None,
        trainer_id: Optional[str] = None,
        hippodrome: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        results = list(self._rows)

        if horse_id is not None:
            results = [row for row in results if row.get("idChe") == horse_id]

        if jockey_id is not None:
            results = [row for row in results if row.get("idJockey") == jockey_id]

        if trainer_id is not None:
            results = [row for row in results if row.get("idEntraineur") == trainer_id]

        if hippodrome is not None:
            hippo_upper = hippodrome.upper()
            results = [
                row for row in results if row.get("hippo") and row["hippo"].upper() == hippo_upper
            ]

        if start_date is not None or end_date is not None:
            dated: List[Dict[str, Any]] = []
            for row in results:
                race_date = row.get("jour")
                if not isinstance(race_date, date):
                    continue
                if start_date is not None and race_date < start_date:
                    continue
                if end_date is not None and race_date > end_date:
                    continue
                dated.append(row)
            results = dated

        return [dict(row) for row in results]

    async def get_partants_course(
        self,
        course_date: date,
        hippodrome: str,
        course_number: int,
    ) -> List[Dict[str, Any]]:
        key = (course_date, hippodrome.upper(), course_number)
        return [dict(item) for item in self._partants.get(key, [])]

    async def leaderboard(
        self,
        entity_type: str,
        *,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        hippodrome: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []

        for row in self._rows:
            race_date = row.get("jour")

            if isinstance(race_date, date):
                if start_date and race_date < start_date:
                    continue
                if end_date and race_date > end_date:
                    continue
            elif start_date or end_date:
                continue

            if hippodrome is not None:
                hippo_value = row.get("hippo")
                if not hippo_value or hippo_value.upper() != hippodrome.upper():
                    continue

            filtered.append(dict(row))

        key_field = {
            "horse": "idChe",
            "jockey": "idJockey",
            "trainer": "idEntraineur",
        }.get(entity_type)

        label_field = {
            "horse": "nom_cheval",
            "jockey": "jockey",
            "trainer": "entraineur",
        }.get(entity_type)

        if not key_field or not label_field:
            return []

        aggregations: Dict[str, Dict[str, Any]] = {}

        for row in filtered:
            identifier = row.get(key_field)
            if not identifier:
                continue

            label = row.get(label_field) or str(identifier)
            entry = aggregations.setdefault(
                str(identifier),
                {
                    "entity_id": str(identifier),
                    "label": label,
                    "sample_size": 0,
                    "wins": 0,
                    "podiums": 0,
                    "positions": [],
                    "last_seen": None,
                },
            )

            entry["label"] = label
            entry["sample_size"] += 1

            position = row.get("cl")
            if isinstance(position, int):
                entry["positions"].append(position)
                if position == 1:
                    entry["wins"] += 1
                if 1 <= position <= 3:
                    entry["podiums"] += 1

            race_date = row.get("jour")
            if isinstance(race_date, date):
                last_seen = entry.get("last_seen")
                if last_seen is None or race_date > last_seen:
                    entry["last_seen"] = race_date

        leaderboard: List[Dict[str, Any]] = []
        for entry in aggregations.values():
            sample_size = entry["sample_size"]
            if not sample_size:
                continue

            win_rate = entry["wins"] / sample_size if sample_size else None
            podium_rate = entry["podiums"] / sample_size if sample_size else None

            positions = [pos for pos in entry["positions"] if isinstance(pos, int)]
            average_finish = sum(positions) / len(positions) if positions else None

            leaderboard.append(
                {
                    "entity_id": entry["entity_id"],
                    "label": entry["label"],
                    "sample_size": sample_size,
                    "wins": entry["wins"],
                    "podiums": entry["podiums"],
                    "win_rate": round(win_rate, 4) if win_rate is not None else None,
                    "podium_rate": round(podium_rate, 4) if podium_rate is not None else None,
                    "average_finish": round(average_finish, 2) if average_finish is not None else None,
                    "last_seen": entry.get("last_seen"),
                }
            )

        leaderboard.sort(
            key=lambda item: (
                -(item["win_rate"] or 0),
                -(item["podium_rate"] or 0),
                -item["sample_size"],
                item["label"],
            )
        )

        return leaderboard[: max(1, limit)]

    async def value_opportunities(
        self,
        *,
        entity_type: str,
        entity_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        hippodrome: Optional[str] = None,
        min_edge: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        filtered = await self.get_races(
            horse_id=entity_id if entity_type == "horse" else None,
            jockey_id=entity_id if entity_type == "jockey" else None,
            trainer_id=entity_id if entity_type == "trainer" else None,
            hippodrome=hippodrome,
            start_date=start_date,
            end_date=end_date,
        )

        def parse(value: Any) -> Optional[float]:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value.replace(",", "."))
                except ValueError:
                    return None
            return None

        samples: List[Dict[str, Any]] = []
        hippodromes: Set[str] = set()
        first_date: Optional[date] = None
        last_date: Optional[date] = None
        entity_label: Optional[str] = None

        for row in filtered:
            actual = parse(row.get("cotedirect") or row.get("cote_direct"))
            implied = parse(row.get("coteprob") or row.get("cote_prob"))

            edge: Optional[float] = None
            if actual is not None and implied is not None:
                edge = round(implied - actual, 4)

            if min_edge is not None and (edge is None or edge < min_edge):
                continue

            race_date = row.get("jour")
            if isinstance(race_date, date):
                first_date = race_date if first_date is None else min(first_date, race_date)
                last_date = race_date if last_date is None else max(last_date, race_date)

            hippo_value = row.get("hippo")
            if isinstance(hippo_value, str):
                hippodromes.add(hippo_value.upper())

            label_field = {
                "horse": "nom_cheval",
                "jockey": "jockey",
                "trainer": "entraineur",
            }.get(entity_type)
            if label_field:
                raw_label = row.get(label_field)
                if isinstance(raw_label, str) and raw_label.strip():
                    entity_label = raw_label.strip()

            position = row.get("cl")
            is_win = position == 1 if isinstance(position, int) else None

            profit: Optional[float] = None
            if actual is not None:
                profit = (actual - 1) if is_win else -1.0

            samples.append(
                {
                    "date": race_date,
                    "hippodrome": hippo_value,
                    "course_number": row.get("prix"),
                    "distance": row.get("dist"),
                    "final_position": position,
                    "odds_actual": actual,
                    "odds_implied": implied,
                    "edge": edge,
                    "is_win": is_win,
                    "profit": profit,
                }
            )

        samples.sort(key=lambda item: item.get("edge") or float("-inf"), reverse=True)

        if limit is not None and limit > 0:
            samples = samples[:limit]

        edges = [item["edge"] for item in samples if item.get("edge") is not None]
        odds_values = [item["odds_actual"] for item in samples if item.get("odds_actual") is not None]
        profits = [item["profit"] for item in samples if item.get("profit") is not None]
        wins = sum(1 for item in samples if item.get("is_win"))
        stake_count = len(profits)

        total_profit = sum(profits) if profits else None
        roi = (total_profit / stake_count) if total_profit is not None and stake_count else None

        summary = {
            "sample_size": len(samples),
            "wins": wins,
            "win_rate": round(wins / len(samples), 4) if samples else None,
            "positive_edges": sum(1 for value in edges if value is not None and value >= 0),
            "negative_edges": sum(1 for value in edges if value is not None and value < 0),
            "average_edge": round(sum(edges) / len(edges), 4) if edges else None,
            "median_edge": round(median(edges), 4) if edges else None,
            "average_odds": round(sum(odds_values) / len(odds_values), 4) if odds_values else None,
            "median_odds": round(median(odds_values), 4) if odds_values else None,
            "stake_count": stake_count,
            "profit": round(total_profit, 4) if total_profit is not None else None,
            "roi": round(roi, 4) if roi is not None else None,
        }

        return {
            "entity_id": entity_id,
            "entity_label": entity_label,
            "date_start": first_date,
            "date_end": last_date,
            "hippodromes": sorted(hippodromes),
            "samples": samples,
            "summary": summary,
        }

    async def performance_calendar(
        self,
        *,
        entity_type: str,
        entity_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        hippodrome: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Construit un agrégat journalier minimal pour les tests."""

        if entity_type == "horse":
            key_field = "idChe"
            label_field = "nom_cheval"
        elif entity_type == "jockey":
            key_field = "idJockey"
            label_field = "jockey"
        else:
            key_field = "idEntraineur"
            label_field = "entraineur"

        hippo_upper = hippodrome.upper() if hippodrome else None
        days: Dict[date, Dict[str, Any]] = {}
        entity_label: Optional[str] = None
        first_date: Optional[date] = None
        last_date: Optional[date] = None

        for row in self._rows:
            if row.get(key_field) != entity_id:
                continue

            race_date = row.get("jour")
            if not isinstance(race_date, date):
                continue

            if start_date and race_date < start_date:
                continue

            if end_date and race_date > end_date:
                continue

            if hippo_upper:
                hippo_value = row.get("hippo")
                if not hippo_value or hippo_value.upper() != hippo_upper:
                    continue

            if entity_label is None:
                label_value = row.get(label_field)
                if isinstance(label_value, str) and label_value.strip():
                    entity_label = label_value.strip()

            bucket = days.setdefault(
                race_date,
                {
                    "hippodromes": set(),
                    "rows": [],
                },
            )
            bucket["rows"].append(row)

            hippo_value = row.get("hippo")
            if isinstance(hippo_value, str):
                bucket["hippodromes"].add(hippo_value.upper())

            if first_date is None or race_date < first_date:
                first_date = race_date

            if last_date is None or race_date > last_date:
                last_date = race_date

        total_races = 0
        total_wins = 0
        total_podiums = 0
        formatted_days: List[Dict[str, Any]] = []

        for race_date, bucket in sorted(days.items(), key=lambda item: item[0]):
            rows = bucket["rows"]
            wins = 0
            podiums = 0
            positions: List[int] = []
            odds: List[float] = []
            race_details: List[Dict[str, Any]] = []

            for row in rows:
                position = row.get("cl")
                if isinstance(position, int):
                    positions.append(position)
                    if position == 1:
                        wins += 1
                    if 1 <= position <= 3:
                        podiums += 1

                raw_odds = row.get("cotedirect") or row.get("coteprob")
                if isinstance(raw_odds, (int, float)):
                    odds.append(float(raw_odds))

                race_details.append(
                    {
                        "hippodrome": row.get("hippo"),
                        "course_number": row.get("prix"),
                        "distance": row.get("dist") if isinstance(row.get("dist"), int) else None,
                        "final_position": position if isinstance(position, int) else None,
                        "odds": float(raw_odds) if isinstance(raw_odds, (int, float)) else None,
                    }
                )

            races = len(rows)
            total_races += races
            total_wins += wins
            total_podiums += podiums

            average_finish = sum(positions) / len(positions) if positions else None
            average_odds = sum(odds) / len(odds) if odds else None

            formatted_days.append(
                {
                    "date": race_date,
                    "hippodromes": sorted(bucket["hippodromes"]),
                    "races": races,
                    "wins": wins,
                    "podiums": podiums,
                    "average_finish": round(average_finish, 2) if average_finish is not None else None,
                    "average_odds": round(average_odds, 2) if average_odds is not None else None,
                    "race_details": race_details,
                }
            )

        return {
            "entity_id": entity_id,
            "entity_label": entity_label,
            "date_start": first_date,
            "date_end": last_date,
            "total_races": total_races,
            "total_wins": total_wins,
            "total_podiums": total_podiums,
            "days": formatted_days,
        }

    async def performance_streaks(
        self,
        *,
        entity_type: str,
        entity_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        hippodrome: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calcule des séries consécutives minimalistes pour les tests API."""

        key_field = {
            "horse": "idChe",
            "jockey": "idJockey",
            "trainer": "idEntraineur",
        }.get(entity_type)

        label_fields = {
            "horse": ("nom_cheval", "cheval"),
            "jockey": ("jockey",),
            "trainer": ("entraineur",),
        }.get(entity_type, tuple())

        if not key_field:
            raise ValueError("Type d'entité non supporté")

        hippo_upper = hippodrome.upper() if hippodrome else None

        filtered: List[Dict[str, Any]] = []
        wins = 0
        podiums = 0
        entity_label: Optional[str] = None
        first_date: Optional[date] = None
        last_date: Optional[date] = None

        for row in self._rows:
            if row.get(key_field) != entity_id:
                continue

            race_date = row.get("jour")
            if not isinstance(race_date, date):
                continue

            if start_date and race_date < start_date:
                continue

            if end_date and race_date > end_date:
                continue

            if hippo_upper:
                hippo_value = row.get("hippo")
                if not hippo_value or hippo_value.upper() != hippo_upper:
                    continue

            if entity_label is None:
                for field in label_fields:
                    value = row.get(field)
                    if isinstance(value, str) and value.strip():
                        entity_label = value.strip()
                        break

            position = row.get("cl")
            if isinstance(position, int):
                if position == 1:
                    wins += 1
                if 1 <= position <= 3:
                    podiums += 1

            if first_date is None or race_date < first_date:
                first_date = race_date

            if last_date is None or race_date > last_date:
                last_date = race_date

            filtered.append(dict(row))

        filtered.sort(key=lambda item: item.get("jour") or date.min)

        total_races = len(filtered)
        if not total_races:
            return {
                "entity_id": entity_id,
                "entity_label": entity_label,
                "total_races": 0,
                "wins": 0,
                "podiums": 0,
                "date_start": None,
                "date_end": None,
                "best_win": None,
                "best_podium": None,
                "current_win": None,
                "current_podium": None,
                "history": [],
            }

        current = {
            "win": {"length": 0, "start": None, "end": None},
            "podium": {"length": 0, "start": None, "end": None},
        }
        best = {
            "win": {"length": 0, "start": None, "end": None},
            "podium": {"length": 0, "start": None, "end": None},
        }
        history: List[Dict[str, Any]] = []

        def update_best(kind: str) -> None:
            tracker = current[kind]
            if tracker["length"] > best[kind]["length"]:
                best[kind] = {
                    "length": tracker["length"],
                    "start": tracker["start"],
                    "end": tracker["end"],
                }

        def close_streak(kind: str) -> None:
            tracker = current[kind]
            if tracker["length"]:
                history.append(
                    {
                        "type": kind,
                        "length": tracker["length"],
                        "start_date": tracker["start"],
                        "end_date": tracker["end"],
                        "is_active": False,
                    }
                )
                tracker["length"] = 0
                tracker["start"] = None
                tracker["end"] = None

        for row in filtered:
            race_date = row.get("jour")
            position = row.get("cl")

            is_win = isinstance(position, int) and position == 1
            is_podium = isinstance(position, int) and 1 <= position <= 3

            if is_win:
                tracker = current["win"]
                if tracker["length"] == 0:
                    tracker["start"] = race_date
                tracker["length"] += 1
                tracker["end"] = race_date
                update_best("win")
            else:
                close_streak("win")

            if is_podium:
                tracker = current["podium"]
                if tracker["length"] == 0:
                    tracker["start"] = race_date
                tracker["length"] += 1
                tracker["end"] = race_date
                update_best("podium")
            else:
                close_streak("podium")

        for kind in ("win", "podium"):
            tracker = current[kind]
            if tracker["length"]:
                history.append(
                    {
                        "type": kind,
                        "length": tracker["length"],
                        "start_date": tracker["start"],
                        "end_date": tracker["end"],
                        "is_active": True,
                    }
                )
                update_best(kind)

        def build_best(kind: str) -> Optional[Dict[str, Any]]:
            data = best[kind]
            if data["length"] <= 0:
                return None
            tracker = current[kind]
            is_active = (
                tracker["length"] == data["length"]
                and tracker["start"] == data["start"]
                and tracker["end"] == data["end"]
            )
            return {
                "type": kind,
                "length": data["length"],
                "start_date": data["start"],
                "end_date": data["end"],
                "is_active": is_active,
            }

        def build_current(kind: str) -> Optional[Dict[str, Any]]:
            tracker = current[kind]
            if tracker["length"] <= 0:
                return None
            return {
                "type": kind,
                "length": tracker["length"],
                "start_date": tracker["start"],
                "end_date": tracker["end"],
                "is_active": True,
            }

        history.sort(
            key=lambda item: (
                item.get("is_active", False),
                item.get("length", 0),
                item.get("end_date") or date.min,
            ),
            reverse=True,
        )

        return {
            "entity_id": entity_id,
            "entity_label": entity_label,
            "total_races": total_races,
            "wins": wins,
            "podiums": podiums,
            "date_start": first_date,
            "date_end": last_date,
            "best_win": build_best("win"),
            "best_podium": build_best("podium"),
            "current_win": build_current("win"),
            "current_podium": build_current("podium"),
            "history": history[:10],
        }

    async def performance_trend(
        self,
        *,
        entity_type: str,
        entity_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        hippodrome: Optional[str] = None,
        granularity: str = "month",
    ) -> Dict[str, Any]:
        key_field = {
            "horse": "idChe",
            "jockey": "idJockey",
            "trainer": "idEntraineur",
        }.get(entity_type)

        label_field = {
            "horse": "nom_cheval",
            "jockey": "jockey",
            "trainer": "entraineur",
        }.get(entity_type)

        if not key_field:
            return {"entity_id": entity_id, "entity_label": None, "points": []}

        hippo_upper = hippodrome.upper() if hippodrome else None

        def period_bounds(race_date: date) -> Tuple[Tuple[int, int], date, date, str]:
            if granularity == "week":
                iso_year, iso_week, _ = race_date.isocalendar()
                start = race_date - timedelta(days=race_date.weekday())
                end = start + timedelta(days=6)
                return (iso_year, iso_week), start, end, f"{iso_year}-S{iso_week:02d}"

            month_last_day = monthrange(race_date.year, race_date.month)[1]
            start = race_date.replace(day=1)
            end = race_date.replace(day=month_last_day)
            return (
                (race_date.year, race_date.month),
                start,
                end,
                f"{race_date.year}-{race_date.month:02d}",
            )

        buckets: Dict[Tuple[int, int], Dict[str, Any]] = {}
        entity_label: Optional[str] = None
        min_date: Optional[date] = None
        max_date: Optional[date] = None

        for row in self._rows:
            if row.get(key_field) != entity_id:
                continue

            race_date = row.get("jour")
            if not isinstance(race_date, date):
                continue

            if start_date and race_date < start_date:
                continue

            if end_date and race_date > end_date:
                continue

            if hippo_upper:
                hippo_value = row.get("hippo")
                if not hippo_value or hippo_value.upper() != hippo_upper:
                    continue

            if entity_label is None:
                label_value = row.get(label_field)
                if isinstance(label_value, str) and label_value.strip():
                    entity_label = label_value.strip()

            if min_date is None or race_date < min_date:
                min_date = race_date

            if max_date is None or race_date > max_date:
                max_date = race_date

            key, start, end, label = period_bounds(race_date)
            bucket = buckets.setdefault(
                key,
                {
                    "period_start": start,
                    "period_end": end,
                    "label": label,
                    "races": 0,
                    "wins": 0,
                    "podiums": 0,
                    "positions": [],
                    "odds": [],
                },
            )

            bucket["races"] += 1

            position = row.get("cl")
            if isinstance(position, int):
                bucket["positions"].append(position)
                if position == 1:
                    bucket["wins"] += 1
                if 1 <= position <= 3:
                    bucket["podiums"] += 1

            odds = row.get("cotedirect") or row.get("coteprob")
            if isinstance(odds, (int, float)):
                bucket["odds"].append(float(odds))

        points: List[Dict[str, Any]] = []
        for bucket in sorted(buckets.values(), key=lambda item: item["period_start"]):
            races = bucket["races"]
            wins = bucket["wins"]
            podiums = bucket["podiums"]
            positions = bucket["positions"]
            odds_values = bucket["odds"]

            average_finish = sum(positions) / len(positions) if positions else None
            average_odds = sum(odds_values) / len(odds_values) if odds_values else None

            points.append(
                {
                    "period_start": bucket["period_start"],
                    "period_end": bucket["period_end"],
                    "label": bucket["label"],
                    "races": races,
                    "wins": wins,
                    "podiums": podiums,
                    "win_rate": round(wins / races, 4) if races else None,
                    "podium_rate": round(podiums / races, 4) if races else None,
                    "average_finish": round(average_finish, 2) if average_finish is not None else None,
                    "average_odds": round(average_odds, 2) if average_odds is not None else None,
                }
            )

        return {
            "entity_id": entity_id,
            "entity_label": entity_label,
            "date_start": min_date,
            "date_end": max_date,
            "points": points,
        }

    async def performance_distribution(
        self,
        *,
        entity_type: str,
        entity_id: str,
        dimension: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        hippodrome: Optional[str] = None,
        distance_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        key_field = {
            "horse": "idChe",
            "jockey": "idJockey",
            "trainer": "idEntraineur",
        }.get(entity_type)

        label_field = {
            "horse": "nom_cheval",
            "jockey": "jockey",
            "trainer": "entraineur",
        }.get(entity_type)

        if not key_field:
            return {"entity_id": entity_id, "entity_label": None, "buckets": []}

        allowed_dimensions = {"distance", "draw", "hippodrome", "discipline"}
        if dimension not in allowed_dimensions:
            return {"entity_id": entity_id, "entity_label": None, "buckets": []}

        if dimension == "distance":
            step = distance_step or 200
            if step <= 0:
                step = 200
        else:
            step = None

        hippo_upper = hippodrome.upper() if hippodrome else None

        def resolve_bucket(row: Dict[str, Any]) -> Optional[Tuple[Any, str]]:
            if dimension == "distance":
                value = row.get("dist")
                if not isinstance(value, int):
                    return None
                assert step is not None
                bucket_start = (value // step) * step
                bucket_end = bucket_start + step - 1
                label = f"{bucket_start}-{bucket_end} m"
                return (bucket_start, label)

            if dimension == "draw":
                value = row.get("numero")
                if value is None:
                    return None
                try:
                    numeric = int(value)
                except (TypeError, ValueError):
                    return None
                return (numeric, f"N° {numeric}")

            if dimension == "hippodrome":
                value = row.get("hippo")
                if not value:
                    return None
                label = str(value).upper()
                return (label, label)

            raw = row.get("typec") or row.get("typecourse") or row.get("discipline") or "Inconnu"
            label = str(raw).strip() or "Inconnu"
            return (label.lower(), label.title())

        buckets: Dict[Any, Dict[str, Any]] = {}
        entity_label: Optional[str] = None
        min_date: Optional[date] = None
        max_date: Optional[date] = None

        for row in self._rows:
            if row.get(key_field) != entity_id:
                continue

            race_date = row.get("jour")
            if isinstance(race_date, date):
                if start_date and race_date < start_date:
                    continue
                if end_date and race_date > end_date:
                    continue
            elif start_date or end_date:
                continue

            if hippo_upper:
                hippo_value = row.get("hippo")
                if not hippo_value or hippo_value.upper() != hippo_upper:
                    continue

            bucket_descriptor = resolve_bucket(row)
            if bucket_descriptor is None:
                continue

            bucket_key, bucket_label = bucket_descriptor

            if entity_label is None:
                label_value = row.get(label_field)
                if isinstance(label_value, str) and label_value.strip():
                    entity_label = label_value.strip()

            if isinstance(race_date, date):
                if min_date is None or race_date < min_date:
                    min_date = race_date
                if max_date is None or race_date > max_date:
                    max_date = race_date

            bucket = buckets.setdefault(
                bucket_key,
                {
                    "label": bucket_label,
                    "races": 0,
                    "wins": 0,
                    "podiums": 0,
                    "positions": [],
                    "odds": [],
                },
            )

            bucket["label"] = bucket_label
            bucket["races"] += 1

            position = row.get("cl")
            if isinstance(position, int):
                bucket["positions"].append(position)
                if position == 1:
                    bucket["wins"] += 1
                if 1 <= position <= 3:
                    bucket["podiums"] += 1

            odds_value = row.get("cotedirect") or row.get("coteprob")
            if isinstance(odds_value, (int, float)):
                bucket["odds"].append(float(odds_value))

        formatted: List[Dict[str, Any]] = []
        for key, bucket in buckets.items():
            races = bucket["races"]
            wins = bucket["wins"]
            podiums = bucket["podiums"]

            positions = [pos for pos in bucket["positions"] if isinstance(pos, (int, float))]
            odds_values = bucket["odds"]

            average_finish = sum(positions) / len(positions) if positions else None
            average_odds = sum(odds_values) / len(odds_values) if odds_values else None

            formatted.append(
                {
                    "key": key,
                    "label": bucket["label"],
                    "races": races,
                    "wins": wins,
                    "podiums": podiums,
                    "win_rate": round(wins / races, 4) if races else None,
                    "podium_rate": round(podiums / races, 4) if races else None,
                    "average_finish": round(average_finish, 2) if average_finish is not None else None,
                    "average_odds": round(average_odds, 2) if average_odds is not None else None,
                }
            )

        formatted.sort(key=lambda item: (item["races"], item["wins"]), reverse=True)

        return {
            "entity_id": entity_id,
            "entity_label": entity_label,
            "date_start": min_date,
            "date_end": max_date,
            "dimension": dimension,
            "buckets": formatted,
        }
    async def search_entities(
        self,
        entity_type: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        normalized = _normalize_text(query.strip())
        if len(normalized) < 2:
            return []

        results: Dict[str, Dict[str, Any]] = {}

        if entity_type == "horse":
            key_field = "idChe"
            label_field = "nom_cheval"
        elif entity_type == "jockey":
            key_field = "idJockey"
            label_field = "jockey"
        elif entity_type == "trainer":
            key_field = "idEntraineur"
            label_field = "entraineur"
        elif entity_type == "hippodrome":
            key_field = "hippo"
            label_field = "hippo"
        else:
            return []

        for row in self._rows:
            identifier = row.get(key_field)
            if not identifier:
                continue

            identifier_str = str(identifier)
            label = row.get(label_field) or identifier_str

            searchable = _normalize_text(f"{identifier_str} {label}")
            if normalized not in searchable:
                continue

            entry = results.setdefault(
                identifier_str.upper() if entity_type == "hippodrome" else identifier_str,
                {
                    "id": identifier_str,
                    "label": label,
                    "metadata": {
                        "total_races": 0,
                        "hippodromes": set(),
                        "course_count": 0,
                    },
                },
            )

            if entity_type == "hippodrome":
                entry["id"] = identifier_str.upper()
                entry["label"] = label
                entry["metadata"]["course_count"] += 1
            else:
                entry["metadata"]["total_races"] += 1
                hippo = row.get("hippo")
                if hippo:
                    entry["metadata"]["hippodromes"].add(str(hippo))

            race_date = row.get("jour")
            if isinstance(race_date, date):
                key_name = "last_meeting" if entity_type == "hippodrome" else "last_seen"
                current = entry["metadata"].get(key_name)
                if current is None or race_date > current:
                    entry["metadata"][key_name] = race_date

        formatted = []
        for entry in results.values():
            metadata = entry["metadata"]
            hippos = metadata.get("hippodromes")
            if isinstance(hippos, set):
                metadata["hippodromes"] = sorted(hippos)

            formatted.append({"id": entry["id"], "label": entry["label"], "metadata": metadata})

        formatted.sort(
            key=lambda item: (
                -(item["metadata"].get("total_races") or item["metadata"].get("course_count") or 0),
                item["label"],
            )
        )

        return formatted[:limit]


@pytest.fixture()
def analytics_rows() -> List[Dict[str, Any]]:
    return [
        {
            "jour": date(2024, 5, 4),
            "hippo": "PARIS",
            "prix": 5,
            "dist": 2100,
            "cl": 1,
            "idChe": "H-1",
            "nom_cheval": "Étoile Filante",
            "idJockey": "J-77",
            "jockey": "Jean Dupont",
            "idEntraineur": "T-42",
            "entraineur": "Marie Martin",
            "cotedirect": 5.0,
            "coteprob": 6.2,
            "numero": 3,
        },
        {
            "jour": date(2024, 5, 4),
            "hippo": "PARIS",
            "prix": 5,
            "dist": 2100,
            "cl": 4,
            "idChe": "H-2",
            "nom_cheval": "Nouveau Départ",
            "idJockey": "J-99",
            "jockey": "Anne Leroy",
            "idEntraineur": "T-99",
            "entraineur": "Claire Dubois",
            "cotedirect": 12.0,
            "coteprob": 10.5,
            "numero": 12,
        },
        {
            "jour": date(2024, 5, 18),
            "hippo": "LYON",
            "prix": 2,
            "dist": 2400,
            "cl": 3,
            "idChe": "H-1",
            "nom_cheval": "Étoile Filante",
            "idJockey": "J-88",
            "jockey": "Paul Petit",
            "idEntraineur": "T-42",
            "entraineur": "Marie Martin",
            "cotedirect": 7.5,
            "coteprob": 8.3,
            "numero": 8,
        },
        {
            "jour": date(2024, 6, 1),
            "hippo": "PARIS",
            "prix": 1,
            "dist": 2200,
            "cl": 2,
            "idChe": "H-2",
            "nom_cheval": "Nouveau Départ",
            "idJockey": "J-77",
            "jockey": "Jean Dupont",
            "idEntraineur": "T-99",
            "entraineur": "Claire Dubois",
            "cotedirect": 10.0,
            "coteprob": 9.0,
            "numero": 4,
        },
        {
            "jour": date(2024, 7, 20),
            "hippo": "CHARTRES",
            "prix": 6,
            "dist": 2000,
            "cl": 8,
            "idChe": "H-3",
            "nom_cheval": "Vent Rapide",
            "idJockey": "J-66",
            "jockey": "Luc Morel",
            "idEntraineur": "T-66",
            "entraineur": "Sophie Bernard",
            "cotedirect": 15.0,
            "coteprob": 14.5,
            "numero": 7,
        },
        {
            "jour": date(2024, 7, 5),
            "hippo": "ANGERS",
            "prix": 2,
            "dist": 2100,
            "cl": 4,
            "idChe": "H-3",
            "nom_cheval": "Vent Rapide",
            "idJockey": "J-66",
            "jockey": "Luc Morel",
            "idEntraineur": "T-66",
            "entraineur": "Sophie Bernard",
            "cotedirect": 7.0,
            "coteprob": 6.8,
            "numero": 2,
        },
        {
            "jour": date(2024, 4, 25),
            "hippo": "CAEN",
            "prix": 4,
            "dist": 2300,
            "cl": 1,
            "idChe": "H-3",
            "nom_cheval": "Vent Rapide",
            "idJockey": "J-66",
            "jockey": "Luc Morel",
            "idEntraineur": "T-66",
            "entraineur": "Sophie Bernard",
            "cotedirect": 2.8,
            "coteprob": 3.0,
            "numero": 5,
        },
        {
            "jour": date(2024, 4, 5),
            "hippo": "CAEN",
            "prix": 3,
            "dist": 2300,
            "cl": 3,
            "idChe": "H-3",
            "nom_cheval": "Vent Rapide",
            "idJockey": "J-66",
            "jockey": "Luc Morel",
            "idEntraineur": "T-66",
            "entraineur": "Sophie Bernard",
            "cotedirect": 5.5,
            "coteprob": 5.0,
            "numero": 8,
        },
        {
            "jour": date(2024, 3, 20),
            "hippo": "LAVAL",
            "prix": 7,
            "dist": 2500,
            "cl": 6,
            "idChe": "H-3",
            "nom_cheval": "Vent Rapide",
            "idJockey": "J-66",
            "jockey": "Luc Morel",
            "idEntraineur": "T-66",
            "entraineur": "Sophie Bernard",
            "cotedirect": 12.0,
            "coteprob": 10.0,
            "numero": 9,
        },
        {
            "jour": date(2024, 3, 5),
            "hippo": "LAVAL",
            "prix": 5,
            "dist": 2500,
            "cl": 1,
            "idChe": "H-3",
            "nom_cheval": "Vent Rapide",
            "idJockey": "J-66",
            "jockey": "Luc Morel",
            "idEntraineur": "T-66",
            "entraineur": "Sophie Bernard",
            "cotedirect": 4.4,
            "coteprob": 4.2,
            "numero": 3,
        },
    ]


@pytest.fixture()
def analytics_partants() -> Dict[Tuple[date, str, int], List[Dict[str, Any]]]:
    return {
        (
            date(2024, 6, 10),
            "Paris",
            4,
        ): [
            {
                "jour": date(2024, 6, 10),
                "hippo": "ParisLongchamp",
                "prix": 4,
                "numero": 5,
                "idChe": "H-1",
                "nom_cheval": "Étoile Filante",
                "idJockey": "J-77",
                "jockey": "Jean Dupont",
                "idEntraineur": "T-42",
                "entraineur": "Marie Martin",
                "cotedirect": 5.2,
                "coteprob": 4.9,
                "musiqueche": "1a2a",
                "recence": 12,
                "vha": 14,
                "coursescheval": 25,
                "victoirescheval": 8,
                "placescheval": 14,
                "coursesjockey": 40,
                "victoiresjockey": 12,
                "placejockey": 20,
                "coursesentraineur": 60,
                "victoiresentraineur": 18,
                "placeentraineur": 28,
                "nbCourseCouple": 6,
                "nbVictCouple": 3,
                "nbPlaceCouple": 5,
                "cheque": 42000,
                "devise": "EUR",
                "dist": 2100,
                "typec": "plat",
            },
            {
                "jour": date(2024, 6, 10),
                "hippo": "ParisLongchamp",
                "prix": 4,
                "numero": 8,
                "idChe": "H-2",
                "nom_cheval": "Nouveau Départ",
                "idJockey": "J-99",
                "jockey": "Anne Leroy",
                "idEntraineur": "T-99",
                "entraineur": "Claire Dubois",
                "cotedirect": 9.5,
                "coteprob": 8.1,
                "musiqueche": "4a3a",
                "recence": 20,
                "vha": 10,
                "coursescheval": 15,
                "victoirescheval": 4,
                "placescheval": 9,
                "coursesjockey": 35,
                "victoiresjockey": 9,
                "placejockey": 16,
                "coursesentraineur": 50,
                "victoiresentraineur": 14,
                "placeentraineur": 26,
                "nbCourseCouple": 3,
                "nbVictCouple": 1,
                "nbPlaceCouple": 2,
            },
        ]
    }


@pytest.fixture()
async def analytics_client(
    analytics_rows: List[Dict[str, Any]],
    analytics_partants: Dict[Tuple[date, str, int], List[Dict[str, Any]]],
):
    async def override_get_client():
        yield StubAspiturfClient(analytics_rows, analytics_partants)

    test_app.dependency_overrides[analytics_module.get_aspiturf_client] = override_get_client

    async with AsyncClient(app=test_app, base_url="http://test") as async_client:
        yield async_client

    test_app.dependency_overrides.pop(analytics_module.get_aspiturf_client, None)


@pytest.mark.anyio("asyncio")
async def test_horse_analytics_returns_expected_summary(analytics_client: AsyncClient):
    response = await analytics_client.get("/api/v1/analytics/horse/H-1")
    assert response.status_code == 200

    payload = response.json()
    assert payload["horse_id"] == "H-1"
    assert payload["horse_name"] == "Étoile Filante"
    assert payload["sample_size"] == 2
    assert payload["wins"] == 1
    assert payload["podiums"] == 2
    assert payload["win_rate"] == 0.5
    assert payload["podium_rate"] == 1.0

    recent_results = payload["recent_results"]
    assert recent_results[0]["date"] == "2024-05-18"
    assert recent_results[0]["hippodrome"] == "LYON"

    hippodrome_labels = {item["label"] for item in payload["hippodrome_breakdown"]}
    assert {"PARIS", "LYON"}.issubset(hippodrome_labels)


@pytest.mark.anyio("asyncio")
async def test_jockey_analytics_applies_hippodrome_filter(analytics_client: AsyncClient):
    response = await analytics_client.get("/api/v1/analytics/jockey/J-77", params={"hippodrome": "paris"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["sample_size"] == 2
    assert payload["wins"] == 1

    horses = {item["label"] for item in payload["horse_breakdown"]}
    assert "Étoile Filante (H-1)" in horses
    assert "Nouveau Départ (H-2)" in horses

    metadata = payload["metadata"]
    assert metadata["hippodrome_filter"] == "paris"


@pytest.mark.anyio("asyncio")
async def test_course_analytics_exposes_partant_statistics(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/course",
        params={
            "course_date": "2024-06-10",
            "hippodrome": "Paris",
            "course_number": 4,
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["hippodrome"] == "ParisLongchamp"
    assert payload["course_number"] == 4
    assert payload["distance"] == 2100
    assert payload["discipline"] == "plat"

    partants = payload["partants"]
    assert len(partants) == 2
    first_partant = partants[0]
    assert first_partant["horse_id"] == "H-1"
    assert first_partant["horse_stats"]["wins"] == 8
    assert first_partant["couple_stats"]["wins"] == 3

    second_partant = partants[1]
    assert second_partant["horse_stats"]["places"] == 9
    assert second_partant["couple_stats"]["sample_size"] == 3

    metadata = payload["metadata"]
    assert metadata["date_start"] == "2024-06-10"
    assert metadata["hippodrome_filter"] == "Paris"


@pytest.mark.anyio("asyncio")
async def test_calendar_endpoint_returns_daily_breakdown(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/calendar",
        params={"entity_type": "horse", "entity_id": "H-1"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["total_races"] == 2
    assert payload["total_wins"] == 1
    assert payload["total_podiums"] == 2

    days = payload["days"]
    assert len(days) == 2
    assert days[0]["date"] == "2024-05-04"
    assert days[0]["wins"] == 1
    assert days[0]["race_details"][0]["hippodrome"] == "PARIS"


@pytest.mark.anyio("asyncio")
async def test_calendar_endpoint_supports_date_filters(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/calendar",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "start_date": "2024-05-10",
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["total_races"] == 1
    assert payload["days"][0]["date"] == "2024-05-18"


@pytest.mark.anyio("asyncio")
async def test_calendar_endpoint_rejects_invalid_date_range(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/calendar",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "start_date": "2024-06-01",
            "end_date": "2024-05-01",
        },
    )
    assert response.status_code == 400


@pytest.mark.anyio("asyncio")
async def test_calendar_endpoint_returns_404_when_no_history(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/calendar",
        params={"entity_type": "horse", "entity_id": "UNKNOWN"},
    )
    assert response.status_code == 404


@pytest.mark.anyio("asyncio")
async def test_value_endpoint_returns_ranked_samples(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/value",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "limit": 10,
            "min_edge": 0.0,
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["entity_type"] == "horse"
    assert payload["entity_id"] == "H-1"
    assert payload["sample_size"] == 2
    assert payload["wins"] == 1
    assert payload["positive_edges"] == 2
    assert payload["negative_edges"] == 0
    assert payload["hippodromes"] == ["LYON", "PARIS"]

    samples = payload["samples"]
    assert len(samples) == 2
    assert samples[0]["edge"] >= samples[1]["edge"]
    assert samples[0]["hippodrome"].upper() == "PARIS"
    assert payload["profit"] == 3.0
    assert payload["roi"] == 1.5


@pytest.mark.anyio("asyncio")
async def test_value_endpoint_applies_threshold_and_filters(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/value",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "min_edge": 1.1,
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["sample_size"] == 1
    assert payload["wins"] == 1
    assert payload["positive_edges"] == 1
    assert payload["samples"][0]["edge"] >= 1.1


@pytest.mark.anyio("asyncio")
async def test_value_endpoint_validates_date_range(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/value",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "start_date": "2024-06-30",
            "end_date": "2024-05-01",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "start_date doit être antérieure ou égale à end_date"


@pytest.mark.anyio("asyncio")
async def test_value_endpoint_returns_404_when_no_sample(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/value",
        params={
            "entity_type": "jockey",
            "entity_id": "UNKNOWN",
        },
    )
    assert response.status_code == 404
    assert (
        response.json()["detail"]
        == "Aucune opportunité de value bet détectée pour cette configuration"
    )


@pytest.mark.anyio("asyncio")
async def test_volatility_endpoint_returns_dispersions(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/volatility",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
        },
    )
    assert response.status_code == 200

    payload = response.json()
    metrics = payload["metrics"]

    assert payload["entity_type"] == "horse"
    assert payload["entity_id"] == "H-1"
    assert metrics["sample_size"] == 2
    assert metrics["wins"] == 1
    assert metrics["win_rate"] == 0.5
    assert metrics["average_finish"] == 2.0
    assert metrics["position_std_dev"] == 1.0
    assert metrics["average_odds"] == 6.25
    assert metrics["average_edge"] == 1.0
    assert metrics["consistency_index"] == 0.5

    races = payload["races"]
    assert len(races) == 2
    assert races[0]["date"] == "2024-05-18"
    assert races[0]["odds_actual"] == 7.5
    assert races[1]["edge"] == 1.2
    assert payload["metadata"]["date_start"] == "2024-05-04"
    assert payload["metadata"]["date_end"] == "2024-05-18"


@pytest.mark.anyio("asyncio")
async def test_volatility_endpoint_validates_dates(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/volatility",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "start_date": "2024-06-30",
            "end_date": "2024-05-01",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "start_date doit être antérieure ou égale à end_date"


@pytest.mark.anyio("asyncio")
async def test_volatility_endpoint_returns_404_on_missing_data(
    analytics_client: AsyncClient,
):
    response = await analytics_client.get(
        "/api/v1/analytics/volatility",
        params={
            "entity_type": "horse",
            "entity_id": "UNKNOWN",
        },
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Aucune course trouvée pour cette entité et cette période"


@pytest.mark.anyio("asyncio")
async def test_momentum_endpoint_returns_comparative_windows(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/momentum",
        params={
            "entity_type": "horse",
            "entity_id": "H-3",
            "window": 3,
            "baseline_window": 3,
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["entity_id"] == "H-3"
    assert payload["recent_window"]["race_count"] == 3
    assert payload["recent_window"]["wins"] == 1
    assert payload["recent_window"]["average_finish"] == 4.33
    assert payload["recent_window"]["roi"] == -0.067

    reference = payload["reference_window"]
    assert reference is not None
    assert reference["race_count"] == 3
    assert reference["podiums"] == 2
    assert reference["average_odds"] == 7.3
    assert reference["roi"] == 0.467

    deltas = payload["deltas"]
    assert deltas["win_rate"] == 0.0
    assert deltas["podium_rate"] == pytest.approx(-0.3334, rel=1e-3)
    assert deltas["average_finish"] == 1.0
    assert deltas["roi"] == pytest.approx(-0.534, rel=1e-3)

    metadata = payload["metadata"]
    assert metadata["date_start"] == "2024-03-05"
    assert metadata["date_end"] == "2024-07-20"


@pytest.mark.anyio("asyncio")
async def test_momentum_endpoint_validates_date_range(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/momentum",
        params={
            "entity_type": "horse",
            "entity_id": "H-3",
            "start_date": "2024-07-01",
            "end_date": "2024-06-01",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "start_date doit être antérieure ou égale à end_date"


@pytest.mark.anyio("asyncio")
async def test_momentum_endpoint_returns_404_on_missing_data(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/momentum",
        params={
            "entity_type": "horse",
            "entity_id": "UNKNOWN",
        },
    )
    assert response.status_code == 404


@pytest.mark.anyio("asyncio")
async def test_form_endpoint_returns_recent_form_metrics(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/form",
        params={"entity_type": "horse", "entity_id": "H-1", "window": 5},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["entity_type"] == "horse"
    assert payload["entity_id"] == "H-1"
    assert payload["total_races"] == 2
    assert payload["wins"] == 1
    assert payload["podiums"] == 2
    assert payload["win_rate"] == 0.5
    assert payload["podium_rate"] == 1.0
    assert payload["average_finish"] == 2.0
    assert payload["average_odds"] == 6.25
    assert payload["median_odds"] == 6.25
    assert payload["best_position"] == 1
    assert payload["consistency_index"] == 0.5
    assert payload["form_score"] == 3.5

    races = payload["races"]
    assert [race["score"] for race in races] == [2, 5]
    assert payload["metadata"]["date_end"] == "2024-05-18"


@pytest.mark.anyio("asyncio")
async def test_form_endpoint_applies_date_filters(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/form",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "window": 5,
            "start_date": "2024-05-10",
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["total_races"] == 1
    assert payload["wins"] == 0
    assert payload["podiums"] == 1
    assert payload["average_finish"] == 3.0
    assert payload["consistency_index"] == 1.0
    assert payload["form_score"] == 2.0


@pytest.mark.anyio("asyncio")
async def test_form_endpoint_returns_404_when_no_history(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/form",
        params={"entity_type": "horse", "entity_id": "H-999"},
    )
    assert response.status_code == 404


@pytest.mark.anyio("asyncio")
async def test_form_endpoint_rejects_invalid_date_range(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/form",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "start_date": "2024-06-01",
            "end_date": "2024-05-01",
        },
    )
    assert response.status_code == 400


@pytest.mark.anyio("asyncio")
async def test_comparisons_endpoint_returns_duel_summary(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/comparisons",
        params=[("type", "horse"), ("ids", "H-1"), ("ids", "H-2")],
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["entity_type"] == "horse"
    assert payload["shared_races"] == 1
    assert {item["entity_id"] for item in payload["entities"]} == {"H-1", "H-2"}

    head_to_head = {
        entry["entity_id"]: entry["head_to_head"][0] if entry["head_to_head"] else None
        for entry in payload["entities"]
    }

    assert head_to_head["H-1"]["opponent_id"] == "H-2"
    assert head_to_head["H-1"]["ahead"] == 1
    assert head_to_head["H-1"]["behind"] == 0
    assert head_to_head["H-2"]["behind"] == 1


@pytest.mark.anyio("asyncio")
async def test_comparisons_endpoint_requires_existing_ids(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/comparisons",
        params=[("type", "horse"), ("ids", "H-1"), ("ids", "H-42")],
    )

    assert response.status_code == 404
    assert "H-42" in response.json()["detail"]


@pytest.mark.anyio("asyncio")
async def test_comparisons_endpoint_rejects_invalid_dates(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/comparisons",
        params=[
            ("type", "horse"),
            ("ids", "H-1"),
            ("ids", "H-2"),
            ("start_date", "2024-06-02"),
            ("end_date", "2024-05-01"),
        ],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "start_date doit être antérieure ou égale à end_date"


@pytest.mark.anyio("asyncio")
async def test_insights_endpoint_builds_leaderboards(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/insights",
        params={
            "start_date": "2024-05-01",
            "end_date": "2024-06-30",
            "limit": 3,
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["metadata"]["date_start"] == "2024-05-01"
    assert payload["metadata"]["date_end"] == "2024-06-30"

    horses = payload["top_horses"]
    assert [item["entity_id"] for item in horses] == ["H-1", "H-2"]
    assert horses[0]["wins"] == 1
    assert horses[0]["podiums"] == 2

    jockeys = payload["top_jockeys"]
    assert jockeys[0]["entity_id"] == "J-77"
    assert jockeys[0]["win_rate"] == 0.5

    trainers = payload["top_trainers"]
    assert trainers[0]["entity_id"] == "T-42"
    assert trainers[0]["podium_rate"] == 1.0


@pytest.mark.anyio("asyncio")
async def test_insights_endpoint_validates_date_range(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/insights",
        params={
            "start_date": "2024-06-30",
            "end_date": "2024-05-01",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "start_date doit être antérieure ou égale à end_date"


@pytest.mark.anyio("asyncio")
async def test_trends_endpoint_returns_monthly_series(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/trends",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "granularity": "month",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["entity_type"] == "horse"
    assert payload["entity_id"] == "H-1"
    assert payload["granularity"] == "month"
    assert payload["metadata"]["date_start"] == "2024-05-04"
    assert payload["metadata"]["date_end"] == "2024-05-18"

    points = payload["points"]
    assert len(points) == 1
    first_point = points[0]
    assert first_point["label"] == "2024-05"
    assert first_point["races"] == 2
    assert first_point["wins"] == 1
    assert first_point["podiums"] == 2
    assert first_point["average_finish"] == 2.0


@pytest.mark.anyio("asyncio")
async def test_trends_endpoint_supports_weekly_breakdown(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/trends",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "granularity": "week",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    points = payload["points"]
    assert len(points) == 2
    labels = {point["label"] for point in points}
    assert labels == {"2024-S18", "2024-S20"}


@pytest.mark.anyio("asyncio")
async def test_trends_endpoint_validates_date_range(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/trends",
        params={
            "entity_type": "jockey",
            "entity_id": "J-77",
            "start_date": "2024-06-15",
            "end_date": "2024-05-01",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "start_date doit être antérieure ou égale à end_date"


@pytest.mark.anyio("asyncio")
async def test_trends_endpoint_returns_404_on_missing_data(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/trends",
        params={
            "entity_type": "trainer",
            "entity_id": "UNKNOWN",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Aucune course trouvée pour cette entité et cette période"


@pytest.mark.anyio("asyncio")
async def test_distributions_endpoint_groups_by_distance(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/distributions",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "dimension": "distance",
            "distance_step": 200,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["entity_type"] == "horse"
    assert payload["entity_id"] == "H-1"
    assert payload["dimension"] == "distance"

    buckets = {bucket["label"]: bucket for bucket in payload["buckets"]}
    assert "2000-2199 m" in buckets
    assert "2400-2599 m" in buckets
    assert buckets["2000-2199 m"]["wins"] == 1
    assert buckets["2000-2199 m"]["win_rate"] == 1.0


@pytest.mark.anyio("asyncio")
async def test_distributions_endpoint_applies_hippodrome_filter(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/distributions",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
            "dimension": "draw",
            "hippodrome": "lyon",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    buckets = payload["buckets"]
    assert len(buckets) == 1
    assert buckets[0]["label"] == "N° 8"
    assert buckets[0]["podiums"] == 1


@pytest.mark.anyio("asyncio")
async def test_distributions_endpoint_returns_404_on_empty_result(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/distributions",
        params={
            "entity_type": "trainer",
            "entity_id": "UNKNOWN",
            "dimension": "hippodrome",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Aucune course trouvée pour cette entité et cette période"


@pytest.mark.anyio("asyncio")
async def test_streaks_endpoint_returns_series(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/streaks",
        params={
            "entity_type": "horse",
            "entity_id": "H-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["total_races"] == 2
    assert payload["wins"] == 1
    assert payload["podiums"] == 2

    best_win = payload["best_win_streak"]
    assert best_win["length"] == 1
    assert best_win["type"] == "win"
    assert best_win["is_active"] is False

    current_podium = payload["current_podium_streak"]
    assert current_podium["length"] == 2
    assert current_podium["type"] == "podium"
    assert current_podium["is_active"] is True

    history = payload["streak_history"]
    types = {item["type"] for item in history}
    assert {"win", "podium"}.issubset(types)


@pytest.mark.anyio("asyncio")
async def test_streaks_endpoint_handles_empty_results(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/streaks",
        params={
            "entity_type": "horse",
            "entity_id": "UNKNOWN",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Aucune course trouvée pour cette entité et cette période"


@pytest.mark.anyio("asyncio")
async def test_streaks_endpoint_validates_date_range(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/streaks",
        params={
            "entity_type": "jockey",
            "entity_id": "J-77",
            "start_date": "2024-06-30",
            "end_date": "2024-05-01",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "start_date doit être antérieure ou égale à end_date"


@pytest.mark.anyio("asyncio")
async def test_search_endpoint_returns_sorted_results(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/search",
        params={"type": "horse", "query": "etoile"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload
    first_result = payload[0]
    assert first_result["id"] == "H-1"
    assert first_result["label"] == "Étoile Filante"
    assert first_result["metadata"]["total_races"] == 2
    assert "PARIS" in first_result["metadata"]["hippodromes"]


@pytest.mark.anyio("asyncio")
async def test_search_endpoint_enforces_min_length(analytics_client: AsyncClient):
    response = await analytics_client.get(
        "/api/v1/analytics/search",
        params={"type": "jockey", "query": "j"},
    )
    assert response.status_code == 422
