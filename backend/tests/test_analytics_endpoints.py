from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("CORS_ORIGINS", "[\"http://localhost:3000\"]")

from app.api.endpoints import analytics as analytics_module


test_app = FastAPI()
test_app.include_router(analytics_module.router, prefix="/api/v1/analytics")

pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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

        return [dict(row) for row in results]

    async def get_partants_course(
        self,
        course_date: date,
        hippodrome: str,
        course_number: int,
    ) -> List[Dict[str, Any]]:
        key = (course_date, hippodrome.upper(), course_number)
        return [dict(item) for item in self._partants.get(key, [])]

    async def search_entities(
        self,
        entity_type: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        normalized = query.strip().lower()
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

            searchable = f"{identifier_str.lower()} {str(label).lower()}"
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
            "coteprob": 4.8,
            "numero": 3,
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
            "coteprob": 6.9,
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
