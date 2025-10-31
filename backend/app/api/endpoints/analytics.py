"""Endpoints REST pour exposer les statistiques enrichies Aspiturf."""

from __future__ import annotations

from datetime import date
from statistics import mean
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import settings
from app.schemas.analytics import (
    AnalyticsMetadata,
    AnalyticsSearchResult,
    AnalyticsSearchType,
    CourseAnalyticsResponse,
    CoupleAnalyticsResponse,
    HorseAnalyticsResponse,
    JockeyAnalyticsResponse,
    PartantInsight,
    PerformanceBreakdown,
    PerformanceSummary,
    RecentRace,
    TrainerAnalyticsResponse,
)
from app.services.aspiturf_client import AspiturfClient, AspiturfConfig


router = APIRouter()


async def get_aspiturf_client() -> AsyncIterator[AspiturfClient]:
    """Dépendance FastAPI pour charger le client Aspiturf configuré."""

    csv_path = settings.ASPITURF_CSV_PATH
    csv_url = settings.ASPITURF_CSV_URL

    if not csv_path and not csv_url:
        raise HTTPException(
            status_code=503,
            detail="Aspiturf n'est pas configuré. Définir ASPITURF_CSV_PATH ou ASPITURF_CSV_URL."
        )

    config = AspiturfConfig(
        csv_delimiter=settings.ASPITURF_CSV_DELIMITER,
        csv_encoding=settings.ASPITURF_CSV_ENCODING,
    )

    client = AspiturfClient(csv_path=csv_path, csv_url=csv_url, config=config)

    try:
        await client.__aenter__()
        yield client
    except ValueError as exc:  # Mauvaise configuration (CSV manquant, etc.)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - renvoyer erreur contrôlée
        raise HTTPException(
            status_code=502,
            detail=f"Impossible de charger les données Aspiturf: {exc}"
        ) from exc
    finally:
        await client.__aexit__(None, None, None)


@router.get(
    "/search",
    response_model=List[AnalyticsSearchResult],
    summary="Rechercher des entités Aspiturf",
    description="Permet de retrouver rapidement les identifiants cheval, jockey, entraineur ou un hippodrome.",
)
async def search_entities(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    search_type: AnalyticsSearchType = Query(
        ..., alias="type", description="Type d'entité à rechercher"
    ),
    query: str = Query(
        ..., min_length=2, description="Terme de recherche (nom ou identifiant)"
    ),
    limit: int = Query(10, ge=1, le=50, description="Nombre maximum de résultats"),
) -> List[AnalyticsSearchResult]:
    """Expose un moteur de recherche simplifié sur les données Aspiturf."""

    raw_results = await client.search_entities(search_type.value, query, limit)

    return [
        AnalyticsSearchResult(
            type=search_type,
            id=item["id"],
            label=item["label"],
            metadata=item.get("metadata", {}),
        )
        for item in raw_results
    ]


def _safe_rate(value: int, total: int) -> Optional[float]:
    if not total:
        return None
    return round(value / total, 4)


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [float(v) for v in values if isinstance(v, (int, float))]
    if not cleaned:
        return None
    return round(mean(cleaned), 3)


def _is_win(row: Dict[str, Any]) -> bool:
    position = row.get('cl')
    return isinstance(position, int) and position == 1


def _is_podium(row: Dict[str, Any]) -> bool:
    position = row.get('cl')
    return isinstance(position, int) and 1 <= position <= 3


def _extract_name(rows: Iterable[Dict[str, Any]], keys: Iterable[str]) -> Optional[str]:
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _sorted_races(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(item: Dict[str, Any]):
        race_date = item.get('jour')
        course_number = item.get('prix') or 0
        numero = item.get('numero') or 0
        # None -> date.min pour tri
        if isinstance(race_date, date):
            return (race_date, course_number, numero)
        return (date.min, course_number, numero)

    return sorted(rows, key=sort_key, reverse=True)


def _recent_results(rows: List[Dict[str, Any]], limit: int = 5) -> List[RecentRace]:
    results: List[RecentRace] = []

    for row in _sorted_races(rows)[:limit]:
        results.append(
            RecentRace(
                date=row.get('jour'),
                hippodrome=row.get('hippo'),
                course_number=row.get('prix'),
                distance=row.get('dist'),
                final_position=row.get('cl'),
                odds=row.get('cotedirect') or row.get('coteprob'),
                is_win=_is_win(row),
                is_podium=_is_podium(row),
            )
        )

    return results


def _compute_breakdown(
    rows: List[Dict[str, Any]],
    key: str,
    label_getter: Optional[Any] = None,
    limit: int = 5,
) -> List[PerformanceBreakdown]:
    groups: Dict[Any, List[Dict[str, Any]]] = {}

    for row in rows:
        group_key = row.get(key)
        groups.setdefault(group_key, []).append(row)

    breakdown: List[PerformanceBreakdown] = []

    for group_key, group_rows in groups.items():
        label = (
            label_getter(group_key, group_rows)
            if label_getter
            else (str(group_key) if group_key is not None else "Inconnu")
        )

        total = len(group_rows)
        wins = sum(1 for r in group_rows if _is_win(r))
        podiums = sum(1 for r in group_rows if _is_podium(r))

        breakdown.append(
            PerformanceBreakdown(
                label=label,
                total=total,
                wins=wins,
                podiums=podiums,
                win_rate=_safe_rate(wins, total),
                podium_rate=_safe_rate(podiums, total),
            )
        )

    breakdown.sort(key=lambda item: (item.total, item.win_rate or 0), reverse=True)
    return breakdown[:limit]


def _build_metadata(rows: List[Dict[str, Any]], hippodrome: Optional[str]) -> AnalyticsMetadata:
    dates = [row.get('jour') for row in rows if isinstance(row.get('jour'), date)]

    return AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=min(dates) if dates else None,
        date_end=max(dates) if dates else None,
    )


def _build_performance_summary(
    total: Optional[int],
    wins: Optional[int],
    places: Optional[int],
) -> Optional[PerformanceSummary]:
    if total in (None, 0) and wins in (None, 0) and places in (None, 0):
        return None

    total_value = int(total or 0)
    wins_value = int(wins or 0)
    places_value = int(places or 0)

    return PerformanceSummary(
        sample_size=total_value,
        wins=wins_value,
        places=places_value,
        win_rate=_safe_rate(wins_value, total_value),
        place_rate=_safe_rate(places_value, total_value),
    )


def _distance_label(_: Any, rows: List[Dict[str, Any]]) -> str:
    distance = rows[0].get('dist') if rows else None
    if isinstance(distance, int):
        return f"{distance} m"
    return "Distance inconnue"


def _hippodrome_label(group_key: Any, _: List[Dict[str, Any]]) -> str:
    if isinstance(group_key, str) and group_key.strip():
        return group_key.upper()
    return "Hippodrome inconnu"


def _horse_label(group_key: Any, rows: List[Dict[str, Any]]) -> str:
    name = _extract_name(rows, ['nom_cheval', 'cheval'])
    if name and group_key:
        return f"{name} ({group_key})"
    if name:
        return name
    if group_key:
        return str(group_key)
    return "Cheval inconnu"


def _jockey_label(group_key: Any, rows: List[Dict[str, Any]]) -> str:
    name = _extract_name(rows, ['jockey'])
    if name and group_key:
        return f"{name} ({group_key})"
    if name:
        return name
    if group_key:
        return str(group_key)
    return "Jockey inconnu"


def _trainer_label(group_key: Any, rows: List[Dict[str, Any]]) -> str:
    name = _extract_name(rows, ['entraineur'])
    if name and group_key:
        return f"{name} ({group_key})"
    if name:
        return name
    if group_key:
        return str(group_key)
    return "Entraineur inconnu"


@router.get("/horse/{horse_id}", response_model=HorseAnalyticsResponse)
async def get_horse_analytics(
    horse_id: str,
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer sur un hippodrome spécifique (optionnel)",
    ),
    client: AspiturfClient = Depends(get_aspiturf_client),
) -> HorseAnalyticsResponse:
    """Retourne les statistiques complètes pour un cheval."""

    rows = await client.get_races(horse_id=horse_id, hippodrome=hippodrome)

    if not rows:
        raise HTTPException(status_code=404, detail="Aucune donnée trouvée pour ce cheval")

    sample_size = len(rows)
    wins = sum(1 for row in rows if _is_win(row))
    podiums = sum(1 for row in rows if _is_podium(row))

    average_finish = _safe_mean(row.get('cl') for row in rows)
    odds_values = [row.get('cotedirect') or row.get('coteprob') for row in rows]
    average_odds = _safe_mean(odds_values)

    horse_name = _extract_name(rows, ['nom_cheval', 'cheval'])

    return HorseAnalyticsResponse(
        horse_id=horse_id,
        horse_name=horse_name,
        sample_size=sample_size,
        wins=wins,
        podiums=podiums,
        win_rate=_safe_rate(wins, sample_size),
        podium_rate=_safe_rate(podiums, sample_size),
        average_finish=average_finish,
        average_odds=average_odds,
        recent_results=_recent_results(rows),
        hippodrome_breakdown=_compute_breakdown(rows, 'hippo', _hippodrome_label),
        distance_breakdown=_compute_breakdown(rows, 'dist', _distance_label),
        metadata=_build_metadata(rows, hippodrome),
    )


@router.get("/jockey/{jockey_id}", response_model=JockeyAnalyticsResponse)
async def get_jockey_analytics(
    jockey_id: str,
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer sur un hippodrome spécifique (optionnel)",
    ),
    client: AspiturfClient = Depends(get_aspiturf_client),
) -> JockeyAnalyticsResponse:
    """Retourne les statistiques complètes pour un jockey."""

    rows = await client.get_races(jockey_id=jockey_id, hippodrome=hippodrome)

    if not rows:
        raise HTTPException(status_code=404, detail="Aucune donnée trouvée pour ce jockey")

    sample_size = len(rows)
    wins = sum(1 for row in rows if _is_win(row))
    podiums = sum(1 for row in rows if _is_podium(row))

    jockey_name = _extract_name(rows, ['jockey'])

    return JockeyAnalyticsResponse(
        jockey_id=jockey_id,
        jockey_name=jockey_name,
        sample_size=sample_size,
        wins=wins,
        podiums=podiums,
        win_rate=_safe_rate(wins, sample_size),
        podium_rate=_safe_rate(podiums, sample_size),
        average_finish=_safe_mean(row.get('cl') for row in rows),
        recent_results=_recent_results(rows),
        horse_breakdown=_compute_breakdown(rows, 'idChe', _horse_label),
        hippodrome_breakdown=_compute_breakdown(rows, 'hippo', _hippodrome_label),
        metadata=_build_metadata(rows, hippodrome),
    )


@router.get("/trainer/{trainer_id}", response_model=TrainerAnalyticsResponse)
async def get_trainer_analytics(
    trainer_id: str,
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer sur un hippodrome spécifique (optionnel)",
    ),
    client: AspiturfClient = Depends(get_aspiturf_client),
) -> TrainerAnalyticsResponse:
    """Retourne les statistiques complètes pour un entraineur."""

    rows = await client.get_races(trainer_id=trainer_id, hippodrome=hippodrome)

    if not rows:
        raise HTTPException(status_code=404, detail="Aucune donnée trouvée pour cet entraineur")

    sample_size = len(rows)
    wins = sum(1 for row in rows if _is_win(row))
    podiums = sum(1 for row in rows if _is_podium(row))

    trainer_name = _extract_name(rows, ['entraineur'])

    return TrainerAnalyticsResponse(
        trainer_id=trainer_id,
        trainer_name=trainer_name,
        sample_size=sample_size,
        wins=wins,
        podiums=podiums,
        win_rate=_safe_rate(wins, sample_size),
        podium_rate=_safe_rate(podiums, sample_size),
        average_finish=_safe_mean(row.get('cl') for row in rows),
        recent_results=_recent_results(rows),
        horse_breakdown=_compute_breakdown(rows, 'idChe', _horse_label),
        hippodrome_breakdown=_compute_breakdown(rows, 'hippo', _hippodrome_label),
        metadata=_build_metadata(rows, hippodrome),
    )


@router.get("/couple", response_model=CoupleAnalyticsResponse)
async def get_couple_analytics(
    horse_id: str = Query(..., description="Identifiant du cheval (idChe)"),
    jockey_id: str = Query(..., description="Identifiant du jockey (idJockey)"),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer sur un hippodrome spécifique (optionnel)",
    ),
    client: AspiturfClient = Depends(get_aspiturf_client),
) -> CoupleAnalyticsResponse:
    """Retourne les statistiques pour un couple cheval-jockey."""

    rows = await client.get_races(
        horse_id=horse_id,
        jockey_id=jockey_id,
        hippodrome=hippodrome,
    )

    if not rows:
        raise HTTPException(status_code=404, detail="Aucune donnée trouvée pour ce couple")

    sample_size = len(rows)
    wins = sum(1 for row in rows if _is_win(row))
    podiums = sum(1 for row in rows if _is_podium(row))

    horse_name = _extract_name(rows, ['nom_cheval', 'cheval'])
    jockey_name = _extract_name(rows, ['jockey'])

    return CoupleAnalyticsResponse(
        horse_id=horse_id,
        jockey_id=jockey_id,
        horse_name=horse_name,
        jockey_name=jockey_name,
        sample_size=sample_size,
        wins=wins,
        podiums=podiums,
        win_rate=_safe_rate(wins, sample_size),
        podium_rate=_safe_rate(podiums, sample_size),
        average_finish=_safe_mean(row.get('cl') for row in rows),
        recent_results=_recent_results(rows),
        metadata=_build_metadata(rows, hippodrome),
    )


@router.get("/course", response_model=CourseAnalyticsResponse)
async def get_course_analytics(
    course_date: str = Query(..., description="Date de la course au format YYYY-MM-DD"),
    hippodrome: str = Query(..., description="Nom de l'hippodrome"),
    course_number: int = Query(..., ge=1, description="Numéro de la course"),
    client: AspiturfClient = Depends(get_aspiturf_client),
) -> CourseAnalyticsResponse:
    """Retourne les informations détaillées d'une course Aspiturf."""

    try:
        target_date = date.fromisoformat(course_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Date invalide. Format attendu: YYYY-MM-DD") from exc

    partants = await client.get_partants_course(target_date, hippodrome, course_number)

    if not partants:
        raise HTTPException(status_code=404, detail="Course introuvable dans les données Aspiturf")

    course_info = partants[0]

    insights: List[PartantInsight] = []

    for p in partants:
        insights.append(
            PartantInsight(
                numero=p.get('numero'),
                horse_id=p.get('idChe'),
                horse_name=_extract_name([p], ['nom_cheval', 'cheval']),
                jockey_id=p.get('idJockey'),
                jockey_name=p.get('jockey'),
                trainer_id=p.get('idEntraineur'),
                trainer_name=p.get('entraineur'),
                odds=p.get('cotedirect'),
                probable_odds=p.get('coteprob'),
                recent_form=p.get('musiqueche'),
                days_since_last_race=p.get('recence'),
                handicap_value=p.get('vha'),
                horse_stats=_build_performance_summary(
                    p.get('coursescheval'),
                    p.get('victoirescheval'),
                    p.get('placescheval'),
                ),
                jockey_stats=_build_performance_summary(
                    p.get('coursesjockey'),
                    p.get('victoiresjockey'),
                    p.get('placejockey'),
                ),
                trainer_stats=_build_performance_summary(
                    p.get('coursesentraineur'),
                    p.get('victoiresentraineur'),
                    p.get('placeentraineur'),
                ),
                couple_stats=_build_performance_summary(
                    p.get('nbCourseCouple'),
                    p.get('nbVictCouple'),
                    p.get('nbPlaceCouple'),
                ),
            )
        )

    return CourseAnalyticsResponse(
        date=target_date,
        hippodrome=course_info.get('hippo') or hippodrome,
        course_number=course_number,
        distance=course_info.get('dist'),
        discipline=course_info.get('typec'),
        allocation=course_info.get('cheque'),
        currency=course_info.get('devise'),
        partants=insights,
        metadata=AnalyticsMetadata(
            hippodrome_filter=hippodrome,
            date_start=target_date,
            date_end=target_date,
        ),
    )

