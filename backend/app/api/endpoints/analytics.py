"""Endpoints REST pour exposer les statistiques enrichies Aspiturf."""

from __future__ import annotations

from datetime import date, timedelta
from itertools import combinations
from statistics import mean, median, pstdev
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import settings
from app.schemas.analytics import (
    AnalyticsMetadata,
    AnalyticsComparisonResponse,
    ComparisonEntitySummary,
    HeadToHeadBreakdown,
    AnalyticsMomentumResponse,
    AnalyticsProgressionResponse,
    AnalyticsFormResponse,
    AnalyticsSearchResult,
    AnalyticsSearchType,
    AnalyticsOddsResponse,
    AnalyticsInsightsResponse,
    AnalyticsStreakResponse,
    AnalyticsCalendarResponse,
    AnalyticsValueResponse,
    AnalyticsVolatilityResponse,
    AnalyticsWorkloadResponse,
    AnalyticsEfficiencyResponse,
    DistributionBucket,
    DistributionDimension,
    FormRace,
    CalendarDaySummary,
    CalendarRaceDetail,
    CourseAnalyticsResponse,
    CoupleAnalyticsResponse,
    HorseAnalyticsResponse,
    JockeyAnalyticsResponse,
    LeaderboardEntry,
    PerformanceStreak,
    PerformanceTrendPoint,
    PerformanceTrendResponse,
    PerformanceDistributionResponse,
    PartantInsight,
    PerformanceBreakdown,
    MomentumSlice,
    MomentumShift,
    ProgressionRace,
    ProgressionSummary,
    OddsBucketMetrics,
    PerformanceSummary,
    RecentRace,
    ValueOpportunitySample,
    VolatilityMetrics,
    VolatilityRaceSample,
    WorkloadSummary,
    WorkloadTimelineEntry,
    EfficiencyMetrics,
    EfficiencySample,
    TrainerAnalyticsResponse,
    TrendEntityType,
    TrendGranularity,
)

try:  # pragma: no cover - optional dependency for tests
    from app.services.aspiturf_client import AspiturfClient, AspiturfConfig
    _ASPITURF_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:  # pragma: no cover - handled via dependency
    AspiturfClient = Any  # type: ignore[assignment]
    AspiturfConfig = Any  # type: ignore[assignment]
    _ASPITURF_IMPORT_ERROR = exc


router = APIRouter()


async def get_aspiturf_client() -> AsyncIterator[AspiturfClient]:
    """Dépendance FastAPI pour charger le client Aspiturf configuré."""

    if _ASPITURF_IMPORT_ERROR is not None:
        raise HTTPException(
            status_code=503,
            detail="Le client Aspiturf n'est pas disponible: installez les dépendances backend.",
        )

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


def _average_position(values: Iterable[int]) -> Optional[float]:
    """Calcule prudemment la position moyenne d'arrivée."""

    positions = [int(value) for value in values if isinstance(value, int)]
    if not positions:
        return None
    return round(sum(positions) / len(positions), 2)


def _parse_float(value: Any) -> Optional[float]:
    """Convertit un champ Aspiturf en float si possible."""

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "."))
        except ValueError:
            return None
    return None


def _rest_bucket(rest_days: int) -> str:
    """Regroupe un nombre de jours de repos dans une tranche lisible."""

    if rest_days <= 7:
        return "0-7 jours"
    if rest_days <= 14:
        return "8-14 jours"
    if rest_days <= 30:
        return "15-30 jours"
    if rest_days <= 60:
        return "31-60 jours"
    return "60+ jours"


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


def _round_optional(value: Optional[float], digits: int = 3) -> Optional[float]:
    """Arrondit prudemment une valeur flottante si elle est définie."""

    if value is None:
        return None
    return round(value, digits)


def _odds_bucket_descriptor(odds: float) -> Tuple[str, str]:
    """Retourne un identifiant stable et un libellé pour segmenter les cotes."""

    if odds < 3.0:
        return ("favorite", "Favori (<3.0)")
    if odds < 6.0:
        return ("contender", "Challenger (3.0-5.9)")
    if odds < 10.0:
        return ("outsider", "Outsider (6.0-9.9)")
    return ("longshot", "Long shot (>=10.0)")


def _implied_probability(odds: Optional[float]) -> Optional[float]:
    """Transforme une cote décimale en probabilité implicite bornée."""

    if odds is None or odds <= 0:
        return None
    return min(round(1.0 / odds, 4), 1.0)


def _podium_probability(probability: Optional[float]) -> Optional[float]:
    """Approxime une probabilité de podium à partir d'une probabilité de victoire."""

    if probability is None:
        return None
    return min(round(probability * 3, 4), 1.0)


def _form_points(position: Optional[int]) -> int:
    """Attribue un score simple de 0 à 5 selon la position finale."""

    if not isinstance(position, int):
        return 0
    if position == 1:
        return 5
    if position == 2:
        return 3
    if position == 3:
        return 2
    if position == 4:
        return 1
    return 0


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


def _form_races(rows: List[Dict[str, Any]]) -> List[FormRace]:
    """Convertit les courses récentes en objets enrichis d'un score de forme."""

    history: List[FormRace] = []

    for row in rows:
        position = row.get('cl') if isinstance(row.get('cl'), int) else None
        odds_value = row.get('cotedirect') or row.get('coteprob')

        history.append(
            FormRace(
                date=row.get('jour'),
                hippodrome=row.get('hippo'),
                course_number=row.get('prix'),
                distance=row.get('dist'),
                final_position=position,
                odds=odds_value,
                is_win=_is_win(row),
                is_podium=_is_podium(row),
                score=_form_points(position),
            )
        )

    return history


def _consistency_index(positions: List[int]) -> Optional[float]:
    """Calcule un indice de constance basé sur l'écart-type des positions."""

    cleaned = [pos for pos in positions if isinstance(pos, int)]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return 1.0

    deviation = pstdev(cleaned)
    # Plus l'écart-type est faible, plus l'indice tend vers 1.
    return round(1 / (1 + deviation), 3)


def _progression_trend(change: Optional[int], has_reference: bool) -> str:
    """Associe un libellé simple à la variation observée."""

    if change is None:
        return "initial" if not has_reference else "unknown"
    if change > 0:
        return "improvement"
    if change < 0:
        return "decline"
    return "stable"


def _progression_analysis(rows: List[Dict[str, Any]]) -> Tuple[List[ProgressionRace], ProgressionSummary]:
    """Analyse la séquence chronologique de courses pour produire les variations."""

    chronological = list(reversed(_sorted_races(rows)))

    entries: List[ProgressionRace] = []
    changes: List[int] = []

    improvements = 0
    declines = 0
    stable = 0
    longest_improvement_streak = 0
    longest_decline_streak = 0
    current_improvement_streak = 0
    current_decline_streak = 0

    previous_position: Optional[int] = None
    reference_defined = False

    for row in chronological:
        position = row.get("cl") if isinstance(row.get("cl"), int) else None
        if position is not None and not reference_defined:
            # La première position rencontrée sert de base de comparaison.
            reference_defined = True

        change_value: Optional[int] = None
        if previous_position is not None and position is not None:
            change_value = previous_position - position
            changes.append(change_value)

            if change_value > 0:
                improvements += 1
                current_improvement_streak += 1
                current_decline_streak = 0
                longest_improvement_streak = max(longest_improvement_streak, current_improvement_streak)
            elif change_value < 0:
                declines += 1
                current_decline_streak += 1
                current_improvement_streak = 0
                longest_decline_streak = max(longest_decline_streak, current_decline_streak)
            else:
                stable += 1
                current_improvement_streak = 0
                current_decline_streak = 0
        else:
            current_improvement_streak = 0
            current_decline_streak = 0

        entries.append(
            ProgressionRace(
                date=row.get("jour") if isinstance(row.get("jour"), date) else None,
                hippodrome=row.get("hippo"),
                course_number=row.get("prix") if isinstance(row.get("prix"), int) else None,
                distance=row.get("dist") if isinstance(row.get("dist"), int) else None,
                final_position=position,
                previous_position=previous_position if previous_position is not None else None,
                change=change_value,
                trend=_progression_trend(change_value, previous_position is not None),
            )
        )

        if position is not None:
            previous_position = position

    races_with_position = [entry for entry in entries if entry.final_position is not None]

    average_change: Optional[float] = None
    if changes:
        average_change = round(sum(changes) / len(changes), 3)

    best_change = max(changes) if changes else None
    worst_change = min(changes) if changes else None
    net_progress = sum(changes) if changes else None

    summary = ProgressionSummary(
        races=len(races_with_position),
        improvements=improvements,
        declines=declines,
        stable=stable,
        average_change=average_change,
        best_change=best_change,
        worst_change=worst_change,
        longest_improvement_streak=longest_improvement_streak,
        longest_decline_streak=longest_decline_streak,
        net_progress=net_progress,
    )

    return entries, summary


def _momentum_slice(label: str, rows: List[Dict[str, Any]]) -> MomentumSlice:
    """Construit un résumé statistique pour une fenêtre temporelle donnée."""

    race_count = len(rows)
    wins = sum(1 for row in rows if _is_win(row))
    podiums = sum(1 for row in rows if _is_podium(row))

    sorted_rows = _sorted_races(rows)

    positions: List[int] = []
    odds_values: List[float] = []
    first_date: Optional[date] = None
    last_date: Optional[date] = None
    profit_sum = 0.0
    bet_count = 0
    races: List[RecentRace] = []

    for row in sorted_rows:
        race_date = row.get("jour")
        if isinstance(race_date, date):
            first_date = race_date if first_date is None else min(first_date, race_date)
            last_date = race_date if last_date is None else max(last_date, race_date)

        position = row.get("cl")
        if isinstance(position, int):
            positions.append(position)

        actual_odds = _parse_float(row.get("cotedirect") or row.get("cote_direct"))
        implied_odds = _parse_float(row.get("coteprob") or row.get("cote_prob"))
        display_odds = actual_odds if actual_odds is not None else implied_odds

        if actual_odds is not None:
            odds_values.append(actual_odds)
            bet_count += 1
            profit_sum += (actual_odds - 1) if _is_win(row) else -1

        races.append(
            RecentRace(
                date=race_date if isinstance(race_date, date) else None,
                hippodrome=row.get("hippo"),
                course_number=row.get("prix"),
                distance=row.get("dist"),
                final_position=position if isinstance(position, int) else None,
                odds=display_odds,
                is_win=_is_win(row),
                is_podium=_is_podium(row),
            )
        )

    average_finish = None
    if positions:
        average_finish = round(sum(positions) / len(positions), 2)

    average_odds = _safe_mean(odds_values)
    roi = None
    if bet_count:
        roi = round(profit_sum / bet_count, 3)

    return MomentumSlice(
        label=label,
        start_date=first_date,
        end_date=last_date,
        race_count=race_count,
        wins=wins,
        podiums=podiums,
        win_rate=_safe_rate(wins, race_count) if race_count else None,
        podium_rate=_safe_rate(podiums, race_count) if race_count else None,
        average_finish=average_finish,
        average_odds=average_odds,
        roi=roi,
        races=races,
    )


def _momentum_shift(recent: MomentumSlice, reference: Optional[MomentumSlice]) -> MomentumShift:
    """Calcule les écarts élémentaires entre deux fenêtres de momentum."""

    if reference is None:
        return MomentumShift()

    def _delta(current: Optional[float], baseline: Optional[float], precision: int) -> Optional[float]:
        if current is None or baseline is None:
            return None
        return round(current - baseline, precision)

    return MomentumShift(
        win_rate=_delta(recent.win_rate, reference.win_rate, 4),
        podium_rate=_delta(recent.podium_rate, reference.podium_rate, 4),
        average_finish=_delta(recent.average_finish, reference.average_finish, 2),
        roi=_delta(recent.roi, reference.roi, 3),
    )


def _to_leaderboard_entries(rows: List[Dict[str, Any]]) -> List[LeaderboardEntry]:
    """Convertit les dictionnaires issus du client en objets de classement typés."""

    entries: List[LeaderboardEntry] = []

    for row in rows:
        entries.append(
            LeaderboardEntry(
                entity_id=str(row.get("entity_id")),
                label=str(row.get("label")),
                sample_size=int(row.get("sample_size", 0)),
                wins=int(row.get("wins", 0)),
                podiums=int(row.get("podiums", 0)),
                win_rate=row.get("win_rate"),
                podium_rate=row.get("podium_rate"),
                average_finish=row.get("average_finish"),
                last_seen=row.get("last_seen"),
            )
        )

    return entries


def _to_streak(payload: Optional[Dict[str, Any]], fallback_type: str) -> Optional[PerformanceStreak]:
    """Convertit un dictionnaire brut en objet de série typé."""

    if not payload:
        return None

    length = payload.get("length")
    if not length:
        return None

    streak_type = payload.get("type", fallback_type)
    if streak_type not in {"win", "podium"}:
        streak_type = fallback_type

    return PerformanceStreak(
        type=streak_type,  # type: ignore[arg-type]
        length=int(length),
        start_date=payload.get("start_date"),
        end_date=payload.get("end_date"),
        is_active=bool(payload.get("is_active", False)),
    )


def _to_trend_points(rows: List[Dict[str, Any]]) -> List[PerformanceTrendPoint]:
    """Cast des agrégats de tendance fournis par le client en schémas Pydantic."""

    points: List[PerformanceTrendPoint] = []

    for row in rows:
        points.append(
            PerformanceTrendPoint(
                period_start=row.get("period_start"),
                period_end=row.get("period_end"),
                label=str(row.get("label")),
                races=int(row.get("races", 0)),
                wins=int(row.get("wins", 0)),
                podiums=int(row.get("podiums", 0)),
                win_rate=row.get("win_rate"),
                podium_rate=row.get("podium_rate"),
                average_finish=row.get("average_finish"),
                average_odds=row.get("average_odds"),
            )
        )

    return points


def _to_distribution_buckets(rows: List[Dict[str, Any]]) -> List[DistributionBucket]:
    """Convertit une liste de dictionnaires en seaux de distribution typés."""

    buckets: List[DistributionBucket] = []

    for row in rows:
        buckets.append(
            DistributionBucket(
                label=str(row.get("label")),
                races=int(row.get("races", 0)),
                wins=int(row.get("wins", 0)),
                podiums=int(row.get("podiums", 0)),
                win_rate=row.get("win_rate"),
                podium_rate=row.get("podium_rate"),
                average_finish=row.get("average_finish"),
                average_odds=row.get("average_odds"),
            )
        )

    return buckets


def _to_calendar_days(rows: List[Dict[str, Any]]) -> List[CalendarDaySummary]:
    """Transforme les agrégats journaliers en objets Pydantic typés."""

    days: List[CalendarDaySummary] = []

    for row in rows:
        raw_details = row.get("race_details", [])
        race_details: List[CalendarRaceDetail] = []

        for detail in raw_details:
            if not isinstance(detail, dict):
                continue

            race_details.append(
                CalendarRaceDetail(
                    hippodrome=detail.get("hippodrome"),
                    course_number=detail.get("course_number"),
                    distance=detail.get("distance"),
                    final_position=detail.get("final_position"),
                    odds=detail.get("odds"),
                )
            )

        hippodromes_raw = row.get("hippodromes") or []
        hippodromes: List[str] = []
        for value in hippodromes_raw:
            if value is None:
                continue
            hippodromes.append(str(value))

        days.append(
            CalendarDaySummary(
                date=row.get("date"),
                hippodromes=hippodromes,
                races=int(row.get("races", 0)),
                wins=int(row.get("wins", 0)),
                podiums=int(row.get("podiums", 0)),
                average_finish=row.get("average_finish"),
                average_odds=row.get("average_odds"),
                race_details=race_details,
            )
        )

    return days


def _to_value_samples(rows: List[Dict[str, Any]]) -> List[ValueOpportunitySample]:
    """Cast des opportunités de value bet en objets typés pour la réponse API."""

    samples: List[ValueOpportunitySample] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        samples.append(
            ValueOpportunitySample(
                date=row.get("date"),
                hippodrome=row.get("hippodrome"),
                course_number=row.get("course_number"),
                distance=row.get("distance"),
                final_position=row.get("final_position"),
                odds_actual=row.get("odds_actual"),
                odds_implied=row.get("odds_implied"),
                edge=row.get("edge"),
                is_win=row.get("is_win"),
                profit=row.get("profit"),
            )
        )

    return samples


@router.get(
    "/insights",
    response_model=AnalyticsInsightsResponse,
    summary="Obtenir les classements multi-entités",
    description="Agrège les meilleures performances chevaux, jockeys et entraineurs sur une période.",
)
async def get_analytics_insights(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer sur un hippodrome spécifique (code Aspiturf)",
    ),
    start_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses disputées à partir de cette date",
    ),
    end_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses disputées jusqu'à cette date",
    ),
    limit: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Nombre maximal d'entrées par classement",
    ),
) -> AnalyticsInsightsResponse:
    """Retourne les classements consolidés pour chaque entité majeure."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    leaderboard_kwargs = {
        "start_date": start_date,
        "end_date": end_date,
        "hippodrome": hippodrome,
        "limit": limit,
    }

    horse_rows = await client.leaderboard("horse", **leaderboard_kwargs)
    jockey_rows = await client.leaderboard("jockey", **leaderboard_kwargs)
    trainer_rows = await client.leaderboard("trainer", **leaderboard_kwargs)

    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=start_date,
        date_end=end_date,
    )

    return AnalyticsInsightsResponse(
        metadata=metadata,
        top_horses=_to_leaderboard_entries(horse_rows),
        top_jockeys=_to_leaderboard_entries(jockey_rows),
        top_trainers=_to_leaderboard_entries(trainer_rows),
    )


@router.get(
    "/comparisons",
    response_model=AnalyticsComparisonResponse,
    summary="Comparer plusieurs entités sur une période",
    description=(
        "Agrège les statistiques principales de plusieurs chevaux, jockeys ou entraîneurs "
        "et calcule un bilan des confrontations directes lorsqu'ils se sont affrontés."
    ),
)
async def compare_entities(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., alias="type", description="Type d'entités à comparer (cheval, jockey, entraîneur)"
    ),
    entity_ids: List[str] = Query(
        ...,
        alias="ids",
        min_items=2,
        description="Liste des identifiants Aspiturf à comparer (au moins deux)",
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer sur un hippodrome spécifique (optionnel)",
    ),
    start_date: Optional[date] = Query(
        default=None,
        description="Ignorer les courses disputées avant cette date",
    ),
    end_date: Optional[date] = Query(
        default=None,
        description="Ignorer les courses disputées après cette date",
    ),
) -> AnalyticsComparisonResponse:
    """Compare plusieurs entités homogènes en exposant un résumé et les duels directs."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    normalized_ids: List[str] = []
    for entity_id in entity_ids:
        candidate = entity_id.strip()
        if candidate and candidate not in normalized_ids:
            normalized_ids.append(candidate)

    if len(normalized_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="Fournir au moins deux identifiants distincts à comparer",
        )

    rows_by_entity: Dict[str, List[Dict[str, Any]]] = {}
    missing: List[str] = []

    for entity_id in normalized_ids:
        race_kwargs = {
            "hippodrome": hippodrome,
            "start_date": start_date,
            "end_date": end_date,
        }

        if entity_type == TrendEntityType.HORSE:
            race_kwargs["horse_id"] = entity_id
        elif entity_type == TrendEntityType.JOCKEY:
            race_kwargs["jockey_id"] = entity_id
        else:
            race_kwargs["trainer_id"] = entity_id

        rows = await client.get_races(**race_kwargs)

        if not rows:
            missing.append(entity_id)
            continue

        rows_by_entity[entity_id] = rows

    if missing:
        raise HTTPException(
            status_code=404,
            detail=(
                "Aucune donnée trouvée pour les identifiants suivants: "
                + ", ".join(sorted(missing))
            ),
        )

    if len(rows_by_entity) < 2:
        raise HTTPException(
            status_code=404,
            detail="Impossible de comparer les entités demandées sur la période sélectionnée",
        )

    summaries, shared_races, metadata = _aggregate_comparison(
        entity_type=AnalyticsSearchType(entity_type.value),
        rows_by_entity=rows_by_entity,
        hippodrome=hippodrome,
        start_date=start_date,
        end_date=end_date,
    )

    return AnalyticsComparisonResponse(
        entity_type=AnalyticsSearchType(entity_type.value),
        shared_races=shared_races,
        entities=summaries,
        metadata=metadata,
    )


@router.get(
    "/form",
    response_model=AnalyticsFormResponse,
    summary="Mesurer la forme récente d'une entité",
    description=(
        "Analyse les N dernières courses d'un cheval, d'un jockey ou d'un entraîneur "
        "pour dériver un score de forme et des indicateurs de constance."
    ),
)
async def get_form_snapshot(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)"
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité analysée"
    ),
    window: int = Query(
        5,
        ge=1,
        le=30,
        description="Nombre de courses à considérer pour calculer la forme",
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer sur un hippodrome spécifique (optionnel)",
    ),
    start_date: Optional[date] = Query(
        default=None,
        description="Ignorer les courses disputées avant cette date",
    ),
    end_date: Optional[date] = Query(
        default=None,
        description="Ignorer les courses disputées après cette date",
    ),
) -> AnalyticsFormResponse:
    """Retourne les indicateurs de forme récente pour l'entité demandée."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400, detail="start_date doit être antérieure ou égale à end_date"
        )

    filters: Dict[str, Any] = {}
    if entity_type == TrendEntityType.HORSE:
        filters["horse_id"] = entity_id
    elif entity_type == TrendEntityType.JOCKEY:
        filters["jockey_id"] = entity_id
    elif entity_type == TrendEntityType.TRAINER:
        filters["trainer_id"] = entity_id
    else:  # pragma: no cover - enum exhaustivité
        raise HTTPException(status_code=400, detail="Type d'entité non supporté")

    if hippodrome:
        filters["hippodrome"] = hippodrome

    rows = await client.get_races(**filters)

    filtered_rows: List[Dict[str, Any]] = []
    for row in rows:
        race_date = row.get("jour")

        if start_date and isinstance(race_date, date) and race_date < start_date:
            continue
        if end_date and isinstance(race_date, date) and race_date > end_date:
            continue
        if start_date and not isinstance(race_date, date):
            # Si une borne temporelle est spécifiée mais que la date est inconnue,
            # on exclut la course pour éviter des résultats biaisés.
            continue
        if end_date and not isinstance(race_date, date):
            continue

        filtered_rows.append(row)

    if not filtered_rows:
        raise HTTPException(status_code=404, detail="Aucune course trouvée pour cette entité")

    sorted_rows = _sorted_races(filtered_rows)
    window_rows = sorted_rows[:window]

    positions = [row.get("cl") for row in window_rows if isinstance(row.get("cl"), int)]
    odds_values = [
        float(row.get("cotedirect") or row.get("coteprob"))
        for row in window_rows
        if isinstance(row.get("cotedirect") or row.get("coteprob"), (int, float))
    ]

    wins = sum(1 for row in window_rows if _is_win(row))
    podiums = sum(1 for row in window_rows if _is_podium(row))
    score_total = sum(_form_points(row.get("cl")) for row in window_rows)

    entity_label = _extract_name(
        window_rows,
        {
            TrendEntityType.HORSE: ("nom_cheval", "cheval"),
            TrendEntityType.JOCKEY: ("jockey",),
            TrendEntityType.TRAINER: ("entraineur",),
        }[entity_type],
    )

    return AnalyticsFormResponse(
        entity_type=entity_type,
        entity_id=entity_id,
        entity_label=entity_label,
        window=window,
        metadata=_build_metadata(window_rows, hippodrome),
        total_races=len(window_rows),
        wins=wins,
        podiums=podiums,
        win_rate=_safe_rate(wins, len(window_rows)),
        podium_rate=_safe_rate(podiums, len(window_rows)),
        average_finish=_safe_mean(positions),
        average_odds=_safe_mean(odds_values),
        median_odds=round(median(odds_values), 3) if odds_values else None,
        best_position=min(positions) if positions else None,
        consistency_index=_consistency_index(positions),
        form_score=round(score_total / len(window_rows), 2) if window_rows else None,
        races=_form_races(window_rows),
    )


@router.get(
    "/trends",
    response_model=PerformanceTrendResponse,
    summary="Visualiser la tendance de performance d'une entité",
    description=(
        "Agrège les courses par semaine ou par mois afin d'identifier la dynamique d'un cheval, "
        "d'un jockey ou d'un entraîneur."
    ),
)
async def get_performance_trends(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)"
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité"
    ),
    granularity: TrendGranularity = Query(
        TrendGranularity.MONTH,
        description="Granularité temporelle de l'agrégation",
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer uniquement les courses disputées dans un hippodrome",
    ),
    start_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses à partir de cette date",
    ),
    end_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses jusqu'à cette date",
    ),
) -> PerformanceTrendResponse:
    """Retourne l'évolution des résultats d'une entité sur une période donnée."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    trend_payload = await client.performance_trend(
        entity_type=entity_type.value,
        entity_id=entity_id,
        start_date=start_date,
        end_date=end_date,
        hippodrome=hippodrome,
        granularity=granularity.value,
    )

    points = _to_trend_points(trend_payload.get("points", []))

    if not points:
        raise HTTPException(
            status_code=404,
            detail="Aucune course trouvée pour cette entité et cette période",
        )

    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=trend_payload.get("date_start"),
        date_end=trend_payload.get("date_end"),
    )

    return PerformanceTrendResponse(
        entity_type=entity_type,
        entity_id=trend_payload.get("entity_id", entity_id),
        entity_label=trend_payload.get("entity_label"),
        granularity=granularity,
        metadata=metadata,
        points=points,
    )


@router.get(
    "/distributions",
    response_model=PerformanceDistributionResponse,
    summary="Analyser la répartition des performances d'une entité",
    description=(
        "Agrège les résultats d'un cheval, jockey ou entraîneur selon une dimension donnée "
        "(distance, numéro de corde, hippodrome ou discipline)."
    ),
)
async def get_performance_distribution(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)"
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité"
    ),
    dimension: DistributionDimension = Query(
        ..., description="Dimension d'agrégation pour la distribution",
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer uniquement les courses disputées dans un hippodrome",
    ),
    start_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses à partir de cette date",
    ),
    end_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses jusqu'à cette date",
    ),
    distance_step: int = Query(
        200,
        ge=50,
        le=1000,
        description="Largeur des intervalles en mètres pour l'analyse par distance",
    ),
) -> PerformanceDistributionResponse:
    """Retourne la distribution des performances selon la dimension choisie."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    # On ne transmet le pas de distance que lorsque l'utilisateur analyse par distance.
    distance_step_value: Optional[int] = None
    if dimension is DistributionDimension.DISTANCE:
        distance_step_value = distance_step

    payload = await client.performance_distribution(
        entity_type=entity_type.value,
        entity_id=entity_id,
        dimension=dimension.value,
        start_date=start_date,
        end_date=end_date,
        hippodrome=hippodrome,
        distance_step=distance_step_value,
    )

    buckets = _to_distribution_buckets(payload.get("buckets", []))

    if not buckets:
        raise HTTPException(
            status_code=404,
            detail="Aucune course trouvée pour cette entité et cette période",
        )

    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=payload.get("date_start"),
        date_end=payload.get("date_end"),
    )

    return PerformanceDistributionResponse(
        entity_type=entity_type,
        entity_id=payload.get("entity_id", entity_id),
        entity_label=payload.get("entity_label"),
        dimension=dimension,
        metadata=metadata,
        buckets=buckets,
    )


@router.get(
    "/calendar",
    response_model=AnalyticsCalendarResponse,
    summary="Visualiser le calendrier de performances quotidiennes",
    description=(
        "Regroupe les courses disputées par une entité Aspiturf par date pour analyser les enchaînements "
        "de résultats, les périodes fastes ou les passages à vide."
    ),
)
async def get_performance_calendar(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)"
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité"
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer uniquement les courses disputées dans un hippodrome",
    ),
    start_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses à partir de cette date",
    ),
    end_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses jusqu'à cette date",
    ),
) -> AnalyticsCalendarResponse:
    """Retourne les performances quotidiennes d'une entité sous forme de calendrier."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    payload = await client.performance_calendar(
        entity_type=entity_type.value,
        entity_id=entity_id,
        start_date=start_date,
        end_date=end_date,
        hippodrome=hippodrome,
    )

    total_races = int(payload.get("total_races") or 0)
    if total_races == 0:
        raise HTTPException(
            status_code=404,
            detail="Aucune course trouvée pour cette entité et cette période",
        )

    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=payload.get("date_start"),
        date_end=payload.get("date_end"),
    )

    days = _to_calendar_days(payload.get("days", []))
    if not days:
        raise HTTPException(
            status_code=404,
            detail="Aucune journée de course exploitable sur la période",
        )

    return AnalyticsCalendarResponse(
        entity_type=entity_type,
        entity_id=payload.get("entity_id", entity_id),
        entity_label=payload.get("entity_label"),
        metadata=metadata,
        total_races=total_races,
        total_wins=int(payload.get("total_wins") or 0),
        total_podiums=int(payload.get("total_podiums") or 0),
        days=days,
    )


@router.get(
    "/value",
    response_model=AnalyticsValueResponse,
    summary="Identifier les opportunités de value bet",
    description=(
        "Compare la cote probable Aspiturf et la cote observée pour repérer les courses offrant un écart "
        "potentiellement intéressant pour un cheval, un jockey ou un entraîneur."
    ),
)
async def get_value_opportunities(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)"
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité"
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer uniquement les courses disputées dans un hippodrome",
    ),
    start_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses à partir de cette date",
    ),
    end_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses jusqu'à cette date",
    ),
    min_edge: Optional[float] = Query(
        default=0.0,
        ge=0.0,
        description="Écarter les courses dont la différence de cote est inférieure à ce seuil",
    ),
    limit: int = Query(
        default=25,
        ge=5,
        le=100,
        description="Nombre maximal de courses retournées, classées par edge décroissant",
    ),
) -> AnalyticsValueResponse:
    """Retourne les courses où la cote observée diffère sensiblement de l'estimation Aspiturf."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    payload = await client.value_opportunities(
        entity_type=entity_type.value,
        entity_id=entity_id,
        start_date=start_date,
        end_date=end_date,
        hippodrome=hippodrome,
        min_edge=min_edge,
        limit=limit,
    )

    samples = _to_value_samples(payload.get("samples", []))

    if not samples:
        raise HTTPException(
            status_code=404,
            detail="Aucune opportunité de value bet détectée pour cette configuration",
        )

    summary = payload.get("summary") or {}

    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=payload.get("date_start"),
        date_end=payload.get("date_end"),
    )

    # Les agrégats sont calculés côté client pour garantir la cohérence entre les différents supports.
    return AnalyticsValueResponse(
        entity_type=entity_type,
        entity_id=payload.get("entity_id", entity_id),
        entity_label=payload.get("entity_label"),
        metadata=metadata,
        sample_size=int(summary.get("sample_size") or len(samples)),
        wins=int(summary.get("wins") or 0),
        win_rate=summary.get("win_rate"),
        positive_edges=int(summary.get("positive_edges") or 0),
        negative_edges=int(summary.get("negative_edges") or 0),
        average_edge=summary.get("average_edge"),
        median_edge=summary.get("median_edge"),
        average_odds=summary.get("average_odds"),
        median_odds=summary.get("median_odds"),
        stake_count=int(summary.get("stake_count") or 0),
        profit=summary.get("profit"),
        roi=summary.get("roi"),
        hippodromes=[str(item) for item in payload.get("hippodromes", []) if item is not None],
        samples=samples,
    )


@router.get(
    "/volatility",
    response_model=AnalyticsVolatilityResponse,
    summary="Mesurer la volatilité des résultats d'une entité",
    description=(
        "Calcule des indicateurs de dispersion (écart-type, indice de constance) sur les positions et cotes "
        "en s'appuyant sur l'historique des courses Aspiturf filtrées."
    ),
)
async def get_volatility_profile(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)"
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité"
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer uniquement les courses disputées dans un hippodrome",
    ),
    start_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses à partir de cette date"
    ),
    end_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses jusqu'à cette date"
    ),
) -> AnalyticsVolatilityResponse:
    """Retourne un profil de volatilité basé sur les positions finales et les cotes."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    rows = await client.get_races(
        horse_id=entity_id if entity_type == TrendEntityType.HORSE else None,
        jockey_id=entity_id if entity_type == TrendEntityType.JOCKEY else None,
        trainer_id=entity_id if entity_type == TrendEntityType.TRAINER else None,
        hippodrome=hippodrome,
        start_date=start_date,
        end_date=end_date,
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Aucune course trouvée pour cette entité et cette période",
        )

    # Tri décroissant pour mettre en avant les courses les plus récentes dans la réponse.
    rows.sort(
        key=lambda item: (
            item.get("jour") if isinstance(item.get("jour"), date) else date.min,
            item.get("prix") if isinstance(item.get("prix"), int) else -1,
        ),
        reverse=True,
    )

    label_fields: Tuple[str, ...]
    if entity_type == TrendEntityType.HORSE:
        label_fields = ("nom_cheval", "cheval")
    elif entity_type == TrendEntityType.JOCKEY:
        label_fields = ("jockey",)
    else:
        label_fields = ("entraineur",)

    entity_label = _extract_name(rows, label_fields)

    wins = sum(1 for row in rows if _is_win(row))
    podiums = sum(1 for row in rows if _is_podium(row))
    sample_size = len(rows)

    positions: List[int] = []
    odds_actual_values: List[float] = []
    edges: List[float] = []
    race_samples: List[VolatilityRaceSample] = []
    first_date: Optional[date] = None
    last_date: Optional[date] = None

    # Parcours des courses retenues afin d'accumuler les statistiques brutes.
    for row in rows:
        race_date = row.get("jour")
        if isinstance(race_date, date):
            first_date = race_date if first_date is None else min(first_date, race_date)
            last_date = race_date if last_date is None else max(last_date, race_date)

        position = row.get("cl")
        if isinstance(position, int):
            positions.append(position)

        actual_odds = _parse_float(row.get("cotedirect") or row.get("cote_direct"))
        implied_odds = _parse_float(row.get("coteprob") or row.get("cote_prob"))

        if actual_odds is not None:
            odds_actual_values.append(actual_odds)

        edge: Optional[float] = None
        if actual_odds is not None and implied_odds is not None:
            edge = round(implied_odds - actual_odds, 4)
            edges.append(edge)

        race_samples.append(
            VolatilityRaceSample(
                date=race_date if isinstance(race_date, date) else None,
                hippodrome=row.get("hippo"),
                course_number=row.get("prix") if isinstance(row.get("prix"), int) else None,
                distance=row.get("dist") if isinstance(row.get("dist"), int) else None,
                final_position=position if isinstance(position, int) else None,
                odds_actual=actual_odds,
                odds_implied=implied_odds,
                edge=edge,
                is_win=_is_win(row),
                is_podium=_is_podium(row),
            )
        )

    average_finish = None
    position_std = None
    if positions:
        average_finish = round(sum(positions) / len(positions), 2)
        if len(positions) >= 2:
            position_std = round(pstdev(positions), 3)

    average_odds = _safe_mean(odds_actual_values)
    odds_std = None
    if len(odds_actual_values) >= 2:
        odds_std = round(pstdev(odds_actual_values), 3)

    average_edge = _safe_mean(edges)
    consistency_index = None
    if position_std is not None:
        consistency_index = round(1 / (1 + position_std), 3)

    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=first_date,
        date_end=last_date,
    )

    metrics = VolatilityMetrics(
        sample_size=sample_size,
        wins=wins,
        podiums=podiums,
        win_rate=_safe_rate(wins, sample_size),
        podium_rate=_safe_rate(podiums, sample_size),
        average_finish=average_finish,
        position_std_dev=position_std,
        average_odds=average_odds,
        odds_std_dev=odds_std,
        average_edge=average_edge,
        consistency_index=consistency_index,
    )

    return AnalyticsVolatilityResponse(
        entity_type=entity_type,
        entity_id=entity_id,
        entity_label=entity_label,
        metadata=metadata,
        metrics=metrics,
        races=race_samples,
    )


@router.get(
    "/efficiency",
    response_model=AnalyticsEfficiencyResponse,
    summary="Comparer victoires attendues et observées d'une entité",
    description=(
        "Met en regard les probabilités implicites issues des cotes et les résultats concrets pour "
        "identifier les profils surperformants ou sous-performants."
    ),
)
async def get_efficiency_profile(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)",
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité",
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer uniquement les courses disputées dans un hippodrome",
    ),
    start_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses à partir de cette date",
    ),
    end_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses jusqu'à cette date",
    ),
) -> AnalyticsEfficiencyResponse:
    """Mesure l'écart entre attentes de marché et réalisations sportives."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    rows = await client.get_races(
        horse_id=entity_id if entity_type == TrendEntityType.HORSE else None,
        jockey_id=entity_id if entity_type == TrendEntityType.JOCKEY else None,
        trainer_id=entity_id if entity_type == TrendEntityType.TRAINER else None,
        hippodrome=hippodrome,
        start_date=start_date,
        end_date=end_date,
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Aucune course trouvée pour cette entité et cette période",
        )

    sorted_rows = _sorted_races(rows)

    label_fields: Tuple[str, ...]
    if entity_type == TrendEntityType.HORSE:
        label_fields = ("nom_cheval", "cheval")
    elif entity_type == TrendEntityType.JOCKEY:
        label_fields = ("jockey",)
    else:
        label_fields = ("entraineur",)

    entity_label = _extract_name(rows, label_fields)

    sample_size = len(rows)
    wins = sum(1 for row in rows if _is_win(row))
    podiums = sum(1 for row in rows if _is_podium(row))

    race_dates: List[date] = []
    odds_values: List[float] = []
    probability_values: List[float] = []
    expected_wins_total = 0.0
    expected_podiums_total = 0.0
    expected_observations = 0
    podium_observations = 0
    stake_count = 0
    profit = 0.0
    samples: List[EfficiencySample] = []

    for row in sorted_rows:
        race_date_value = row.get("jour")
        race_date = race_date_value if isinstance(race_date_value, date) else None
        if race_date is not None:
            race_dates.append(race_date)

        odds = _parse_float(row.get("cotedirect") or row.get("cote_direct"))
        win_probability = _implied_probability(odds)
        podium_probability = _podium_probability(win_probability)
        is_win = _is_win(row)
        is_podium = _is_podium(row)

        if odds is not None:
            odds_values.append(odds)
            stake_count += 1
            profit += (odds - 1.0) if is_win else -1.0

        if win_probability is not None:
            expected_observations += 1
            expected_wins_total += win_probability
            probability_values.append(win_probability)

        if podium_probability is not None:
            podium_observations += 1
            expected_podiums_total += podium_probability

        edge = None
        if win_probability is not None:
            edge = round((1.0 if is_win else 0.0) - win_probability, 4)

        samples.append(
            EfficiencySample(
                date=race_date,
                hippodrome=row.get("hippo"),
                course_number=row.get("prix") if isinstance(row.get("prix"), int) else None,
                odds=_round_optional(odds, 2),
                expected_win_probability=win_probability,
                expected_podium_probability=podium_probability,
                finish_position=row.get("cl") if isinstance(row.get("cl"), int) else None,
                is_win=is_win,
                is_podium=is_podium,
                edge=edge,
            )
        )

    expected_wins_value = (
        round(expected_wins_total, 2) if expected_observations else None
    )
    expected_podiums_value = (
        round(expected_podiums_total, 2) if podium_observations else None
    )

    win_delta = (
        round(wins - expected_wins_total, 2) if expected_observations else None
    )
    podium_delta = (
        round(podiums - expected_podiums_total, 2) if podium_observations else None
    )

    average_odds = _safe_mean(odds_values)
    average_win_probability = _safe_mean(probability_values)

    roi = round(profit / stake_count, 3) if stake_count else None

    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=min(race_dates) if race_dates else None,
        date_end=max(race_dates) if race_dates else None,
    )

    metrics = EfficiencyMetrics(
        sample_size=sample_size,
        wins=wins,
        expected_wins=expected_wins_value,
        win_delta=win_delta,
        podiums=podiums,
        expected_podiums=expected_podiums_value,
        podium_delta=podium_delta,
        average_odds=average_odds,
        average_expected_win_probability=average_win_probability,
        stake_count=stake_count,
        profit=round(profit, 2) if stake_count else None,
        roi=roi,
    )

    return AnalyticsEfficiencyResponse(
        entity_type=entity_type,
        entity_id=entity_id,
        entity_label=entity_label,
        metadata=metadata,
        metrics=metrics,
        samples=samples,
    )


@router.get(
    "/odds",
    response_model=AnalyticsOddsResponse,
    summary="Segmenter les performances par tranches de cotes",
    description=(
        "Ventile les courses en quatre segments de cotes (favori, challenger, outsider, long shot) "
        "et calcule pour chacun les taux de réussite et retours sur investissement."
    ),
)
async def get_odds_profile(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)"
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité"
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Limiter l'analyse aux courses disputées sur un hippodrome",
    ),
    start_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses à partir de cette date"
    ),
    end_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses jusqu'à cette date"
    ),
) -> AnalyticsOddsResponse:
    """Analyse les performances d'une entité selon les segments de cotes."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    rows = await client.get_races(
        horse_id=entity_id if entity_type == TrendEntityType.HORSE else None,
        jockey_id=entity_id if entity_type == TrendEntityType.JOCKEY else None,
        trainer_id=entity_id if entity_type == TrendEntityType.TRAINER else None,
        hippodrome=hippodrome,
        start_date=start_date,
        end_date=end_date,
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Aucune course trouvée pour cette entité et cette période",
        )

    rows.sort(
        key=lambda item: (
            item.get("jour") if isinstance(item.get("jour"), date) else date.min,
            item.get("prix") if isinstance(item.get("prix"), int) else -1,
        ),
        reverse=True,
    )

    if entity_type == TrendEntityType.HORSE:
        label_fields = ("nom_cheval", "cheval")
    elif entity_type == TrendEntityType.JOCKEY:
        label_fields = ("jockey",)
    else:
        label_fields = ("entraineur",)

    entity_label = _extract_name(rows, label_fields)

    bucket_definitions: List[Tuple[str, str]] = [
        ("favorite", "Favori (<3.0)"),
        ("contender", "Challenger (3.0-5.9)"),
        ("outsider", "Outsider (6.0-9.9)"),
        ("longshot", "Long shot (>=10.0)"),
    ]

    bucket_stats: Dict[str, Dict[str, Any]] = {
        key: {
            "label": label,
            "sample_size": 0,
            "wins": 0,
            "podiums": 0,
            "positions": [],
            "odds": [],
            "profit": 0.0,
        }
        for key, label in bucket_definitions
    }

    races_with_odds = 0
    missing_odds = 0
    overall_positions: List[int] = []
    overall_odds: List[float] = []
    overall_wins = 0
    overall_podiums = 0
    overall_profit = 0.0
    race_dates: List[date] = []

    for row in rows:
        race_date = row.get("jour")
        if isinstance(race_date, date):
            race_dates.append(race_date)

        odds = _parse_float(row.get("cotedirect") or row.get("cote_direct"))
        if odds is None or odds <= 0:
            missing_odds += 1
            continue

        bucket_key, bucket_label = _odds_bucket_descriptor(odds)
        stats = bucket_stats.setdefault(
            bucket_key,
            {
                "label": bucket_label,
                "sample_size": 0,
                "wins": 0,
                "podiums": 0,
                "positions": [],
                "odds": [],
                "profit": 0.0,
            },
        )

        stats["label"] = bucket_label
        stats["sample_size"] += 1
        races_with_odds += 1

        is_win = _is_win(row)
        is_podium = _is_podium(row)

        if is_win:
            stats["wins"] += 1
            overall_wins += 1
        if is_podium:
            stats["podiums"] += 1
            overall_podiums += 1

        position = row.get("cl")
        if isinstance(position, int):
            stats.setdefault("positions", []).append(position)
            overall_positions.append(position)

        stats.setdefault("odds", []).append(odds)
        overall_odds.append(odds)

        profit_delta = (odds - 1.0) if is_win else -1.0
        stats["profit"] = stats.get("profit", 0.0) + profit_delta
        overall_profit += profit_delta

    bucket_metrics: List[OddsBucketMetrics] = []
    for key, fallback_label in bucket_definitions:
        stats = bucket_stats.get(key, {"label": fallback_label})
        sample_size = int(stats.get("sample_size", 0))
        wins = int(stats.get("wins", 0))
        podiums = int(stats.get("podiums", 0))
        positions = stats.get("positions", [])
        odds_values = stats.get("odds", [])
        profit_value = stats.get("profit", 0.0)

        bucket_metrics.append(
            OddsBucketMetrics(
                label=str(stats.get("label", fallback_label)),
                sample_size=sample_size,
                wins=wins,
                podiums=podiums,
                win_rate=_safe_rate(wins, sample_size),
                podium_rate=_safe_rate(podiums, sample_size),
                average_finish=_average_position(positions),
                average_odds=_safe_mean(odds_values),
                profit=round(profit_value, 2) if sample_size else None,
                roi=round(profit_value / sample_size, 3) if sample_size else None,
            )
        )

    overall_metric = OddsBucketMetrics(
        label="Global (toutes cotes)",
        sample_size=races_with_odds,
        wins=overall_wins,
        podiums=overall_podiums,
        win_rate=_safe_rate(overall_wins, races_with_odds),
        podium_rate=_safe_rate(overall_podiums, races_with_odds),
        average_finish=_average_position(overall_positions),
        average_odds=_safe_mean(overall_odds),
        profit=round(overall_profit, 2) if races_with_odds else None,
        roi=round(overall_profit / races_with_odds, 3) if races_with_odds else None,
    )

    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=min(race_dates) if race_dates else None,
        date_end=max(race_dates) if race_dates else None,
    )

    return AnalyticsOddsResponse(
        entity_type=entity_type,
        entity_id=entity_id,
        entity_label=entity_label,
        metadata=metadata,
        total_races=len(rows),
        races_with_odds=races_with_odds,
        races_without_odds=missing_odds,
        buckets=bucket_metrics,
        overall=overall_metric,
    )


@router.get(
    "/workload",
    response_model=AnalyticsWorkloadResponse,
    summary="Mesurer la charge de travail et les temps de repos d'une entité",
    description=(
        "Analyse les espacements entre les courses successives pour un cheval, un jockey ou un "
        "entraîneur afin de dégager des tendances d'activité (repos moyens, pics de participation)."
    ),
)
async def get_workload_profile(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)"
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité"
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer uniquement les courses disputées dans un hippodrome",
    ),
    start_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses à partir de cette date"
    ),
    end_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses jusqu'à cette date"
    ),
) -> AnalyticsWorkloadResponse:
    """Calcule les rythmes de participation récents d'une entité Aspiturf."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    rows = await client.get_races(
        horse_id=entity_id if entity_type == TrendEntityType.HORSE else None,
        jockey_id=entity_id if entity_type == TrendEntityType.JOCKEY else None,
        trainer_id=entity_id if entity_type == TrendEntityType.TRAINER else None,
        hippodrome=hippodrome,
        start_date=start_date,
        end_date=end_date,
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Aucune course trouvée pour cette entité et cette période",
        )

    # Tri chronologique pour pouvoir mesurer l'écart de jours entre deux engagements successifs.
    rows.sort(
        key=lambda item: (
            item.get("jour") if isinstance(item.get("jour"), date) else date.min,
            item.get("prix") if isinstance(item.get("prix"), int) else -1,
        )
    )

    label_fields: Tuple[str, ...]
    if entity_type == TrendEntityType.HORSE:
        label_fields = ("nom_cheval", "cheval")
    elif entity_type == TrendEntityType.JOCKEY:
        label_fields = ("jockey",)
    else:
        label_fields = ("entraineur",)

    entity_label = _extract_name(rows, label_fields)

    wins = sum(1 for row in rows if _is_win(row))
    podiums = sum(1 for row in rows if _is_podium(row))
    sample_size = len(rows)

    rest_values: List[int] = []
    rest_distribution: Dict[str, int] = {}
    timeline_entries: List[WorkloadTimelineEntry] = []
    race_dates: List[date] = []
    previous_date: Optional[date] = None

    for row in rows:
        race_date_value = row.get("jour")
        race_date = race_date_value if isinstance(race_date_value, date) else None
        if race_date is not None:
            race_dates.append(race_date)

        rest_days: Optional[int] = None
        if race_date is not None and previous_date is not None:
            delta_days = (race_date - previous_date).days
            if delta_days >= 0:
                rest_days = delta_days
                rest_values.append(delta_days)
                bucket = _rest_bucket(delta_days)
                rest_distribution[bucket] = rest_distribution.get(bucket, 0) + 1

        if race_date is not None:
            previous_date = race_date

        timeline_entries.append(
            WorkloadTimelineEntry(
                date=race_date,
                hippodrome=row.get("hippo"),
                course_number=row.get("prix") if isinstance(row.get("prix"), int) else None,
                distance=row.get("dist") if isinstance(row.get("dist"), int) else None,
                final_position=row.get("cl") if isinstance(row.get("cl"), int) else None,
                rest_days=rest_days,
                odds=_parse_float(row.get("cotedirect") or row.get("cote_direct")),
                is_win=_is_win(row),
                is_podium=_is_podium(row),
            )
        )

    first_date = min(race_dates) if race_dates else None
    last_date = max(race_dates) if race_dates else None

    races_last_30 = 0
    races_last_90 = 0
    if last_date is not None:
        thirty_limit = last_date - timedelta(days=30)
        ninety_limit = last_date - timedelta(days=90)
        for race_date in race_dates:
            if race_date >= thirty_limit:
                races_last_30 += 1
            if race_date >= ninety_limit:
                races_last_90 += 1

    average_rest = _safe_mean(rest_values)
    median_rest = round(median(rest_values), 1) if rest_values else None
    shortest_rest = min(rest_values) if rest_values else None
    longest_rest = max(rest_values) if rest_values else None

    average_monthly_races: Optional[float] = None
    if first_date is not None and last_date is not None:
        total_days = (last_date - first_date).days
        months_span = max(total_days / 30.0, 1.0)
        average_monthly_races = round(sample_size / months_span, 2)

    ordered_distribution = {
        label: rest_distribution[label]
        for label in ("0-7 jours", "8-14 jours", "15-30 jours", "31-60 jours", "60+ jours")
        if label in rest_distribution
    }

    summary = WorkloadSummary(
        sample_size=sample_size,
        wins=wins,
        podiums=podiums,
        win_rate=_safe_rate(wins, sample_size),
        podium_rate=_safe_rate(podiums, sample_size),
        average_rest_days=average_rest,
        median_rest_days=median_rest,
        shortest_rest_days=shortest_rest,
        longest_rest_days=longest_rest,
        races_last_30_days=races_last_30,
        races_last_90_days=races_last_90,
        average_monthly_races=average_monthly_races,
        rest_distribution=ordered_distribution,
    )

    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=first_date,
        date_end=last_date,
    )

    timeline_entries.sort(
        key=lambda entry: (
            entry.date if isinstance(entry.date, date) else date.min,
            entry.course_number or -1,
        ),
        reverse=True,
    )

    return AnalyticsWorkloadResponse(
        entity_type=entity_type,
        entity_id=entity_id,
        entity_label=entity_label,
        metadata=metadata,
        summary=summary,
        timeline=timeline_entries,
    )


@router.get(
    "/progression",
    response_model=AnalyticsProgressionResponse,
    summary="Mesurer la progression course par course",
    description=(
        "Analyse les variations de classement d'un cheval, jockey ou entraîneur sur ses courses "
        "successives afin d'identifier les phases d'amélioration ou de creux."
    ),
)
async def get_progression_analysis(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)"
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité"
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer uniquement les courses disputées dans un hippodrome",
    ),
    start_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses à partir de cette date"
    ),
    end_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses jusqu'à cette date"
    ),
) -> AnalyticsProgressionResponse:
    """Retourne une analyse des variations de position d'arrivée d'une entité."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    rows = await client.get_races(
        horse_id=entity_id if entity_type == TrendEntityType.HORSE else None,
        jockey_id=entity_id if entity_type == TrendEntityType.JOCKEY else None,
        trainer_id=entity_id if entity_type == TrendEntityType.TRAINER else None,
        hippodrome=hippodrome,
        start_date=start_date,
        end_date=end_date,
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Aucune course trouvée pour cette entité et cette période",
        )

    entries, summary = _progression_analysis(rows)
    meaningful_entries = [entry for entry in entries if entry.change is not None]

    if summary.races < 2 or not meaningful_entries:
        raise HTTPException(
            status_code=404,
            detail="Au moins deux courses avec position valide sont nécessaires pour analyser la progression",
        )

    if entity_type == TrendEntityType.HORSE:
        label_fields: Tuple[str, ...] = ("nom_cheval", "cheval")
    elif entity_type == TrendEntityType.JOCKEY:
        label_fields = ("jockey",)
    else:
        label_fields = ("entraineur",)

    entity_label = _extract_name(rows, label_fields)

    dates = [entry.date for entry in entries if isinstance(entry.date, date)]
    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=min(dates) if dates else None,
        date_end=max(dates) if dates else None,
    )

    ordered_entries = sorted(
        entries,
        key=lambda item: (
            item.date if isinstance(item.date, date) else date.min,
            item.course_number or -1,
        ),
        reverse=True,
    )

    return AnalyticsProgressionResponse(
        entity_type=entity_type,
        entity_id=entity_id,
        entity_label=entity_label,
        metadata=metadata,
        summary=summary,
        races=ordered_entries,
    )


@router.get(
    "/momentum",
    response_model=AnalyticsMomentumResponse,
    summary="Comparer la dynamique récente à une période de référence",
    description=(
        "Analyse la forme actuelle d'une entité Aspiturf en la comparant à la période précédente "
        "afin de détecter un changement de tendance sur les dernières courses."
    ),
)
async def get_momentum_profile(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)"
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité"
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer uniquement les courses disputées dans un hippodrome",
    ),
    start_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses à partir de cette date"
    ),
    end_date: Optional[date] = Query(
        default=None, description="Inclure uniquement les courses jusqu'à cette date"
    ),
    window: int = Query(
        5,
        ge=1,
        le=50,
        description="Nombre de courses à inclure dans la fenêtre récente",
    ),
    baseline_window: Optional[int] = Query(
        default=None,
        ge=1,
        le=50,
        description=(
            "Nombre de courses à utiliser pour la période de référence. "
            "Par défaut, la même taille que la fenêtre récente est utilisée."
        ),
    ),
) -> AnalyticsMomentumResponse:
    """Retourne un comparatif de momentum basé sur deux fenêtres glissantes."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    rows = await client.get_races(
        horse_id=entity_id if entity_type == TrendEntityType.HORSE else None,
        jockey_id=entity_id if entity_type == TrendEntityType.JOCKEY else None,
        trainer_id=entity_id if entity_type == TrendEntityType.TRAINER else None,
        hippodrome=hippodrome,
        start_date=start_date,
        end_date=end_date,
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Aucune course trouvée pour cette entité et cette période",
        )

    sorted_rows = _sorted_races(rows)

    window_size = max(1, min(window, len(sorted_rows)))
    reference_size = baseline_window or window_size

    recent_rows = sorted_rows[:window_size]
    reference_rows = sorted_rows[window_size : window_size + reference_size]

    if entity_type == TrendEntityType.HORSE:
        label_fields: Tuple[str, ...] = ("nom_cheval", "cheval")
    elif entity_type == TrendEntityType.JOCKEY:
        label_fields = ("jockey",)
    else:
        label_fields = ("entraineur",)

    entity_label = _extract_name(sorted_rows, label_fields)

    recent_label = (
        "Dernière course" if len(recent_rows) == 1 else f"Dernières {len(recent_rows)} courses"
    )
    reference_label = None
    if reference_rows:
        reference_label = (
            "Période précédente"
            if len(reference_rows) == 1
            else f"Période précédente ({len(reference_rows)} courses)"
        )

    recent_window = _momentum_slice(recent_label, recent_rows)
    reference_window = (
        _momentum_slice(reference_label, reference_rows) if reference_rows and reference_label else None
    )
    deltas = _momentum_shift(recent_window, reference_window)

    all_dates = [row.get("jour") for row in sorted_rows if isinstance(row.get("jour"), date)]
    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=min(all_dates) if all_dates else None,
        date_end=max(all_dates) if all_dates else None,
    )

    return AnalyticsMomentumResponse(
        entity_type=entity_type,
        entity_id=entity_id,
        entity_label=entity_label,
        metadata=metadata,
        recent_window=recent_window,
        reference_window=reference_window,
        deltas=deltas,
    )


@router.get(
    "/streaks",
    response_model=AnalyticsStreakResponse,
    summary="Analyser les séries de résultats d'une entité",
    description=(
        "Identifie les séries de victoires/podiums consécutifs pour un cheval, un jockey ou un entraîneur."
    ),
)
async def get_performance_streaks(
    *,
    client: AspiturfClient = Depends(get_aspiturf_client),
    entity_type: TrendEntityType = Query(
        ..., description="Type d'entité à analyser (cheval, jockey ou entraîneur)"
    ),
    entity_id: str = Query(
        ..., min_length=1, description="Identifiant Aspiturf de l'entité"
    ),
    hippodrome: Optional[str] = Query(
        default=None,
        description="Filtrer uniquement les courses disputées dans un hippodrome",
    ),
    start_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses à partir de cette date",
    ),
    end_date: Optional[date] = Query(
        default=None,
        description="Inclure uniquement les courses jusqu'à cette date",
    ),
) -> AnalyticsStreakResponse:
    """Retourne les principales séries de résultats observées pour une entité."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date doit être antérieure ou égale à end_date",
        )

    payload = await client.performance_streaks(
        entity_type=entity_type.value,
        entity_id=entity_id,
        start_date=start_date,
        end_date=end_date,
        hippodrome=hippodrome,
    )

    total_races = int(payload.get("total_races") or 0)
    if total_races == 0:
        raise HTTPException(
            status_code=404,
            detail="Aucune course trouvée pour cette entité et cette période",
        )

    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=payload.get("date_start"),
        date_end=payload.get("date_end"),
    )

    best_win = _to_streak(payload.get("best_win"), "win")
    best_podium = _to_streak(payload.get("best_podium"), "podium")
    current_win = _to_streak(payload.get("current_win"), "win")
    current_podium = _to_streak(payload.get("current_podium"), "podium")

    history_items: List[PerformanceStreak] = []
    for item in payload.get("history", []):
        streak = _to_streak(item, item.get("type", "win"))  # type: ignore[arg-type]
        if streak is not None:
            history_items.append(streak)

    return AnalyticsStreakResponse(
        entity_type=entity_type,
        entity_id=payload.get("entity_id", entity_id),
        entity_label=payload.get("entity_label"),
        metadata=metadata,
        total_races=total_races,
        wins=int(payload.get("wins") or 0),
        podiums=int(payload.get("podiums") or 0),
        best_win_streak=best_win,
        best_podium_streak=best_podium,
        current_win_streak=current_win,
        current_podium_streak=current_podium,
        streak_history=history_items,
    )


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


def _aggregate_comparison(
    *,
    entity_type: AnalyticsSearchType,
    rows_by_entity: Dict[str, List[Dict[str, Any]]],
    hippodrome: Optional[str],
    start_date: Optional[date],
    end_date: Optional[date],
) -> Tuple[List[ComparisonEntitySummary], int, AnalyticsMetadata]:
    """Construit les statistiques comparatives et le métadata associés."""

    if entity_type == AnalyticsSearchType.HORSE:
        label_fields = ("nom_cheval", "cheval")
    elif entity_type == AnalyticsSearchType.JOCKEY:
        label_fields = ("jockey",)
    else:
        label_fields = ("entraineur",)

    observed_start: Optional[date] = None
    observed_end: Optional[date] = None

    # Construit un index des courses afin de calculer les confrontations directes.
    race_participations: Dict[Tuple[Any, Any, Any], Dict[str, Dict[str, Any]]] = {}

    for entity_id, rows in rows_by_entity.items():
        for row in rows:
            race_date = row.get("jour")
            if isinstance(race_date, date):
                if observed_start is None or race_date < observed_start:
                    observed_start = race_date
                if observed_end is None or race_date > observed_end:
                    observed_end = race_date

            race_key = (
                row.get("jour"),
                row.get("hippo"),
                row.get("prix") or row.get("numcourse") or row.get("course"),
            )
            race_participations.setdefault(race_key, {})[entity_id] = row

    shared_races = sum(1 for rows in race_participations.values() if len(rows) >= 2)

    pair_stats: Dict[Tuple[str, str], Dict[str, int]] = {}

    for race_rows in race_participations.values():
        if len(race_rows) < 2:
            continue

        for left_id, right_id in combinations(race_rows.keys(), 2):
            left_row = race_rows[left_id]
            right_row = race_rows[right_id]

            left_stats = pair_stats.setdefault(
                (left_id, right_id), {"meetings": 0, "ahead": 0, "behind": 0, "ties": 0}
            )
            right_stats = pair_stats.setdefault(
                (right_id, left_id), {"meetings": 0, "ahead": 0, "behind": 0, "ties": 0}
            )

            left_stats["meetings"] += 1
            right_stats["meetings"] += 1

            left_pos = left_row.get("cl") if isinstance(left_row.get("cl"), int) else None
            right_pos = right_row.get("cl") if isinstance(right_row.get("cl"), int) else None

            if left_pos is None or right_pos is None:
                left_stats["ties"] += 1
                right_stats["ties"] += 1
            elif left_pos < right_pos:
                left_stats["ahead"] += 1
                right_stats["behind"] += 1
            elif left_pos > right_pos:
                left_stats["behind"] += 1
                right_stats["ahead"] += 1
            else:
                left_stats["ties"] += 1
                right_stats["ties"] += 1

    summaries: List[ComparisonEntitySummary] = []

    for entity_id, rows in rows_by_entity.items():
        sample_size = len(rows)
        wins = sum(1 for row in rows if _is_win(row))
        podiums = sum(1 for row in rows if _is_podium(row))
        positions = [row.get("cl") for row in rows if isinstance(row.get("cl"), int)]

        label = _extract_name(rows, label_fields)
        best_finish = min(positions) if positions else None
        last_seen = None

        for row in rows:
            race_date = row.get("jour")
            if isinstance(race_date, date):
                if last_seen is None or race_date > last_seen:
                    last_seen = race_date

        matchups = [
            HeadToHeadBreakdown(
                opponent_id=opponent,
                meetings=stats.get("meetings", 0),
                ahead=stats.get("ahead", 0),
                behind=stats.get("behind", 0),
                ties=stats.get("ties", 0),
            )
            for (source, opponent), stats in pair_stats.items()
            if source == entity_id
        ]

        matchups.sort(key=lambda item: (-item.meetings, item.opponent_id))

        summaries.append(
            ComparisonEntitySummary(
                entity_id=entity_id,
                label=label,
                sample_size=sample_size,
                wins=wins,
                podiums=podiums,
                win_rate=_safe_rate(wins, sample_size),
                podium_rate=_safe_rate(podiums, sample_size),
                average_finish=_safe_mean(positions),
                best_finish=best_finish,
                last_seen=last_seen,
                head_to_head=matchups,
            )
        )

    summaries.sort(key=lambda item: (item.entity_id))

    metadata = AnalyticsMetadata(
        hippodrome_filter=hippodrome,
        date_start=observed_start or start_date,
        date_end=observed_end or end_date,
    )

    return summaries, shared_races, metadata


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

