# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""Schémas Pydantic pour les endpoints d'analytics Aspiturf."""

from enum import Enum
from datetime import date as date_type
from typing import Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class AnalyticsSearchType(str, Enum):
    """Types d'entités disponibles pour la recherche analytics."""

    HORSE = "horse"
    JOCKEY = "jockey"
    TRAINER = "trainer"
    HIPPODROME = "hippodrome"


class AnalyticsSearchMetadata(BaseModel):
    """Métadonnées accompagnant un résultat de recherche analytics."""

    total_races: Optional[int] = Field(
        default=None,
        ge=0,
        description="Nombre total de courses trouvées pour l'entité",
    )
    hippodromes: List[str] = Field(
        default_factory=list,
        description="Liste des hippodromes les plus fréquents",
    )
    last_seen: Optional[date_type] = Field(
        default=None,
        description="Dernière apparition de l'entité dans les données",
    )
    course_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Nombre de courses correspondant à l'hippodrome",
    )
    last_meeting: Optional[date_type] = Field(
        default=None,
        description="Dernière réunion disponible pour l'hippodrome",
    )
    disciplines: List[str] = Field(
        default_factory=list,
        description="Disciplines ou types de courses associés",
    )


class AnalyticsSearchResult(BaseModel):
    """Résultat retourné par l'endpoint de recherche analytics."""

    type: AnalyticsSearchType = Field(..., description="Type d'entité trouvée")
    id: str = Field(..., description="Identifiant principal de l'entité")
    label: str = Field(..., description="Libellé lisible par un humain")
    metadata: AnalyticsSearchMetadata = Field(
        default_factory=AnalyticsSearchMetadata,
        description="Métadonnées complémentaires sur l'entité",
    )


class AnalyticsMetadata(BaseModel):
    """Métadonnées communes aux réponses analytics."""

    hippodrome_filter: Optional[str] = Field(
        default=None,
        description="Filtre d'hippodrome appliqué sur les données"
    )
    date_start: Optional[date_type] = Field(
        default=None,
        description="Première date trouvée dans l'échantillon"
    )
    date_end: Optional[date_type] = Field(
        default=None,
        description="Dernière date trouvée dans l'échantillon"
    )


class PerformanceBreakdown(BaseModel):
    """Statistiques agrégées par catégorie (hippodrome, distance, etc.)."""

    label: str = Field(..., description="Libellé de la catégorie")
    total: int = Field(..., ge=0, description="Nombre de courses dans la catégorie")
    wins: int = Field(..., ge=0, description="Nombre de victoires")
    podiums: int = Field(..., ge=0, description="Nombre de podiums (top 3)")
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de victoire (0-1)"
    )
    podium_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de podium (0-1)"
    )


class RecentRace(BaseModel):
    """Informations synthétiques sur une course récente."""

    date: Optional[date_type] = Field(default=None, description="Date de la course")
    hippodrome: Optional[str] = Field(default=None, description="Nom de l'hippodrome")
    course_number: Optional[int] = Field(default=None, description="Numéro de la course")
    distance: Optional[int] = Field(default=None, description="Distance en mètres")
    final_position: Optional[int] = Field(default=None, description="Position finale")
    odds: Optional[float] = Field(default=None, description="Cote PMU relevée")
    is_win: bool = Field(..., description="Indique si la course est une victoire")
    is_podium: bool = Field(..., description="Indique si la course est un podium (top 3)")


class FormRace(RecentRace):
    """Course récente agrémentée d'un score de forme (0-5)."""

    score: int = Field(
        ...,
        ge=0,
        le=5,
        description="Indice de forme pondéré en fonction de la position finale",
    )


class PerformanceSummary(BaseModel):
    """Résumé rapide de performances (cheval, jockey, entraineur, couple)."""

    sample_size: Optional[int] = Field(default=None, ge=0, description="Nombre de courses")
    wins: Optional[int] = Field(default=None, ge=0, description="Nombre de victoires")
    places: Optional[int] = Field(default=None, ge=0, description="Nombre de places")
    win_rate: Optional[float] = Field(default=None, ge=0, le=1, description="Taux de victoire")
    place_rate: Optional[float] = Field(default=None, ge=0, le=1, description="Taux de place")


class LeaderboardEntry(BaseModel):
    """Entrée de classement synthétique pour une entité Aspiturf."""

    entity_id: str = Field(
        ...,
        description="Identifiant unique de l'entité (cheval, jockey ou entraineur)",
    )
    label: str = Field(..., description="Nom lisible de l'entité")
    sample_size: int = Field(..., ge=0, description="Nombre de courses étudiées")
    wins: int = Field(..., ge=0, description="Nombre de victoires dans l'échantillon")
    podiums: int = Field(..., ge=0, description="Nombre de podiums dans l'échantillon")
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de victoire (0-1) calculé sur l'échantillon",
    )
    podium_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de podium (0-1) calculé sur l'échantillon",
    )
    average_finish: Optional[float] = Field(
        default=None,
        description="Position moyenne à l'arrivée sur la période",
    )
    last_seen: Optional[date_type] = Field(
        default=None,
        description="Dernière date observée pour l'entité dans la période analysée",
    )


class HeadToHeadBreakdown(BaseModel):
    """Bilan d'une confrontation directe avec une entité opposée."""

    opponent_id: str = Field(..., description="Identifiant de l'adversaire comparé")
    meetings: int = Field(
        ...,
        ge=0,
        description="Nombre de courses communes observées avec l'adversaire",
    )
    ahead: int = Field(
        ...,
        ge=0,
        description="Occurrences où l'entité termine devant l'adversaire",
    )
    behind: int = Field(
        ...,
        ge=0,
        description="Occurrences où l'entité termine derrière l'adversaire",
    )
    ties: int = Field(
        ...,
        ge=0,
        description="Courses sans classement exploitable ou positions identiques",
    )


class ComparisonEntitySummary(BaseModel):
    """Résumé statistique d'une entité dans une comparaison multi-id."""

    entity_id: str = Field(..., description="Identifiant unique de l'entité")
    label: Optional[str] = Field(
        default=None,
        description="Libellé lisible de l'entité (nom ou alias)",
    )
    sample_size: int = Field(..., ge=0, description="Nombre de courses analysées")
    wins: int = Field(..., ge=0, description="Nombre de victoires sur la période")
    podiums: int = Field(..., ge=0, description="Nombre de podiums sur la période")
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de victoire rapporté à l'échantillon",
    )
    podium_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de podium rapporté à l'échantillon",
    )
    average_finish: Optional[float] = Field(
        default=None,
        description="Position moyenne à l'arrivée",
    )
    best_finish: Optional[int] = Field(
        default=None,
        ge=1,
        description="Meilleure place obtenue dans la période",
    )
    last_seen: Optional[date_type] = Field(
        default=None,
        description="Dernière date observée pour l'entité",
    )
    head_to_head: List[HeadToHeadBreakdown] = Field(
        default_factory=list,
        description="Bilan des confrontations directes contre les autres entités",
    )


class AnalyticsComparisonResponse(BaseModel):
    """Réponse retournée par le comparateur d'entités analytics."""

    entity_type: AnalyticsSearchType = Field(
        ..., description="Type d'entités comparées (cheval, jockey ou entraîneur)"
    )
    shared_races: int = Field(
        ...,
        ge=0,
        description="Nombre de courses où au moins deux entités étaient présentes",
    )
    entities: List[ComparisonEntitySummary] = Field(
        ...,
        description="Résumés statistiques par entité sélectionnée",
    )
    metadata: AnalyticsMetadata = Field(
        default_factory=AnalyticsMetadata,
        description="Plage temporelle et filtres appliqués",
    )


class HorseAnalyticsResponse(BaseModel):
    """Réponse analytics pour un cheval."""

    horse_id: str = Field(..., description="Identifiant Aspiturf du cheval")
    horse_name: Optional[str] = Field(default=None, description="Nom du cheval")
    sample_size: int = Field(..., ge=0, description="Nombre de courses analysées")
    wins: int = Field(..., ge=0, description="Nombre de victoires")
    podiums: int = Field(..., ge=0, description="Nombre de podiums (top 3)")
    win_rate: Optional[float] = Field(default=None, ge=0, le=1)
    podium_rate: Optional[float] = Field(default=None, ge=0, le=1)
    average_finish: Optional[float] = Field(
        default=None,
        description="Position moyenne à l'arrivée"
    )
    average_odds: Optional[float] = Field(default=None, description="Cote moyenne observée")
    recent_results: List[RecentRace] = Field(default_factory=list)
    hippodrome_breakdown: List[PerformanceBreakdown] = Field(default_factory=list)
    distance_breakdown: List[PerformanceBreakdown] = Field(default_factory=list)
    metadata: AnalyticsMetadata = Field(default_factory=AnalyticsMetadata)


class JockeyAnalyticsResponse(BaseModel):
    """Réponse analytics pour un jockey."""

    jockey_id: str = Field(..., description="Identifiant Aspiturf du jockey")
    jockey_name: Optional[str] = Field(default=None, description="Nom du jockey")
    sample_size: int = Field(..., ge=0)
    wins: int = Field(..., ge=0)
    podiums: int = Field(..., ge=0)
    win_rate: Optional[float] = Field(default=None, ge=0, le=1)
    podium_rate: Optional[float] = Field(default=None, ge=0, le=1)
    average_finish: Optional[float] = Field(default=None)
    recent_results: List[RecentRace] = Field(default_factory=list)
    horse_breakdown: List[PerformanceBreakdown] = Field(default_factory=list)
    hippodrome_breakdown: List[PerformanceBreakdown] = Field(default_factory=list)
    metadata: AnalyticsMetadata = Field(default_factory=AnalyticsMetadata)


class TrainerAnalyticsResponse(BaseModel):
    """Réponse analytics pour un entraineur."""

    trainer_id: str = Field(..., description="Identifiant Aspiturf de l'entraineur")
    trainer_name: Optional[str] = Field(default=None, description="Nom de l'entraineur")
    sample_size: int = Field(..., ge=0)
    wins: int = Field(..., ge=0)
    podiums: int = Field(..., ge=0)
    win_rate: Optional[float] = Field(default=None, ge=0, le=1)
    podium_rate: Optional[float] = Field(default=None, ge=0, le=1)
    average_finish: Optional[float] = Field(default=None)
    recent_results: List[RecentRace] = Field(default_factory=list)
    horse_breakdown: List[PerformanceBreakdown] = Field(default_factory=list)
    hippodrome_breakdown: List[PerformanceBreakdown] = Field(default_factory=list)
    metadata: AnalyticsMetadata = Field(default_factory=AnalyticsMetadata)


class CoupleAnalyticsResponse(BaseModel):
    """Réponse analytics pour un couple cheval-jockey."""

    horse_id: str = Field(...)
    jockey_id: str = Field(...)
    horse_name: Optional[str] = Field(default=None)
    jockey_name: Optional[str] = Field(default=None)
    sample_size: int = Field(..., ge=0)
    wins: int = Field(..., ge=0)
    podiums: int = Field(..., ge=0)
    win_rate: Optional[float] = Field(default=None, ge=0, le=1)
    podium_rate: Optional[float] = Field(default=None, ge=0, le=1)
    average_finish: Optional[float] = Field(default=None)
    recent_results: List[RecentRace] = Field(default_factory=list)
    metadata: AnalyticsMetadata = Field(default_factory=AnalyticsMetadata)


class PartantInsight(BaseModel):
    """Informations détaillées pour un partant d'une course."""

    numero: Optional[int] = Field(default=None, description="Numéro du partant")
    horse_id: Optional[str] = Field(default=None)
    horse_name: Optional[str] = Field(default=None)
    jockey_id: Optional[str] = Field(default=None)
    jockey_name: Optional[str] = Field(default=None)
    trainer_id: Optional[str] = Field(default=None)
    trainer_name: Optional[str] = Field(default=None)
    odds: Optional[float] = Field(default=None, description="Cote PMU actuelle")
    probable_odds: Optional[float] = Field(default=None, description="Cote probable")
    recent_form: Optional[str] = Field(default=None, description="Musique du cheval")
    days_since_last_race: Optional[int] = Field(default=None, ge=0)
    handicap_value: Optional[int] = Field(default=None)
    jockey_stats: Optional[PerformanceSummary] = Field(default=None)
    trainer_stats: Optional[PerformanceSummary] = Field(default=None)
    horse_stats: Optional[PerformanceSummary] = Field(default=None)
    couple_stats: Optional[PerformanceSummary] = Field(default=None)


class CourseAnalyticsResponse(BaseModel):
    """Réponse détaillée pour une course Aspiturf spécifique."""

    date: date_type = Field(..., description="Date de la course")
    hippodrome: str = Field(..., description="Nom de l'hippodrome")
    course_number: int = Field(..., ge=1, description="Numéro de la course")
    distance: Optional[int] = Field(default=None, description="Distance en mètres")
    discipline: Optional[str] = Field(default=None, description="Type de course Aspiturf")
    allocation: Optional[float] = Field(default=None, description="Allocation totale")
    currency: Optional[str] = Field(default=None, description="Devise de l'allocation")
    partants: List[PartantInsight] = Field(default_factory=list)
    metadata: AnalyticsMetadata = Field(default_factory=AnalyticsMetadata)


class AnalyticsInsightsResponse(BaseModel):
    """Réponse agrégée pour les classements trans-entités."""

    metadata: AnalyticsMetadata = Field(default_factory=AnalyticsMetadata)
    top_horses: List[LeaderboardEntry] = Field(
        default_factory=list,
        description="Classement des chevaux les plus performants",
    )
    top_jockeys: List[LeaderboardEntry] = Field(
        default_factory=list,
        description="Classement des jockeys les plus performants",
    )
    top_trainers: List[LeaderboardEntry] = Field(
        default_factory=list,
        description="Classement des entraineurs les plus performants",
    )


class TrendGranularity(str, Enum):
    """Résolutions temporelles disponibles pour les tendances de performance."""

    WEEK = "week"
    MONTH = "month"


class TrendEntityType(str, Enum):
    """Entités compatibles avec l'analyse de tendance."""

    HORSE = "horse"
    JOCKEY = "jockey"
    TRAINER = "trainer"


class PerformanceTrendPoint(BaseModel):
    """Mesure agrégée sur une période temporelle."""

    period_start: date_type = Field(..., description="Date de début de la période")
    period_end: date_type = Field(..., description="Date de fin de la période")
    label: str = Field(..., description="Libellé lisible de la période (AAAA-MM, AAAA-Sxx)")
    races: int = Field(..., ge=0, description="Nombre de courses disputées")
    wins: int = Field(..., ge=0, description="Nombre de victoires sur la période")
    podiums: int = Field(..., ge=0, description="Nombre de podiums sur la période")
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de victoire sur la période",
    )
    podium_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de podium sur la période",
    )
    average_finish: Optional[float] = Field(
        default=None,
        description="Position moyenne à l'arrivée",
    )
    average_odds: Optional[float] = Field(
        default=None,
        description="Cote moyenne observée (directe ou probable)",
    )


class PerformanceTrendResponse(BaseModel):
    """Réponse retournée par l'endpoint /analytics/trends."""

    entity_type: TrendEntityType = Field(..., description="Type de l'entité analysée")
    entity_id: str = Field(..., description="Identifiant Aspiturf de l'entité")
    entity_label: Optional[str] = Field(default=None, description="Libellé humain de l'entité")
    granularity: TrendGranularity = Field(..., description="Granularité choisie pour l'agrégation")
    metadata: AnalyticsMetadata = Field(default_factory=AnalyticsMetadata)
    points: List[PerformanceTrendPoint] = Field(
        default_factory=list,
        description="Liste des mesures agrégées par période",
    )


class PerformanceStreak(BaseModel):
    """Description d'une série de résultats consécutifs."""

    type: Literal["win", "podium"] = Field(
        ...,
        description="Nature de la série (victoires ou podiums)",
    )
    length: int = Field(
        ..., ge=1, description="Nombre d'occurrences consécutives dans la série"
    )
    start_date: Optional[date_type] = Field(
        default=None,
        description="Date de début de la série",
    )
    end_date: Optional[date_type] = Field(
        default=None,
        description="Date de fin de la série",
    )
    is_active: bool = Field(
        default=False,
        description="Indique si la série est toujours en cours",
    )


class AnalyticsStreakResponse(BaseModel):
    """Réponse synthétique pour les séries de résultats d'une entité."""

    entity_type: TrendEntityType = Field(..., description="Type d'entité analysée")
    entity_id: str = Field(..., description="Identifiant Aspiturf de l'entité")
    entity_label: Optional[str] = Field(
        default=None,
        description="Nom lisible de l'entité (si disponible)",
    )
    metadata: AnalyticsMetadata = Field(default_factory=AnalyticsMetadata)
    total_races: int = Field(
        ..., ge=0, description="Nombre de courses retenues pour l'analyse"
    )
    wins: int = Field(..., ge=0, description="Total de victoires sur l'échantillon")
    podiums: int = Field(
        ..., ge=0, description="Total de podiums sur l'échantillon"
    )
    best_win_streak: Optional[PerformanceStreak] = Field(
        default=None,
        description="Meilleure série de victoires observée",
    )
    best_podium_streak: Optional[PerformanceStreak] = Field(
        default=None,
        description="Meilleure série de podiums observée",
    )
    current_win_streak: Optional[PerformanceStreak] = Field(
        default=None,
        description="Série de victoires en cours (si existante)",
    )
    current_podium_streak: Optional[PerformanceStreak] = Field(
        default=None,
        description="Série de podiums en cours (si existante)",
    )
    streak_history: List[PerformanceStreak] = Field(
        default_factory=list,
        description="Historique des principales séries détectées",
    )


class DistributionDimension(str, Enum):
    """Dimensions disponibles pour l'analyse de distribution."""

    DISTANCE = "distance"
    DRAW = "draw"
    HIPPODROME = "hippodrome"
    DISCIPLINE = "discipline"


class DistributionBucket(BaseModel):
    """Agrégat statistique pour une catégorie de distribution."""

    label: str = Field(..., description="Libellé de la catégorie agrégée")
    races: int = Field(..., ge=0, description="Nombre de courses dans la catégorie")
    wins: int = Field(..., ge=0, description="Nombre de victoires dans la catégorie")
    podiums: int = Field(..., ge=0, description="Nombre de podiums dans la catégorie")
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de victoire (0-1) calculé sur la catégorie",
    )
    podium_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de podium (0-1) calculé sur la catégorie",
    )
    average_finish: Optional[float] = Field(
        default=None,
        description="Position moyenne à l'arrivée dans la catégorie",
    )
    average_odds: Optional[float] = Field(
        default=None,
        description="Cote moyenne observée dans la catégorie",
    )


class PerformanceDistributionResponse(BaseModel):
    """Réponse pour l'analyse des distributions de performance."""

    entity_type: TrendEntityType = Field(
        ..., description="Type d'entité analysée (cheval, jockey ou entraineur)"
    )
    entity_id: str = Field(..., description="Identifiant Aspiturf de l'entité analysée")
    entity_label: Optional[str] = Field(
        default=None, description="Nom lisible de l'entité analysée"
    )
    dimension: DistributionDimension = Field(
        ..., description="Dimension d'agrégation utilisée pour la distribution"
    )
    metadata: AnalyticsMetadata = Field(
        default_factory=AnalyticsMetadata,
        description="Métadonnées décrivant les filtres appliqués",
    )
    buckets: List[DistributionBucket] = Field(
        default_factory=list,
        description="Liste des agrégats calculés par catégorie",
    )


class SeasonalityGranularity(str, Enum):
    """Granularité disponible pour l'analyse de saisonnalité."""

    MONTH = "month"
    WEEKDAY = "weekday"


class SeasonalityBucket(BaseModel):
    """Statistiques agrégées pour une période saisonnière (mois ou jour)."""

    key: str = Field(..., description="Identifiant technique du seau de saisonnalité")
    label: str = Field(..., description="Libellé lisible par un humain (ex: 'Mars')")
    races: int = Field(..., ge=0, description="Nombre de courses observées")
    wins: int = Field(..., ge=0, description="Nombre de victoires dans la période")
    podiums: int = Field(..., ge=0, description="Nombre de podiums dans la période")
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de victoire calculé sur le seau",
    )
    podium_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de podium calculé sur le seau",
    )
    average_finish: Optional[float] = Field(
        default=None,
        description="Position moyenne à l'arrivée sur la période",
    )
    average_odds: Optional[float] = Field(
        default=None,
        description="Cote moyenne observée sur la période",
    )


class AnalyticsSeasonalityResponse(BaseModel):
    """Réponse de l'endpoint /analytics/seasonality."""

    entity_type: TrendEntityType = Field(..., description="Type d'entité analysée")
    entity_id: str = Field(..., description="Identifiant unique de l'entité analysée")
    entity_label: Optional[str] = Field(
        default=None, description="Libellé lisible de l'entité s'il est disponible"
    )
    granularity: SeasonalityGranularity = Field(
        ..., description="Granularité temporelle choisie (mois ou jour de semaine)"
    )
    metadata: AnalyticsMetadata = Field(
        default_factory=AnalyticsMetadata,
        description="Métadonnées associées aux filtres et à la période analysée",
    )
    total_races: int = Field(..., ge=0, description="Nombre total de courses étudiées")
    total_wins: int = Field(..., ge=0, description="Nombre total de victoires enregistrées")
    total_podiums: int = Field(..., ge=0, description="Nombre total de podiums enregistrés")
    buckets: List[SeasonalityBucket] = Field(
        default_factory=list,
        description="Détail des performances par période de saisonnalité",
    )


class CalendarRaceDetail(BaseModel):
    """Détail d'une course utilisée dans le calendrier de performances."""

    hippodrome: Optional[str] = Field(
        default=None,
        description="Libellé de l'hippodrome ayant accueilli la course",
    )
    course_number: Optional[int] = Field(
        default=None,
        description="Numéro officiel de la course dans la réunion",
    )
    distance: Optional[int] = Field(
        default=None,
        description="Distance officielle de la course en mètres",
    )
    final_position: Optional[int] = Field(
        default=None,
        description="Classement final de l'entité sur cette course",
    )
    odds: Optional[float] = Field(
        default=None,
        description="Cote observée (rapport PMU ou cote probable)",
    )


class CalendarDaySummary(BaseModel):
    """Synthèse des performances sur une journée donnée."""

    date: date_type = Field(..., description="Date de la réunion analysée")
    hippodromes: List[str] = Field(
        default_factory=list,
        description="Liste des hippodromes concernés sur la journée",
    )
    races: int = Field(..., ge=0, description="Nombre de courses disputées")
    wins: int = Field(..., ge=0, description="Nombre de victoires sur la journée")
    podiums: int = Field(..., ge=0, description="Nombre de podiums sur la journée")
    average_finish: Optional[float] = Field(
        default=None,
        description="Position moyenne enregistrée sur les courses disputées",
    )
    average_odds: Optional[float] = Field(
        default=None,
        description="Cote moyenne relevée sur la journée",
    )
    race_details: List[CalendarRaceDetail] = Field(
        default_factory=list,
        description="Détail des courses prises en compte",
    )


class AnalyticsCalendarResponse(BaseModel):
    """Réponse structurée pour l'endpoint /analytics/calendar."""

    entity_type: TrendEntityType = Field(
        ..., description="Type de l'entité analysée (cheval, jockey ou entraîneur)"
    )
    entity_id: str = Field(..., description="Identifiant Aspiturf de l'entité")
    entity_label: Optional[str] = Field(
        default=None,
        description="Libellé humain de l'entité analysée",
    )
    metadata: AnalyticsMetadata = Field(default_factory=AnalyticsMetadata)
    total_races: int = Field(..., ge=0, description="Nombre total de courses agrégées")
    total_wins: int = Field(..., ge=0, description="Nombre total de victoires")
    total_podiums: int = Field(..., ge=0, description="Nombre total de podiums")
    days: List[CalendarDaySummary] = Field(
        default_factory=list,
        description="Liste des journées analysées avec leurs statistiques",
    )


class AnalyticsFormResponse(BaseModel):
    """Résumé synthétique de la forme récente d'une entité Aspiturf."""

    entity_type: TrendEntityType = Field(
        ..., description="Type d'entité analysée (cheval, jockey ou entraîneur)"
    )
    entity_id: str = Field(..., description="Identifiant Aspiturf de l'entité")
    entity_label: Optional[str] = Field(
        default=None, description="Libellé lisible associé à l'entité"
    )
    window: int = Field(
        ..., ge=1, le=50, description="Nombre de courses prises en compte pour la forme"
    )
    metadata: AnalyticsMetadata = Field(
        default_factory=AnalyticsMetadata,
        description="Métadonnées relatives aux filtres appliqués",
    )
    total_races: int = Field(
        ..., ge=0, description="Nombre de courses réellement analysées dans la fenêtre"
    )
    wins: int = Field(..., ge=0, description="Victoires recensées dans la fenêtre")
    podiums: int = Field(
        ..., ge=0, description="Podiums recensés dans la fenêtre"
    )
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de victoire sur l'échantillon (0-1)",
    )
    podium_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de podium sur l'échantillon (0-1)",
    )
    average_finish: Optional[float] = Field(
        default=None, description="Position moyenne à l'arrivée"
    )
    average_odds: Optional[float] = Field(
        default=None, description="Cote moyenne observée"
    )
    median_odds: Optional[float] = Field(
        default=None,
        description="Médiane des cotes relevées dans la fenêtre",
    )
    best_position: Optional[int] = Field(
        default=None,
        description="Meilleure position obtenue sur la période étudiée",
    )
    consistency_index: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Indice de constance compris entre 0 (irrégulier) et 1 (très stable)",
    )
    form_score: Optional[float] = Field(
        default=None,
        description="Score de forme moyen (0-5) calculé à partir des positions finales",
    )
    races: List[FormRace] = Field(
        default_factory=list,
        description="Historique des courses triées de la plus récente à la plus ancienne",
    )


class ValueOpportunitySample(BaseModel):
    """Détail d'une course présentant un potentiel écart de cote."""

    date: Optional[date_type] = Field(
        default=None, description="Date de la course analysée pour l'entité"
    )
    hippodrome: Optional[str] = Field(
        default=None, description="Nom de l'hippodrome ayant accueilli la course"
    )
    course_number: Optional[int] = Field(
        default=None, description="Numéro de course dans la réunion"
    )
    distance: Optional[int] = Field(
        default=None, description="Distance officielle de l'épreuve en mètres"
    )
    final_position: Optional[int] = Field(
        default=None, description="Classement final enregistré dans la ligne Aspiturf"
    )
    odds_actual: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cote directe observée (rapport PMU) si disponible",
    )
    odds_implied: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cote probable estimée dans les données Aspiturf",
    )
    edge: Optional[float] = Field(
        default=None,
        description="Différence entre la cote probable et la cote observée",
    )
    is_win: Optional[bool] = Field(
        default=None, description="Indique si l'entité a remporté la course"
    )
    profit: Optional[float] = Field(
        default=None,
        description="Gain/perte théorique pour une mise unitaire en cas de cote disponible",
    )


class AnalyticsValueResponse(BaseModel):
    """Résumé des opportunités de value bet pour une entité Aspiturf."""

    entity_type: TrendEntityType = Field(
        ..., description="Type d'entité analysée (cheval, jockey ou entraîneur)"
    )
    entity_id: str = Field(..., description="Identifiant Aspiturf de l'entité")
    entity_label: Optional[str] = Field(
        default=None, description="Libellé lisible correspondant à l'entité"
    )
    metadata: AnalyticsMetadata = Field(
        default_factory=AnalyticsMetadata,
        description="Métadonnées décrivant les filtres appliqués",
    )
    sample_size: int = Field(
        ..., ge=0, description="Nombre de courses retenues pour l'analyse"
    )
    wins: int = Field(
        ..., ge=0, description="Nombre de victoires sur les courses retenues"
    )
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de victoire calculé sur l'échantillon (0-1)",
    )
    positive_edges: int = Field(
        ..., ge=0, description="Nombre de courses où l'écart de cote est positif"
    )
    negative_edges: int = Field(
        ..., ge=0, description="Nombre de courses où l'écart de cote est négatif"
    )
    average_edge: Optional[float] = Field(
        default=None,
        description="Écart moyen entre cote probable et cote observée",
    )
    median_edge: Optional[float] = Field(
        default=None, description="Écart médian de cote sur l'échantillon"
    )
    average_odds: Optional[float] = Field(
        default=None, description="Cote directe moyenne observée"
    )
    median_odds: Optional[float] = Field(
        default=None, description="Cote directe médiane observée"
    )
    stake_count: int = Field(
        ..., ge=0, description="Nombre de courses disposant d'une cote directe"
    )
    profit: Optional[float] = Field(
        default=None,
        description="Profit cumulé théorique pour une mise unitaire",
    )
    roi: Optional[float] = Field(
        default=None,
        description="Retour sur investissement théorique (profit / mises)",
    )
    hippodromes: List[str] = Field(
        default_factory=list,
        description="Liste des hippodromes rencontrés dans l'échantillon",
    )
    samples: List[ValueOpportunitySample] = Field(
        default_factory=list,
        description="Liste des courses classées par écart de cote décroissant",
    )


class VolatilityRaceSample(BaseModel):
    """Historique d'une course utilisée pour le calcul de la volatilité."""

    date: Optional[date_type] = Field(
        default=None, description="Date de la course associée à l'entité"
    )
    hippodrome: Optional[str] = Field(
        default=None, description="Hippodrome où la course s'est disputée"
    )
    course_number: Optional[int] = Field(
        default=None, description="Numéro de course dans la réunion"
    )
    distance: Optional[int] = Field(
        default=None, description="Distance officielle de l'épreuve en mètres"
    )
    final_position: Optional[int] = Field(
        default=None, description="Classement final de l'entité"
    )
    odds_actual: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cote observée (rapport PMU) si disponible",
    )
    odds_implied: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cote probable Aspiturf convertie en décimal",
    )
    edge: Optional[float] = Field(
        default=None, description="Différence cote probable - cote observée"
    )
    is_win: bool = Field(..., description="Indique si l'entité a gagné la course")
    is_podium: bool = Field(
        ..., description="Indique si l'entité a terminé dans les trois premiers"
    )


class VolatilityMetrics(BaseModel):
    """Métriques statistiques décrivant la volatilité des performances."""

    sample_size: int = Field(
        ..., ge=0, description="Nombre de courses retenues pour l'analyse"
    )
    wins: int = Field(..., ge=0, description="Nombre de victoires sur l'échantillon")
    podiums: int = Field(
        ..., ge=0, description="Nombre de podiums enregistrés sur l'échantillon"
    )
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de victoire calculé sur l'échantillon",
    )
    podium_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de podium calculé sur l'échantillon",
    )
    average_finish: Optional[float] = Field(
        default=None, description="Position moyenne à l'arrivée"
    )
    position_std_dev: Optional[float] = Field(
        default=None, description="Écart-type des positions à l'arrivée"
    )
    average_odds: Optional[float] = Field(
        default=None, description="Cote observée moyenne"
    )
    odds_std_dev: Optional[float] = Field(
        default=None, description="Écart-type des cotes observées"
    )
    average_edge: Optional[float] = Field(
        default=None, description="Écart moyen entre cote probable et cote observée"
    )
    consistency_index: Optional[float] = Field(
        default=None,
        description="Indice synthétique de régularité basé sur la dispersion",
    )


class AnalyticsVolatilityResponse(BaseModel):
    """Réponse de l'endpoint `/analytics/volatility`."""

    entity_type: TrendEntityType = Field(
        ..., description="Type d'entité analysée (cheval, jockey ou entraîneur)"
    )
    entity_id: str = Field(..., description="Identifiant Aspiturf de l'entité")
    entity_label: Optional[str] = Field(
        default=None, description="Libellé lisible correspondant à l'entité"
    )
    metadata: AnalyticsMetadata = Field(
        default_factory=AnalyticsMetadata,
        description="Métadonnées décrivant les filtres appliqués",
    )
    metrics: VolatilityMetrics = Field(
        ..., description="Métriques calculées à partir des courses filtrées"
    )
    races: List[VolatilityRaceSample] = Field(
        default_factory=list,
        description="Liste des courses retenues, triées par date décroissante",
    )


class OddsBucketMetrics(BaseModel):
    """Statistiques agrégées pour une tranche de cotes homogène."""

    label: str = Field(..., description="Libellé lisible du segment de cotes")
    sample_size: int = Field(
        ..., ge=0, description="Nombre de courses disposant d'une cote dans le segment"
    )
    wins: int = Field(..., ge=0, description="Victoires enregistrées dans le segment")
    podiums: int = Field(
        ..., ge=0, description="Arrivées dans les trois premiers pour le segment"
    )
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Proportion de victoires calculée sur le segment",
    )
    podium_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Proportion de podiums calculée sur le segment",
    )
    average_finish: Optional[float] = Field(
        default=None, description="Position moyenne d'arrivée dans le segment"
    )
    average_odds: Optional[float] = Field(
        default=None, description="Cote directe moyenne observée"
    )
    profit: Optional[float] = Field(
        default=None,
        description="Gain théorique cumulé en misant 1 unité sur chaque course",
    )
    roi: Optional[float] = Field(
        default=None,
        description="Retour sur investissement moyen (profit / nombre de mises)",
    )


class AnalyticsOddsResponse(BaseModel):
    """Réponse normalisée pour l'analyse de segments de cotes."""

    entity_type: TrendEntityType = Field(..., description="Type d'entité analysée")
    entity_id: str = Field(..., description="Identifiant Aspiturf analysé")
    entity_label: Optional[str] = Field(
        default=None, description="Libellé lisible de l'entité"
    )
    metadata: AnalyticsMetadata = Field(
        default_factory=AnalyticsMetadata,
        description="Métadonnées issues des filtres appliqués",
    )
    total_races: int = Field(
        ..., ge=0, description="Nombre total de courses correspondant au filtre"
    )
    races_with_odds: int = Field(
        ..., ge=0, description="Nombre de courses disposant d'une cote exploitable"
    )
    races_without_odds: int = Field(
        ..., ge=0, description="Nombre de courses ignorées faute de cote exploitable"
    )
    buckets: List[OddsBucketMetrics] = Field(
        default_factory=list,
        description="Statistiques calculées pour chaque segment de cotes",
    )
    overall: OddsBucketMetrics = Field(
        ..., description="Synthèse globale des courses disposant d'une cote",
    )


class EfficiencySample(BaseModel):
    """Course annotée avec les probabilités implicites d'un engagement."""

    date: Optional[date_type] = Field(
        default=None, description="Date de la course associée à l'entité",
    )
    hippodrome: Optional[str] = Field(
        default=None, description="Hippodrome où la course s'est disputée",
    )
    course_number: Optional[int] = Field(
        default=None, description="Numéro de la course dans la réunion",
    )
    odds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cote directe observée au départ de la course",
    )
    expected_win_probability: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Probabilité implicite de victoire dérivée de la cote",
    )
    expected_podium_probability: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Approximation de la probabilité de podium",
    )
    finish_position: Optional[int] = Field(
        default=None, description="Classement final de l'entité",
    )
    is_win: bool = Field(..., description="Indique si l'entité a remporté la course")
    is_podium: bool = Field(
        ..., description="Indique si l'entité a terminé sur le podium",
    )
    edge: Optional[float] = Field(
        default=None,
        description="Différence entre le résultat observé (0/1) et la probabilité de victoire",
    )


class EfficiencyMetrics(BaseModel):
    """Synthèse des gains attendus et observés d'une entité."""

    sample_size: int = Field(
        ..., ge=0, description="Nombre total de courses retenues",
    )
    wins: int = Field(..., ge=0, description="Nombre de victoires observées")
    expected_wins: Optional[float] = Field(
        default=None,
        ge=0,
        description="Somme des probabilités de victoire implicites",
    )
    win_delta: Optional[float] = Field(
        default=None,
        description="Différence entre victoires observées et attendues",
    )
    podiums: int = Field(..., ge=0, description="Nombre de podiums observés")
    expected_podiums: Optional[float] = Field(
        default=None,
        ge=0,
        description="Somme des probabilités approximatives de podium",
    )
    podium_delta: Optional[float] = Field(
        default=None,
        description="Différence entre podiums observés et attendus",
    )
    average_odds: Optional[float] = Field(
        default=None, description="Cote moyenne calculée sur les courses renseignées",
    )
    average_expected_win_probability: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Probabilité implicite moyenne de victoire",
    )
    stake_count: int = Field(
        ..., ge=0, description="Nombre de courses avec une cote exploitable",
    )
    profit: Optional[float] = Field(
        default=None, description="Profit théorique pour une mise unitaire",
    )
    roi: Optional[float] = Field(
        default=None,
        description="Retour sur investissement théorique (profit / mises)",
    )


class AnalyticsEfficiencyResponse(BaseModel):
    """Réponse structurée de l'endpoint `/analytics/efficiency`."""

    entity_type: TrendEntityType = Field(
        ..., description="Type d'entité analysée (cheval, jockey ou entraîneur)",
    )
    entity_id: str = Field(..., description="Identifiant Aspiturf de l'entité")
    entity_label: Optional[str] = Field(
        default=None, description="Libellé humain de l'entité analysée",
    )
    metadata: AnalyticsMetadata = Field(
        default_factory=AnalyticsMetadata,
        description="Métadonnées décrivant les filtres appliqués",
    )
    metrics: EfficiencyMetrics = Field(
        ..., description="Synthèse des performances attendues vs observées",
    )
    samples: List[EfficiencySample] = Field(
        default_factory=list,
        description="Détails course par course avec probabilités implicites",
    )


class WorkloadTimelineEntry(BaseModel):
    """Course annotée du repos observé avant l'engagement."""

    date: Optional[date_type] = Field(
        default=None, description="Date de la course étudiée",
    )
    hippodrome: Optional[str] = Field(
        default=None, description="Hippodrome ayant accueilli la course",
    )
    course_number: Optional[int] = Field(
        default=None, description="Numéro officiel de la course dans la réunion",
    )
    distance: Optional[int] = Field(
        default=None, description="Distance totale de l'épreuve en mètres",
    )
    final_position: Optional[int] = Field(
        default=None, description="Classement final obtenu",
    )
    rest_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Nombre de jours de repos depuis la participation précédente",
    )
    odds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cote directe observée (si disponible)",
    )
    is_win: bool = Field(
        default=False,
        description="Indique si l'entité a remporté la course",
    )
    is_podium: bool = Field(
        default=False,
        description="Indique si l'entité a terminé sur le podium",
    )


class WorkloadSummary(BaseModel):
    """Synthèse d'activité calculée sur l'historique d'une entité."""

    sample_size: int = Field(
        ..., ge=0, description="Nombre de courses retenues dans la période",
    )
    wins: int = Field(..., ge=0, description="Nombre de victoires dans la période")
    podiums: int = Field(
        ..., ge=0, description="Nombre de podiums (top 3) sur la période",
    )
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de victoire rapporté à l'échantillon",
    )
    podium_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Taux de podium rapporté à l'échantillon",
    )
    average_rest_days: Optional[float] = Field(
        default=None,
        ge=0,
        description="Repos moyen observé entre deux participations",
    )
    median_rest_days: Optional[float] = Field(
        default=None,
        ge=0,
        description="Repos médian observé entre deux participations",
    )
    shortest_rest_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Plus court repos observé",
    )
    longest_rest_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Plus long repos observé",
    )
    races_last_30_days: int = Field(
        ..., ge=0, description="Courses disputées sur les 30 derniers jours",
    )
    races_last_90_days: int = Field(
        ..., ge=0, description="Courses disputées sur les 90 derniers jours",
    )
    average_monthly_races: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fréquence moyenne de participation (courses/mois)",
    )
    rest_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Répartition du repos par tranches (ex: '0-7 jours': 3)",
    )


class AnalyticsWorkloadResponse(BaseModel):
    """Réponse structurée pour l'endpoint `/analytics/workload`."""

    entity_type: TrendEntityType = Field(
        ..., description="Type d'entité analysée (cheval, jockey ou entraîneur)",
    )
    entity_id: str = Field(..., description="Identifiant Aspiturf de l'entité")
    entity_label: Optional[str] = Field(
        default=None, description="Libellé lisible correspondant à l'entité",
    )
    metadata: AnalyticsMetadata = Field(
        default_factory=AnalyticsMetadata,
        description="Métadonnées globales sur l'échantillon considéré",
    )
    summary: WorkloadSummary = Field(
        ..., description="Synthèse d'activité et d'espacement entre les courses",
    )
    timeline: List[WorkloadTimelineEntry] = Field(
        default_factory=list,
        description="Chronologie annotée des courses analysées",
    )


ProgressionTrend = Literal["improvement", "decline", "stable", "initial", "unknown"]


class ProgressionRace(BaseModel):
    """Course successive utilisée pour mesurer l'évolution d'une entité."""

    date: Optional[date_type] = Field(
        default=None,
        description="Date de la course considérée",
    )
    hippodrome: Optional[str] = Field(
        default=None,
        description="Hippodrome où la course s'est disputée",
    )
    course_number: Optional[int] = Field(
        default=None,
        description="Numéro de course Aspiturf (prix)",
    )
    distance: Optional[int] = Field(
        default=None,
        description="Distance officielle de la course",
    )
    final_position: Optional[int] = Field(
        default=None,
        ge=1,
        description="Position finale (1 = victoire)",
    )
    previous_position: Optional[int] = Field(
        default=None,
        ge=1,
        description="Position lors de la course précédente considérée",
    )
    change: Optional[int] = Field(
        default=None,
        description="Différence de rang vs la course précédente (positive = amélioration)",
    )
    trend: ProgressionTrend = Field(
        default="unknown",
        description="Type d'évolution constatée sur cette course",
    )


class ProgressionSummary(BaseModel):
    """Synthèse statistique des variations de résultats dans le temps."""

    races: int = Field(
        0,
        ge=0,
        description="Nombre total de courses prises en compte",
    )
    improvements: int = Field(
        0,
        ge=0,
        description="Nombre de courses avec amélioration de la position finale",
    )
    declines: int = Field(
        0,
        ge=0,
        description="Nombre de courses avec régression",
    )
    stable: int = Field(
        0,
        ge=0,
        description="Nombre de courses avec position identique à la précédente",
    )
    average_change: Optional[float] = Field(
        default=None,
        description="Variation moyenne (positive = tendance à progresser)",
    )
    best_change: Optional[int] = Field(
        default=None,
        description="Plus forte amélioration constatée",
    )
    worst_change: Optional[int] = Field(
        default=None,
        description="Plus forte régression constatée",
    )
    longest_improvement_streak: int = Field(
        0,
        ge=0,
        description="Série consécutive maximale d'améliorations",
    )
    longest_decline_streak: int = Field(
        0,
        ge=0,
        description="Série consécutive maximale de régressions",
    )
    net_progress: Optional[int] = Field(
        default=None,
        description="Somme algébrique des variations (positive = progression globale)",
    )


class AnalyticsProgressionResponse(BaseModel):
    """Réponse détaillant la dynamique course par course d'une entité."""

    entity_type: TrendEntityType = Field(
        ...,
        description="Type d'entité analysée (cheval, jockey, entraîneur)",
    )
    entity_id: str = Field(..., description="Identifiant Aspiturf de l'entité")
    entity_label: Optional[str] = Field(
        default=None,
        description="Libellé de l'entité si disponible",
    )
    metadata: AnalyticsMetadata = Field(
        default_factory=AnalyticsMetadata,
        description="Métadonnées issues du filtre (dates, hippodrome)",
    )
    summary: ProgressionSummary = Field(
        default_factory=ProgressionSummary,
        description="Synthèse chiffrée de l'évolution",
    )
    races: List[ProgressionRace] = Field(
        default_factory=list,
        description="Liste des courses chronologiques avec les variations observées",
    )


class MomentumSlice(BaseModel):
    """Vue consolidée d'une période d'observation pour le momentum."""

    label: str = Field(..., description="Nom lisible de la fenêtre d'analyse")
    start_date: Optional[date_type] = Field(
        default=None, description="Première date incluse dans la fenêtre"
    )
    end_date: Optional[date_type] = Field(
        default=None, description="Dernière date incluse dans la fenêtre"
    )
    race_count: int = Field(
        ..., ge=0, description="Nombre de courses prises en compte dans la période"
    )
    wins: int = Field(..., ge=0, description="Victoires relevées sur la période")
    podiums: int = Field(..., ge=0, description="Podiums relevés sur la période")
    win_rate: Optional[float] = Field(
        default=None, ge=0, le=1, description="Taux de victoire (0-1)"
    )
    podium_rate: Optional[float] = Field(
        default=None, ge=0, le=1, description="Taux de podium (0-1)"
    )
    average_finish: Optional[float] = Field(
        default=None, description="Position moyenne d'arrivée"
    )
    average_odds: Optional[float] = Field(
        default=None, description="Cote directe moyenne observée"
    )
    roi: Optional[float] = Field(
        default=None,
        description="Retour sur investissement théorique (mise unitaire)"
    )
    races: List[RecentRace] = Field(
        default_factory=list,
        description="Détail des courses composant la fenêtre, du plus récent au plus ancien",
    )


class MomentumShift(BaseModel):
    """Différences élémentaires entre deux fenêtres de momentum."""

    win_rate: Optional[float] = Field(
        default=None,
        description="Écart de taux de victoire entre la fenêtre récente et la fenêtre de référence",
    )
    podium_rate: Optional[float] = Field(
        default=None,
        description="Écart de taux de podium entre les deux périodes",
    )
    average_finish: Optional[float] = Field(
        default=None,
        description="Variation de la position moyenne (valeurs négatives = amélioration)",
    )
    roi: Optional[float] = Field(
        default=None,
        description="Écart de ROI théorique calculé sur les courses disposant d'une cote",
    )


class AnalyticsMomentumResponse(BaseModel):
    """Réponse structurée pour l'endpoint `/analytics/momentum`."""

    entity_type: TrendEntityType = Field(
        ..., description="Type d'entité analysée (cheval, jockey ou entraîneur)"
    )
    entity_id: str = Field(..., description="Identifiant Aspiturf de l'entité")
    entity_label: Optional[str] = Field(
        default=None, description="Libellé lisible correspondant à l'entité"
    )
    metadata: AnalyticsMetadata = Field(
        default_factory=AnalyticsMetadata,
        description="Métadonnées globales sur l'échantillon considéré",
    )
    recent_window: MomentumSlice = Field(
        ..., description="Fenêtre d'analyse la plus récente"
    )
    reference_window: Optional[MomentumSlice] = Field(
        default=None, description="Fenêtre de comparaison (baseline)"
    )
    deltas: MomentumShift = Field(
        default_factory=MomentumShift,
        description="Écarts calculés entre les fenêtres récente et de référence",
    )