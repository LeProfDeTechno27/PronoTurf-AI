"""Schémas Pydantic pour les endpoints d'analytics Aspiturf."""

from enum import Enum
from datetime import date as date_type
from typing import List, Optional

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

