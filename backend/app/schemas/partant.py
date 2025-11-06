# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Schémas Pydantic pour les partants (participants/runners)
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from app.schemas.horse import HorseSimple
from app.schemas.jockey import JockeySimple
from app.schemas.trainer import TrainerSimple
from app.schemas.course import CourseSimple


# =============================
# Base Schemas
# =============================

class PartantBase(BaseModel):
    """Schéma de base pour un partant"""
    course_id: int = Field(..., description="ID de la course")
    horse_id: int = Field(..., description="ID du cheval")
    jockey_id: Optional[int] = Field(None, description="ID du jockey")
    trainer_id: int = Field(..., description="ID de l'entraîneur")
    numero_corde: int = Field(..., ge=1, le=30, description="Numéro de corde/départ")
    poids_porte: Optional[Decimal] = Field(None, description="Poids porté en kg")
    handicap_value: Optional[int] = Field(None, description="Valeur de handicap")
    equipment: Optional[Dict[str, Any]] = Field(None, description="Équipement du cheval")
    days_since_last_race: Optional[int] = Field(None, ge=0, description="Jours depuis dernière course")
    recent_form: Optional[str] = Field(None, max_length=50, description="Forme récente (ex: 1-3-2-5)")
    odds_pmu: Optional[Decimal] = Field(None, description="Cote PMU")
    final_position: Optional[int] = Field(None, ge=1, description="Position finale")
    disqualified: bool = Field(default=False, description="Disqualifié")


# =============================
# Create/Update Schemas
# =============================

class PartantCreate(PartantBase):
    """Schéma pour créer un partant"""
    pass


class PartantUpdate(BaseModel):
    """Schéma pour mettre à jour un partant"""
    course_id: Optional[int] = None
    horse_id: Optional[int] = None
    jockey_id: Optional[int] = None
    trainer_id: Optional[int] = None
    numero_corde: Optional[int] = Field(None, ge=1, le=30)
    poids_porte: Optional[Decimal] = None
    handicap_value: Optional[int] = None
    equipment: Optional[Dict[str, Any]] = None
    days_since_last_race: Optional[int] = Field(None, ge=0)
    recent_form: Optional[str] = Field(None, max_length=50)
    odds_pmu: Optional[Decimal] = None
    final_position: Optional[int] = Field(None, ge=1)
    disqualified: Optional[bool] = None


# =============================
# Response Schemas
# =============================

class PartantResponse(PartantBase):
    """Schéma de réponse pour un partant"""
    partant_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PartantWithRelations(PartantResponse):
    """Schéma de partant avec toutes ses relations"""
    horse: HorseSimple
    jockey: Optional[JockeySimple] = None
    trainer: TrainerSimple

    class Config:
        from_attributes = True


class PartantSimple(BaseModel):
    """Schéma simplifié pour un partant (pour les relations)"""
    partant_id: int
    numero_corde: int
    odds_pmu: Optional[Decimal] = None
    horse: HorseSimple

    class Config:
        from_attributes = True


class PartantList(BaseModel):
    """Schéma pour une liste de partants"""
    total: int
    partants: list[PartantWithRelations]


class PartantDetailResponse(PartantWithRelations):
    """Schéma détaillé pour un partant avec statistiques"""
    course: CourseSimple
    equipment_list: Optional[List[str]] = Field(None, description="Liste d'équipement")
    has_oeilleres: bool = Field(default=False, description="Porte des œillères")
    is_favorite: bool = Field(default=False, description="Est favori")
    odds_category: str = Field(..., description="Catégorie de cote")
    recent_form_list: Optional[List[int]] = Field(None, description="Forme récente en liste")
    average_recent_position: Optional[float] = Field(None, description="Position moyenne récente")
    has_won_recently: bool = Field(default=False, description="A gagné récemment")
    weight_display: Optional[str] = Field(None, description="Poids formaté")
    rest_days_category: str = Field(..., description="Catégorie de jours de repos")
    is_finished: bool = Field(default=False, description="A terminé la course")
    result_display: str = Field(..., description="Affichage du résultat")

    class Config:
        from_attributes = True


# =============================
# Batch Operations
# =============================

class PartantBatchCreate(BaseModel):
    """Schéma pour créer plusieurs partants en batch"""
    course_id: int = Field(..., description="ID de la course")
    partants: List[PartantCreate] = Field(..., description="Liste des partants à créer")


class PartantResultUpdate(BaseModel):
    """Schéma pour mettre à jour les résultats d'un partant"""
    final_position: Optional[int] = Field(None, ge=1, description="Position finale")
    disqualified: bool = Field(default=False, description="Disqualifié")


class PartantOddsUpdate(BaseModel):
    """Schéma pour mettre à jour uniquement la cote"""
    odds_pmu: Decimal = Field(..., description="Nouvelle cote PMU")