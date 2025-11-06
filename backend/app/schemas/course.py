# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Schémas Pydantic pour les courses
"""

from typing import Optional
from datetime import time, datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from app.models.course import Discipline, SurfaceType, StartType, CourseStatus
from app.schemas.reunion import ReunionSimple


# =============================
# Base Schemas
# =============================

class CourseBase(BaseModel):
    """Schéma de base pour une course"""
    reunion_id: int = Field(..., description="ID de la réunion")
    course_number: int = Field(..., ge=1, le=20, description="Numéro de la course (1-20)")
    course_name: Optional[str] = Field(None, max_length=300, description="Nom de la course")
    discipline: Discipline = Field(..., description="Discipline hippique")
    distance: int = Field(..., gt=0, description="Distance en mètres")
    prize_money: Optional[Decimal] = Field(None, description="Allocation en euros")
    race_category: Optional[str] = Field(None, max_length=100, description="Catégorie de course")
    race_class: Optional[str] = Field(None, max_length=50, description="Classe de course")
    surface_type: SurfaceType = Field(..., description="Type de surface")
    start_type: StartType = Field(default=StartType.STALLE, description="Type de départ")
    scheduled_time: time = Field(..., description="Heure prévue")
    actual_start_time: Optional[time] = Field(None, description="Heure de départ réelle")
    number_of_runners: Optional[int] = Field(None, ge=0, description="Nombre de partants")
    status: CourseStatus = Field(default=CourseStatus.SCHEDULED, description="Statut de la course")


# =============================
# Create/Update Schemas
# =============================

class CourseCreate(CourseBase):
    """Schéma pour créer une course"""
    pass


class CourseUpdate(BaseModel):
    """Schéma pour mettre à jour une course"""
    reunion_id: Optional[int] = None
    course_number: Optional[int] = Field(None, ge=1, le=20)
    course_name: Optional[str] = Field(None, max_length=300)
    discipline: Optional[Discipline] = None
    distance: Optional[int] = Field(None, gt=0)
    prize_money: Optional[Decimal] = None
    race_category: Optional[str] = Field(None, max_length=100)
    race_class: Optional[str] = Field(None, max_length=50)
    surface_type: Optional[SurfaceType] = None
    start_type: Optional[StartType] = None
    scheduled_time: Optional[time] = None
    actual_start_time: Optional[time] = None
    number_of_runners: Optional[int] = Field(None, ge=0)
    status: Optional[CourseStatus] = None


# =============================
# Response Schemas
# =============================

class CourseResponse(CourseBase):
    """Schéma de réponse pour une course"""
    course_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CourseWithReunion(CourseResponse):
    """Schéma de course avec les données de la réunion"""
    reunion: ReunionSimple

    class Config:
        from_attributes = True


class CourseSimple(BaseModel):
    """Schéma simplifié pour une course (pour les relations)"""
    course_id: int
    course_number: int
    course_name: Optional[str] = None
    discipline: Discipline
    distance: int
    scheduled_time: time
    status: CourseStatus

    class Config:
        from_attributes = True


class CourseList(BaseModel):
    """Schéma pour une liste de courses"""
    total: int
    courses: list[CourseWithReunion]


class CourseDetailResponse(CourseWithReunion):
    """Schéma détaillé pour une course avec statistiques"""
    number_of_partants: int = Field(..., description="Nombre de partants confirmés")
    has_pronostic: bool = Field(default=False, description="Pronostic disponible")

    class Config:
        from_attributes = True


# =============================
# Filter/Query Schemas
# =============================

class CourseFilter(BaseModel):
    """Schéma pour filtrer les courses"""
    reunion_id: Optional[int] = None
    discipline: Optional[Discipline] = None
    status: Optional[CourseStatus] = None
    hippodrome_id: Optional[int] = None
    min_distance: Optional[int] = None
    max_distance: Optional[int] = None
    surface_type: Optional[SurfaceType] = None

    class Config:
        use_enum_values = True