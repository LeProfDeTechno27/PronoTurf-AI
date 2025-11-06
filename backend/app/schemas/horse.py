# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Schémas Pydantic pour les chevaux
"""

from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.horse import Gender


# =============================
# Base Schemas
# =============================

class HorseBase(BaseModel):
    """Schéma de base pour un cheval"""
    official_id: Optional[str] = Field(None, max_length=50, description="ID officiel du cheval")
    name: str = Field(..., max_length=200, description="Nom du cheval")
    birth_year: Optional[int] = Field(None, ge=1990, le=2030, description="Année de naissance")
    gender: Gender = Field(..., description="Genre du cheval")
    coat_color: Optional[str] = Field(None, max_length=50, description="Couleur de la robe")
    breed: Optional[str] = Field(None, max_length=100, description="Race")
    sire: Optional[str] = Field(None, max_length=200, description="Père")
    dam: Optional[str] = Field(None, max_length=200, description="Mère")
    owner: Optional[str] = Field(None, max_length=300, description="Propriétaire")


# =============================
# Create/Update Schemas
# =============================

class HorseCreate(HorseBase):
    """Schéma pour créer un cheval"""
    pass


class HorseUpdate(BaseModel):
    """Schéma pour mettre à jour un cheval"""
    official_id: Optional[str] = Field(None, max_length=50)
    name: Optional[str] = Field(None, max_length=200)
    birth_year: Optional[int] = Field(None, ge=1990, le=2030)
    gender: Optional[Gender] = None
    coat_color: Optional[str] = Field(None, max_length=50)
    breed: Optional[str] = Field(None, max_length=100)
    sire: Optional[str] = Field(None, max_length=200)
    dam: Optional[str] = Field(None, max_length=200)
    owner: Optional[str] = Field(None, max_length=300)


# =============================
# Response Schemas
# =============================

class HorseResponse(HorseBase):
    """Schéma de réponse pour un cheval"""
    horse_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class HorseSimple(BaseModel):
    """Schéma simplifié pour un cheval (pour les relations)"""
    horse_id: int
    name: str
    birth_year: Optional[int] = None
    gender: Gender

    class Config:
        from_attributes = True


class HorseList(BaseModel):
    """Schéma pour une liste de chevaux"""
    total: int
    horses: list[HorseResponse]


class HorseDetailResponse(HorseResponse):
    """Schéma détaillé pour un cheval avec statistiques"""
    total_races: int = Field(default=0, description="Nombre total de courses")
    total_wins: int = Field(default=0, description="Nombre de victoires")
    win_rate: Optional[float] = Field(None, description="Taux de victoires en %")

    class Config:
        from_attributes = True