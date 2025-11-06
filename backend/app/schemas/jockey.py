# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Schémas Pydantic pour les jockeys
"""

from typing import Optional
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field


# =============================
# Base Schemas
# =============================

class JockeyBase(BaseModel):
    """Schéma de base pour un jockey"""
    official_id: Optional[str] = Field(None, max_length=50, description="ID officiel du jockey")
    first_name: str = Field(..., max_length=100, description="Prénom")
    last_name: str = Field(..., max_length=100, description="Nom")
    birth_date: Optional[date] = Field(None, description="Date de naissance")
    nationality: Optional[str] = Field(None, max_length=50, description="Nationalité")
    weight: Optional[Decimal] = Field(None, description="Poids en kg")
    career_wins: int = Field(default=0, ge=0, description="Nombre de victoires")
    career_places: int = Field(default=0, ge=0, description="Nombre de places")
    career_starts: int = Field(default=0, ge=0, description="Nombre de courses")


# =============================
# Create/Update Schemas
# =============================

class JockeyCreate(JockeyBase):
    """Schéma pour créer un jockey"""
    pass


class JockeyUpdate(BaseModel):
    """Schéma pour mettre à jour un jockey"""
    official_id: Optional[str] = Field(None, max_length=50)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    birth_date: Optional[date] = None
    nationality: Optional[str] = Field(None, max_length=50)
    weight: Optional[Decimal] = None
    career_wins: Optional[int] = Field(None, ge=0)
    career_places: Optional[int] = Field(None, ge=0)
    career_starts: Optional[int] = Field(None, ge=0)


# =============================
# Response Schemas
# =============================

class JockeyResponse(JockeyBase):
    """Schéma de réponse pour un jockey"""
    jockey_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class JockeySimple(BaseModel):
    """Schéma simplifié pour un jockey (pour les relations)"""
    jockey_id: int
    first_name: str
    last_name: str
    nationality: Optional[str] = None

    class Config:
        from_attributes = True


class JockeyList(BaseModel):
    """Schéma pour une liste de jockeys"""
    total: int
    jockeys: list[JockeyResponse]


class JockeyDetailResponse(JockeyResponse):
    """Schéma détaillé pour un jockey avec statistiques calculées"""
    age: Optional[int] = Field(None, description="Âge")
    win_rate: Optional[float] = Field(None, description="Taux de victoires en %")
    place_rate: Optional[float] = Field(None, description="Taux de places en %")

    class Config:
        from_attributes = True