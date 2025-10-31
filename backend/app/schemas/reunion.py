"""
Schémas Pydantic pour les réunions
"""

from typing import Optional, Dict, Any
from datetime import date, datetime

from pydantic import BaseModel, Field

from app.models.reunion import ReunionStatus
from app.schemas.hippodrome import HippodromeSimple


# =============================
# Base Schemas
# =============================

class ReunionBase(BaseModel):
    """Schéma de base pour une réunion"""
    hippodrome_id: int = Field(..., description="ID de l'hippodrome")
    reunion_date: date = Field(..., description="Date de la réunion")
    reunion_number: int = Field(..., ge=1, le=20, description="Numéro de la réunion (1-20)")
    status: ReunionStatus = Field(default=ReunionStatus.SCHEDULED, description="Statut de la réunion")
    api_source: Optional[str] = Field(None, max_length=50, description="Source API des données")
    weather_conditions: Optional[Dict[str, Any]] = Field(None, description="Conditions météo")


# =============================
# Create/Update Schemas
# =============================

class ReunionCreate(ReunionBase):
    """Schéma pour créer une réunion"""
    pass


class ReunionUpdate(BaseModel):
    """Schéma pour mettre à jour une réunion"""
    hippodrome_id: Optional[int] = None
    reunion_date: Optional[date] = None
    reunion_number: Optional[int] = Field(None, ge=1, le=20)
    status: Optional[ReunionStatus] = None
    api_source: Optional[str] = Field(None, max_length=50)
    weather_conditions: Optional[Dict[str, Any]] = None


# =============================
# Response Schemas
# =============================

class ReunionResponse(ReunionBase):
    """Schéma de réponse pour une réunion"""
    reunion_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ReunionWithHippodrome(ReunionResponse):
    """Schéma de réponion avec les données de l'hippodrome"""
    hippodrome: HippodromeSimple

    class Config:
        from_attributes = True


class ReunionSimple(BaseModel):
    """Schéma simplifié pour une réunion (pour les relations)"""
    reunion_id: int
    reunion_date: date
    reunion_number: int
    status: ReunionStatus

    class Config:
        from_attributes = True


class ReunionList(BaseModel):
    """Schéma pour une liste de réunions"""
    total: int
    reunions: list[ReunionWithHippodrome]


class ReunionDetailResponse(ReunionWithHippodrome):
    """Schéma détaillé pour une réunion avec le nombre de courses"""
    number_of_courses: int = Field(..., description="Nombre de courses dans la réunion")

    class Config:
        from_attributes = True
