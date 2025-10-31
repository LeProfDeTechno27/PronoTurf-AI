"""
Schémas Pydantic pour les hippodromes
"""

from typing import Optional
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from app.models.hippodrome import TrackType


# =============================
# Base Schemas
# =============================

class HippodromeBase(BaseModel):
    """Schéma de base pour un hippodrome"""
    code: str = Field(..., max_length=10, description="Code unique de l'hippodrome")
    name: str = Field(..., max_length=200, description="Nom de l'hippodrome")
    city: Optional[str] = Field(None, max_length=100, description="Ville")
    country: str = Field(default="France", max_length=50, description="Pays")
    track_type: TrackType = Field(..., description="Type de piste")
    track_length: Optional[int] = Field(None, description="Longueur de piste en mètres")
    track_surface: Optional[str] = Field(None, max_length=50, description="Surface de la piste")
    latitude: Optional[Decimal] = Field(None, description="Latitude GPS")
    longitude: Optional[Decimal] = Field(None, description="Longitude GPS")


# =============================
# Create/Update Schemas
# =============================

class HippodromeCreate(HippodromeBase):
    """Schéma pour créer un hippodrome"""
    pass


class HippodromeUpdate(BaseModel):
    """Schéma pour mettre à jour un hippodrome"""
    code: Optional[str] = Field(None, max_length=10)
    name: Optional[str] = Field(None, max_length=200)
    city: Optional[str] = Field(None, max_length=100)
    country: Optional[str] = Field(None, max_length=50)
    track_type: Optional[TrackType] = None
    track_length: Optional[int] = None
    track_surface: Optional[str] = Field(None, max_length=50)
    latitude: Optional[Decimal] = None
    longitude: Optional[Decimal] = None


# =============================
# Response Schemas
# =============================

class HippodromeResponse(HippodromeBase):
    """Schéma de réponse pour un hippodrome"""
    hippodrome_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class HippodromeList(BaseModel):
    """Schéma pour une liste d'hippodromes"""
    total: int
    hippodromes: list[HippodromeResponse]


class HippodromeSimple(BaseModel):
    """Schéma simplifié pour un hippodrome (pour les relations)"""
    hippodrome_id: int
    code: str
    name: str
    city: Optional[str] = None
    track_type: TrackType

    class Config:
        from_attributes = True
