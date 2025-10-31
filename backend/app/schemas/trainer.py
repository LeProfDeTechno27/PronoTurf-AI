"""
Schémas Pydantic pour les entraîneurs
"""

from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field


# =============================
# Base Schemas
# =============================

class TrainerBase(BaseModel):
    """Schéma de base pour un entraîneur"""
    official_id: Optional[str] = Field(None, max_length=50, description="ID officiel de l'entraîneur")
    first_name: str = Field(..., max_length=100, description="Prénom")
    last_name: str = Field(..., max_length=100, description="Nom")
    stable_name: Optional[str] = Field(None, max_length=200, description="Nom de l'écurie")
    nationality: Optional[str] = Field(None, max_length=50, description="Nationalité")
    career_wins: int = Field(default=0, ge=0, description="Nombre de victoires")
    career_places: int = Field(default=0, ge=0, description="Nombre de places")
    career_starts: int = Field(default=0, ge=0, description="Nombre de courses")


# =============================
# Create/Update Schemas
# =============================

class TrainerCreate(TrainerBase):
    """Schéma pour créer un entraîneur"""
    pass


class TrainerUpdate(BaseModel):
    """Schéma pour mettre à jour un entraîneur"""
    official_id: Optional[str] = Field(None, max_length=50)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    stable_name: Optional[str] = Field(None, max_length=200)
    nationality: Optional[str] = Field(None, max_length=50)
    career_wins: Optional[int] = Field(None, ge=0)
    career_places: Optional[int] = Field(None, ge=0)
    career_starts: Optional[int] = Field(None, ge=0)


# =============================
# Response Schemas
# =============================

class TrainerResponse(TrainerBase):
    """Schéma de réponse pour un entraîneur"""
    trainer_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class TrainerSimple(BaseModel):
    """Schéma simplifié pour un entraîneur (pour les relations)"""
    trainer_id: int
    first_name: str
    last_name: str
    stable_name: Optional[str] = None

    class Config:
        from_attributes = True


class TrainerList(BaseModel):
    """Schéma pour une liste d'entraîneurs"""
    total: int
    trainers: list[TrainerResponse]


class TrainerDetailResponse(TrainerResponse):
    """Schéma détaillé pour un entraîneur avec statistiques calculées"""
    win_rate: Optional[float] = Field(None, description="Taux de victoires en %")
    place_rate: Optional[float] = Field(None, description="Taux de places en %")

    class Config:
        from_attributes = True
