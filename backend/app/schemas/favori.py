# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""Schémas Pydantic pour les favoris"""

from typing import Optional, Any, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field

from app.models.favori import EntityType


class FavoriBase(BaseModel):
    """Schéma de base pour un favori"""

    entity_type: EntityType = Field(..., description="Type d'entité")
    entity_id: int = Field(..., gt=0, description="ID de l'entité")
    alert_enabled: bool = Field(default=True, description="Alertes activées")


class FavoriCreate(FavoriBase):
    """Schéma pour créer un favori"""

    pass


class FavoriUpdate(BaseModel):
    """Schéma pour mettre à jour un favori"""

    alert_enabled: Optional[bool] = None


class FavoriResponse(FavoriBase):
    """Schéma de réponse pour un favori"""

    favori_id: int
    user_id: int
    added_at: datetime

    class Config:
        from_attributes = True


class FavoriWithDetailsResponse(FavoriResponse):
    """Schéma de réponse détaillé incluant les infos de l'entité"""

    entity_name: Optional[str] = Field(None, description="Nom de l'entité")
    entity_details: Optional[Dict[str, Any]] = Field(
        None, description="Détails supplémentaires de l'entité"
    )


class FavorisListResponse(BaseModel):
    """Réponse contenant la liste des favoris par catégorie"""

    horses: List[FavoriWithDetailsResponse] = Field(
        default_factory=list, description="Chevaux favoris"
    )
    jockeys: List[FavoriWithDetailsResponse] = Field(
        default_factory=list, description="Jockeys favoris"
    )
    trainers: List[FavoriWithDetailsResponse] = Field(
        default_factory=list, description="Entraîneurs favoris"
    )
    hippodromes: List[FavoriWithDetailsResponse] = Field(
        default_factory=list, description="Hippodromes favoris"
    )
    total_count: int = Field(..., description="Nombre total de favoris")


class FavoriAlert(BaseModel):
    """Alerte générée pour un favori"""

    favori_id: int
    entity_type: EntityType
    entity_id: int
    entity_name: str
    alert_type: str = Field(
        ..., description="Type d'alerte (course_today, value_bet, etc.)"
    )
    message: str = Field(..., description="Message de l'alerte")
    course_id: Optional[int] = None
    pronostic_id: Optional[int] = None


class FavoriAlertUpdate(BaseModel):
    """Payload pour activer/désactiver les alertes d'un favori"""

    alert_enabled: bool = Field(..., description="Nouveau statut de l'alerte")
