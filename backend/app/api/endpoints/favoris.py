"""
Endpoints API pour la gestion des favoris des utilisateurs

Routes disponibles:
- GET /api/v1/favoris : Liste des favoris
- POST /api/v1/favoris : Ajouter un favori
- DELETE /api/v1/favoris/{favori_id} : Supprimer un favori
- PATCH /api/v1/favoris/{favori_id}/alert : Toggle alerte
- GET /api/v1/favoris/by-type/{entity_type} : Favoris par type
- GET /api/v1/favoris/{favori_id}/details : Détails d'un favori
"""

from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from sqlalchemy.orm import selectinload

from app.core.database import get_async_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.favori import Favori, EntityType
from app.models.horse import Horse
from app.models.jockey import Jockey
from app.models.trainer import Trainer
from app.models.hippodrome import Hippodrome
from app.schemas.favori import (
    FavoriCreate,
    FavoriResponse,
    FavoriWithDetailsResponse,
    FavoriAlertUpdate
)

router = APIRouter()


@router.get("/", response_model=List[FavoriResponse])
async def get_all_favoris(
    entity_type: Optional[EntityType] = Query(None, description="Filtrer par type d'entité"),
    skip: int = Query(0, ge=0, description="Nombre d'entrées à ignorer"),
    limit: int = Query(100, ge=1, le=500, description="Nombre d'entrées à retourner"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Récupère tous les favoris de l'utilisateur

    Args:
        entity_type: Filtrer par type d'entité (horse, jockey, trainer, hippodrome)
        skip: Pagination - nombre d'entrées à ignorer
        limit: Pagination - nombre d'entrées à retourner

    Returns:
        Liste des favoris
    """
    stmt = (
        select(Favori)
        .where(Favori.user_id == current_user.user_id)
        .order_by(desc(Favori.added_at))
    )

    if entity_type:
        stmt = stmt.where(Favori.entity_type == entity_type)

    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    favoris = result.scalars().all()

    return [
        FavoriResponse(
            favori_id=favori.favori_id,
            user_id=favori.user_id,
            entity_type=favori.entity_type,
            entity_id=favori.entity_id,
            alert_enabled=favori.alert_enabled,
            added_at=favori.added_at
        )
        for favori in favoris
    ]


@router.post("/", response_model=FavoriResponse, status_code=status.HTTP_201_CREATED)
async def add_favori(
    favori_data: FavoriCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Ajoute une entité aux favoris

    Args:
        favori_data: Données du favori (type et ID d'entité)

    Returns:
        Favori créé

    Raises:
        HTTPException 404: Si l'entité n'existe pas
        HTTPException 409: Si déjà en favoris
    """
    # Vérifier que l'entité existe
    entity_exists = await _check_entity_exists(
        db, favori_data.entity_type, favori_data.entity_id
    )

    if not entity_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{favori_data.entity_type.value} avec ID {favori_data.entity_id} introuvable"
        )

    # Vérifier si déjà en favoris
    stmt = select(Favori).where(
        and_(
            Favori.user_id == current_user.user_id,
            Favori.entity_type == favori_data.entity_type,
            Favori.entity_id == favori_data.entity_id
        )
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cette entité est déjà dans vos favoris"
        )

    # Créer le favori
    favori = Favori(
        user_id=current_user.user_id,
        entity_type=favori_data.entity_type,
        entity_id=favori_data.entity_id,
        alert_enabled=favori_data.alert_enabled
    )

    db.add(favori)
    await db.commit()
    await db.refresh(favori)

    return FavoriResponse(
        favori_id=favori.favori_id,
        user_id=favori.user_id,
        entity_type=favori.entity_type,
        entity_id=favori.entity_id,
        alert_enabled=favori.alert_enabled,
        added_at=favori.added_at
    )


@router.delete("/{favori_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_favori(
    favori_id: int = Path(..., description="ID du favori"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Supprime un favori

    Args:
        favori_id: ID du favori à supprimer

    Raises:
        HTTPException 404: Si le favori n'existe pas
        HTTPException 403: Si le favori n'appartient pas à l'utilisateur
    """
    stmt = select(Favori).where(Favori.favori_id == favori_id)
    result = await db.execute(stmt)
    favori = result.scalar_one_or_none()

    if not favori:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Favori {favori_id} introuvable"
        )

    if favori.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ce favori ne vous appartient pas"
        )

    await db.delete(favori)
    await db.commit()


@router.patch("/{favori_id}/alert", response_model=FavoriResponse)
async def toggle_alert(
    favori_id: int = Path(..., description="ID du favori"),
    alert_update: FavoriAlertUpdate = ...,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Active ou désactive les alertes pour un favori

    Args:
        favori_id: ID du favori
        alert_update: Nouveau statut des alertes

    Returns:
        Favori mis à jour

    Raises:
        HTTPException 404: Si le favori n'existe pas
        HTTPException 403: Si le favori n'appartient pas à l'utilisateur
    """
    stmt = select(Favori).where(Favori.favori_id == favori_id)
    result = await db.execute(stmt)
    favori = result.scalar_one_or_none()

    if not favori:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Favori {favori_id} introuvable"
        )

    if favori.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ce favori ne vous appartient pas"
        )

    favori.alert_enabled = alert_update.alert_enabled

    await db.commit()
    await db.refresh(favori)

    return FavoriResponse(
        favori_id=favori.favori_id,
        user_id=favori.user_id,
        entity_type=favori.entity_type,
        entity_id=favori.entity_id,
        alert_enabled=favori.alert_enabled,
        added_at=favori.added_at
    )


@router.get("/by-type/{entity_type}", response_model=List[FavoriWithDetailsResponse])
async def get_favoris_by_type(
    entity_type: EntityType = Path(..., description="Type d'entité"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Récupère tous les favoris d'un type spécifique avec détails

    Args:
        entity_type: Type d'entité (horse, jockey, trainer, hippodrome)

    Returns:
        Liste des favoris avec informations détaillées de l'entité
    """
    stmt = (
        select(Favori)
        .where(
            and_(
                Favori.user_id == current_user.user_id,
                Favori.entity_type == entity_type
            )
        )
        .order_by(desc(Favori.added_at))
    )

    result = await db.execute(stmt)
    favoris = result.scalars().all()

    # Récupérer les détails pour chaque favori
    favoris_with_details = []
    for favori in favoris:
        entity_details = await _get_entity_details(db, favori.entity_type, favori.entity_id)

        favoris_with_details.append(
            FavoriWithDetailsResponse(
                favori_id=favori.favori_id,
                user_id=favori.user_id,
                entity_type=favori.entity_type,
                entity_id=favori.entity_id,
                alert_enabled=favori.alert_enabled,
                added_at=favori.added_at,
                entity_details=entity_details
            )
        )

    return favoris_with_details


@router.get("/{favori_id}/details", response_model=FavoriWithDetailsResponse)
async def get_favori_details(
    favori_id: int = Path(..., description="ID du favori"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Récupère les détails d'un favori spécifique

    Args:
        favori_id: ID du favori

    Returns:
        Favori avec informations détaillées de l'entité

    Raises:
        HTTPException 404: Si le favori n'existe pas
        HTTPException 403: Si le favori n'appartient pas à l'utilisateur
    """
    stmt = select(Favori).where(Favori.favori_id == favori_id)
    result = await db.execute(stmt)
    favori = result.scalar_one_or_none()

    if not favori:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Favori {favori_id} introuvable"
        )

    if favori.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ce favori ne vous appartient pas"
        )

    entity_details = await _get_entity_details(db, favori.entity_type, favori.entity_id)

    return FavoriWithDetailsResponse(
        favori_id=favori.favori_id,
        user_id=favori.user_id,
        entity_type=favori.entity_type,
        entity_id=favori.entity_id,
        alert_enabled=favori.alert_enabled,
        added_at=favori.added_at,
        entity_details=entity_details
    )


@router.delete("/by-entity", status_code=status.HTTP_204_NO_CONTENT)
async def remove_favori_by_entity(
    entity_type: EntityType = Query(..., description="Type d'entité"),
    entity_id: int = Query(..., description="ID de l'entité"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Supprime un favori en spécifiant l'entité directement

    Args:
        entity_type: Type d'entité
        entity_id: ID de l'entité

    Raises:
        HTTPException 404: Si le favori n'existe pas
    """
    stmt = select(Favori).where(
        and_(
            Favori.user_id == current_user.user_id,
            Favori.entity_type == entity_type,
            Favori.entity_id == entity_id
        )
    )
    result = await db.execute(stmt)
    favori = result.scalar_one_or_none()

    if not favori:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Favori {entity_type.value}:{entity_id} introuvable"
        )

    await db.delete(favori)
    await db.commit()


# Fonctions utilitaires privées

async def _check_entity_exists(
    db: AsyncSession,
    entity_type: EntityType,
    entity_id: int
) -> bool:
    """Vérifie qu'une entité existe dans la base"""
    if entity_type == EntityType.HORSE:
        stmt = select(Horse).where(Horse.horse_id == entity_id)
    elif entity_type == EntityType.JOCKEY:
        stmt = select(Jockey).where(Jockey.jockey_id == entity_id)
    elif entity_type == EntityType.TRAINER:
        stmt = select(Trainer).where(Trainer.trainer_id == entity_id)
    elif entity_type == EntityType.HIPPODROME:
        stmt = select(Hippodrome).where(Hippodrome.hippodrome_id == entity_id)
    else:
        return False

    result = await db.execute(stmt)
    entity = result.scalar_one_or_none()
    return entity is not None


async def _get_entity_details(
    db: AsyncSession,
    entity_type: EntityType,
    entity_id: int
) -> Optional[dict]:
    """Récupère les détails d'une entité"""
    if entity_type == EntityType.HORSE:
        stmt = select(Horse).where(Horse.horse_id == entity_id)
        result = await db.execute(stmt)
        horse = result.scalar_one_or_none()
        if horse:
            return {
                "id": horse.horse_id,
                "name": horse.name,
                "birth_year": horse.birth_year,
                "gender": horse.gender.value if horse.gender else None,
                "coat_color": horse.coat_color,
                "breed": horse.breed,
                "sire": horse.sire,
                "dam": horse.dam,
                "owner": horse.owner
            }

    elif entity_type == EntityType.JOCKEY:
        stmt = select(Jockey).where(Jockey.jockey_id == entity_id)
        result = await db.execute(stmt)
        jockey = result.scalar_one_or_none()
        if jockey:
            return {
                "id": jockey.jockey_id,
                "name": jockey.full_name,
                "first_name": jockey.first_name,
                "last_name": jockey.last_name,
                "nationality": jockey.nationality,
                "weight": float(jockey.weight) if jockey.weight else None,
                "total_races": jockey.total_races,
                "total_wins": jockey.total_wins,
                "win_rate": jockey.win_rate
            }

    elif entity_type == EntityType.TRAINER:
        stmt = select(Trainer).where(Trainer.trainer_id == entity_id)
        result = await db.execute(stmt)
        trainer = result.scalar_one_or_none()
        if trainer:
            return {
                "id": trainer.trainer_id,
                "name": trainer.full_name,
                "first_name": trainer.first_name,
                "last_name": trainer.last_name,
                "stable_name": trainer.stable_name,
                "nationality": trainer.nationality,
                "total_races": trainer.total_races,
                "total_wins": trainer.total_wins,
                "win_rate": trainer.win_rate
            }

    elif entity_type == EntityType.HIPPODROME:
        stmt = select(Hippodrome).where(Hippodrome.hippodrome_id == entity_id)
        result = await db.execute(stmt)
        hippodrome = result.scalar_one_or_none()
        if hippodrome:
            return {
                "id": hippodrome.hippodrome_id,
                "name": hippodrome.name,
                "code": hippodrome.code,
                "city": hippodrome.city,
                "country": hippodrome.country,
                "track_type": hippodrome.track_type.value if hippodrome.track_type else None,
                "latitude": float(hippodrome.latitude) if hippodrome.latitude else None,
                "longitude": float(hippodrome.longitude) if hippodrome.longitude else None
            }

    return None
