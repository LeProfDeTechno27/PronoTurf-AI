"""
Router pour les endpoints des hippodromes
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models import Hippodrome, User
from app.models.hippodrome import TrackType
from app.schemas.hippodrome import (
    HippodromeCreate,
    HippodromeUpdate,
    HippodromeResponse,
    HippodromeList,
)

router = APIRouter()


@router.get("/", response_model=HippodromeList)
async def list_hippodromes(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    track_type: Optional[TrackType] = None,
    country: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Liste tous les hippodromes avec pagination et filtres
    """
    query = select(Hippodrome)

    # Apply filters
    if track_type:
        query = query.where(Hippodrome.track_type == track_type)
    if country:
        query = query.where(Hippodrome.country == country)
    if search:
        query = query.where(
            Hippodrome.name.ilike(f"%{search}%") |
            Hippodrome.city.ilike(f"%{search}%")
        )

    # Order by name
    query = query.order_by(Hippodrome.name)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Get paginated results
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    hippodromes = result.scalars().all()

    return HippodromeList(
        total=total,
        hippodromes=[HippodromeResponse.model_validate(h) for h in hippodromes]
    )


@router.get("/{hippodrome_id}", response_model=HippodromeResponse)
async def get_hippodrome(
    hippodrome_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère un hippodrome spécifique
    """
    query = select(Hippodrome).where(Hippodrome.hippodrome_id == hippodrome_id)
    result = await db.execute(query)
    hippodrome = result.scalar_one_or_none()

    if not hippodrome:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hippodrome with id {hippodrome_id} not found"
        )

    return HippodromeResponse.model_validate(hippodrome)


@router.get("/code/{code}", response_model=HippodromeResponse)
async def get_hippodrome_by_code(
    code: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère un hippodrome par son code
    """
    query = select(Hippodrome).where(Hippodrome.code == code.upper())
    result = await db.execute(query)
    hippodrome = result.scalar_one_or_none()

    if not hippodrome:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hippodrome with code {code} not found"
        )

    return HippodromeResponse.model_validate(hippodrome)


@router.post("/", response_model=HippodromeResponse, status_code=status.HTTP_201_CREATED)
async def create_hippodrome(
    hippodrome_in: HippodromeCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Crée un nouvel hippodrome (Admin uniquement)
    """
    # Check if code already exists
    existing_query = select(Hippodrome).where(Hippodrome.code == hippodrome_in.code)
    existing_result = await db.execute(existing_query)
    existing = existing_result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Hippodrome with code {hippodrome_in.code} already exists"
        )

    # Create hippodrome
    hippodrome = Hippodrome(**hippodrome_in.model_dump())
    db.add(hippodrome)
    await db.commit()
    await db.refresh(hippodrome)

    return HippodromeResponse.model_validate(hippodrome)


@router.put("/{hippodrome_id}", response_model=HippodromeResponse)
async def update_hippodrome(
    hippodrome_id: int,
    hippodrome_in: HippodromeUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Met à jour un hippodrome (Admin uniquement)
    """
    # Get hippodrome
    query = select(Hippodrome).where(Hippodrome.hippodrome_id == hippodrome_id)
    result = await db.execute(query)
    hippodrome = result.scalar_one_or_none()

    if not hippodrome:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hippodrome with id {hippodrome_id} not found"
        )

    # Update fields
    update_data = hippodrome_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(hippodrome, field, value)

    await db.commit()
    await db.refresh(hippodrome)

    return HippodromeResponse.model_validate(hippodrome)


@router.delete("/{hippodrome_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_hippodrome(
    hippodrome_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Supprime un hippodrome (Admin uniquement)
    """
    # Get hippodrome
    query = select(Hippodrome).where(Hippodrome.hippodrome_id == hippodrome_id)
    result = await db.execute(query)
    hippodrome = result.scalar_one_or_none()

    if not hippodrome:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hippodrome with id {hippodrome_id} not found"
        )

    await db.delete(hippodrome)
    await db.commit()

    return None
