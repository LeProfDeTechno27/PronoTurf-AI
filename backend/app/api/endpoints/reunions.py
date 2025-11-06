# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Router pour les endpoints des réunions
"""

from typing import Optional
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models import Reunion, Hippodrome, Course, User
from app.models.reunion import ReunionStatus
from app.schemas.reunion import (
    ReunionCreate,
    ReunionUpdate,
    ReunionResponse,
    ReunionWithHippodrome,
    ReunionList,
    ReunionDetailResponse,
)
from app.services import WeatherService

router = APIRouter()


@router.get("/today", response_model=ReunionList)
async def get_today_reunions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère les réunions du jour
    """
    today = date.today()

    query = (
        select(Reunion)
        .where(Reunion.reunion_date == today)
        .options(selectinload(Reunion.hippodrome))
        .order_by(Reunion.reunion_number)
    )

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Get paginated results
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    reunions = result.scalars().all()

    return ReunionList(
        total=total,
        reunions=[ReunionWithHippodrome.model_validate(r) for r in reunions]
    )


@router.get("/date/{reunion_date}", response_model=ReunionList)
async def get_reunions_by_date(
    reunion_date: date,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    status_filter: Optional[ReunionStatus] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère les réunions pour une date spécifique
    """
    query = (
        select(Reunion)
        .where(Reunion.reunion_date == reunion_date)
        .options(selectinload(Reunion.hippodrome))
        .order_by(Reunion.reunion_number)
    )

    if status_filter:
        query = query.where(Reunion.status == status_filter)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Get paginated results
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    reunions = result.scalars().all()

    return ReunionList(
        total=total,
        reunions=[ReunionWithHippodrome.model_validate(r) for r in reunions]
    )


@router.get("/{reunion_id}", response_model=ReunionDetailResponse)
async def get_reunion(
    reunion_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère les détails d'une réunion spécifique
    """
    query = (
        select(Reunion)
        .where(Reunion.reunion_id == reunion_id)
        .options(
            selectinload(Reunion.hippodrome),
            selectinload(Reunion.courses)
        )
    )

    result = await db.execute(query)
    reunion = result.scalar_one_or_none()

    if not reunion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reunion with id {reunion_id} not found"
        )

    # Build detailed response
    reunion_dict = ReunionWithHippodrome.model_validate(reunion).model_dump()
    reunion_dict["number_of_courses"] = len(reunion.courses)

    return ReunionDetailResponse(**reunion_dict)


@router.get("/{reunion_id}/courses")
async def get_reunion_courses(
    reunion_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère toutes les courses d'une réunion
    """
    # Verify reunion exists
    reunion_query = select(Reunion).where(Reunion.reunion_id == reunion_id)
    reunion_result = await db.execute(reunion_query)
    reunion = reunion_result.scalar_one_or_none()

    if not reunion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reunion with id {reunion_id} not found"
        )

    # Get courses
    courses_query = (
        select(Course)
        .where(Course.reunion_id == reunion_id)
        .order_by(Course.course_number)
    )

    result = await db.execute(courses_query)
    courses = result.scalars().all()

    return {
        "reunion_id": reunion_id,
        "reunion_date": reunion.reunion_date,
        "hippodrome": reunion.hippodrome,
        "total_courses": len(courses),
        "courses": courses
    }


@router.get("/{reunion_id}/weather")
async def get_reunion_weather(
    reunion_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère les conditions météo pour une réunion
    """
    # Get reunion with hippodrome
    query = (
        select(Reunion)
        .where(Reunion.reunion_id == reunion_id)
        .options(selectinload(Reunion.hippodrome))
    )

    result = await db.execute(query)
    reunion = result.scalar_one_or_none()

    if not reunion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reunion with id {reunion_id} not found"
        )

    # Get weather from service
    weather_service = WeatherService()

    weather = await weather_service.get_weather_for_hippodrome(
        hippodrome_code=reunion.hippodrome.code,
        hippodrome_name=reunion.hippodrome.name,
        latitude=reunion.hippodrome.latitude,
        longitude=reunion.hippodrome.longitude,
        target_date=reunion.reunion_date
    )

    return {
        "reunion_id": reunion_id,
        "reunion_date": reunion.reunion_date,
        "hippodrome": {
            "code": reunion.hippodrome.code,
            "name": reunion.hippodrome.name,
            "city": reunion.hippodrome.city,
        },
        "weather": weather
    }


@router.post("/", response_model=ReunionResponse, status_code=status.HTTP_201_CREATED)
async def create_reunion(
    reunion_in: ReunionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Crée une nouvelle réunion (Admin uniquement)
    """
    # Verify hippodrome exists
    hippodrome_query = select(Hippodrome).where(
        Hippodrome.hippodrome_id == reunion_in.hippodrome_id
    )
    hippodrome_result = await db.execute(hippodrome_query)
    hippodrome = hippodrome_result.scalar_one_or_none()

    if not hippodrome:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hippodrome with id {reunion_in.hippodrome_id} not found"
        )

    # Check if reunion already exists
    existing_query = select(Reunion).where(
        and_(
            Reunion.hippodrome_id == reunion_in.hippodrome_id,
            Reunion.reunion_date == reunion_in.reunion_date,
            Reunion.reunion_number == reunion_in.reunion_number
        )
    )
    existing_result = await db.execute(existing_query)
    existing = existing_result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reunion already exists for this hippodrome, date, and number"
        )

    # Create reunion
    reunion = Reunion(**reunion_in.model_dump())
    db.add(reunion)
    await db.commit()
    await db.refresh(reunion)

    return ReunionResponse.model_validate(reunion)


@router.put("/{reunion_id}", response_model=ReunionResponse)
async def update_reunion(
    reunion_id: int,
    reunion_in: ReunionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Met à jour une réunion (Admin uniquement)
    """
    # Get reunion
    query = select(Reunion).where(Reunion.reunion_id == reunion_id)
    result = await db.execute(query)
    reunion = result.scalar_one_or_none()

    if not reunion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reunion with id {reunion_id} not found"
        )

    # Update fields
    update_data = reunion_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(reunion, field, value)

    await db.commit()
    await db.refresh(reunion)

    return ReunionResponse.model_validate(reunion)


@router.delete("/{reunion_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_reunion(
    reunion_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Supprime une réunion (Admin uniquement)
    """
    # Get reunion
    query = select(Reunion).where(Reunion.reunion_id == reunion_id)
    result = await db.execute(query)
    reunion = result.scalar_one_or_none()

    if not reunion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reunion with id {reunion_id} not found"
        )

    await db.delete(reunion)
    await db.commit()

    return None