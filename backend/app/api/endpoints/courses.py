"""
Router pour les endpoints des courses hippiques
"""

from typing import List, Optional
from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.deps import get_current_user, get_current_subscriber
from app.models import Course, Reunion, Hippodrome, Partant, User
from app.models.course import CourseStatus, Discipline, SurfaceType
from app.schemas.course import (
    CourseCreate,
    CourseUpdate,
    CourseResponse,
    CourseWithReunion,
    CourseList,
    CourseDetailResponse,
    CourseFilter,
)

router = APIRouter()


@router.get("/today", response_model=CourseList)
async def get_today_courses(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    discipline: Optional[Discipline] = None,
    status: Optional[CourseStatus] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère les courses du jour
    """
    today = date.today()

    # Build query
    query = (
        select(Course)
        .join(Reunion)
        .where(Reunion.reunion_date == today)
        .options(
            selectinload(Course.reunion).selectinload(Reunion.hippodrome)
        )
        .order_by(Course.scheduled_time)
    )

    # Apply filters
    if discipline:
        query = query.where(Course.discipline == discipline)
    if status:
        query = query.where(Course.status == status)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Get paginated results
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    courses = result.scalars().all()

    return CourseList(
        total=total,
        courses=[CourseWithReunion.model_validate(course) for course in courses]
    )


@router.get("/upcoming", response_model=CourseList)
async def get_upcoming_courses(
    days: int = Query(7, ge=1, le=30, description="Nombre de jours à venir"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    discipline: Optional[Discipline] = None,
    hippodrome_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère les courses à venir dans les N prochains jours
    """
    today = date.today()
    future_date = date.fromordinal(today.toordinal() + days)

    # Build query
    query = (
        select(Course)
        .join(Reunion)
        .where(
            and_(
                Reunion.reunion_date >= today,
                Reunion.reunion_date <= future_date,
                Course.status == CourseStatus.SCHEDULED
            )
        )
        .options(
            selectinload(Course.reunion).selectinload(Reunion.hippodrome)
        )
        .order_by(Reunion.reunion_date, Course.scheduled_time)
    )

    # Apply filters
    if discipline:
        query = query.where(Course.discipline == discipline)
    if hippodrome_id:
        query = query.where(Reunion.hippodrome_id == hippodrome_id)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Get paginated results
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    courses = result.scalars().all()

    return CourseList(
        total=total,
        courses=[CourseWithReunion.model_validate(course) for course in courses]
    )


@router.get("/{course_id}", response_model=CourseDetailResponse)
async def get_course(
    course_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère les détails d'une course spécifique
    """
    query = (
        select(Course)
        .where(Course.course_id == course_id)
        .options(
            selectinload(Course.reunion).selectinload(Reunion.hippodrome),
            selectinload(Course.partants),
            selectinload(Course.pronostics)
        )
    )

    result = await db.execute(query)
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Course with id {course_id} not found"
        )

    # Build detailed response
    course_dict = CourseWithReunion.model_validate(course).model_dump()
    course_dict["number_of_partants"] = len(course.partants)
    course_dict["has_pronostic"] = bool(course.pronostics)

    return CourseDetailResponse(**course_dict)


@router.get("/{course_id}/partants")
async def get_course_partants(
    course_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Récupère les partants d'une course
    """
    # Verify course exists
    course_query = select(Course).where(Course.course_id == course_id)
    course_result = await db.execute(course_query)
    course = course_result.scalar_one_or_none()

    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Course with id {course_id} not found"
        )

    # Get partants with relations
    partants_query = (
        select(Partant)
        .where(Partant.course_id == course_id)
        .options(
            selectinload(Partant.horse),
            selectinload(Partant.jockey),
            selectinload(Partant.trainer)
        )
        .order_by(Partant.numero_corde)
    )

    result = await db.execute(partants_query)
    partants = result.scalars().all()

    return {
        "course_id": course_id,
        "course_name": course.course_name,
        "total_partants": len(partants),
        "partants": partants
    }


@router.post("/", response_model=CourseResponse, status_code=status.HTTP_201_CREATED)
async def create_course(
    course_in: CourseCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Crée une nouvelle course (Admin uniquement via dependency)
    """
    # Verify reunion exists
    reunion_query = select(Reunion).where(Reunion.reunion_id == course_in.reunion_id)
    reunion_result = await db.execute(reunion_query)
    reunion = reunion_result.scalar_one_or_none()

    if not reunion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reunion with id {course_in.reunion_id} not found"
        )

    # Create course
    course = Course(**course_in.model_dump())
    db.add(course)
    await db.commit()
    await db.refresh(course)

    return CourseResponse.model_validate(course)


@router.put("/{course_id}", response_model=CourseResponse)
async def update_course(
    course_id: int,
    course_in: CourseUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Met à jour une course (Admin uniquement via dependency)
    """
    # Get course
    query = select(Course).where(Course.course_id == course_id)
    result = await db.execute(query)
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Course with id {course_id} not found"
        )

    # Update fields
    update_data = course_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(course, field, value)

    await db.commit()
    await db.refresh(course)

    return CourseResponse.model_validate(course)


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course(
    course_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Supprime une course (Admin uniquement via dependency)
    """
    # Get course
    query = select(Course).where(Course.course_id == course_id)
    result = await db.execute(query)
    course = result.scalar_one_or_none()

    if not course:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Course with id {course_id} not found"
        )

    await db.delete(course)
    await db.commit()

    return None


@router.post("/filter", response_model=CourseList)
async def filter_courses(
    filters: CourseFilter,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    Filtre les courses selon plusieurs critères
    """
    # Build base query
    query = (
        select(Course)
        .join(Reunion)
        .options(
            selectinload(Course.reunion).selectinload(Reunion.hippodrome)
        )
    )

    # Apply filters
    conditions = []

    if filters.reunion_id:
        conditions.append(Course.reunion_id == filters.reunion_id)
    if filters.discipline:
        conditions.append(Course.discipline == filters.discipline)
    if filters.status:
        conditions.append(Course.status == filters.status)
    if filters.hippodrome_id:
        conditions.append(Reunion.hippodrome_id == filters.hippodrome_id)
    if filters.min_distance:
        conditions.append(Course.distance >= filters.min_distance)
    if filters.max_distance:
        conditions.append(Course.distance <= filters.max_distance)
    if filters.surface_type:
        conditions.append(Course.surface_type == filters.surface_type)

    if conditions:
        query = query.where(and_(*conditions))

    # Order by date and time
    query = query.order_by(Reunion.reunion_date, Course.scheduled_time)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Get paginated results
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    courses = result.scalars().all()

    return CourseList(
        total=total,
        courses=[CourseWithReunion.model_validate(course) for course in courses]
    )
