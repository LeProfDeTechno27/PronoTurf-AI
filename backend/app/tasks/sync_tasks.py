# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Tâches Celery pour la synchronisation des données PMU
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, Any

from celery import Task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.tasks.celery_app import celery_app
from app.core.config import settings
from app.models import Course, Reunion, Hippodrome, Partant
from app.models.course import CourseStatus
from app.models.reunion import ReunionStatus
from app.services import PMUService, WeatherService

logger = logging.getLogger(__name__)

# Create async engine and session factory for Celery tasks
engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class AsyncTask(Task):
    """Base task class for async operations"""

    async def run_async(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.run_async(*args, **kwargs))


@celery_app.task(base=AsyncTask, bind=True, max_retries=3)
async def sync_daily_programs(self):
    """
    Synchronise les programmes de courses pour aujourd'hui et demain
    """
    logger.info("Starting daily program synchronization")

    async with async_session_factory() as db:
        pmu_service = PMUService(db)

        # Synchroniser aujourd'hui
        today = date.today()
        try:
            stats_today = await pmu_service.sync_program_for_date(today)
            logger.info(f"Synced program for {today}: {stats_today}")
        except Exception as e:
            logger.error(f"Error syncing program for {today}: {e}")
            raise self.retry(exc=e, countdown=300)  # Retry in 5 minutes

        # Synchroniser demain
        tomorrow = today + timedelta(days=1)
        try:
            stats_tomorrow = await pmu_service.sync_program_for_date(tomorrow)
            logger.info(f"Synced program for {tomorrow}: {stats_tomorrow}")
        except Exception as e:
            logger.error(f"Error syncing program for {tomorrow}: {e}")

        # Récupérer la météo pour toutes les réunions d'aujourd'hui
        await sync_weather_for_date(db, today)

    logger.info("Daily program synchronization completed")
    return {"today": stats_today, "tomorrow": stats_tomorrow}


@celery_app.task(base=AsyncTask, bind=True, max_retries=5)
async def update_odds(self):
    """
    Met à jour les cotes PMU pour les courses du jour
    """
    logger.info("Starting odds update")

    async with async_session_factory() as db:
        today = date.today()

        # Récupérer toutes les courses du jour non terminées
        query = (
            select(Course)
            .join(Reunion)
            .where(
                Reunion.reunion_date == today,
                Course.status.in_([CourseStatus.SCHEDULED, CourseStatus.RUNNING])
            )
        )

        result = await db.execute(query)
        courses = result.scalars().all()

        updated_count = 0
        error_count = 0

        pmu_service = PMUService(db)

        for course in courses:
            try:
                # Récupérer les détails de la course depuis PMU
                course_id = f"R{course.reunion.reunion_number}C{course.course_number}"
                course_data = await pmu_service.fetch_course_details(course_id)

                # Mettre à jour les cotes des partants
                partants_data = course_data.get("partants", [])

                for partant_data in partants_data:
                    numero = partant_data.get("numero")
                    new_odds = partant_data.get("cote")

                    if numero and new_odds:
                        # Trouver le partant correspondant
                        partant_query = select(Partant).where(
                            Partant.course_id == course.course_id,
                            Partant.numero_corde == numero
                        )
                        partant_result = await db.execute(partant_query)
                        partant = partant_result.scalar_one_or_none()

                        if partant:
                            partant.odds_pmu = new_odds
                            updated_count += 1

                await db.commit()

            except Exception as e:
                logger.error(f"Error updating odds for course {course.course_id}: {e}")
                error_count += 1
                await db.rollback()

    logger.info(f"Odds update completed: {updated_count} updated, {error_count} errors")
    return {"updated": updated_count, "errors": error_count}


@celery_app.task(base=AsyncTask, bind=True, max_retries=3)
async def check_race_results(self):
    """
    Vérifie et met à jour les résultats des courses terminées
    """
    logger.info("Starting race results check")

    async with async_session_factory() as db:
        today = date.today()

        # Récupérer les courses du jour en cours ou programmées
        query = (
            select(Course)
            .join(Reunion)
            .where(
                Reunion.reunion_date == today,
                Course.status.in_([CourseStatus.SCHEDULED, CourseStatus.RUNNING])
            )
        )

        result = await db.execute(query)
        courses = result.scalars().all()

        finished_count = 0
        error_count = 0

        pmu_service = PMUService(db)

        for course in courses:
            try:
                # Récupérer les résultats depuis PMU
                course_id = f"R{course.reunion.reunion_number}C{course.course_number}"
                course_data = await pmu_service.fetch_course_details(course_id)

                # Vérifier si la course est terminée
                course_status = course_data.get("statut")
                if course_status and course_status.lower() == "finished":
                    # Mettre à jour le statut de la course
                    course.status = CourseStatus.FINISHED

                    # Mettre à jour les résultats des partants
                    results = course_data.get("arrivee", [])

                    for position, partant_data in enumerate(results, start=1):
                        numero = partant_data.get("numero")

                        if numero:
                            partant_query = select(Partant).where(
                                Partant.course_id == course.course_id,
                                Partant.numero_corde == numero
                            )
                            partant_result = await db.execute(partant_query)
                            partant = partant_result.scalar_one_or_none()

                            if partant:
                                partant.final_position = position
                                partant.disqualified = partant_data.get("disqualified", False)

                    finished_count += 1
                    await db.commit()

                    logger.info(f"Updated results for course {course.course_id}")

            except Exception as e:
                logger.error(f"Error checking results for course {course.course_id}: {e}")
                error_count += 1
                await db.rollback()

        # Mettre à jour le statut des réunions
        await update_reunion_status(db, today)

    logger.info(f"Race results check completed: {finished_count} finished, {error_count} errors")
    return {"finished": finished_count, "errors": error_count}


@celery_app.task(base=AsyncTask)
async def sync_specific_date(target_date: str):
    """
    Synchronise le programme pour une date spécifique

    Args:
        target_date: Date au format YYYY-MM-DD
    """
    logger.info(f"Starting program synchronization for {target_date}")

    target = date.fromisoformat(target_date)

    async with async_session_factory() as db:
        pmu_service = PMUService(db)

        try:
            stats = await pmu_service.sync_program_for_date(target)
            logger.info(f"Synced program for {target}: {stats}")

            # Récupérer la météo
            await sync_weather_for_date(db, target)

            return stats

        except Exception as e:
            logger.error(f"Error syncing program for {target}: {e}")
            raise


async def sync_weather_for_date(db: AsyncSession, target_date: date):
    """
    Synchronise les données météo pour toutes les réunions d'une date

    Args:
        db: Session de base de données
        target_date: Date cible
    """
    logger.info(f"Syncing weather data for {target_date}")

    weather_service = WeatherService()

    # Récupérer toutes les réunions de la date
    query = (
        select(Reunion)
        .join(Hippodrome)
        .where(Reunion.reunion_date == target_date)
    )

    result = await db.execute(query)
    reunions = result.scalars().all()

    for reunion in reunions:
        try:
            if reunion.hippodrome.latitude and reunion.hippodrome.longitude:
                weather = await weather_service.get_weather_for_hippodrome(
                    hippodrome_code=reunion.hippodrome.code,
                    hippodrome_name=reunion.hippodrome.name,
                    latitude=reunion.hippodrome.latitude,
                    longitude=reunion.hippodrome.longitude,
                    target_date=target_date
                )

                reunion.weather_conditions = weather
                logger.info(f"Updated weather for reunion {reunion.reunion_id}")

        except Exception as e:
            logger.error(f"Error fetching weather for reunion {reunion.reunion_id}: {e}")

    await db.commit()


async def update_reunion_status(db: AsyncSession, target_date: date):
    """
    Met à jour le statut des réunions selon l'état de leurs courses

    Args:
        db: Session de base de données
        target_date: Date cible
    """
    query = select(Reunion).where(Reunion.reunion_date == target_date)
    result = await db.execute(query)
    reunions = result.scalars().all()

    for reunion in reunions:
        # Compter les courses par statut
        courses_query = select(Course).where(Course.reunion_id == reunion.reunion_id)
        courses_result = await db.execute(courses_query)
        courses = courses_result.scalars().all()

        if not courses:
            continue

        finished_count = sum(1 for c in courses if c.status == CourseStatus.FINISHED)
        running_count = sum(1 for c in courses if c.status == CourseStatus.RUNNING)
        total_count = len(courses)

        # Déterminer le nouveau statut
        if finished_count == total_count:
            reunion.status = ReunionStatus.COMPLETED
        elif running_count > 0 or finished_count > 0:
            reunion.status = ReunionStatus.ONGOING
        else:
            reunion.status = ReunionStatus.SCHEDULED

    await db.commit()


@celery_app.task(base=AsyncTask)
async def cleanup_old_data(days_to_keep: int = 90):
    """
    Nettoie les anciennes données (courses terminées de plus de X jours)

    Args:
        days_to_keep: Nombre de jours de données à conserver
    """
    logger.info(f"Starting data cleanup (keeping last {days_to_keep} days)")

    cutoff_date = date.today() - timedelta(days=days_to_keep)

    async with async_session_factory() as db:
        # Cette tâche pourrait supprimer:
        # - Les anciennes courses terminées
        # - Les anciens paris simulés
        # - Les anciennes notifications lues
        # Pour l'instant, on garde tout pour l'historique

        logger.info("Data cleanup completed (no action taken - keeping all data)")

    return {"cutoff_date": cutoff_date.isoformat(), "action": "none"}