# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Configuration de l'application Celery pour PronoTurf
"""

from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "pronoturf",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.sync_tasks", "app.tasks.ml_tasks", "app.tasks.notification_tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Paris",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Celery Beat schedule (tâches planifiées)
celery_app.conf.beat_schedule = {
    # Synchronisation quotidienne des programmes à 6h du matin
    "sync-daily-programs": {
        "task": "app.tasks.sync_tasks.sync_daily_programs",
        "schedule": crontab(hour=6, minute=0),
        "args": (),
    },
    # Mise à jour des cotes toutes les 30 minutes pendant la journée
    "update-odds": {
        "task": "app.tasks.sync_tasks.update_odds",
        "schedule": crontab(minute="*/30", hour="9-22"),
        "args": (),
    },
    # Vérification des résultats toutes les heures
    "check-race-results": {
        "task": "app.tasks.sync_tasks.check_race_results",
        "schedule": crontab(minute=0),
        "args": (),
    },
    # Génération des pronostics à 7h du matin
    "generate-daily-predictions": {
        "task": "app.tasks.ml_tasks.generate_daily_predictions",
        "schedule": crontab(hour=7, minute=0),
        "args": (),
    },
    # Entraînement hebdomadaire du modèle ML (dimanche à 2h du matin)
    "train-ml-model": {
        "task": "app.tasks.ml_tasks.train_ml_model",
        "schedule": crontab(hour=2, minute=0, day_of_week=0),
        "args": (),
    },
    # Rapport quotidien à 20h
    "send-daily-reports": {
        "task": "app.tasks.notification_tasks.send_daily_reports",
        "schedule": crontab(hour=20, minute=0),
        "args": (),
    },
}

# Task routes (pour diriger certaines tâches vers des workers spécifiques)
celery_app.conf.task_routes = {
    "app.tasks.sync_tasks.*": {"queue": "sync"},
    "app.tasks.ml_tasks.*": {"queue": "ml"},
    "app.tasks.notification_tasks.*": {"queue": "notifications"},
}