"""
Tâches Celery pour PronoTurf
"""

from .celery_app import celery_app
from . import sync_tasks, ml_tasks, notification_tasks

__all__ = [
    "celery_app",
    "sync_tasks",
    "ml_tasks",
    "notification_tasks",
]
