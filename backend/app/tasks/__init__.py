# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

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