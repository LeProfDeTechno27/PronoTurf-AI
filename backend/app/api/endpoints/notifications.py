"""
Endpoints API pour la gestion des notifications utilisateur

Routes disponibles:
- GET /api/v1/notifications : Liste des notifications
- GET /api/v1/notifications/unread : Notifications non lues
- PATCH /api/v1/notifications/{notification_id}/read : Marquer comme lu
- PATCH /api/v1/notifications/read-all : Tout marquer comme lu
- DELETE /api/v1/notifications/{notification_id} : Supprimer
- DELETE /api/v1/notifications/clear : Supprimer toutes les lues
- GET /api/v1/notifications/stats : Statistiques
"""

from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func, or_

from app.core.database import get_async_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.notification import (
    Notification,
    NotificationType,
    NotificationChannel,
    NotificationStatus
)
from app.schemas.notification import (
    NotificationResponse,
    NotificationStatsResponse,
    NotificationMarkReadRequest
)

router = APIRouter()


@router.get("/", response_model=List[NotificationResponse])
async def get_notifications(
    skip: int = Query(0, ge=0, description="Nombre d'entrées à ignorer"),
    limit: int = Query(50, ge=1, le=200, description="Nombre d'entrées à retourner"),
    notification_type: Optional[NotificationType] = Query(None, description="Filtrer par type"),
    status_filter: Optional[NotificationStatus] = Query(None, description="Filtrer par statut"),
    unread_only: bool = Query(False, description="Seulement les non lues"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Récupère les notifications de l'utilisateur

    Args:
        skip: Pagination - nombre d'entrées à ignorer
        limit: Pagination - nombre d'entrées à retourner
        notification_type: Filtrer par type de notification
        status_filter: Filtrer par statut (pending, sent, failed)
        unread_only: Ne retourner que les notifications non lues

    Returns:
        Liste des notifications
    """
    stmt = (
        select(Notification)
        .where(Notification.user_id == current_user.user_id)
        .order_by(desc(Notification.sent_at))
    )

    # Appliquer les filtres
    if notification_type:
        stmt = stmt.where(Notification.notification_type == notification_type)

    if status_filter:
        stmt = stmt.where(Notification.status == status_filter)

    if unread_only:
        stmt = stmt.where(Notification.read_at.is_(None))

    # Pagination
    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    notifications = result.scalars().all()

    return [
        NotificationResponse(
            notification_id=notif.notification_id,
            user_id=notif.user_id,
            notification_type=notif.notification_type,
            title=notif.title,
            message=notif.message,
            related_course_id=notif.related_course_id,
            related_pronostic_id=notif.related_pronostic_id,
            sent_via=notif.sent_via,
            sent_at=notif.sent_at,
            read_at=notif.read_at,
            status=notif.status,
            is_read=notif.is_read
        )
        for notif in notifications
    ]


@router.get("/unread", response_model=List[NotificationResponse])
async def get_unread_notifications(
    limit: int = Query(50, ge=1, le=200, description="Nombre max de notifications"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Récupère uniquement les notifications non lues

    Args:
        limit: Nombre maximum de notifications à retourner

    Returns:
        Liste des notifications non lues
    """
    stmt = (
        select(Notification)
        .where(
            and_(
                Notification.user_id == current_user.user_id,
                Notification.read_at.is_(None)
            )
        )
        .order_by(desc(Notification.sent_at))
        .limit(limit)
    )

    result = await db.execute(stmt)
    notifications = result.scalars().all()

    return [
        NotificationResponse(
            notification_id=notif.notification_id,
            user_id=notif.user_id,
            notification_type=notif.notification_type,
            title=notif.title,
            message=notif.message,
            related_course_id=notif.related_course_id,
            related_pronostic_id=notif.related_pronostic_id,
            sent_via=notif.sent_via,
            sent_at=notif.sent_at,
            read_at=notif.read_at,
            status=notif.status,
            is_read=notif.is_read
        )
        for notif in notifications
    ]


@router.patch("/{notification_id}/read", response_model=NotificationResponse)
async def mark_notification_as_read(
    notification_id: int = Path(..., description="ID de la notification"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Marque une notification comme lue

    Args:
        notification_id: ID de la notification

    Returns:
        Notification mise à jour

    Raises:
        HTTPException 404: Si la notification n'existe pas
        HTTPException 403: Si la notification n'appartient pas à l'utilisateur
    """
    stmt = select(Notification).where(Notification.notification_id == notification_id)
    result = await db.execute(stmt)
    notification = result.scalar_one_or_none()

    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notification {notification_id} introuvable"
        )

    if notification.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cette notification ne vous appartient pas"
        )

    # Marquer comme lue
    notification.mark_as_read()

    await db.commit()
    await db.refresh(notification)

    return NotificationResponse(
        notification_id=notification.notification_id,
        user_id=notification.user_id,
        notification_type=notification.notification_type,
        title=notification.title,
        message=notification.message,
        related_course_id=notification.related_course_id,
        related_pronostic_id=notification.related_pronostic_id,
        sent_via=notification.sent_via,
        sent_at=notification.sent_at,
        read_at=notification.read_at,
        status=notification.status,
        is_read=notification.is_read
    )


@router.patch("/read-all", status_code=status.HTTP_200_OK)
async def mark_all_as_read(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Marque toutes les notifications comme lues

    Returns:
        Nombre de notifications marquées comme lues
    """
    stmt = (
        select(Notification)
        .where(
            and_(
                Notification.user_id == current_user.user_id,
                Notification.read_at.is_(None)
            )
        )
    )

    result = await db.execute(stmt)
    notifications = result.scalars().all()

    count = 0
    for notification in notifications:
        notification.mark_as_read()
        count += 1

    await db.commit()

    return {"message": f"{count} notification(s) marquée(s) comme lue(s)", "count": count}


@router.delete("/{notification_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_notification(
    notification_id: int = Path(..., description="ID de la notification"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Supprime une notification

    Args:
        notification_id: ID de la notification

    Raises:
        HTTPException 404: Si la notification n'existe pas
        HTTPException 403: Si la notification n'appartient pas à l'utilisateur
    """
    stmt = select(Notification).where(Notification.notification_id == notification_id)
    result = await db.execute(stmt)
    notification = result.scalar_one_or_none()

    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notification {notification_id} introuvable"
        )

    if notification.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cette notification ne vous appartient pas"
        )

    await db.delete(notification)
    await db.commit()


@router.delete("/clear", status_code=status.HTTP_200_OK)
async def clear_read_notifications(
    older_than_days: int = Query(7, ge=1, le=365, description="Supprimer les notifications lues plus anciennes que X jours"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Supprime toutes les notifications lues anciennes

    Args:
        older_than_days: Supprimer les notifications lues de plus de X jours

    Returns:
        Nombre de notifications supprimées
    """
    cutoff_date = datetime.now() - timedelta(days=older_than_days)

    stmt = (
        select(Notification)
        .where(
            and_(
                Notification.user_id == current_user.user_id,
                Notification.read_at.isnot(None),
                Notification.read_at < cutoff_date
            )
        )
    )

    result = await db.execute(stmt)
    notifications = result.scalars().all()

    count = len(notifications)
    for notification in notifications:
        await db.delete(notification)

    await db.commit()

    return {"message": f"{count} notification(s) supprimée(s)", "count": count}


@router.get("/stats", response_model=NotificationStatsResponse)
async def get_notification_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Récupère les statistiques de notifications

    Returns:
        Statistiques complètes des notifications
    """
    # Compter les notifications par catégorie
    total_stmt = (
        select(func.count(Notification.notification_id))
        .where(Notification.user_id == current_user.user_id)
    )
    result = await db.execute(total_stmt)
    total_notifications = result.scalar() or 0

    unread_stmt = (
        select(func.count(Notification.notification_id))
        .where(
            and_(
                Notification.user_id == current_user.user_id,
                Notification.read_at.is_(None)
            )
        )
    )
    result = await db.execute(unread_stmt)
    unread_count = result.scalar() or 0

    # Par type
    type_stmt = (
        select(
            Notification.notification_type,
            func.count(Notification.notification_id).label("count")
        )
        .where(Notification.user_id == current_user.user_id)
        .group_by(Notification.notification_type)
    )
    result = await db.execute(type_stmt)
    by_type = {row.notification_type.value: row.count for row in result.all()}

    # Par canal
    channel_stmt = (
        select(
            Notification.sent_via,
            func.count(Notification.notification_id).label("count")
        )
        .where(Notification.user_id == current_user.user_id)
        .group_by(Notification.sent_via)
    )
    result = await db.execute(channel_stmt)
    by_channel = {row.sent_via.value: row.count for row in result.all()}

    # Par statut
    status_stmt = (
        select(
            Notification.status,
            func.count(Notification.notification_id).label("count")
        )
        .where(Notification.user_id == current_user.user_id)
        .group_by(Notification.status)
    )
    result = await db.execute(status_stmt)
    by_status = {row.status.value: row.count for row in result.all()}

    # Dernières 24h
    last_24h_cutoff = datetime.now() - timedelta(hours=24)
    recent_stmt = (
        select(func.count(Notification.notification_id))
        .where(
            and_(
                Notification.user_id == current_user.user_id,
                Notification.sent_at >= last_24h_cutoff
            )
        )
    )
    result = await db.execute(recent_stmt)
    last_24h_count = result.scalar() or 0

    return NotificationStatsResponse(
        total_notifications=total_notifications,
        unread_count=unread_count,
        read_count=total_notifications - unread_count,
        by_type=by_type,
        by_channel=by_channel,
        by_status=by_status,
        last_24h_count=last_24h_count
    )


@router.get("/recent", response_model=List[NotificationResponse])
async def get_recent_notifications(
    hours: int = Query(24, ge=1, le=168, description="Dernières X heures"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Récupère les notifications récentes

    Args:
        hours: Nombre d'heures en arrière (défaut: 24h, max: 7 jours)

    Returns:
        Liste des notifications récentes
    """
    cutoff_date = datetime.now() - timedelta(hours=hours)

    stmt = (
        select(Notification)
        .where(
            and_(
                Notification.user_id == current_user.user_id,
                Notification.sent_at >= cutoff_date
            )
        )
        .order_by(desc(Notification.sent_at))
    )

    result = await db.execute(stmt)
    notifications = result.scalars().all()

    return [
        NotificationResponse(
            notification_id=notif.notification_id,
            user_id=notif.user_id,
            notification_type=notif.notification_type,
            title=notif.title,
            message=notif.message,
            related_course_id=notif.related_course_id,
            related_pronostic_id=notif.related_pronostic_id,
            sent_via=notif.sent_via,
            sent_at=notif.sent_at,
            read_at=notif.read_at,
            status=notif.status,
            is_read=notif.is_read
        )
        for notif in notifications
    ]
