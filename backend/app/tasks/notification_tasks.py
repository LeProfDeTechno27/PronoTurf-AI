"""
Tâches Celery pour l'envoi de notifications

Tâches automatisées pour envoyer des notifications via Telegram et Email:
- Notifications de pronostics
- Alertes value bets
- Rappels de courses
- Résultats
- Rapports quotidiens
- Alertes bankroll
- Alertes favoris
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal

from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasks.celery_app import celery_app
from app.core.database import get_async_db
from app.models.user import User
from app.models.notification import Notification, NotificationType, NotificationChannel, NotificationStatus
from app.models.pronostic import Pronostic
from app.models.course import Course
from app.models.pari_simule import PariSimule
from app.models.favori import Favori, EntityType
from app.services.telegram_service import telegram_service
from app.services.email_service import email_service

logger = logging.getLogger(__name__)


@celery_app.task(name="send_new_pronostic_notifications")
def send_new_pronostic_notifications(pronostic_id: int) -> Dict[str, Any]:
    """
    Envoie des notifications pour un nouveau pronostic

    Args:
        pronostic_id: ID du pronostic

    Returns:
        Statistiques d'envoi
    """
    import asyncio
    return asyncio.run(_send_new_pronostic_notifications_async(pronostic_id))


async def _send_new_pronostic_notifications_async(pronostic_id: int) -> Dict[str, Any]:
    """Version async de send_new_pronostic_notifications"""
    stats = {
        "pronostic_id": pronostic_id,
        "notifications_sent": 0,
        "telegram_sent": 0,
        "email_sent": 0,
        "errors": 0
    }

    async for db in get_async_db():
        try:
            # Récupérer le pronostic avec la course
            stmt = (
                select(Pronostic)
                .where(Pronostic.pronostic_id == pronostic_id)
            )
            result = await db.execute(stmt)
            pronostic = result.scalar_one_or_none()

            if not pronostic:
                logger.error(f"Pronostic {pronostic_id} not found")
                return stats

            # Récupérer la course
            stmt = select(Course).where(Course.course_id == pronostic.course_id)
            result = await db.execute(stmt)
            course = result.scalar_one_or_none()

            if not course:
                logger.error(f"Course {pronostic.course_id} not found")
                return stats

            # Récupérer les utilisateurs abonnés aux notifications
            stmt = select(User).where(
                and_(
                    User.is_active == True,
                    or_(
                        User.telegram_notifications_enabled == True,
                        User.email_notifications_enabled == True
                    )
                )
            )
            result = await db.execute(stmt)
            users = result.scalars().all()

            # Préparer les données de la course
            course_info = {
                "course_id": course.course_id,
                "course_name": course.course_name,
                "hippodrome": course.reunion.hippodrome.name if course.reunion else "N/A",
                "scheduled_time": str(course.scheduled_time) if course.scheduled_time else "N/A"
            }

            # Préparer les prédictions (top 5)
            predictions = []
            if pronostic.gagnant_predicted:
                for pred in pronostic.gagnant_predicted[:5]:
                    predictions.append({
                        "horse_name": pred.get("horse_name", "N/A"),
                        "numero_corde": pred.get("numero", "?"),
                        "confidence_score": pred.get("confidence", 0) * 100,
                        "is_value_bet": pred.get("is_value_bet", False)
                    })

            # Envoyer les notifications
            for user in users:
                # Telegram
                if user.telegram_notifications_enabled and user.telegram_chat_id:
                    success = await telegram_service.send_pronostic_notification(
                        chat_id=user.telegram_chat_id,
                        course_info=course_info,
                        predictions=predictions
                    )

                    if success:
                        stats["telegram_sent"] += 1
                        stats["notifications_sent"] += 1
                    else:
                        stats["errors"] += 1

                # Email
                if user.email_notifications_enabled and user.email:
                    success = await email_service.send_pronostic_email(
                        to_email=user.email,
                        course_info=course_info,
                        predictions=predictions
                    )

                    if success:
                        stats["email_sent"] += 1
                        stats["notifications_sent"] += 1
                    else:
                        stats["errors"] += 1

                # Créer entrée de notification dans la BDD
                notification = Notification(
                    user_id=user.user_id,
                    notification_type=NotificationType.PRONOSTIC,
                    title=f"Nouveau Pronostic - {course_info['hippodrome']}",
                    message=f"Pronostic disponible pour {course_info['course_name']}",
                    related_course_id=course.course_id,
                    related_pronostic_id=pronostic.pronostic_id,
                    sent_via=NotificationChannel.IN_APP,
                    status=NotificationStatus.SENT
                )
                db.add(notification)

            await db.commit()

            logger.info(f"Pronostic notifications sent: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error sending pronostic notifications: {str(e)}")
            stats["errors"] += 1
            return stats


@celery_app.task(name="send_daily_reports")
def send_daily_reports() -> Dict[str, Any]:
    """
    Envoie les rapports quotidiens aux utilisateurs

    Returns:
        Statistiques d'envoi
    """
    import asyncio
    return asyncio.run(_send_daily_reports_async())


async def _send_daily_reports_async() -> Dict[str, Any]:
    """Version async de send_daily_reports"""
    stats = {
        "reports_sent": 0,
        "telegram_sent": 0,
        "email_sent": 0,
        "errors": 0
    }

    today = date.today()
    yesterday = today - timedelta(days=1)

    async for db in get_async_db():
        try:
            # Récupérer les utilisateurs avec notifications activées
            stmt = select(User).where(
                and_(
                    User.is_active == True,
                    or_(
                        User.telegram_notifications_enabled == True,
                        User.email_notifications_enabled == True
                    )
                )
            )
            result = await db.execute(stmt)
            users = result.scalars().all()

            for user in users:
                # Calculer les stats du jour
                stmt = (
                    select(PariSimule)
                    .where(
                        and_(
                            PariSimule.user_id == user.user_id,
                            PariSimule.created_at >= yesterday
                        )
                    )
                )
                result = await db.execute(stmt)
                paris = result.scalars().all()

                total_bets = len(paris)
                if total_bets == 0:
                    continue  # Skip si pas de paris

                wins = sum(1 for p in paris if p.resultat == "won")
                losses = sum(1 for p in paris if p.resultat == "lost")
                win_rate = (wins / total_bets * 100) if total_bets > 0 else 0

                total_mise = sum(p.montant for p in paris)
                total_gain = sum(p.gain or Decimal("0") for p in paris)
                profit = total_gain - total_mise
                roi = (profit / total_mise * 100) if total_mise > 0 else 0

                # Préparer le rapport
                report = {
                    "date": yesterday.strftime("%d/%m/%Y"),
                    "total_bets": total_bets,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": win_rate,
                    "total_mise": total_mise,
                    "total_gain": total_gain,
                    "profit": profit,
                    "roi": roi,
                    "current_bankroll": user.current_bankroll,
                    "bankroll_change": profit
                }

                # Envoyer Telegram
                if user.telegram_notifications_enabled and user.telegram_chat_id:
                    success = await telegram_service.send_daily_report(
                        chat_id=user.telegram_chat_id,
                        report=report
                    )
                    if success:
                        stats["telegram_sent"] += 1

                # Envoyer Email
                if user.email_notifications_enabled and user.email:
                    success = await email_service.send_daily_report_email(
                        to_email=user.email,
                        report=report
                    )
                    if success:
                        stats["email_sent"] += 1

                stats["reports_sent"] += 1

                # Créer notification dans la BDD
                notification = Notification(
                    user_id=user.user_id,
                    notification_type=NotificationType.DAILY_REPORT,
                    title=f"Rapport Quotidien - {report['date']}",
                    message=f"{total_bets} paris | Profit: {profit:+.2f}€",
                    sent_via=NotificationChannel.IN_APP,
                    status=NotificationStatus.SENT
                )
                db.add(notification)

            await db.commit()

            logger.info(f"Daily reports sent: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error sending daily reports: {str(e)}")
            stats["errors"] += 1
            return stats


@celery_app.task(name="send_bankroll_alerts")
def send_bankroll_alerts() -> Dict[str, Any]:
    """
    Envoie des alertes de bankroll critique

    Returns:
        Statistiques d'envoi
    """
    import asyncio
    return asyncio.run(_send_bankroll_alerts_async())


async def _send_bankroll_alerts_async() -> Dict[str, Any]:
    """Version async de send_bankroll_alerts"""
    stats = {
        "alerts_sent": 0,
        "telegram_sent": 0,
        "email_sent": 0,
        "errors": 0
    }

    async for db in get_async_db():
        try:
            # Récupérer les utilisateurs avec bankroll critique
            stmt = select(User).where(
                and_(
                    User.is_active == True,
                    User.is_bankroll_critical == True,
                    or_(
                        User.telegram_notifications_enabled == True,
                        User.email_notifications_enabled == True
                    )
                )
            )
            result = await db.execute(stmt)
            users = result.scalars().all()

            for user in users:
                current_bankroll = user.current_bankroll
                initial_bankroll = user.initial_bankroll
                percentage = user.bankroll_percentage

                # Envoyer Telegram
                if user.telegram_notifications_enabled and user.telegram_chat_id:
                    success = await telegram_service.send_bankroll_alert(
                        chat_id=user.telegram_chat_id,
                        current_bankroll=current_bankroll,
                        initial_bankroll=initial_bankroll,
                        percentage=percentage
                    )
                    if success:
                        stats["telegram_sent"] += 1

                # Envoyer Email
                if user.email_notifications_enabled and user.email:
                    success = await email_service.send_bankroll_alert_email(
                        to_email=user.email,
                        current_bankroll=current_bankroll,
                        initial_bankroll=initial_bankroll,
                        percentage=percentage
                    )
                    if success:
                        stats["email_sent"] += 1

                stats["alerts_sent"] += 1

                # Créer notification dans la BDD
                notification = Notification(
                    user_id=user.user_id,
                    notification_type=NotificationType.RESULT,
                    title="⚠️ Alerte Bankroll Critique",
                    message=f"Votre bankroll est à {percentage:.1f}%",
                    sent_via=NotificationChannel.IN_APP,
                    status=NotificationStatus.SENT
                )
                db.add(notification)

            await db.commit()

            logger.info(f"Bankroll alerts sent: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error sending bankroll alerts: {str(e)}")
            stats["errors"] += 1
            return stats
