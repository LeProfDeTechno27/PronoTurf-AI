"""
Schémas Pydantic pour les notifications
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.models.notification import NotificationType, NotificationChannel, NotificationStatus


class NotificationBase(BaseModel):
    """Schéma de base pour une notification"""
    notification_type: NotificationType = Field(..., description="Type de notification")
    title: str = Field(..., min_length=1, max_length=300, description="Titre")
    message: str = Field(..., min_length=1, description="Message")
    related_course_id: Optional[int] = Field(None, description="ID de la course liée")
    related_pronostic_id: Optional[int] = Field(None, description="ID du pronostic lié")
    sent_via: NotificationChannel = Field(..., description="Canal d'envoi")


class NotificationCreate(NotificationBase):
    """Schéma pour créer une notification"""
    user_id: int = Field(..., description="ID de l'utilisateur")


class NotificationUpdate(BaseModel):
    """Schéma pour mettre à jour une notification"""
    status: Optional[NotificationStatus] = None
    read_at: Optional[datetime] = None


class NotificationResponse(NotificationBase):
    """Schéma de réponse pour une notification"""
    notification_id: int
    user_id: int
    sent_at: datetime
    read_at: Optional[datetime]
    status: NotificationStatus

    # Propriétés calculées
    is_read: bool
    is_sent: bool
    is_pending: bool

    class Config:
        from_attributes = True


class NotificationPreferences(BaseModel):
    """Préférences de notification d'un utilisateur"""
    email_enabled: bool = Field(default=True, description="Notifications par email activées")
    telegram_enabled: bool = Field(default=False, description="Notifications Telegram activées")
    in_app_enabled: bool = Field(default=True, description="Notifications in-app activées")

    # Types de notifications activées
    pronostic_enabled: bool = Field(default=True, description="Nouveaux pronostics")
    value_bet_enabled: bool = Field(default=True, description="Value bets détectés")
    race_reminder_enabled: bool = Field(default=False, description="Rappels de courses")
    result_enabled: bool = Field(default=True, description="Résultats de courses")
    daily_report_enabled: bool = Field(default=True, description="Rapports quotidiens")

    # Seuils
    value_bet_min_edge: float = Field(default=0.10, ge=0, le=1, description="Edge minimum pour value bet")
    race_reminder_minutes: int = Field(default=30, ge=5, le=120, description="Minutes avant la course pour le rappel")


class BulkNotificationRequest(BaseModel):
    """Requête pour envoyer une notification à plusieurs utilisateurs"""
    user_ids: list[int] = Field(..., min_length=1, description="Liste des IDs utilisateurs")
    notification_type: NotificationType
    title: str = Field(..., min_length=1, max_length=300)
    message: str = Field(..., min_length=1)
    sent_via: NotificationChannel
    related_course_id: Optional[int] = None
    related_pronostic_id: Optional[int] = None


class NotificationStats(BaseModel):
    """Statistiques de notifications pour un utilisateur"""
    total_notifications: int
    unread_count: int
    by_type: dict[str, int] = Field(..., description="Nombre par type")
    by_channel: dict[str, int] = Field(..., description="Nombre par canal")
    recent_notifications: list[NotificationResponse] = Field(default_factory=list, description="Notifications récentes")
