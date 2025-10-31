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


class NotificationMarkReadRequest(BaseModel):
    """Requête pour marquer plusieurs notifications comme lues."""

    notification_ids: list[int] = Field(
        ..., min_length=1, description="Identifiants des notifications à marquer comme lues"
    )


class NotificationStatsResponse(BaseModel):
    """Statistiques agrégées des notifications utilisateur."""

    total_notifications: int = Field(..., description="Nombre total de notifications envoyées")
    unread_count: int = Field(..., description="Nombre de notifications non lues")
    read_count: int = Field(..., description="Nombre de notifications lues")
    by_type: dict[str, int] = Field(default_factory=dict, description="Répartition par type")
    by_channel: dict[str, int] = Field(default_factory=dict, description="Répartition par canal")
    by_status: dict[str, int] = Field(default_factory=dict, description="Répartition par statut")
    last_24h_count: int = Field(..., description="Notifications envoyées sur les 24 dernières heures")


class TelegramLinkRequest(BaseModel):
    """Requête de liaison du bot Telegram."""

    chat_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Identifiant du chat Telegram (numérique)",
    )


class TelegramLinkResponse(BaseModel):
    """Réponse lors de la liaison ou déliaison d'un chat Telegram."""

    chat_id: Optional[str] = Field(None, description="Identifiant du chat Telegram enregistré")
    telegram_enabled: bool = Field(..., description="Telegram actif pour l'utilisateur")
    test_message_sent: Optional[bool] = Field(
        None,
        description="Indique si le message de confirmation a été envoyé avec succès",
    )


class TelegramStatusResponse(BaseModel):
    """État courant de la configuration Telegram pour l'utilisateur."""

    chat_id: Optional[str] = Field(None, description="Identifiant de chat associé")
    telegram_enabled: bool = Field(..., description="Notifications Telegram activées")
    bot_configured: bool = Field(..., description="Statut global du bot Telegram")
