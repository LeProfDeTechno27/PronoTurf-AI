"""
Modèle SQLAlchemy pour la table notifications
"""

import enum
from typing import Optional
from datetime import datetime

from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, Enum as SQLEnum, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class NotificationType(str, enum.Enum):
    """Enumération des types de notification"""
    PRONOSTIC = "pronostic"
    VALUE_BET = "value_bet"
    RACE_REMINDER = "race_reminder"
    RESULT = "result"
    DAILY_REPORT = "daily_report"


class NotificationChannel(str, enum.Enum):
    """Enumération des canaux de notification"""
    EMAIL = "email"
    TELEGRAM = "telegram"
    IN_APP = "in_app"


class NotificationStatus(str, enum.Enum):
    """Enumération des statuts de notification"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class Notification(Base):
    """
    Modèle pour les notifications des utilisateurs
    """
    __tablename__ = "notifications"

    notification_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    notification_type = Column(
        SQLEnum(NotificationType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    title = Column(String(300), nullable=False)
    message = Column(Text, nullable=False)
    related_course_id = Column(Integer, ForeignKey("courses.course_id", ondelete="SET NULL"), nullable=True)
    related_pronostic_id = Column(Integer, ForeignKey("pronostics.pronostic_id", ondelete="SET NULL"), nullable=True)
    sent_via = Column(
        SQLEnum(NotificationChannel, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    sent_at = Column(TIMESTAMP, server_default=func.now())
    read_at = Column(TIMESTAMP, nullable=True)
    status = Column(
        SQLEnum(NotificationStatus, values_callable=lambda x: [e.value for e in x]),
        default=NotificationStatus.PENDING,
        nullable=False,
        index=True
    )

    # Relationships
    user = relationship("User", back_populates="notifications")
    course = relationship("Course")
    pronostic = relationship("Pronostic")

    def __repr__(self) -> str:
        return f"<Notification(id={self.notification_id}, user_id={self.user_id}, type={self.notification_type}, status={self.status})>"

    @property
    def is_read(self) -> bool:
        """Vérifie si la notification a été lue"""
        return self.read_at is not None

    @property
    def is_sent(self) -> bool:
        """Vérifie si la notification a été envoyée"""
        return self.status == NotificationStatus.SENT

    @property
    def is_pending(self) -> bool:
        """Vérifie si la notification est en attente"""
        return self.status == NotificationStatus.PENDING

    @property
    def notification_type_display(self) -> str:
        """Retourne le type de notification formaté en français"""
        type_names = {
            NotificationType.PRONOSTIC: "Nouveau pronostic",
            NotificationType.VALUE_BET: "Value bet détecté",
            NotificationType.RACE_REMINDER: "Rappel de course",
            NotificationType.RESULT: "Résultat de course",
            NotificationType.DAILY_REPORT: "Rapport quotidien",
        }
        return type_names.get(self.notification_type, self.notification_type.value)

    @property
    def channel_display(self) -> str:
        """Retourne le canal formaté en français"""
        channel_names = {
            NotificationChannel.EMAIL: "Email",
            NotificationChannel.TELEGRAM: "Telegram",
            NotificationChannel.IN_APP: "Application",
        }
        return channel_names.get(self.sent_via, self.sent_via.value)

    def mark_as_read(self):
        """Marque la notification comme lue"""
        if not self.read_at:
            self.read_at = datetime.now()

    def mark_as_sent(self):
        """Marque la notification comme envoyée"""
        self.status = NotificationStatus.SENT

    def mark_as_failed(self):
        """Marque la notification comme échouée"""
        self.status = NotificationStatus.FAILED
