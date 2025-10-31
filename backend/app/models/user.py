"""
Modèle SQLAlchemy pour la table users
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
import enum

from sqlalchemy import Column, Integer, String, Boolean, TIMESTAMP, Enum as SQLEnum, DECIMAL
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, synonym

from app.core.database import Base


class UserRole(str, enum.Enum):
    """Enumération des rôles utilisateurs"""
    ADMIN = "admin"
    SUBSCRIBER = "subscriber"
    GUEST = "guest"


class BankrollStrategy(str, enum.Enum):
    """Enumération des stratégies de gestion de bankroll"""
    KELLY = "kelly"
    FLAT = "flat"
    MARTINGALE = "martingale"


class User(Base):
    """
    Modèle utilisateur
    """
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    role = Column(
        SQLEnum(UserRole, values_callable=lambda x: [e.value for e in x]),
        default=UserRole.GUEST,
        nullable=False,
        index=True
    )
    # Informations Telegram
    telegram_id = Column(String(100), unique=True, nullable=True, index=True)
    telegram_chat_id = synonym("telegram_id")
    telegram_notifications_enabled = Column(Boolean, default=False)
    telegram_linked_at = Column(TIMESTAMP, nullable=True)

    # Notifications Email
    email_notifications_enabled = Column(Boolean, default=True)
    profile_picture_url = Column(String(500), nullable=True)
    initial_bankroll = Column(DECIMAL(10, 2), default=1000.00)
    current_bankroll = Column(DECIMAL(10, 2), default=1000.00)
    preferred_strategy = Column(
        SQLEnum(BankrollStrategy, values_callable=lambda x: [e.value for e in x]),
        default=BankrollStrategy.FLAT
    )
    created_at = Column(TIMESTAMP, server_default=func.now())
    last_login = Column(TIMESTAMP, nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationships
    paris_simules = relationship("PariSimule", back_populates="user", cascade="all, delete-orphan")
    bankroll_history = relationship("BankrollHistory", back_populates="user", cascade="all, delete-orphan")
    favoris = relationship("Favori", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id={self.user_id}, email={self.email}, role={self.role})>"

    @property
    def full_name(self) -> str:
        """Retourne le nom complet de l'utilisateur"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.email

    @property
    def is_admin(self) -> bool:
        """Vérifie si l'utilisateur est admin"""
        return self.role == UserRole.ADMIN

    @property
    def is_subscriber(self) -> bool:
        """Vérifie si l'utilisateur est abonné (ou admin)"""
        return self.role in [UserRole.SUBSCRIBER, UserRole.ADMIN]

    @property
    def is_guest(self) -> bool:
        """Vérifie si l'utilisateur est invité"""
        return self.role == UserRole.GUEST

    @property
    def has_telegram_notifications(self) -> bool:
        """Indique si l'utilisateur a correctement lié Telegram."""
        return bool(self.telegram_notifications_enabled and self.telegram_chat_id)

    @property
    def bankroll_percentage(self) -> float:
        """Calcule le pourcentage du bankroll actuel par rapport à l'initial"""
        if self.initial_bankroll and self.initial_bankroll > 0:
            return float((self.current_bankroll / self.initial_bankroll) * 100)
        return 0.0

    @property
    def is_bankroll_critical(self) -> bool:
        """Vérifie si le bankroll est critique (< 20% de l'initial)"""
        return self.bankroll_percentage < 20.0
