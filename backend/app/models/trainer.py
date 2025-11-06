# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Modèle SQLAlchemy pour la table trainers (entraîneurs)
"""

from typing import Optional

from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Trainer(Base):
    """
    Modèle pour les entraîneurs
    """
    __tablename__ = "trainers"

    trainer_id = Column(Integer, primary_key=True, autoincrement=True)
    official_id = Column(String(50), unique=True, nullable=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False, index=True)
    stable_name = Column(String(200), nullable=True)  # Nom de l'écurie
    nationality = Column(String(50), nullable=True)
    career_wins = Column(Integer, default=0)  # Nombre de victoires
    career_places = Column(Integer, default=0)  # Nombre de places
    career_starts = Column(Integer, default=0)  # Nombre de courses
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    partants = relationship("Partant", back_populates="trainer")

    def __repr__(self) -> str:
        return f"<Trainer(id={self.trainer_id}, name={self.full_name})>"

    @property
    def full_name(self) -> str:
        """Retourne le nom complet de l'entraîneur"""
        return f"{self.first_name} {self.last_name}"

    @property
    def display_name(self) -> str:
        """Retourne le nom d'affichage avec écurie si disponible"""
        if self.stable_name:
            return f"{self.full_name} ({self.stable_name})"
        return self.full_name

    @property
    def win_rate(self) -> Optional[float]:
        """Calcule le taux de victoires"""
        if self.career_starts and self.career_starts > 0:
            return (self.career_wins / self.career_starts) * 100
        return None

    @property
    def place_rate(self) -> Optional[float]:
        """Calcule le taux de places (top 3)"""
        if self.career_starts and self.career_starts > 0:
            return (self.career_places / self.career_starts) * 100
        return None

    @property
    def win_rate_formatted(self) -> Optional[str]:
        """Retourne le taux de victoires formaté"""
        rate = self.win_rate
        if rate is not None:
            return f"{rate:.2f}%"
        return None

    @property
    def place_rate_formatted(self) -> Optional[str]:
        """Retourne le taux de places formaté"""
        rate = self.place_rate
        if rate is not None:
            return f"{rate:.2f}%"
        return None

    @property
    def stats_summary(self) -> str:
        """Retourne un résumé des statistiques"""
        return f"{self.career_wins}V-{self.career_places}P/{self.career_starts}C"