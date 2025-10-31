"""
Modèle SQLAlchemy pour la table jockeys
"""

from typing import Optional
from datetime import date

from sqlalchemy import Column, Integer, String, TIMESTAMP, Date, DECIMAL
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Jockey(Base):
    """
    Modèle pour les jockeys
    """
    __tablename__ = "jockeys"

    jockey_id = Column(Integer, primary_key=True, autoincrement=True)
    official_id = Column(String(50), unique=True, nullable=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False, index=True)
    birth_date = Column(Date, nullable=True)
    nationality = Column(String(50), nullable=True)
    weight = Column(DECIMAL(4, 1), nullable=True)  # Poids en kg
    career_wins = Column(Integer, default=0)  # Nombre de victoires
    career_places = Column(Integer, default=0)  # Nombre de places
    career_starts = Column(Integer, default=0)  # Nombre de courses
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    partants = relationship("Partant", back_populates="jockey")

    def __repr__(self) -> str:
        return f"<Jockey(id={self.jockey_id}, name={self.full_name})>"

    @property
    def full_name(self) -> str:
        """Retourne le nom complet du jockey"""
        return f"{self.first_name} {self.last_name}"

    @property
    def age(self) -> Optional[int]:
        """Calcule l'âge du jockey"""
        if self.birth_date:
            from datetime import datetime
            today = datetime.now().date()
            age = today.year - self.birth_date.year
            if today.month < self.birth_date.month or (today.month == self.birth_date.month and today.day < self.birth_date.day):
                age -= 1
            return age
        return None

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
