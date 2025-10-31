"""
Modèle SQLAlchemy pour la table reunions
"""

from datetime import date
from typing import Optional
import enum

from sqlalchemy import Column, Integer, String, TIMESTAMP, Date, ForeignKey, Enum as SQLEnum, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class ReunionStatus(str, enum.Enum):
    """Enumération des statuts de réunion"""
    SCHEDULED = "scheduled"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Reunion(Base):
    """
    Modèle pour les réunions hippiques
    Une réunion regroupe plusieurs courses sur un hippodrome à une date donnée
    """
    __tablename__ = "reunions"

    reunion_id = Column(Integer, primary_key=True, autoincrement=True)
    hippodrome_id = Column(Integer, ForeignKey("hippodromes.hippodrome_id", ondelete="CASCADE"), nullable=False)
    reunion_date = Column(Date, nullable=False, index=True)
    reunion_number = Column(Integer, nullable=False)  # Numéro de la réunion dans la journée
    status = Column(
        SQLEnum(ReunionStatus, values_callable=lambda x: [e.value for e in x]),
        default=ReunionStatus.SCHEDULED,
        nullable=False
    )
    api_source = Column(String(50), nullable=True)  # Source API (turfinfo, aspiturf, etc.)
    weather_conditions = Column(JSON, nullable=True)  # Conditions météo
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationships
    hippodrome = relationship("Hippodrome", back_populates="reunions")
    courses = relationship("Course", back_populates="reunion", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Reunion(id={self.reunion_id}, hippodrome_id={self.hippodrome_id}, date={self.reunion_date}, R{self.reunion_number})>"

    @property
    def is_today(self) -> bool:
        """Vérifie si la réunion est aujourd'hui"""
        from datetime import date as dt_date
        return self.reunion_date == dt_date.today()

    @property
    def is_completed(self) -> bool:
        """Vérifie si la réunion est terminée"""
        return self.status == ReunionStatus.COMPLETED

    @property
    def nombre_courses(self) -> int:
        """Retourne le nombre de courses dans cette réunion"""
        return len(self.courses) if self.courses else 0
