"""
Modèle SQLAlchemy pour la table hippodromes
"""

from typing import Optional
import enum

from sqlalchemy import Column, Integer, String, TIMESTAMP, Enum as SQLEnum, DECIMAL
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class TrackType(str, enum.Enum):
    """Enumération des types de piste"""
    PLAT = "plat"
    TROT = "trot"
    OBSTACLES = "obstacles"
    MIXTE = "mixte"


class Hippodrome(Base):
    """
    Modèle pour les hippodromes
    """
    __tablename__ = "hippodromes"

    hippodrome_id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    city = Column(String(100), nullable=True)
    country = Column(String(50), default="France")
    track_type = Column(
        SQLEnum(TrackType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    track_length = Column(Integer, nullable=True)  # en mètres
    track_surface = Column(String(50), nullable=True)
    latitude = Column(DECIMAL(10, 8), nullable=True)
    longitude = Column(DECIMAL(11, 8), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    reunions = relationship("Reunion", back_populates="hippodrome", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Hippodrome(id={self.hippodrome_id}, code={self.code}, name={self.name})>"

    @property
    def coordinates(self) -> Optional[tuple[float, float]]:
        """Retourne les coordonnées GPS (latitude, longitude)"""
        if self.latitude and self.longitude:
            return (float(self.latitude), float(self.longitude))
        return None
