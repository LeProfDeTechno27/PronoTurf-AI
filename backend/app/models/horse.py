# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Modèle SQLAlchemy pour la table horses (chevaux)
"""

import enum
from typing import Optional

from sqlalchemy import Column, Integer, String, TIMESTAMP, Enum as SQLEnum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Gender(str, enum.Enum):
    """Enumération des genres de chevaux"""
    MALE = "male"
    FEMALE = "female"
    HONGRE = "hongre"  # Castré


class Horse(Base):
    """
    Modèle pour les chevaux
    """
    __tablename__ = "horses"

    horse_id = Column(Integer, primary_key=True, autoincrement=True)
    official_id = Column(String(50), unique=True, nullable=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    birth_year = Column(Integer, nullable=True)
    gender = Column(
        SQLEnum(Gender, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    coat_color = Column(String(50), nullable=True)  # Robe du cheval
    breed = Column(String(100), nullable=True)  # Race
    sire = Column(String(200), nullable=True)  # Père
    dam = Column(String(200), nullable=True)  # Mère
    owner = Column(String(300), nullable=True)  # Propriétaire
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    partants = relationship("Partant", back_populates="horse", cascade="all, delete-orphan")
    performances = relationship("PerformanceHistorique", back_populates="horse", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Horse(id={self.horse_id}, name={self.name}, birth_year={self.birth_year})>"

    @property
    def age(self) -> Optional[int]:
        """Calcule l'âge du cheval"""
        if self.birth_year:
            from datetime import datetime
            current_year = datetime.now().year
            return current_year - int(self.birth_year)
        return None

    @property
    def full_name(self) -> str:
        """Retourne le nom complet du cheval avec son année de naissance"""
        if self.birth_year:
            return f"{self.name} ({self.birth_year})"
        return self.name

    @property
    def display_gender(self) -> str:
        """Retourne le genre en français"""
        gender_fr = {
            Gender.MALE: "Mâle",
            Gender.FEMALE: "Femelle",
            Gender.HONGRE: "Hongre"
        }
        return gender_fr.get(self.gender, self.gender.value)