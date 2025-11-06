# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Modèle SQLAlchemy pour la table courses
"""

from datetime import time
from typing import Optional
import enum

from sqlalchemy import Column, Integer, String, TIMESTAMP, Time, ForeignKey, Enum as SQLEnum, DECIMAL
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Discipline(str, enum.Enum):
    """Enumération des disciplines hippiques"""
    PLAT = "plat"
    TROT_MONTE = "trot_monte"
    TROT_ATTELE = "trot_attele"
    HAIES = "haies"
    STEEPLE = "steeple"
    CROSS = "cross"


class SurfaceType(str, enum.Enum):
    """Enumération des types de surface"""
    PELOUSE = "pelouse"
    PISTE = "piste"
    SABLE = "sable"
    FIBRE = "fibre"


class StartType(str, enum.Enum):
    """Enumération des types de départ"""
    AUTOSTART = "autostart"
    VOLTE = "volte"
    ELASTIQUE = "elastique"
    STALLE = "stalle"
    CORDE = "corde"


class CourseStatus(str, enum.Enum):
    """Enumération des statuts de course"""
    SCHEDULED = "scheduled"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class Course(Base):
    """
    Modèle pour les courses hippiques
    """
    __tablename__ = "courses"

    course_id = Column(Integer, primary_key=True, autoincrement=True)
    reunion_id = Column(Integer, ForeignKey("reunions.reunion_id", ondelete="CASCADE"), nullable=False)
    course_number = Column(Integer, nullable=False)  # Numéro de la course dans la réunion
    course_name = Column(String(300), nullable=True)
    discipline = Column(
        SQLEnum(Discipline, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    distance = Column(Integer, nullable=False)  # Distance en mètres
    prize_money = Column(DECIMAL(10, 2), nullable=True)  # Allocation en euros
    race_category = Column(String(100), nullable=True)  # Catégorie (Groupe I, II, Listed, etc.)
    race_class = Column(String(50), nullable=True)  # Classe de la course
    surface_type = Column(
        SQLEnum(SurfaceType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    start_type = Column(
        SQLEnum(StartType, values_callable=lambda x: [e.value for e in x]),
        default=StartType.STALLE
    )
    scheduled_time = Column(Time, nullable=False, index=True)
    actual_start_time = Column(Time, nullable=True)
    number_of_runners = Column(Integer, nullable=True)  # Nombre de partants
    status = Column(
        SQLEnum(CourseStatus, values_callable=lambda x: [e.value for e in x]),
        default=CourseStatus.SCHEDULED,
        nullable=False,
        index=True
    )
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationships
    reunion = relationship("Reunion", back_populates="courses")
    partants = relationship("Partant", back_populates="course", cascade="all, delete-orphan")
    pronostics = relationship("Pronostic", back_populates="course", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Course(id={self.course_id}, R{self.reunion_id}C{self.course_number}, {self.course_name})>"

    @property
    def full_name(self) -> str:
        """Retourne le nom complet de la course"""
        if self.course_name:
            return f"R{self.reunion.reunion_number}C{self.course_number} - {self.course_name}"
        return f"R{self.reunion.reunion_number}C{self.course_number}"

    @property
    def is_finished(self) -> bool:
        """Vérifie si la course est terminée"""
        return self.status == CourseStatus.FINISHED

    @property
    def is_scheduled(self) -> bool:
        """Vérifie si la course est programmée"""
        return self.status == CourseStatus.SCHEDULED

    @property
    def distance_km(self) -> float:
        """Retourne la distance en kilomètres"""
        return self.distance / 1000.0

    @property
    def prize_money_formatted(self) -> Optional[str]:
        """Retourne l'allocation formatée"""
        if self.prize_money:
            return f"{float(self.prize_money):,.2f} €"
        return None