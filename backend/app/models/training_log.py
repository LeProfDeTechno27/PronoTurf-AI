# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Modèle SQLAlchemy pour la table training_logs
"""

import enum
from typing import Optional

from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, Enum as SQLEnum, JSON, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class TrainingStatus(str, enum.Enum):
    """Enumération des statuts d'entraînement"""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingLog(Base):
    """
    Modèle pour logger les sessions d'entraînement des modèles ML
    """
    __tablename__ = "training_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("ml_models.model_id", ondelete="CASCADE"), nullable=False, index=True)
    run_date = Column(TIMESTAMP, server_default=func.now())
    duration_seconds = Column(Integer, nullable=True)
    status = Column(
        SQLEnum(TrainingStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        index=True
    )
    error_message = Column(Text, nullable=True)
    metrics = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    model = relationship("MLModel", back_populates="training_logs")

    def __repr__(self) -> str:
        return f"<TrainingLog(id={self.log_id}, model_id={self.model_id}, status={self.status})>"

    @property
    def duration_display(self) -> str:
        """Retourne la durée formatée"""
        if self.duration_seconds:
            minutes = self.duration_seconds // 60
            seconds = self.duration_seconds % 60
            return f"{minutes}m {seconds}s"
        return "N/A"