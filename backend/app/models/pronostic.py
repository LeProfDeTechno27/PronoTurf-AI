# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Modèle SQLAlchemy pour la table pronostics
"""

from typing import Optional

from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, DECIMAL, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Pronostic(Base):
    """
    Modèle pour les pronostics générés par le modèle ML
    """
    __tablename__ = "pronostics"

    pronostic_id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(Integer, ForeignKey("courses.course_id", ondelete="CASCADE"), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    generated_at = Column(TIMESTAMP, server_default=func.now())
    gagnant_predicted = Column(JSON, nullable=True)
    place_predicted = Column(JSON, nullable=True)
    tierce_predicted = Column(JSON, nullable=True)
    quarte_predicted = Column(JSON, nullable=True)
    quinte_predicted = Column(JSON, nullable=True)
    confidence_score = Column(DECIMAL(5, 2), nullable=True)
    value_bet_detected = Column(Boolean, default=False)
    shap_values = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    course = relationship("Course", back_populates="pronostics")
    partant_predictions = relationship("PartantPrediction", back_populates="pronostic", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Pronostic(id={self.pronostic_id}, course_id={self.course_id}, version={self.model_version})>"