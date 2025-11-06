# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Modèle SQLAlchemy pour la table partant_predictions
"""

from typing import Optional

from sqlalchemy import Column, Integer, TIMESTAMP, ForeignKey, DECIMAL, JSON, String
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class PartantPrediction(Base):
    """
    Modèle pour les prédictions individuelles de chaque partant
    """
    __tablename__ = "partant_predictions"

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    pronostic_id = Column(Integer, ForeignKey("pronostics.pronostic_id", ondelete="CASCADE"), nullable=False, index=True)
    partant_id = Column(Integer, ForeignKey("partants.partant_id", ondelete="CASCADE"), nullable=False, index=True)
    predicted_position = Column(Integer, nullable=True)
    win_probability = Column(DECIMAL(5, 4), nullable=True)
    place_probability = Column(DECIMAL(5, 4), nullable=True)
    confidence_score = Column(DECIMAL(5, 2), nullable=True)

    # Note: confidence_level is stored as a string representation of the score
    # We'll use a hybrid approach where we store both
    confidence_level = Column(String(20), nullable=True)  # 'low', 'medium', 'high', 'very_high'

    shap_values = Column(JSON, nullable=True)  # Alias pour shap_contributions
    shap_contributions = Column(JSON, nullable=True)
    top_positive_features = Column(JSON, nullable=True)
    top_negative_features = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    pronostic = relationship("Pronostic", back_populates="partant_predictions")
    partant = relationship("Partant", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<PartantPrediction(id={self.prediction_id}, partant_id={self.partant_id}, win_prob={self.win_probability})>"