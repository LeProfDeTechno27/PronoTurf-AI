# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Modèle SQLAlchemy pour la table ml_models
"""

from typing import Optional

from sqlalchemy import Column, Integer, String, TIMESTAMP, DECIMAL, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class MLModel(Base):
    """
    Modèle pour stocker les informations des modèles ML entraînés
    """
    __tablename__ = "ml_models"

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=True)  # Nom du modèle (ex: "horse_racing_gradient_boosting")
    version = Column(String(50), unique=True, nullable=False)
    algorithm = Column(String(100), nullable=False)
    training_date = Column(TIMESTAMP, server_default=func.now())
    training_samples = Column(Integer, nullable=True)
    features_used = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)

    # Métriques de performance
    accuracy = Column(DECIMAL(5, 4), nullable=True)
    precision_score = Column(DECIMAL(5, 4), nullable=True)
    recall_score = Column(DECIMAL(5, 4), nullable=True)
    f1_score = Column(DECIMAL(5, 4), nullable=True)
    roc_auc = Column(DECIMAL(5, 4), nullable=True)

    # Stockage des métriques complètes en JSON
    performance_metrics = Column(JSON, nullable=True)

    file_path = Column(String(500), nullable=False)
    is_active = Column(Boolean, default=False, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    training_logs = relationship("TrainingLog", back_populates="model", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<MLModel(id={self.model_id}, version={self.version}, algorithm={self.algorithm}, active={self.is_active})>"

    @property
    def display_name(self) -> str:
        """Retourne un nom d'affichage pour le modèle"""
        return f"{self.model_name or self.algorithm} v{self.version}"

    @property
    def performance_summary(self) -> str:
        """Retourne un résumé des performances"""
        if self.accuracy and self.roc_auc:
            return f"Accuracy: {float(self.accuracy):.2%}, ROC-AUC: {float(self.roc_auc):.2%}"
        return "Performance metrics not available"