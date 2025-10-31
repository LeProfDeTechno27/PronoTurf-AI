"""
Modèle SQLAlchemy pour la table favoris
"""

import enum
from typing import Optional

from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, Enum as SQLEnum, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class EntityType(str, enum.Enum):
    """Enumération des types d'entités pouvant être mises en favori"""
    HORSE = "horse"
    JOCKEY = "jockey"
    TRAINER = "trainer"
    HIPPODROME = "hippodrome"


class Favori(Base):
    """
    Modèle pour les favoris des utilisateurs
    """
    __tablename__ = "favoris"

    favori_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    entity_type = Column(
        SQLEnum(EntityType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        index=True
    )
    entity_id = Column(Integer, nullable=False, index=True)
    alert_enabled = Column(Boolean, default=True)
    added_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="favoris")

    def __repr__(self) -> str:
        return f"<Favori(id={self.favori_id}, user_id={self.user_id}, type={self.entity_type}, entity_id={self.entity_id})>"

    @property
    def entity_type_display(self) -> str:
        """Retourne le type d'entité formaté en français"""
        type_names = {
            EntityType.HORSE: "Cheval",
            EntityType.JOCKEY: "Jockey",
            EntityType.TRAINER: "Entraîneur",
            EntityType.HIPPODROME: "Hippodrome",
        }
        return type_names.get(self.entity_type, self.entity_type.value)

    def get_entity(self, db):
        """
        Récupère l'entité correspondante depuis la base de données

        Args:
            db: Session de base de données

        Returns:
            Instance de l'entité (Horse, Jockey, Trainer, ou Hippodrome)
        """
        from app.models.horse import Horse
        from app.models.jockey import Jockey
        from app.models.trainer import Trainer
        from app.models.hippodrome import Hippodrome

        entity_models = {
            EntityType.HORSE: Horse,
            EntityType.JOCKEY: Jockey,
            EntityType.TRAINER: Trainer,
            EntityType.HIPPODROME: Hippodrome,
        }

        model = entity_models.get(self.entity_type)
        if model:
            if self.entity_type == EntityType.HORSE:
                return db.query(model).filter(model.horse_id == self.entity_id).first()
            elif self.entity_type == EntityType.JOCKEY:
                return db.query(model).filter(model.jockey_id == self.entity_id).first()
            elif self.entity_type == EntityType.TRAINER:
                return db.query(model).filter(model.trainer_id == self.entity_id).first()
            elif self.entity_type == EntityType.HIPPODROME:
                return db.query(model).filter(model.hippodrome_id == self.entity_id).first()

        return None
