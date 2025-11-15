from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from app.core.database import Base


class PerformanceHistorique(Base):
    __tablename__ = "performance_historique"

    # identifiant auto-incrémenté
    performance_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # clé étrangère vers horse.horse_id
    horse_id = Column(Integer, ForeignKey("horses.horse_id"))
    # relation inverse vers Horse (déjà déclarée dans Horse)
    horse = relationship("Horse", back_populates="performances")
