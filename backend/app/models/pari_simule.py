"""
Modèle SQLAlchemy pour la table paris_simules
"""

import enum
from typing import Optional, List
from datetime import datetime

from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, DECIMAL, Enum as SQLEnum, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class BetType(str, enum.Enum):
    """Enumération des types de paris"""
    GAGNANT = "gagnant"
    PLACE = "place"
    TIERCE = "tierce"
    QUARTE = "quarte"
    QUINTE = "quinte"
    COUPLE = "couple"
    TRIO = "trio"


class BettingStrategy(str, enum.Enum):
    """Enumération des stratégies de mise"""
    KELLY = "kelly"
    FLAT = "flat"
    MARTINGALE = "martingale"
    MANUAL = "manual"


class BetResult(str, enum.Enum):
    """Enumération des résultats de paris"""
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    CANCELLED = "cancelled"


class PariSimule(Base):
    """
    Modèle pour les paris simulés des utilisateurs
    """
    __tablename__ = "paris_simules"

    pari_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    course_id = Column(Integer, ForeignKey("courses.course_id", ondelete="CASCADE"), nullable=False, index=True)
    bet_type = Column(
        SQLEnum(BetType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    bet_amount = Column(DECIMAL(8, 2), nullable=False)
    selected_horses = Column(JSON, nullable=False)  # Liste des numéros de chevaux
    strategy_used = Column(
        SQLEnum(BettingStrategy, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    pronostic_id = Column(Integer, ForeignKey("pronostics.pronostic_id", ondelete="SET NULL"), nullable=True)
    placed_at = Column(TIMESTAMP, server_default=func.now())
    result = Column(
        SQLEnum(BetResult, values_callable=lambda x: [e.value for e in x]),
        default=BetResult.PENDING,
        nullable=False,
        index=True
    )
    payout = Column(DECIMAL(10, 2), default=0.00)
    net_profit = Column(DECIMAL(10, 2), nullable=True)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="paris_simules")
    course = relationship("Course")
    pronostic = relationship("Pronostic")
    bankroll_entries = relationship("BankrollHistory", back_populates="pari", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<PariSimule(id={self.pari_id}, user_id={self.user_id}, type={self.bet_type}, amount={self.bet_amount})>"

    @property
    def selected_horses_list(self) -> Optional[List[int]]:
        """Retourne la liste des numéros de chevaux sélectionnés"""
        if self.selected_horses and isinstance(self.selected_horses, list):
            return self.selected_horses
        return None

    @property
    def is_won(self) -> bool:
        """Vérifie si le pari est gagné"""
        return self.result == BetResult.WON

    @property
    def is_pending(self) -> bool:
        """Vérifie si le pari est en attente"""
        return self.result == BetResult.PENDING

    @property
    def roi(self) -> Optional[float]:
        """Calcule le ROI (Return on Investment) en pourcentage"""
        if self.bet_amount and float(self.bet_amount) > 0:
            profit = float(self.net_profit or 0)
            amount = float(self.bet_amount)
            return (profit / amount) * 100
        return None

    @property
    def display_amount(self) -> str:
        """Retourne le montant formaté"""
        if self.bet_amount:
            return f"{float(self.bet_amount):.2f} €"
        return "0.00 €"

    @property
    def display_payout(self) -> str:
        """Retourne le gain formaté"""
        if self.payout:
            return f"{float(self.payout):.2f} €"
        return "0.00 €"

    @property
    def display_profit(self) -> str:
        """Retourne le profit net formaté"""
        if self.net_profit:
            profit = float(self.net_profit)
            sign = "+" if profit >= 0 else ""
            return f"{sign}{profit:.2f} €"
        return "0.00 €"

    @property
    def bet_type_display(self) -> str:
        """Retourne le type de pari formaté"""
        type_names = {
            BetType.GAGNANT: "Gagnant",
            BetType.PLACE: "Placé",
            BetType.TIERCE: "Tiercé",
            BetType.QUARTE: "Quarté",
            BetType.QUINTE: "Quinté",
            BetType.COUPLE: "Couplé",
            BetType.TRIO: "Trio",
        }
        return type_names.get(self.bet_type, self.bet_type.value)

    @property
    def strategy_display(self) -> str:
        """Retourne la stratégie formatée"""
        strategy_names = {
            BettingStrategy.KELLY: "Kelly Criterion",
            BettingStrategy.FLAT: "Flat Betting",
            BettingStrategy.MARTINGALE: "Martingale",
            BettingStrategy.MANUAL: "Manuel",
        }
        return strategy_names.get(self.strategy_used, self.strategy_used.value)
