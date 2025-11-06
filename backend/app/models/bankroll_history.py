# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Modèle SQLAlchemy pour la table bankroll_history
"""

import enum
from typing import Optional
from datetime import date as date_type

from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, DECIMAL, Enum as SQLEnum, Date, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class TransactionType(str, enum.Enum):
    """Enumération des types de transaction"""
    BET = "bet"
    WIN = "win"
    LOSS = "loss"
    RESET = "reset"
    ADJUSTMENT = "adjustment"


class BankrollHistory(Base):
    """
    Modèle pour l'historique de bankroll des utilisateurs
    """
    __tablename__ = "bankroll_history"

    history_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    transaction_date = Column(Date, nullable=False, index=True)
    transaction_type = Column(
        SQLEnum(TransactionType, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    amount = Column(DECIMAL(10, 2), nullable=False)
    balance_after = Column(DECIMAL(10, 2), nullable=False)
    pari_id = Column(Integer, ForeignKey("paris_simules.pari_id", ondelete="SET NULL"), nullable=True, index=True)
    description = Column(String(500), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="bankroll_history")
    pari = relationship("PariSimule", back_populates="bankroll_entries")

    def __repr__(self) -> str:
        return f"<BankrollHistory(id={self.history_id}, user_id={self.user_id}, type={self.transaction_type}, amount={self.amount})>"

    @property
    def amount_display(self) -> str:
        """Retourne le montant formaté avec signe"""
        if self.amount:
            amount = float(self.amount)
            sign = "+" if amount >= 0 else ""
            return f"{sign}{amount:.2f} €"
        return "0.00 €"

    @property
    def balance_display(self) -> str:
        """Retourne le solde formaté"""
        if self.balance_after:
            return f"{float(self.balance_after):.2f} €"
        return "0.00 €"

    @property
    def transaction_type_display(self) -> str:
        """Retourne le type de transaction formaté en français"""
        type_names = {
            TransactionType.BET: "Pari placé",
            TransactionType.WIN: "Gain",
            TransactionType.LOSS: "Perte",
            TransactionType.RESET: "Réinitialisation",
            TransactionType.ADJUSTMENT: "Ajustement",
        }
        return type_names.get(self.transaction_type, self.transaction_type.value)

    @property
    def is_positive(self) -> bool:
        """Vérifie si la transaction est positive"""
        return self.amount and float(self.amount) > 0

    @property
    def is_negative(self) -> bool:
        """Vérifie si la transaction est négative"""
        return self.amount and float(self.amount) < 0