"""
Schémas Pydantic pour la gestion de bankroll
"""

from typing import List, Optional
from datetime import date as date_type, datetime
from pydantic import BaseModel, Field
from decimal import Decimal

from app.models.bankroll_history import TransactionType


class BankrollHistoryBase(BaseModel):
    """Schéma de base pour l'historique de bankroll"""
    transaction_date: date_type = Field(..., description="Date de la transaction")
    transaction_type: TransactionType = Field(..., description="Type de transaction")
    amount: Decimal = Field(..., description="Montant de la transaction")
    balance_after: Decimal = Field(..., ge=0, description="Solde après transaction")
    pari_id: Optional[int] = Field(None, description="ID du pari associé")
    description: Optional[str] = Field(None, max_length=500, description="Description")


class BankrollHistoryCreate(BaseModel):
    """Schéma pour créer une entrée d'historique"""
    transaction_type: TransactionType
    amount: Decimal
    pari_id: Optional[int] = None
    description: Optional[str] = Field(None, max_length=500)


class BankrollHistoryResponse(BankrollHistoryBase):
    """Schéma de réponse pour l'historique de bankroll"""
    history_id: int
    user_id: int
    created_at: datetime

    # Propriétés calculées
    is_positive: bool
    is_negative: bool

    class Config:
        from_attributes = True


class BankrollSummary(BaseModel):
    """Résumé du bankroll d'un utilisateur"""
    user_id: int
    initial_bankroll: Decimal = Field(..., description="Bankroll initial")
    current_bankroll: Decimal = Field(..., description="Bankroll actuel")
    total_deposited: Decimal = Field(..., description="Total des dépôts")
    total_withdrawn: Decimal = Field(..., description="Total des retraits")
    total_bet: Decimal = Field(..., description="Total parié")
    total_won: Decimal = Field(..., description="Total des gains")
    net_profit: Decimal = Field(..., description="Profit net")
    roi: float = Field(..., description="ROI global (%)")
    bankroll_percentage: float = Field(..., description="Pourcentage du bankroll initial")
    is_critical: bool = Field(..., description="Bankroll critique (< 20%)")


class BankrollReset(BaseModel):
    """Schéma pour réinitialiser le bankroll"""
    new_amount: Decimal = Field(..., gt=0, description="Nouveau montant initial")
    reason: Optional[str] = Field(None, max_length=500, description="Raison de la réinitialisation")


class BankrollAdjustment(BaseModel):
    """Schéma pour ajuster manuellement le bankroll"""
    amount: Decimal = Field(..., description="Montant de l'ajustement (positif ou négatif)")
    reason: str = Field(..., min_length=5, max_length=500, description="Raison de l'ajustement")


class BankrollChart(BaseModel):
    """Données pour le graphique d'évolution du bankroll"""
    dates: List[str] = Field(..., description="Liste des dates")
    balances: List[float] = Field(..., description="Soldes correspondants")
    profits: List[float] = Field(..., description="Profits cumulés")


class BankrollPeriodStats(BaseModel):
    """Statistiques de bankroll sur une période"""
    period_start: date_type
    period_end: date_type
    starting_balance: Decimal
    ending_balance: Decimal
    total_bets: int
    total_bet_amount: Decimal
    total_won: Decimal
    net_profit: Decimal
    roi: float
    best_day: Optional[date_type] = Field(None, description="Meilleur jour")
    best_day_profit: Optional[Decimal] = None
    worst_day: Optional[date_type] = Field(None, description="Pire jour")
    worst_day_loss: Optional[Decimal] = None
