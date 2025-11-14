# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Schémas Pydantic pour la gestion de bankroll
"""

from typing import List, Optional
from datetime import date as date_type, datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from app.models.bankroll_history import TransactionType
from app.models.user import BankrollStrategy


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


class BankrollResponse(BaseModel):
    """État actuel de la bankroll d'un utilisateur"""

    user_id: int = Field(..., description="Identifiant utilisateur")
    initial_bankroll: Decimal = Field(..., ge=0, description="Bankroll initiale")
    current_bankroll: Decimal = Field(..., ge=0, description="Bankroll actuelle")
    profit_loss: Decimal = Field(..., description="Gain ou perte net(e)")
    profit_loss_percentage: float = Field(..., description="Performance en pourcentage")
    preferred_strategy: Optional[BankrollStrategy] = Field(
        None, description="Stratégie de gestion de bankroll"
    )
    is_critical: bool = Field(..., description="Bankroll sous le seuil critique")
    last_updated: datetime = Field(..., description="Horodatage de mise à jour")


class BankrollResetRequest(BaseModel):
    """Payload pour réinitialiser complètement la bankroll"""

    new_initial_amount: Decimal = Field(
        ..., gt=0, description="Nouveau montant initial de la bankroll"
    )
    reason: Optional[str] = Field(
        None, max_length=500, description="Raison facultative de la réinitialisation"
    )


class BankrollAdjustRequest(BaseModel):
    """Payload pour ajuster la bankroll d'un utilisateur (admin)"""

    user_id: int = Field(..., description="Identifiant utilisateur ciblé")
    adjustment_amount: Decimal = Field(
        ..., description="Montant (positif ou négatif) à appliquer"
    )
    reason: Optional[str] = Field(
        None, max_length=500, description="Raison de l'ajustement"
    )


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


class BankrollStatsResponse(BaseModel):
    """Statistiques agrégées de la bankroll"""

    user_id: int
    initial_bankroll: Decimal
    current_bankroll: Decimal
    peak_bankroll: Decimal
    bottom_bankroll: Decimal
    net_profit: Decimal
    roi_percentage: float
    total_transactions: int
    total_bets: int
    total_gains: Decimal
    total_losses: Decimal
    win_rate: float
    preferred_strategy: Optional[BankrollStrategy]
    is_critical: bool


class BankrollPeriodStatsResponse(BaseModel):
    """Statistiques agrégées sur une période donnée"""

    period: str = Field(..., description="Identifiant de la période (jour/semaine/mois)")
    period_start: date_type
    period_end: date_type
    gains: Decimal
    losses: Decimal
    net_profit: Decimal
    transactions: int
    ending_balance: Decimal
