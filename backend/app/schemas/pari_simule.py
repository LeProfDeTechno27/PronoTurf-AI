"""
Schémas Pydantic pour les paris simulés
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from decimal import Decimal

from app.models.pari_simule import BetType, BettingStrategy, BetResult


class PariSimuleBase(BaseModel):
    """Schéma de base pour un pari simulé"""
    course_id: int = Field(..., description="ID de la course")
    bet_type: BetType = Field(..., description="Type de pari")
    bet_amount: Decimal = Field(..., gt=0, description="Montant du pari")
    selected_horses: List[int] = Field(..., min_length=1, description="Numéros des chevaux sélectionnés")
    strategy_used: BettingStrategy = Field(..., description="Stratégie de mise utilisée")
    pronostic_id: Optional[int] = Field(None, description="ID du pronostic utilisé (optionnel)")

    @field_validator('selected_horses')
    @classmethod
    def validate_selected_horses(cls, v, info):
        """Valide que le nombre de chevaux correspond au type de pari"""
        bet_type = info.data.get('bet_type')
        if not bet_type:
            return v

        min_horses = {
            BetType.GAGNANT: 1,
            BetType.PLACE: 1,
            BetType.TIERCE: 3,
            BetType.QUARTE: 4,
            BetType.QUINTE: 5,
            BetType.COUPLE: 2,
            BetType.TRIO: 3,
        }

        required = min_horses.get(bet_type, 1)
        if len(v) < required:
            raise ValueError(f"Le {bet_type.value} requiert au moins {required} chevaux")

        return v


class PariSimuleCreate(PariSimuleBase):
    """Schéma pour créer un pari simulé"""
    pass


class PariSimuleUpdate(BaseModel):
    """Schéma pour mettre à jour un pari simulé"""
    result: Optional[BetResult] = None
    payout: Optional[Decimal] = Field(None, ge=0)
    net_profit: Optional[Decimal] = None


class PariSimuleResponse(PariSimuleBase):
    """Schéma de réponse pour un pari simulé"""
    pari_id: int
    user_id: int
    placed_at: datetime
    result: BetResult
    payout: Decimal
    net_profit: Optional[Decimal]
    updated_at: datetime

    # Propriétés calculées
    roi: Optional[float] = Field(None, description="ROI en pourcentage")
    is_won: bool = Field(..., description="Pari gagné")
    is_pending: bool = Field(..., description="Pari en attente")

    class Config:
        from_attributes = True


class PariSimuleWithDetails(PariSimuleResponse):
    """Schéma de réponse détaillé incluant les infos de la course"""
    course_name: Optional[str] = None
    hippodrome_name: Optional[str] = None
    reunion_date: Optional[datetime] = None


class BettingStats(BaseModel):
    """Statistiques de paris pour un utilisateur"""
    total_bets: int = Field(..., description="Nombre total de paris")
    total_amount_bet: Decimal = Field(..., description="Montant total parié")
    total_won: int = Field(..., description="Nombre de paris gagnés")
    total_lost: int = Field(..., description="Nombre de paris perdus")
    total_pending: int = Field(..., description="Nombre de paris en attente")
    total_payout: Decimal = Field(..., description="Gains totaux")
    total_profit: Decimal = Field(..., description="Profit net total")
    win_rate: float = Field(..., description="Taux de réussite (%)")
    average_roi: float = Field(..., description="ROI moyen (%)")
    best_bet: Optional[PariSimuleResponse] = Field(None, description="Meilleur pari")
    worst_bet: Optional[PariSimuleResponse] = Field(None, description="Pire pari")


class KellyCriterionCalculation(BaseModel):
    """Calcul du montant optimal selon le critère de Kelly"""
    probability: float = Field(..., ge=0, le=1, description="Probabilité de victoire")
    odds: float = Field(..., gt=1, description="Cote du cheval")
    current_bankroll: Decimal = Field(..., gt=0, description="Bankroll actuel")
    kelly_fraction: float = Field(default=0.25, ge=0, le=1, description="Fraction de Kelly (0.25 = quart de Kelly)")


class KellyCriterionResponse(BaseModel):
    """Réponse du calcul de Kelly"""
    recommended_amount: Decimal = Field(..., description="Montant recommandé")
    kelly_percentage: float = Field(..., description="Pourcentage de Kelly (%)")
    edge: float = Field(..., description="Edge (avantage) (%)")
    is_favorable: bool = Field(..., description="Le pari est-il favorable")
    warning: Optional[str] = Field(None, description="Avertissement si applicable")
