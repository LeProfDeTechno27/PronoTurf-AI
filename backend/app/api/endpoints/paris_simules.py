# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Endpoints API pour les paris simulés

Permet aux utilisateurs de placer des paris simulés et de suivre leurs performances.
"""

import logging
from typing import List, Optional
from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.core.database import get_db
from app.models.pari_simule import PariSimule, BetResult
from app.models.bankroll_history import BankrollHistory, TransactionType
from app.models.user import User
from app.models.course import Course, CourseStatus
from app.schemas.pari_simule import (
    PariSimuleCreate,
    PariSimuleUpdate,
    PariSimuleResponse,
    PariSimuleWithDetails,
    BettingStats,
    KellyCriterionCalculation,
    KellyCriterionResponse
)
from app.api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=PariSimuleResponse)
async def create_pari_simule(
    pari_data: PariSimuleCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Crée un nouveau pari simulé pour l'utilisateur connecté

    Vérifie que:
    - La course existe
    - Le bankroll de l'utilisateur est suffisant
    - Met à jour le bankroll après le pari
    """
    # Vérifier que la course existe et n'est pas terminée
    course = db.query(Course).filter(Course.course_id == pari_data.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    if course.status == CourseStatus.FINISHED:
        raise HTTPException(status_code=400, detail="Cannot bet on a finished race")

    # Vérifier le bankroll
    if current_user.current_bankroll < pari_data.bet_amount:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient bankroll. Current: {current_user.current_bankroll}€, Required: {pari_data.bet_amount}€"
        )

    # Créer le pari
    pari = PariSimule(
        user_id=current_user.user_id,
        course_id=pari_data.course_id,
        bet_type=pari_data.bet_type,
        bet_amount=pari_data.bet_amount,
        selected_horses=pari_data.selected_horses,
        strategy_used=pari_data.strategy_used,
        pronostic_id=pari_data.pronostic_id,
        result=BetResult.PENDING,
        payout=0.00,
        net_profit=-pari_data.bet_amount
    )

    db.add(pari)
    db.flush()

    # Mettre à jour le bankroll
    current_user.current_bankroll -= pari_data.bet_amount

    # Créer l'entrée dans l'historique de bankroll
    history = BankrollHistory(
        user_id=current_user.user_id,
        transaction_date=date.today(),
        transaction_type=TransactionType.BET,
        amount=-pari_data.bet_amount,
        balance_after=current_user.current_bankroll,
        pari_id=pari.pari_id,
        description=f"Pari {pari.bet_type.value} sur course {pari_data.course_id}"
    )

    db.add(history)
    db.commit()
    db.refresh(pari)

    logger.info(f"Pari created: user={current_user.user_id}, pari={pari.pari_id}, amount={pari_data.bet_amount}")

    # Préparer la réponse avec propriétés calculées
    response = PariSimuleResponse.model_validate(pari)
    response.roi = pari.roi
    response.is_won = pari.is_won
    response.is_pending = pari.is_pending

    return response


@router.get("/", response_model=List[PariSimuleWithDetails])
async def get_user_paris(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: Optional[BetResult] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Récupère tous les paris de l'utilisateur connecté avec pagination
    """
    query = db.query(PariSimule).filter(PariSimule.user_id == current_user.user_id)

    if status:
        query = query.filter(PariSimule.result == status)

    paris = query.order_by(PariSimule.placed_at.desc()).offset(skip).limit(limit).all()

    # Enrichir avec les détails de la course
    results = []
    for pari in paris:
        response = PariSimuleWithDetails.model_validate(pari)
        response.roi = pari.roi
        response.is_won = pari.is_won
        response.is_pending = pari.is_pending

        if pari.course:
            response.course_name = pari.course.full_name
            if pari.course.reunion and pari.course.reunion.hippodrome:
                response.hippodrome_name = pari.course.reunion.hippodrome.name
            if pari.course.reunion:
                response.reunion_date = pari.course.reunion.reunion_date

        results.append(response)

    return results


@router.get("/{pari_id}", response_model=PariSimuleWithDetails)
async def get_pari_by_id(
    pari_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Récupère un pari spécifique par son ID
    """
    pari = db.query(PariSimule).filter(
        and_(
            PariSimule.pari_id == pari_id,
            PariSimule.user_id == current_user.user_id
        )
    ).first()

    if not pari:
        raise HTTPException(status_code=404, detail="Pari not found")

    response = PariSimuleWithDetails.model_validate(pari)
    response.roi = pari.roi
    response.is_won = pari.is_won
    response.is_pending = pari.is_pending

    if pari.course:
        response.course_name = pari.course.full_name
        if pari.course.reunion and pari.course.reunion.hippodrome:
            response.hippodrome_name = pari.course.reunion.hippodrome.name
        if pari.course.reunion:
            response.reunion_date = pari.course.reunion.reunion_date

    return response


@router.get("/stats/summary", response_model=BettingStats)
async def get_betting_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Récupère les statistiques de paris de l'utilisateur connecté
    """
    paris = db.query(PariSimule).filter(PariSimule.user_id == current_user.user_id).all()

    if not paris:
        return BettingStats(
            total_bets=0,
            total_amount_bet=0,
            total_won=0,
            total_lost=0,
            total_pending=0,
            total_payout=0,
            total_profit=0,
            win_rate=0,
            average_roi=0
        )

    total_amount = sum(float(p.bet_amount) for p in paris)
    total_payout = sum(float(p.payout) for p in paris)
    total_profit = sum(float(p.net_profit or 0) for p in paris)

    won_paris = [p for p in paris if p.result == BetResult.WON]
    lost_paris = [p for p in paris if p.result == BetResult.LOST]
    pending_paris = [p for p in paris if p.result == BetResult.PENDING]

    finished_paris = won_paris + lost_paris
    win_rate = (len(won_paris) / len(finished_paris) * 100) if finished_paris else 0

    # Calculer ROI moyen
    rois = [p.roi for p in finished_paris if p.roi is not None]
    average_roi = sum(rois) / len(rois) if rois else 0

    # Meilleur et pire pari
    best_bet = max(finished_paris, key=lambda p: p.net_profit or 0) if finished_paris else None
    worst_bet = min(finished_paris, key=lambda p: p.net_profit or 0) if finished_paris else None

    stats = BettingStats(
        total_bets=len(paris),
        total_amount_bet=total_amount,
        total_won=len(won_paris),
        total_lost=len(lost_paris),
        total_pending=len(pending_paris),
        total_payout=total_payout,
        total_profit=total_profit,
        win_rate=win_rate,
        average_roi=average_roi
    )

    if best_bet:
        best_response = PariSimuleResponse.model_validate(best_bet)
        best_response.roi = best_bet.roi
        best_response.is_won = best_bet.is_won
        best_response.is_pending = best_bet.is_pending
        stats.best_bet = best_response

    if worst_bet:
        worst_response = PariSimuleResponse.model_validate(worst_bet)
        worst_response.roi = worst_bet.roi
        worst_response.is_won = worst_bet.is_won
        worst_response.is_pending = worst_bet.is_pending
        stats.worst_bet = worst_response

    return stats


@router.post("/kelly-criterion", response_model=KellyCriterionResponse)
async def calculate_kelly_criterion(
    calc: KellyCriterionCalculation,
    current_user: User = Depends(get_current_user)
):
    """
    Calcule le montant optimal à miser selon le critère de Kelly

    Formula: f = (bp - q) / b
    où:
    - f = fraction du bankroll à miser
    - b = odds - 1
    - p = probabilité de victoire
    - q = 1 - p (probabilité de perte)
    """
    b = calc.odds - 1
    p = calc.probability
    q = 1 - p

    # Calculer la fraction de Kelly
    kelly_f = (b * p - q) / b if b > 0 else 0

    # Appliquer la fraction de Kelly (ex: 0.25 pour quarter-Kelly)
    adjusted_kelly = kelly_f * calc.kelly_fraction

    # Calculer le montant
    recommended_amount = float(calc.current_bankroll) * adjusted_kelly if adjusted_kelly > 0 else 0

    # Calculer l'edge
    expected_value = (p * calc.odds) - 1
    edge = expected_value * 100

    is_favorable = edge > 0

    warning = None
    if not is_favorable:
        warning = "Ce pari n'est pas favorable (edge négatif). Ne pas miser."
    elif kelly_f > 0.20:
        warning = "Attention: Kelly fraction élevée. Utilisez une fraction réduite (quarter-Kelly recommandé)."

    return KellyCriterionResponse(
        recommended_amount=max(0, recommended_amount),
        kelly_percentage=kelly_f * 100,
        edge=edge,
        is_favorable=is_favorable,
        warning=warning
    )


@router.delete("/{pari_id}")
async def cancel_pari(
    pari_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Annule un pari en attente et rembourse le bankroll
    """
    pari = db.query(PariSimule).filter(
        and_(
            PariSimule.pari_id == pari_id,
            PariSimule.user_id == current_user.user_id
        )
    ).first()

    if not pari:
        raise HTTPException(status_code=404, detail="Pari not found")

    if pari.result != BetResult.PENDING:
        raise HTTPException(status_code=400, detail="Can only cancel pending bets")

    # Vérifier si la course est déjà commencée
    if pari.course and pari.course.status != CourseStatus.SCHEDULED:
        raise HTTPException(status_code=400, detail="Cannot cancel bet for a started race")

    # Rembourser le bankroll
    current_user.current_bankroll += pari.bet_amount

    # Marquer comme annulé
    pari.result = BetResult.CANCELLED
    pari.net_profit = 0

    # Créer l'entrée dans l'historique
    history = BankrollHistory(
        user_id=current_user.user_id,
        transaction_date=date.today(),
        transaction_type=TransactionType.ADJUSTMENT,
        amount=pari.bet_amount,
        balance_after=current_user.current_bankroll,
        pari_id=pari.pari_id,
        description=f"Annulation pari {pari.pari_id}"
    )

    db.add(history)
    db.commit()

    logger.info(f"Pari cancelled: pari={pari_id}, user={current_user.user_id}")

    return {"message": "Pari annulé avec succès", "bankroll_refunded": float(pari.bet_amount)}