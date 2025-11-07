# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Endpoints API pour la gestion de la bankroll des utilisateurs

Routes disponibles:
- GET /api/v1/bankroll/current : Bankroll actuel
- GET /api/v1/bankroll/history : Historique des transactions
- POST /api/v1/bankroll/reset : Réinitialiser la bankroll
- POST /api/v1/bankroll/adjust : Ajuster la bankroll (admin)
- GET /api/v1/bankroll/stats : Statistiques de bankroll
- GET /api/v1/bankroll/stats/period : Stats par période
"""

from typing import List, Optional
from datetime import date, datetime, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, case, true
from sqlalchemy.orm import selectinload

from app.core.database import get_async_db
from app.api.deps import get_current_user, get_current_admin
from app.models.user import User, BankrollStrategy
from app.models.bankroll_history import BankrollHistory, TransactionType
from app.schemas.bankroll import (
    BankrollResponse,
    BankrollHistoryResponse,
    BankrollResetRequest,
    BankrollAdjustRequest,
    BankrollStatsResponse,
    BankrollPeriodStatsResponse
)

router = APIRouter()


@router.get("/current", response_model=BankrollResponse)
async def get_current_bankroll(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Récupère la bankroll actuelle de l'utilisateur

    Returns:
        Informations sur la bankroll actuelle
    """
    # Récupérer l'utilisateur avec ses données à jour
    stmt = select(User).where(User.user_id == current_user.user_id)
    result = await db.execute(stmt)
    user = result.scalar_one()

    return BankrollResponse(
        user_id=user.user_id,
        initial_bankroll=user.initial_bankroll,
        current_bankroll=user.current_bankroll,
        profit_loss=user.current_bankroll - user.initial_bankroll,
        profit_loss_percentage=user.bankroll_percentage - 100,
        preferred_strategy=user.preferred_strategy,
        is_critical=user.is_bankroll_critical,
        last_updated=datetime.now()
    )



@router.get("/history", response_model=List[BankrollHistoryResponse])
async def get_bankroll_history(
    skip: int = Query(0, ge=0, description="Nombre d'entrées à ignorer"),
    limit: int = Query(50, ge=1, le=200, description="Nombre d'entrées à retourner"),
    transaction_type: Optional[TransactionType] = Query(None, description="Filtrer par type de transaction"),
    start_date: Optional[date] = Query(None, description="Date de début"),
    end_date: Optional[date] = Query(None, description="Date de fin"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Récupère l'historique des transactions de bankroll

    Args:
        skip: Nombre d'entrées à ignorer (pagination)
        limit: Nombre d'entrées à retourner
        transaction_type: Filtrer par type de transaction
        start_date: Date de début (optionnel)
        end_date: Date de fin (optionnel)

    Returns:
        Liste des transactions de bankroll
    """
    # Construire la requête
    stmt = (
        select(BankrollHistory)
        .where(BankrollHistory.user_id == current_user.user_id)
        .options(selectinload(BankrollHistory.pari))
        .order_by(desc(BankrollHistory.created_at))
    )

    # Appliquer les filtres
    if transaction_type:
        stmt = stmt.where(BankrollHistory.transaction_type == transaction_type)

    if start_date:
        stmt = stmt.where(BankrollHistory.transaction_date >= start_date)

    if end_date:
        stmt = stmt.where(BankrollHistory.transaction_date <= end_date)

    # Pagination
    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    history_entries = result.scalars().all()

    return [
        BankrollHistoryResponse(
            history_id=entry.history_id,
            user_id=entry.user_id,
            transaction_date=entry.transaction_date,
            transaction_type=entry.transaction_type,
            amount=entry.amount,
            balance_after=entry.balance_after,
            pari_id=entry.pari_id,
            description=entry.description,
            created_at=entry.created_at
        )
        for entry in history_entries
    ]


@router.post("/reset", response_model=BankrollResponse)
async def reset_bankroll(
    reset_data: BankrollResetRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Réinitialise la bankroll de l'utilisateur

    Args:
        reset_data: Données de réinitialisation (nouveau montant initial)

    Returns:
        Nouvelle bankroll après réinitialisation
    """
    # Récupérer l'utilisateur
    stmt = select(User).where(User.user_id == current_user.user_id)
    result = await db.execute(stmt)
    user = result.scalar_one()

    old_bankroll = user.current_bankroll
    new_bankroll = reset_data.new_initial_amount

    # Mettre à jour la bankroll
    user.initial_bankroll = new_bankroll
    user.current_bankroll = new_bankroll

    # Enregistrer dans l'historique
    history_entry = BankrollHistory(
        user_id=user.user_id,
        transaction_date=date.today(),
        transaction_type=TransactionType.RESET,
        amount=Decimal("0.00"),  # Pas de gain/perte pour un reset
        balance_after=new_bankroll,
        description=f"Réinitialisation de bankroll: {old_bankroll} → {new_bankroll} €"
    )

    db.add(history_entry)
    await db.commit()
    await db.refresh(user)

    return BankrollResponse(
        user_id=user.user_id,
        initial_bankroll=user.initial_bankroll,
        current_bankroll=user.current_bankroll,
        profit_loss=Decimal("0.00"),
        profit_loss_percentage=0.0,
        preferred_strategy=user.preferred_strategy,
        is_critical=False,
        last_updated=datetime.now()
    )


@router.post("/adjust", response_model=BankrollResponse)
async def adjust_bankroll(
    adjust_data: BankrollAdjustRequest,
    current_admin: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Ajuste manuellement la bankroll d'un utilisateur (admin uniquement)

    Args:
        adjust_data: Données d'ajustement (user_id, montant, raison)

    Returns:
        Bankroll après ajustement
    """
    # Récupérer l'utilisateur cible
    stmt = select(User).where(User.user_id == adjust_data.user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Utilisateur {adjust_data.user_id} introuvable"
        )

    old_bankroll = user.current_bankroll

    # Appliquer l'ajustement
    new_bankroll = old_bankroll + adjust_data.adjustment_amount
    if new_bankroll < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"L'ajustement rendrait la bankroll négative ({new_bankroll})"
        )

    user.current_bankroll = new_bankroll

    # Enregistrer dans l'historique
    history_entry = BankrollHistory(
        user_id=user.user_id,
        transaction_date=date.today(),
        transaction_type=TransactionType.ADJUSTMENT,
        amount=adjust_data.adjustment_amount,
        balance_after=new_bankroll,
        description=adjust_data.reason or f"Ajustement manuel par admin {current_admin.user_id}"
    )

    db.add(history_entry)
    await db.commit()
    await db.refresh(user)

    return BankrollResponse(
        user_id=user.user_id,
        initial_bankroll=user.initial_bankroll,
        current_bankroll=user.current_bankroll,
        profit_loss=user.current_bankroll - user.initial_bankroll,
        profit_loss_percentage=user.bankroll_percentage - 100,
        preferred_strategy=user.preferred_strategy,
        is_critical=user.is_bankroll_critical,
        last_updated=datetime.now()
    )


@router.get("/stats", response_model=BankrollStatsResponse)
async def get_bankroll_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Récupère les statistiques globales de bankroll de l'utilisateur

    Returns:
        Statistiques complètes de gestion de bankroll
    """
    # Stats globales de l'historique (agrégation + récupération utilisateur en une requête)
    stats_subquery = (
        select(
            func.count(BankrollHistory.history_id).label("total_transactions"),
            func.sum(
                case((BankrollHistory.amount > 0, BankrollHistory.amount), else_=0)
            ).label("total_gains"),
            func.sum(
                case(
                    (BankrollHistory.amount < 0, func.abs(BankrollHistory.amount)),
                    else_=0
                )
            ).label("total_losses"),
            func.sum(
                case((BankrollHistory.transaction_type == TransactionType.BET, 1), else_=0)
            ).label("total_bets"),
            func.sum(
                case((BankrollHistory.transaction_type == TransactionType.WIN, 1), else_=0)
            ).label("total_wins"),
            func.sum(
                case((BankrollHistory.transaction_type == TransactionType.LOSS, 1), else_=0)
            ).label("total_losses_count"),
            func.max(BankrollHistory.balance_after).label("peak_bankroll"),
            func.min(BankrollHistory.balance_after).label("bottom_bankroll"),
        )
        .where(BankrollHistory.user_id == current_user.user_id)
        .select_from(BankrollHistory)
        .subquery()
    )

    stmt = (
        select(
            User,
            stats_subquery.c.total_transactions,
            stats_subquery.c.total_gains,
            stats_subquery.c.total_losses,
            stats_subquery.c.total_bets,
            stats_subquery.c.total_wins,
            stats_subquery.c.total_losses_count,
            stats_subquery.c.peak_bankroll,
            stats_subquery.c.bottom_bankroll,
        )
        .outerjoin(stats_subquery, true())
        .where(User.user_id == current_user.user_id)
    )

    result = await db.execute(stmt)
    row = result.one()
    user: User = row.User

    total_transactions = int(row.total_transactions or 0)
    total_gains = Decimal(str(row.total_gains or 0))
    total_losses = Decimal(str(row.total_losses or 0))
    total_bets = int(row.total_bets or 0)
    total_wins = int(row.total_wins or 0)
    total_losses_count = int(row.total_losses_count or 0)
    peak_bankroll = (
        row.peak_bankroll if row.peak_bankroll is not None else user.current_bankroll
    )
    bottom_bankroll = (
        row.bottom_bankroll if row.bottom_bankroll is not None else user.current_bankroll
    )

    # Calculs
    net_profit = user.current_bankroll - user.initial_bankroll
    roi_percentage = float((net_profit / user.initial_bankroll * 100)) if user.initial_bankroll > 0 else 0.0
    win_rate = float((total_wins / total_bets * 100)) if total_bets > 0 else 0.0

    return BankrollStatsResponse(
        user_id=user.user_id,
        initial_bankroll=user.initial_bankroll,
        current_bankroll=user.current_bankroll,
        peak_bankroll=Decimal(str(peak_bankroll)),
        bottom_bankroll=Decimal(str(bottom_bankroll)),
        net_profit=net_profit,
        roi_percentage=roi_percentage,
        total_transactions=total_transactions,
        total_bets=total_bets,
        total_gains=total_gains,
        total_losses=total_losses,
        win_rate=win_rate,
        preferred_strategy=user.preferred_strategy,
        is_critical=user.is_bankroll_critical
    )


@router.get("/stats/period", response_model=List[BankrollPeriodStatsResponse])
async def get_period_stats(
    period: str = Query("week", regex="^(day|week|month)$", description="Période de regroupement"),
    start_date: Optional[date] = Query(None, description="Date de début"),
    end_date: Optional[date] = Query(None, description="Date de fin"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Récupère les statistiques de bankroll par période

    Args:
        period: Période de regroupement (day, week, month)
        start_date: Date de début (défaut: 30 jours en arrière)
        end_date: Date de fin (défaut: aujourd'hui)

    Returns:
        Liste des statistiques par période
    """
    # Dates par défaut
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Formater selon la période
    if period == "day":
        date_format = "%Y-%m-%d"
    elif period == "week":
        date_format = "%Y-%U"  # Année-Semaine
    else:  # month
        date_format = "%Y-%m"

    # Requête groupée par période
    stmt = (
        select(
            func.date_format(BankrollHistory.transaction_date, date_format).label("period"),
            func.min(BankrollHistory.transaction_date).label("period_start"),
            func.max(BankrollHistory.transaction_date).label("period_end"),
            func.sum(
                func.IF(BankrollHistory.amount > 0, BankrollHistory.amount, 0)
            ).label("gains"),
            func.sum(
                func.IF(BankrollHistory.amount < 0, func.abs(BankrollHistory.amount), 0)
            ).label("losses"),
            func.count(BankrollHistory.history_id).label("transactions"),
            func.max(BankrollHistory.balance_after).label("period_ending_balance")
        )
        .where(
            and_(
                BankrollHistory.user_id == current_user.user_id,
                BankrollHistory.transaction_date >= start_date,
                BankrollHistory.transaction_date <= end_date
            )
        )
        .group_by("period")
        .order_by("period")
    )

    result = await db.execute(stmt)
    period_stats = result.all()

    return [
        BankrollPeriodStatsResponse(
            period=row.period,
            period_start=row.period_start,
            period_end=row.period_end,
            gains=Decimal(str(row.gains or 0)),
            losses=Decimal(str(row.losses or 0)),
            net_profit=Decimal(str(row.gains or 0)) - Decimal(str(row.losses or 0)),
            transactions=row.transactions,
            ending_balance=Decimal(str(row.period_ending_balance))
        )
        for row in period_stats
    ]


@router.patch("/strategy", response_model=BankrollResponse)
async def update_betting_strategy(
    strategy: BankrollStrategy,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Met à jour la stratégie de gestion de bankroll préférée

    Args:
        strategy: Nouvelle stratégie (kelly, flat, martingale)

    Returns:
        Bankroll mise à jour
    """
    # Récupérer l'utilisateur
    stmt = select(User).where(User.user_id == current_user.user_id)
    result = await db.execute(stmt)
    user = result.scalar_one()

    # Mettre à jour la stratégie
    user.preferred_strategy = strategy

    await db.commit()
    await db.refresh(user)

    return BankrollResponse(
        user_id=user.user_id,
        initial_bankroll=user.initial_bankroll,
        current_bankroll=user.current_bankroll,
        profit_loss=user.current_bankroll - user.initial_bankroll,
        profit_loss_percentage=user.bankroll_percentage - 100,
        preferred_strategy=user.preferred_strategy,
        is_critical=user.is_bankroll_critical,
        last_updated=datetime.now()
    )
