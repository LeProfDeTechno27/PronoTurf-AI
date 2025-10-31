"""
Service de notifications Telegram

Gère l'envoi de notifications aux utilisateurs via Telegram Bot API.
Supporte plusieurs types de messages : pronostics, value bets, résultats, alertes.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from app.core.config import settings

logger = logging.getLogger(__name__)


class TelegramNotificationService:
    """
    Service pour envoyer des notifications via Telegram

    Usage:
        service = TelegramNotificationService()
        await service.send_pronostic_notification(
            chat_id=user_telegram_id,
            course_info={...}
        )
    """

    def __init__(self, bot_token: Optional[str] = None):
        """
        Initialise le service Telegram

        Args:
            bot_token: Token du bot Telegram (défaut: depuis config)
        """
        self.bot_token = bot_token or settings.TELEGRAM_BOT_TOKEN

        if not self.bot_token or self.bot_token == "your-telegram-bot-token":
            logger.warning(
                "Telegram bot token not configured. "
                "Notifications will not be sent."
            )
            self.bot = None
        else:
            self.bot = Bot(token=self.bot_token)

    @property
    def is_configured(self) -> bool:
        """Indique si le bot Telegram est correctement initialisé."""
        return self.bot is not None

    async def send_message(
        self,
        chat_id: str,
        message: str,
        parse_mode: str = ParseMode.HTML,
        disable_notification: bool = False
    ) -> bool:
        """
        Envoie un message Telegram

        Args:
            chat_id: ID du chat Telegram
            message: Contenu du message
            parse_mode: Mode de parsing (HTML ou Markdown)
            disable_notification: Silencieux si True

        Returns:
            True si succès, False sinon
        """
        if not self.bot:
            logger.warning("Telegram bot not configured. Message not sent.")
            return False

        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_notification=disable_notification,
                disable_web_page_preview=True
            )

            logger.info(f"Telegram message sent to {chat_id}")
            return True

        except TelegramError as e:
            logger.error(f"Telegram error sending to {chat_id}: {str(e)}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error sending to {chat_id}: {str(e)}")
            return False

    async def send_link_confirmation(self, chat_id: str, user_label: str) -> bool:
        """Envoie un message de bienvenue lors de la liaison du compte Telegram."""
        message = (
            "🤖 <b>PronoTurf</b>\n\n"
            f"Bonjour {user_label}! Votre compte est désormais lié à notre bot Telegram.\n"
            "Vous recevrez ici vos pronostics, alertes value bets et rappels personnalisés.\n\n"
            "Vous pouvez à tout moment répondre /stop pour désactiver les notifications."
        )
        return await self.send_message(chat_id=chat_id, message=message)

    async def send_pronostic_notification(
        self,
        chat_id: str,
        course_info: Dict[str, Any],
        predictions: List[Dict[str, Any]]
    ) -> bool:
        """
        Envoie une notification de nouveau pronostic

        Args:
            chat_id: ID du chat Telegram
            course_info: Informations de la course
            predictions: Liste des prédictions

        Returns:
            True si succès
        """
        # Format course time
        course_time = course_info.get("scheduled_time", "N/A")
        hippodrome = course_info.get("hippodrome", "N/A")
        course_name = course_info.get("course_name", "N/A")
        course_id = course_info.get("course_id", "N/A")

        # Build message
        message = f"🏇 <b>Nouveau Pronostic Disponible</b>\n\n"
        message += f"📍 <b>{hippodrome}</b>\n"
        message += f"🏆 {course_name}\n"
        message += f"⏰ Départ: {course_time}\n\n"

        # Top 3 predictions
        message += "🎯 <b>Top 3 Recommandations:</b>\n"

        for i, pred in enumerate(predictions[:3], 1):
            horse_name = pred.get("horse_name", "N/A")
            confidence = pred.get("confidence_score", 0)
            numero = pred.get("numero_corde", "?")

            # Emoji pour le podium
            medal = ["🥇", "🥈", "🥉"][i - 1]

            message += f"{medal} N°{numero} - {horse_name} "
            message += f"(Confiance: {confidence:.1f}%)\n"

        # Value bet indicator
        if any(p.get("is_value_bet", False) for p in predictions):
            message += "\n💎 <b>Value Bet détecté !</b>\n"

        # Link to full details
        message += f"\n👉 Voir détails complets: {settings.FRONTEND_URL}/courses/{course_id}"

        return await self.send_message(chat_id, message)

    async def send_value_bet_alert(
        self,
        chat_id: str,
        course_info: Dict[str, Any],
        value_bet: Dict[str, Any]
    ) -> bool:
        """
        Envoie une alerte de value bet détecté

        Args:
            chat_id: ID du chat Telegram
            course_info: Informations de la course
            value_bet: Informations du value bet

        Returns:
            True si succès
        """
        course_time = course_info.get("scheduled_time", "N/A")
        hippodrome = course_info.get("hippodrome", "N/A")
        horse_name = value_bet.get("horse_name", "N/A")
        numero = value_bet.get("numero_corde", "?")
        model_prob = value_bet.get("model_probability", 0) * 100
        odds = value_bet.get("odds_pmu", 0)
        edge = value_bet.get("edge", 0) * 100

        message = f"💎 <b>VALUE BET DÉTECTÉ !</b>\n\n"
        message += f"📍 {hippodrome} - {course_time}\n"
        message += f"🐴 N°{numero} - <b>{horse_name}</b>\n\n"
        message += f"📊 Probabilité modèle: {model_prob:.1f}%\n"
        message += f"💰 Cote PMU: {odds:.1f}\n"
        message += f"✨ Edge: +{edge:.1f}%\n\n"
        message += f"⚡️ <b>OPPORTUNITÉ DÉTECTÉE !</b>"

        return await self.send_message(chat_id, message)

    async def send_race_reminder(
        self,
        chat_id: str,
        course_info: Dict[str, Any],
        minutes_before: int = 15
    ) -> bool:
        """
        Envoie un rappel avant le départ d'une course

        Args:
            chat_id: ID du chat Telegram
            course_info: Informations de la course
            minutes_before: Minutes avant le départ

        Returns:
            True si succès
        """
        course_time = course_info.get("scheduled_time", "N/A")
        hippodrome = course_info.get("hippodrome", "N/A")
        course_name = course_info.get("course_name", "N/A")
        course_id = course_info.get("course_id", "N/A")

        message = f"⏰ <b>Rappel de Course</b>\n\n"
        message += f"📍 {hippodrome}\n"
        message += f"🏆 {course_name}\n"
        message += f"🚀 Départ dans {minutes_before} minutes !\n"
        message += f"⏰ Heure: {course_time}\n\n"
        message += f"👉 Voir pronostic: {settings.FRONTEND_URL}/courses/{course_id}"

        return await self.send_message(chat_id, message, disable_notification=False)

    async def send_race_result(
        self,
        chat_id: str,
        course_info: Dict[str, Any],
        result: Dict[str, Any],
        user_had_bet: bool = False,
        bet_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Envoie le résultat d'une course

        Args:
            chat_id: ID du chat Telegram
            course_info: Informations de la course
            result: Résultat de la course
            user_had_bet: L'utilisateur avait un pari
            bet_result: Résultat du pari si applicable

        Returns:
            True si succès
        """
        hippodrome = course_info.get("hippodrome", "N/A")
        course_name = course_info.get("course_name", "N/A")
        arrivee = result.get("arrivee", [])

        message = f"🏁 <b>Résultat de Course</b>\n\n"
        message += f"📍 {hippodrome}\n"
        message += f"🏆 {course_name}\n\n"

        # Top 5 arrivée
        message += "🏁 <b>Arrivée:</b>\n"
        for i, numero in enumerate(arrivee[:5], 1):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i - 1]
            message += f"{medal} N°{numero}\n"

        # User bet result
        if user_had_bet and bet_result:
            message += "\n"
            is_win = bet_result.get("is_win", False)
            gain = bet_result.get("gain", Decimal("0"))
            mise = bet_result.get("mise", Decimal("0"))

            if is_win:
                message += f"✅ <b>PARI GAGNANT !</b>\n"
                message += f"💰 Gain: +{gain}€ (Mise: {mise}€)\n"
                profit = gain - mise
                message += f"📈 Bénéfice: +{profit}€"
            else:
                message += f"❌ <b>Pari perdant</b>\n"
                message += f"💸 Perte: -{mise}€"

        # Non-partants
        non_partants = result.get("non_partants", [])
        if non_partants:
            message += f"\n\n⚠️ Non-partants: {', '.join(map(str, non_partants))}"

        return await self.send_message(chat_id, message)

    async def send_daily_report(
        self,
        chat_id: str,
        report: Dict[str, Any]
    ) -> bool:
        """
        Envoie le rapport quotidien

        Args:
            chat_id: ID du chat Telegram
            report: Données du rapport

        Returns:
            True si succès
        """
        date = report.get("date", datetime.now().strftime("%d/%m/%Y"))
        total_bets = report.get("total_bets", 0)
        wins = report.get("wins", 0)
        losses = report.get("losses", 0)
        win_rate = report.get("win_rate", 0)
        total_mise = report.get("total_mise", Decimal("0"))
        total_gain = report.get("total_gain", Decimal("0"))
        profit = report.get("profit", Decimal("0"))
        roi = report.get("roi", 0)

        message = f"📊 <b>Rapport Quotidien - {date}</b>\n\n"

        # Stats paris
        message += f"🎲 <b>Paris du jour:</b>\n"
        message += f"• Total: {total_bets} paris\n"
        message += f"• Gagnants: {wins} ✅\n"
        message += f"• Perdants: {losses} ❌\n"
        message += f"• Win Rate: {win_rate:.1f}%\n\n"

        # Stats financières
        message += f"💰 <b>Finances:</b>\n"
        message += f"• Total misé: {total_mise}€\n"
        message += f"• Total gagné: {total_gain}€\n"

        # Profit/Loss
        if profit >= 0:
            message += f"• <b>Bénéfice: +{profit}€ 📈</b>\n"
        else:
            message += f"• <b>Perte: {profit}€ 📉</b>\n"

        message += f"• ROI: {roi:+.1f}%\n\n"

        # Bankroll
        current_bankroll = report.get("current_bankroll", Decimal("0"))
        bankroll_change = report.get("bankroll_change", Decimal("0"))

        message += f"💼 <b>Bankroll actuelle:</b> {current_bankroll}€\n"

        if bankroll_change >= 0:
            message += f"📈 Evolution: +{bankroll_change}€"
        else:
            message += f"📉 Evolution: {bankroll_change}€"

        return await self.send_message(chat_id, message)

    async def send_bankroll_alert(
        self,
        chat_id: str,
        current_bankroll: Decimal,
        initial_bankroll: Decimal,
        percentage: float
    ) -> bool:
        """
        Envoie une alerte de bankroll critique

        Args:
            chat_id: ID du chat Telegram
            current_bankroll: Bankroll actuel
            initial_bankroll: Bankroll initial
            percentage: Pourcentage actuel

        Returns:
            True si succès
        """
        message = f"⚠️ <b>ALERTE BANKROLL CRITIQUE</b>\n\n"
        message += f"💼 Bankroll actuelle: {current_bankroll}€\n"
        message += f"📊 Progression: {percentage:.1f}%\n"
        message += f"🎯 Initial: {initial_bankroll}€\n\n"

        if percentage < 50:
            message += "🚨 <b>Votre bankroll est en danger !</b>\n"
            message += "Recommandation: Réduisez vos mises ou arrêtez temporairement.\n"
        elif percentage < 80:
            message += "⚠️ <b>Attention à votre gestion</b>\n"
            message += "Recommandation: Soyez prudent sur les prochains paris.\n"

        message += "\n💡 Conseil: Réévaluez votre stratégie de mise."

        return await self.send_message(chat_id, message)

    async def send_favori_alert(
        self,
        chat_id: str,
        favori_type: str,
        favori_name: str,
        course_info: Dict[str, Any]
    ) -> bool:
        """
        Envoie une alerte quand un favori court

        Args:
            chat_id: ID du chat Telegram
            favori_type: Type de favori (horse, jockey, trainer)
            favori_name: Nom du favori
            course_info: Informations de la course

        Returns:
            True si succès
        """
        type_emoji = {
            "horse": "🐴",
            "jockey": "🏇",
            "trainer": "👨‍🏫",
            "hippodrome": "📍"
        }.get(favori_type, "⭐")

        type_label = {
            "horse": "Cheval",
            "jockey": "Jockey",
            "trainer": "Entraîneur",
            "hippodrome": "Hippodrome"
        }.get(favori_type, "Favori")

        course_time = course_info.get("scheduled_time", "N/A")
        hippodrome = course_info.get("hippodrome", "N/A")
        course_name = course_info.get("course_name", "N/A")
        course_id = course_info.get("course_id", "N/A")

        message = f"{type_emoji} <b>Alerte Favori !</b>\n\n"
        message += f"⭐ Votre {type_label.lower()} favori <b>{favori_name}</b> court aujourd'hui !\n\n"
        message += f"📍 {hippodrome}\n"
        message += f"🏆 {course_name}\n"
        message += f"⏰ {course_time}\n\n"
        message += f"👉 Voir pronostic: {settings.FRONTEND_URL}/courses/{course_id}"

        return await self.send_message(chat_id, message)


# Instance globale du service
telegram_service = TelegramNotificationService()
