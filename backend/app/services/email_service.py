# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Service de notifications Email

Gère l'envoi d'emails aux utilisateurs avec templates HTML.
Supporte plusieurs types : pronostics, résultats, rapports, alertes.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import aiosmtplib

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailNotificationService:
    """
    Service pour envoyer des notifications par email

    Usage:
        service = EmailNotificationService()
        await service.send_pronostic_email(
            to_email="user@example.com",
            course_info={...}
        )
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None
    ):
        """
        Initialise le service Email

        Args:
            smtp_host: Serveur SMTP (défaut: depuis config)
            smtp_port: Port SMTP (défaut: depuis config)
            smtp_username: Utilisateur SMTP
            smtp_password: Mot de passe SMTP
            from_email: Email expéditeur
        """
        self.smtp_host = smtp_host or getattr(settings, "SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or getattr(settings, "SMTP_PORT", 587)
        self.smtp_username = smtp_username or getattr(settings, "SMTP_USERNAME", "")
        self.smtp_password = smtp_password or getattr(settings, "SMTP_PASSWORD", "")
        self.from_email = from_email or getattr(
            settings, "SMTP_FROM_EMAIL", "noreply@pronoturf.ai"
        )

        # Vérifier configuration
        if not all([self.smtp_username, self.smtp_password]):
            logger.warning(
                "SMTP credentials not configured. Emails will not be sent."
            )
            self.enabled = False
        else:
            self.enabled = True

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """
        Envoie un email HTML

        Args:
            to_email: Adresse email destinataire
            subject: Sujet de l'email
            html_content: Contenu HTML
            text_content: Contenu texte alternatif

        Returns:
            True si succès, False sinon
        """
        if not self.enabled:
            logger.warning("Email service not configured. Email not sent.")
            return False

        try:
            # Créer message multipart
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.from_email
            message["To"] = to_email

            # Ajouter version texte si fournie
            if text_content:
                part1 = MIMEText(text_content, "plain")
                message.attach(part1)

            # Ajouter version HTML
            part2 = MIMEText(html_content, "html")
            message.attach(part2)

            # Envoyer via SMTP
            async with aiosmtplib.SMTP(
                hostname=self.smtp_host,
                port=self.smtp_port,
                use_tls=False
            ) as smtp:
                await smtp.starttls()
                await smtp.login(self.smtp_username, self.smtp_password)
                await smtp.send_message(message)

            logger.info(f"Email sent to {to_email}")
            return True

        except aiosmtplib.SMTPException as e:
            logger.error(f"SMTP error sending to {to_email}: {str(e)}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error sending email to {to_email}: {str(e)}")
            return False

    def _build_html_template(
        self,
        title: str,
        content: str,
        cta_text: Optional[str] = None,
        cta_link: Optional[str] = None
    ) -> str:
        """
        Génère un template HTML responsive

        Args:
            title: Titre de l'email
            content: Contenu HTML du corps
            cta_text: Texte du bouton call-to-action
            cta_link: Lien du bouton

        Returns:
            HTML complet
        """
        cta_button = ""
        if cta_text and cta_link:
            cta_button = f"""
            <div style="text-align: center; margin: 30px 0;">
                <a href="{cta_link}"
                   style="background-color: #2563eb; color: white; padding: 12px 24px;
                          text-decoration: none; border-radius: 6px; display: inline-block;
                          font-weight: bold;">
                    {cta_text}
                </a>
            </div>
            """

        return f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: Arial, sans-serif;
                     background-color: #f3f4f6; color: #1f2937;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white;
                        padding: 20px; border-radius: 8px; margin-top: 20px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);">

                <!-- Header -->
                <div style="text-align: center; padding: 20px 0; border-bottom: 2px solid #2563eb;">
                    <h1 style="margin: 0; color: #2563eb;">🏇 PronoTurf</h1>
                    <p style="margin: 5px 0 0 0; color: #6b7280; font-size: 14px;">
                        Pronostics Hippiques Intelligents
                    </p>
                </div>

                <!-- Title -->
                <h2 style="color: #1f2937; margin: 30px 0 20px 0;">
                    {title}
                </h2>

                <!-- Content -->
                <div style="color: #4b5563; line-height: 1.6;">
                    {content}
                </div>

                <!-- CTA Button -->
                {cta_button}

                <!-- Footer -->
                <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb;
                            text-align: center; color: #9ca3af; font-size: 12px;">
                    <p>© 2025 PronoTurf - Tous droits réservés</p>
                    <p style="margin: 10px 0 0 0;">
                        Vous recevez cet email car vous êtes abonné aux notifications PronoTurf.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

    async def send_pronostic_email(
        self,
        to_email: str,
        course_info: Dict[str, Any],
        predictions: List[Dict[str, Any]]
    ) -> bool:
        """
        Envoie un email de nouveau pronostic

        Args:
            to_email: Adresse email destinataire
            course_info: Informations de la course
            predictions: Liste des prédictions

        Returns:
            True si succès
        """
        course_time = course_info.get("scheduled_time", "N/A")
        hippodrome = course_info.get("hippodrome", "N/A")
        course_name = course_info.get("course_name", "N/A")
        course_id = course_info.get("course_id", "N/A")

        # Build predictions HTML
        predictions_html = "<ul style='padding-left: 20px;'>"
        for i, pred in enumerate(predictions[:5], 1):
            horse_name = pred.get("horse_name", "N/A")
            confidence = pred.get("confidence_score", 0)
            numero = pred.get("numero_corde", "?")

            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i - 1]

            predictions_html += f"""
            <li style='margin: 10px 0;'>
                {medal} <strong>N°{numero} - {horse_name}</strong>
                <br><span style='color: #6b7280; font-size: 14px;'>
                    Confiance: {confidence:.1f}%
                </span>
            </li>
            """
        predictions_html += "</ul>"

        # Value bet indicator
        value_bet_alert = ""
        if any(p.get("is_value_bet", False) for p in predictions):
            value_bet_alert = """
            <div style='background-color: #fef3c7; border-left: 4px solid #f59e0b;
                        padding: 12px; margin: 20px 0; border-radius: 4px;'>
                <strong>💎 Value Bet détecté !</strong>
                <br>Opportunité intéressante identifiée par le modèle.
            </div>
            """

        content = f"""
        <p><strong>📍 Hippodrome:</strong> {hippodrome}</p>
        <p><strong>🏆 Course:</strong> {course_name}</p>
        <p><strong>⏰ Départ:</strong> {course_time}</p>

        {value_bet_alert}

        <h3 style='color: #1f2937; margin: 25px 0 15px 0;'>
            🎯 Top Recommandations:
        </h3>
        {predictions_html}

        <p style='color: #6b7280; font-size: 14px; margin-top: 20px;'>
            Les pronostics sont basés sur notre modèle d'IA analysant
            45+ facteurs pour chaque cheval.
        </p>
        """

        html = self._build_html_template(
            title="Nouveau Pronostic Disponible",
            content=content,
            cta_text="Voir Détails Complets",
            cta_link=f"{settings.FRONTEND_URL}/courses/{course_id}"
        )

        return await self.send_email(
            to_email=to_email,
            subject=f"🏇 Nouveau Pronostic - {hippodrome}",
            html_content=html
        )

    async def send_daily_report_email(
        self,
        to_email: str,
        report: Dict[str, Any]
    ) -> bool:
        """
        Envoie le rapport quotidien par email

        Args:
            to_email: Adresse email destinataire
            report: Données du rapport

        Returns:
            True si succès
        """
        date = report.get("date", datetime.now().strftime("%d/%m/%Y"))
        total_bets = report.get("total_bets", 0)
        wins = report.get("wins", 0)
        losses = report.get("losses", 0)
        win_rate = report.get("win_rate", 0)
        profit = report.get("profit", Decimal("0"))
        roi = report.get("roi", 0)
        current_bankroll = report.get("current_bankroll", Decimal("0"))

        # Stats color
        profit_color = "#10b981" if profit >= 0 else "#ef4444"
        profit_icon = "📈" if profit >= 0 else "📉"

        content = f"""
        <div style='background-color: #f9fafb; padding: 20px; border-radius: 8px; margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0; color: #1f2937;'>
                📊 Statistiques du {date}
            </h3>

            <table style='width: 100%; border-collapse: collapse;'>
                <tr>
                    <td style='padding: 8px 0;'>🎲 <strong>Paris effectués:</strong></td>
                    <td style='text-align: right; padding: 8px 0;'>{total_bets}</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0;'>✅ <strong>Paris gagnants:</strong></td>
                    <td style='text-align: right; padding: 8px 0;'>{wins}</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0;'>❌ <strong>Paris perdants:</strong></td>
                    <td style='text-align: right; padding: 8px 0;'>{losses}</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0;'>📊 <strong>Win Rate:</strong></td>
                    <td style='text-align: right; padding: 8px 0;'>{win_rate:.1f}%</td>
                </tr>
            </table>
        </div>

        <div style='background-color: {profit_color}15; padding: 20px;
                    border-radius: 8px; border-left: 4px solid {profit_color}; margin: 20px 0;'>
            <h3 style='margin: 0 0 15px 0; color: #1f2937;'>
                💰 Performance Financière
            </h3>

            <p style='margin: 10px 0;'>
                <strong>Résultat du jour:</strong>
                <span style='color: {profit_color}; font-size: 20px; font-weight: bold;'>
                    {profit_icon} {profit:+.2f}€
                </span>
            </p>

            <p style='margin: 10px 0;'>
                <strong>ROI:</strong> <span style='color: {profit_color};'>{roi:+.1f}%</span>
            </p>

            <p style='margin: 10px 0;'>
                <strong>Bankroll actuelle:</strong> {current_bankroll}€
            </p>
        </div>

        <p style='color: #6b7280; font-size: 14px; margin-top: 25px;'>
            💡 <strong>Conseil:</strong> Analysez vos performances pour ajuster
            votre stratégie et améliorer vos résultats.
        </p>
        """

        html = self._build_html_template(
            title=f"Rapport Quotidien - {date}",
            content=content,
            cta_text="Voir Dashboard Complet",
            cta_link=f"{settings.FRONTEND_URL}/dashboard"
        )

        return await self.send_email(
            to_email=to_email,
            subject=f"📊 Votre Rapport Quotidien - {date}",
            html_content=html
        )

    async def send_bankroll_alert_email(
        self,
        to_email: str,
        current_bankroll: Decimal,
        initial_bankroll: Decimal,
        percentage: float
    ) -> bool:
        """
        Envoie une alerte de bankroll critique par email

        Args:
            to_email: Adresse email destinataire
            current_bankroll: Bankroll actuel
            initial_bankroll: Bankroll initial
            percentage: Pourcentage actuel

        Returns:
            True si succès
        """
        alert_level = "CRITIQUE" if percentage < 50 else "ATTENTION"
        alert_color = "#ef4444" if percentage < 50 else "#f59e0b"

        content = f"""
        <div style='background-color: {alert_color}15; padding: 20px;
                    border-radius: 8px; border-left: 4px solid {alert_color};
                    text-align: center;'>
            <h3 style='color: {alert_color}; margin: 0 0 15px 0; font-size: 24px;'>
                ⚠️ ALERTE BANKROLL {alert_level}
            </h3>

            <p style='font-size: 32px; font-weight: bold; color: {alert_color}; margin: 20px 0;'>
                {percentage:.1f}%
            </p>

            <p style='margin: 15px 0;'>
                <strong>Bankroll actuelle:</strong> {current_bankroll}€
            </p>
            <p style='margin: 15px 0;'>
                <strong>Bankroll initiale:</strong> {initial_bankroll}€
            </p>
        </div>

        <div style='margin-top: 25px; padding: 15px; background-color: #f9fafb;
                    border-radius: 8px;'>
            <h4 style='margin: 0 0 10px 0; color: #1f2937;'>
                💡 Recommandations:
            </h4>
            <ul style='margin: 10px 0; padding-left: 20px; color: #4b5563;'>
                <li>Réduisez le montant de vos mises</li>
                <li>Passez à une stratégie plus conservative (Flat Betting)</li>
                <li>Réévaluez votre sélection de courses</li>
                <li>Envisagez une pause pour analyser vos résultats</li>
            </ul>
        </div>

        <p style='color: #6b7280; font-size: 14px; margin-top: 20px;
                  border-top: 1px solid #e5e7eb; padding-top: 15px;'>
            ⚡ Cette alerte a été déclenchée automatiquement pour protéger votre capital.
            Consultez votre dashboard pour plus de détails.
        </p>
        """

        html = self._build_html_template(
            title="Alerte Bankroll",
            content=content,
            cta_text="Voir Mon Compte",
            cta_link=f"{settings.FRONTEND_URL}/bankroll"
        )

        return await self.send_email(
            to_email=to_email,
            subject=f"⚠️ Alerte Bankroll {alert_level} - Action Requise",
            html_content=html
        )


# Instance globale du service
email_service = EmailNotificationService()