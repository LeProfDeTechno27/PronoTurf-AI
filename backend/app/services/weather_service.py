# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Service de récupération des données météorologiques
Utilise l'API Open-Meteo pour obtenir les conditions météo
"""

import httpx
from typing import Dict, Any, Optional
from datetime import date, datetime
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class WeatherService:
    """
    Service pour récupérer les données météo depuis Open-Meteo API
    Documentation: https://open-meteo.com/en/docs
    """

    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.timeout = 10.0

    async def get_weather(
        self,
        latitude: Decimal,
        longitude: Decimal,
        target_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Récupère les conditions météo pour des coordonnées GPS données

        Args:
            latitude: Latitude GPS
            longitude: Longitude GPS
            target_date: Date cible (si None, utilise aujourd'hui)

        Returns:
            Dictionnaire contenant les données météo
        """
        if target_date is None:
            target_date = date.today()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode",
                    "hourly": "temperature_2m,precipitation,windspeed_10m,weathercode",
                    "timezone": "Europe/Paris",
                    "start_date": target_date.isoformat(),
                    "end_date": target_date.isoformat(),
                }

                logger.info(f"Fetching weather data for {latitude}, {longitude} on {target_date}")
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()

                data = response.json()

                # Extraire et formater les données pertinentes
                weather_info = self._format_weather_data(data, target_date)

                logger.info(f"Successfully fetched weather data")
                return weather_info

        except httpx.HTTPError as e:
            logger.error(f"HTTP error while fetching weather data: {e}")
            return self._get_default_weather()
        except Exception as e:
            logger.error(f"Unexpected error while fetching weather data: {e}")
            return self._get_default_weather()

    async def get_weather_for_hippodrome(
        self,
        hippodrome_code: str,
        hippodrome_name: str,
        latitude: Optional[Decimal],
        longitude: Optional[Decimal],
        target_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Récupère les conditions météo pour un hippodrome

        Args:
            hippodrome_code: Code de l'hippodrome
            hippodrome_name: Nom de l'hippodrome
            latitude: Latitude GPS
            longitude: Longitude GPS
            target_date: Date cible

        Returns:
            Dictionnaire contenant les données météo
        """
        if latitude is None or longitude is None:
            logger.warning(f"No GPS coordinates for hippodrome {hippodrome_code}")
            return self._get_default_weather()

        weather = await self.get_weather(latitude, longitude, target_date)
        weather["hippodrome_code"] = hippodrome_code
        weather["hippodrome_name"] = hippodrome_name

        return weather

    def _format_weather_data(self, raw_data: Dict[str, Any], target_date: date) -> Dict[str, Any]:
        """
        Formate les données brutes de l'API Open-Meteo

        Args:
            raw_data: Données brutes de l'API
            target_date: Date cible

        Returns:
            Données météo formatées
        """
        daily = raw_data.get("daily", {})
        hourly = raw_data.get("hourly", {})

        # Trouver l'heure de midi (12:00) pour les données horaires
        noon_index = None
        if "time" in hourly:
            for idx, time_str in enumerate(hourly["time"]):
                if "12:00" in time_str:
                    noon_index = idx
                    break

        weather_data = {
            "date": target_date.isoformat(),
            "temperature_max": daily.get("temperature_2m_max", [None])[0],
            "temperature_min": daily.get("temperature_2m_min", [None])[0],
            "precipitation_sum": daily.get("precipitation_sum", [None])[0],
            "wind_speed_max": daily.get("windspeed_10m_max", [None])[0],
            "weather_code": daily.get("weathercode", [None])[0],
            "weather_description": self._get_weather_description(
                daily.get("weathercode", [None])[0]
            ),
            "temperature_at_noon": None,
            "wind_speed_at_noon": None,
            "track_condition": "good",  # Valeur par défaut
        }

        # Ajouter les données à midi si disponibles
        if noon_index is not None:
            weather_data["temperature_at_noon"] = hourly.get("temperature_2m", [])[noon_index]
            weather_data["wind_speed_at_noon"] = hourly.get("windspeed_10m", [])[noon_index]

        # Déterminer l'état de la piste basé sur les précipitations
        precipitation = weather_data["precipitation_sum"]
        if precipitation is not None:
            if precipitation == 0:
                weather_data["track_condition"] = "good"
            elif precipitation < 5:
                weather_data["track_condition"] = "soft"
            elif precipitation < 15:
                weather_data["track_condition"] = "heavy"
            else:
                weather_data["track_condition"] = "very_heavy"

        return weather_data

    def _get_weather_description(self, weather_code: Optional[int]) -> str:
        """
        Convertit le code météo WMO en description française

        WMO Weather interpretation codes (WW):
        0: Ciel dégagé
        1, 2, 3: Peu nuageux à couvert
        45, 48: Brouillard
        51, 53, 55: Bruine
        61, 63, 65: Pluie
        71, 73, 75: Neige
        80, 81, 82: Averses de pluie
        95, 96, 99: Orage

        Args:
            weather_code: Code météo WMO

        Returns:
            Description en français
        """
        if weather_code is None:
            return "Inconnu"

        weather_descriptions = {
            0: "Ciel dégagé",
            1: "Peu nuageux",
            2: "Partiellement nuageux",
            3: "Couvert",
            45: "Brouillard",
            48: "Brouillard givrant",
            51: "Bruine légère",
            53: "Bruine modérée",
            55: "Bruine dense",
            56: "Bruine verglaçante légère",
            57: "Bruine verglaçante dense",
            61: "Pluie légère",
            63: "Pluie modérée",
            65: "Pluie forte",
            66: "Pluie verglaçante légère",
            67: "Pluie verglaçante forte",
            71: "Chute de neige légère",
            73: "Chute de neige modérée",
            75: "Chute de neige forte",
            77: "Grains de neige",
            80: "Averses de pluie légères",
            81: "Averses de pluie modérées",
            82: "Averses de pluie violentes",
            85: "Averses de neige légères",
            86: "Averses de neige fortes",
            95: "Orage",
            96: "Orage avec grêle légère",
            99: "Orage avec grêle forte",
        }

        return weather_descriptions.get(weather_code, f"Code météo {weather_code}")

    def _get_default_weather(self) -> Dict[str, Any]:
        """
        Retourne des données météo par défaut en cas d'erreur

        Returns:
            Dictionnaire avec données par défaut
        """
        return {
            "date": date.today().isoformat(),
            "temperature_max": None,
            "temperature_min": None,
            "precipitation_sum": None,
            "wind_speed_max": None,
            "weather_code": None,
            "weather_description": "Non disponible",
            "temperature_at_noon": None,
            "wind_speed_at_noon": None,
            "track_condition": "good",
            "error": "Weather data unavailable",
        }

    def get_track_condition_impact(self, track_condition: str) -> Dict[str, float]:
        """
        Retourne l'impact de l'état de la piste sur les performances

        Args:
            track_condition: État de la piste

        Returns:
            Dictionnaire avec les facteurs d'impact
        """
        impact_factors = {
            "good": {
                "speed_factor": 1.0,
                "stamina_importance": 1.0,
                "weight_impact": 1.0,
            },
            "soft": {
                "speed_factor": 0.95,
                "stamina_importance": 1.1,
                "weight_impact": 1.05,
            },
            "heavy": {
                "speed_factor": 0.90,
                "stamina_importance": 1.25,
                "weight_impact": 1.15,
            },
            "very_heavy": {
                "speed_factor": 0.85,
                "stamina_importance": 1.40,
                "weight_impact": 1.25,
            },
        }

        return impact_factors.get(track_condition, impact_factors["good"])