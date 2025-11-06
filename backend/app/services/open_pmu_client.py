# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Client asynchrone pour l'API Open-PMU (sans clé API)

Fournit un accès aux résultats officiels PMU:
- Arrivées des courses par date
- Filtrage par hippodrome
- Filtrage par nom de prix
- Rapports PMU officiels
- Non-partants

API publique gratuite hébergée sur Vercel.
Données stables (24h cache recommandé).
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from urllib.parse import quote

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)


class OpenPMUClient:
    """
    Client asynchrone pour l'API Open-PMU

    Usage:
        async with OpenPMUClient() as client:
            arrivees = await client.get_arrivees_by_date(date_obj)
            course = await client.get_arrivee_by_prix(date_obj, "PRIX DE FRANCE")
    """

    BASE_URL = "https://open-pmu-api.vercel.app/api"

    def __init__(
        self,
        timeout: float = 30.0,
        max_connections: int = 10,
        max_keepalive_connections: int = 5
    ):
        """
        Initialise le client Open-PMU

        Args:
            timeout: Timeout des requêtes en secondes
            max_connections: Nombre max de connexions simultanées
            max_keepalive_connections: Nombre max de connexions keepalive
        """
        self.base_url = self.BASE_URL
        self.timeout = timeout

        # Configuration du client HTTP
        self.limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections
        )

        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Context manager entry"""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            limits=self.limits,
            follow_redirects=True,
            headers={
                "User-Agent": "PronoGold/1.0",
                "Accept": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._client:
            await self._client.aclose()

    @staticmethod
    def format_date_open_pmu(date_obj: date) -> str:
        """
        Formate une date au format Open-PMU: DD/MM/YYYY

        Args:
            date_obj: Date à formater

        Returns:
            Date au format DD/MM/YYYY (ex: 15/01/2025)
        """
        return date_obj.strftime("%d/%m/%Y")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Any:
        """
        Effectue une requête HTTP avec retry automatique

        Args:
            endpoint: Endpoint de l'API (relatif à base_url)
            params: Paramètres de query string

        Returns:
            Réponse JSON parsée (peut être Dict ou List)

        Raises:
            httpx.HTTPStatusError: Si erreur HTTP
            httpx.TimeoutException: Si timeout après retries
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        logger.debug(f"Open-PMU request: {endpoint} with params {params}")

        try:
            response = await self._client.get(endpoint, params=params)
            response.raise_for_status()

            data = response.json()
            logger.debug(f"Open-PMU response OK: {endpoint}")
            return data

        except httpx.HTTPStatusError as e:
            logger.error(f"Open-PMU HTTP error {e.response.status_code}: {endpoint}")
            raise
        except httpx.TimeoutException:
            logger.error(f"Open-PMU timeout: {endpoint}")
            raise
        except Exception as e:
            logger.error(f"Open-PMU unexpected error: {endpoint} - {str(e)}")
            raise

    async def get_arrivees_by_date(self, date_obj: date) -> List[Dict[str, Any]]:
        """
        Récupère toutes les arrivées pour une date

        Args:
            date_obj: Date des courses

        Returns:
            Liste de toutes les arrivées du jour

        Example response structure:
            [
                {
                    "date": "15/01/2025",
                    "hippodrome": "VINCENNES",
                    "numero_course": 1,
                    "prix": "PRIX DE FRANCE",
                    "discipline": "Trot Attelé",
                    "distance": 2100,
                    "arrivee": [1, 7, 3, 12, 5],
                    "non_partants": [8],
                    "rapports": {
                        "simple_gagnant": [
                            {"cheval": 1, "rapport": 8.5}
                        ],
                        "simple_place": [
                            {"cheval": 1, "rapport": 2.3},
                            {"cheval": 7, "rapport": 3.1},
                            {"cheval": 3, "rapport": 4.2}
                        ],
                        "couple_gagnant": [
                            {"combinaison": "1-7", "rapport": 45.2}
                        ],
                        "couple_place": [...],
                        "trio": [
                            {"combinaison": "1-7-3", "rapport": 152.4}
                        ],
                        "tierce": [...],
                        "quarte": [...],
                        "quinte": [...]
                    }
                }
            ]
        """
        date_str = self.format_date_open_pmu(date_obj)
        params = {"date": date_str}

        logger.info(f"Fetching arrivées for {date_obj}")
        return await self._make_request("/arrivees", params=params)

    async def get_arrivees_by_hippodrome(
        self,
        date_obj: date,
        hippodrome_name: str
    ) -> List[Dict[str, Any]]:
        """
        Récupère les arrivées d'un hippodrome pour une date

        Args:
            date_obj: Date des courses
            hippodrome_name: Nom de l'hippodrome (ex: "VINCENNES", "LONGCHAMP")

        Returns:
            Liste des arrivées de l'hippodrome

        Note:
            Le nom de l'hippodrome est sensible à la casse.
            Formats courants: "VINCENNES", "LONGCHAMP", "CHANTILLY", etc.
        """
        date_str = self.format_date_open_pmu(date_obj)
        params = {
            "date": date_str,
            "hippo": hippodrome_name
        }

        logger.info(f"Fetching arrivées for hippodrome {hippodrome_name} on {date_obj}")
        return await self._make_request("/arrivees", params=params)

    async def get_arrivee_by_prix(
        self,
        date_obj: date,
        prix_name: str
    ) -> Dict[str, Any]:
        """
        Récupère l'arrivée d'une course par son nom

        Args:
            date_obj: Date de la course
            prix_name: Nom du prix (ex: "PRIX DE FRANCE")

        Returns:
            Arrivée de la course spécifiée

        Note:
            Le nom du prix doit être exact (sensible à la casse).
            Retourne un seul objet (pas une liste).
        """
        date_str = self.format_date_open_pmu(date_obj)
        params = {
            "date": date_str,
            "prix": prix_name
        }

        logger.info(f"Fetching arrivée for prix '{prix_name}' on {date_obj}")
        result = await self._make_request("/arrivees", params=params)

        # L'API peut retourner une liste ou un objet selon les cas
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return result

    async def get_rapports_course(
        self,
        date_obj: date,
        hippodrome_name: str,
        numero_course: int
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère les rapports PMU d'une course spécifique

        Args:
            date_obj: Date de la course
            hippodrome_name: Nom de l'hippodrome
            numero_course: Numéro de la course

        Returns:
            Rapports PMU de la course ou None si non trouvée
        """
        arrivees = await self.get_arrivees_by_hippodrome(date_obj, hippodrome_name)

        for arrivee in arrivees:
            if arrivee.get("numero_course") == numero_course:
                return arrivee.get("rapports")

        logger.warning(
            f"Course R{numero_course} not found for {hippodrome_name} on {date_obj}"
        )
        return None

    async def check_non_partants(
        self,
        date_obj: date,
        hippodrome_name: str,
        numero_course: int
    ) -> List[int]:
        """
        Récupère la liste des non-partants d'une course

        Args:
            date_obj: Date de la course
            hippodrome_name: Nom de l'hippodrome
            numero_course: Numéro de la course

        Returns:
            Liste des numéros de chevaux non-partants
        """
        arrivees = await self.get_arrivees_by_hippodrome(date_obj, hippodrome_name)

        for arrivee in arrivees:
            if arrivee.get("numero_course") == numero_course:
                return arrivee.get("non_partants", [])

        logger.warning(
            f"Course R{numero_course} not found for {hippodrome_name} on {date_obj}"
        )
        return []


# Fonctions utilitaires pour faciliter l'usage

async def fetch_daily_results(date_obj: Optional[date] = None) -> List[Dict[str, Any]]:
    """
    Fonction utilitaire pour récupérer tous les résultats du jour

    Args:
        date_obj: Date des résultats (défaut: aujourd'hui)

    Returns:
        Liste de tous les résultats
    """
    if date_obj is None:
        date_obj = date.today()

    async with OpenPMUClient() as client:
        return await client.get_arrivees_by_date(date_obj)


async def fetch_hippodrome_results(
    hippodrome_name: str,
    date_obj: Optional[date] = None
) -> List[Dict[str, Any]]:
    """
    Fonction utilitaire pour récupérer les résultats d'un hippodrome

    Args:
        hippodrome_name: Nom de l'hippodrome
        date_obj: Date des résultats (défaut: aujourd'hui)

    Returns:
        Liste des résultats de l'hippodrome
    """
    if date_obj is None:
        date_obj = date.today()

    async with OpenPMUClient() as client:
        return await client.get_arrivees_by_hippodrome(date_obj, hippodrome_name)


async def fetch_course_result(
    date_obj: date,
    hippodrome_name: str,
    numero_course: int
) -> Optional[Dict[str, Any]]:
    """
    Fonction utilitaire pour récupérer le résultat complet d'une course

    Args:
        date_obj: Date de la course
        hippodrome_name: Nom de l'hippodrome
        numero_course: Numéro de la course

    Returns:
        Résultat complet avec arrivée et rapports, ou None
    """
    async with OpenPMUClient() as client:
        arrivees = await client.get_arrivees_by_hippodrome(date_obj, hippodrome_name)

        for arrivee in arrivees:
            if arrivee.get("numero_course") == numero_course:
                return arrivee

        return None