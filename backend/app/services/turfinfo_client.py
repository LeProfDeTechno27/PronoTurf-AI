"""
Client asynchrone pour l'API TurfInfo (sans clé API)

Fournit un accès optimisé aux données PMU via TurfInfo:
- Programme des courses du jour
- Partants et informations détaillées
- Performances détaillées des chevaux
- Rapports PMU officiels

Deux endpoints disponibles:
- Offline: Optimisé pour bornes physiques (données essentielles)
- Online: Plus détaillé pour applications web
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from enum import Enum

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)


class TurfinfoEndpoint(str, Enum):
    """Type d'endpoint TurfInfo"""
    OFFLINE = "offline"  # Optimisé, données essentielles
    ONLINE = "online"    # Détaillé, plus d'informations


class TurfinfoClient:
    """
    Client asynchrone pour l'API TurfInfo

    Usage:
        async with TurfinfoClient() as client:
            programme = await client.get_programme_jour()
            partants = await client.get_partants_course(date_obj, 1, 1)
    """

    # URLs de base selon le type d'endpoint
    BASE_URLS = {
        TurfinfoEndpoint.OFFLINE: "https://offline.turfinfo.api.pmu.fr/rest/client/7",
        TurfinfoEndpoint.ONLINE: "https://online.turfinfo.api.pmu.fr/rest/client/61"
    }

    def __init__(
        self,
        endpoint_type: TurfinfoEndpoint = TurfinfoEndpoint.ONLINE,
        timeout: float = 30.0,
        max_connections: int = 10,
        max_keepalive_connections: int = 5
    ):
        """
        Initialise le client TurfInfo

        Args:
            endpoint_type: Type d'endpoint (OFFLINE ou ONLINE)
            timeout: Timeout des requêtes en secondes
            max_connections: Nombre max de connexions simultanées
            max_keepalive_connections: Nombre max de connexions keepalive
        """
        self.endpoint_type = endpoint_type
        self.base_url = self.BASE_URLS[endpoint_type]
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
    def format_date_turfinfo(date_obj: date) -> str:
        """
        Formate une date au format TurfInfo: JJMMAAAA

        Args:
            date_obj: Date à formater

        Returns:
            Date au format JJMMAAAA (ex: 15012025)
        """
        return date_obj.strftime("%d%m%Y")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """
        Effectue une requête HTTP avec retry automatique

        Args:
            endpoint: Endpoint de l'API (relatif à base_url)

        Returns:
            Réponse JSON parsée

        Raises:
            httpx.HTTPStatusError: Si erreur HTTP
            httpx.TimeoutException: Si timeout après retries
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        logger.debug(f"TurfInfo request: {endpoint}")

        try:
            response = await self._client.get(endpoint)
            response.raise_for_status()

            data = response.json()
            logger.debug(f"TurfInfo response OK: {endpoint}")
            return data

        except httpx.HTTPStatusError as e:
            logger.error(f"TurfInfo HTTP error {e.response.status_code}: {endpoint}")
            raise
        except httpx.TimeoutException:
            logger.error(f"TurfInfo timeout: {endpoint}")
            raise
        except Exception as e:
            logger.error(f"TurfInfo unexpected error: {endpoint} - {str(e)}")
            raise

    async def get_programme_jour(self, date_obj: Optional[date] = None) -> Dict[str, Any]:
        """
        Récupère le programme des courses pour une date

        Args:
            date_obj: Date du programme (par défaut: aujourd'hui)

        Returns:
            Programme complet avec toutes les réunions et courses

        Example response structure:
            {
                "programme": {
                    "reunions": [
                        {
                            "numOfficiel": 1,
                            "hippodrome": {"code": "M", "libelleCourt": "PAR-VINCENNES"},
                            "courses": [
                                {
                                    "numOrdre": 1,
                                    "libelle": "PRIX DE ...",
                                    "heureDepart": "13:50",
                                    "discipline": "TROT_ATTELE",
                                    "nombrePartants": 16
                                }
                            ]
                        }
                    ]
                }
            }
        """
        if date_obj is None:
            date_obj = date.today()

        date_str = self.format_date_turfinfo(date_obj)
        endpoint = f"/programme/{date_str}"

        logger.info(f"Fetching programme for {date_obj}")
        return await self._make_request(endpoint)

    async def get_partants_course(
        self,
        date_obj: date,
        reunion_num: int,
        course_num: int
    ) -> Dict[str, Any]:
        """
        Récupère les partants d'une course

        Args:
            date_obj: Date de la course
            reunion_num: Numéro de réunion
            course_num: Numéro de course

        Returns:
            Liste des partants avec informations détaillées

        Example response structure:
            {
                "participants": [
                    {
                        "numPmu": 1,
                        "cheval": {
                            "nom": "HORSE NAME",
                            "sexe": "M",
                            "age": 4
                        },
                        "driver": {"nom": "JOCKEY NAME"},
                        "entraineur": {"nom": "TRAINER NAME"},
                        "indicateurInedit": false,
                        "deferre": "4MMBFFF",
                        "musique": "0a 1a 4a 2a",
                        "nombreCourses": 45,
                        "nombreVictoires": 8,
                        "nombrePlaces": 18
                    }
                ]
            }
        """
        date_str = self.format_date_turfinfo(date_obj)
        endpoint = f"/programme/{date_str}/R{reunion_num}/C{course_num}/participants"

        logger.info(f"Fetching partants for R{reunion_num}C{course_num} on {date_obj}")
        return await self._make_request(endpoint)

    async def get_performances_detaillees(
        self,
        date_obj: date,
        reunion_num: int,
        course_num: int
    ) -> Dict[str, Any]:
        """
        Récupère les performances détaillées des partants

        Contient l'historique complet des dernières courses de chaque cheval
        avec résultats, conditions, gains, etc.

        Args:
            date_obj: Date de la course
            reunion_num: Numéro de réunion
            course_num: Numéro de course

        Returns:
            Performances détaillées de tous les partants

        Example response structure:
            {
                "participants": [
                    {
                        "numPmu": 1,
                        "cheval": {"nom": "HORSE NAME"},
                        "performances": [
                            {
                                "date": "2025-01-15",
                                "hippodrome": "PAR-VINCENNES",
                                "distance": 2100,
                                "place": 1,
                                "ordreArrivee": "1-3-7-2",
                                "nombrePartants": 16,
                                "cote": 8.5,
                                "allocation": 15000,
                                "gain": 7500
                            }
                        ]
                    }
                ]
            }
        """
        date_str = self.format_date_turfinfo(date_obj)
        endpoint = f"/programme/{date_str}/R{reunion_num}/C{course_num}/performances-detaillees/pretty"

        logger.info(f"Fetching detailed performances for R{reunion_num}C{course_num} on {date_obj}")
        return await self._make_request(endpoint)

    async def get_rapports_definitifs(
        self,
        date_obj: date,
        reunion_num: int,
        course_num: int
    ) -> Dict[str, Any]:
        """
        Récupère les rapports PMU définitifs d'une course (après arrivée)

        Args:
            date_obj: Date de la course
            reunion_num: Numéro de réunion
            course_num: Numéro de course

        Returns:
            Rapports PMU avec arrivée officielle et gains

        Example response structure:
            {
                "ordreArrivee": [1, 7, 3, 12, 5],
                "nonPartants": [8],
                "rapportsDirect": [
                    {
                        "typePari": "SIMPLE_GAGNANT",
                        "numPmu": 1,
                        "rapport": 8.5
                    },
                    {
                        "typePari": "SIMPLE_PLACE",
                        "numPmu": 1,
                        "rapport": 2.3
                    }
                ],
                "rapportsCouple": [...],
                "rapportsTrio": [...]
            }
        """
        date_str = self.format_date_turfinfo(date_obj)
        endpoint = f"/programme/{date_str}/R{reunion_num}/C{course_num}/rapports-definitifs"

        logger.info(f"Fetching final reports for R{reunion_num}C{course_num} on {date_obj}")
        return await self._make_request(endpoint)

    async def get_cotes_probables(
        self,
        date_obj: date,
        reunion_num: int,
        course_num: int
    ) -> Dict[str, Any]:
        """
        Récupère les cotes probables PMU avant le départ

        Args:
            date_obj: Date de la course
            reunion_num: Numéro de réunion
            course_num: Numéro de course

        Returns:
            Cotes probables pour chaque partant

        Example response structure:
            {
                "participants": [
                    {
                        "numPmu": 1,
                        "rapport": 8.5
                    }
                ]
            }
        """
        date_str = self.format_date_turfinfo(date_obj)
        endpoint = f"/programme/{date_str}/R{reunion_num}/C{course_num}/cotes-probables"

        logger.info(f"Fetching probable odds for R{reunion_num}C{course_num} on {date_obj}")
        return await self._make_request(endpoint)


# Fonctions utilitaires pour faciliter l'usage

async def fetch_daily_programme(
    date_obj: Optional[date] = None,
    endpoint_type: TurfinfoEndpoint = TurfinfoEndpoint.ONLINE
) -> Dict[str, Any]:
    """
    Fonction utilitaire pour récupérer le programme du jour

    Args:
        date_obj: Date du programme (défaut: aujourd'hui)
        endpoint_type: Type d'endpoint à utiliser

    Returns:
        Programme complet
    """
    async with TurfinfoClient(endpoint_type=endpoint_type) as client:
        return await client.get_programme_jour(date_obj)


async def fetch_course_details(
    date_obj: date,
    reunion_num: int,
    course_num: int,
    include_performances: bool = True,
    endpoint_type: TurfinfoEndpoint = TurfinfoEndpoint.ONLINE
) -> Dict[str, Any]:
    """
    Fonction utilitaire pour récupérer toutes les infos d'une course

    Args:
        date_obj: Date de la course
        reunion_num: Numéro de réunion
        course_num: Numéro de course
        include_performances: Inclure les performances détaillées
        endpoint_type: Type d'endpoint à utiliser

    Returns:
        Dict avec partants et performances (si demandé)
    """
    async with TurfinfoClient(endpoint_type=endpoint_type) as client:
        result = {
            "partants": await client.get_partants_course(date_obj, reunion_num, course_num)
        }

        if include_performances:
            result["performances"] = await client.get_performances_detaillees(
                date_obj, reunion_num, course_num
            )

        return result
