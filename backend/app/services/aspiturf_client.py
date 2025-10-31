"""
Client asynchrone pour les données Aspiturf (format CSV)

Fournit un accès complet aux données hippiques Aspiturf:
- Données ultra-détaillées (120+ colonnes)
- Statistiques chevaux, jockeys, entraineurs
- Historiques complets sur 365 jours
- Performances par hippodrome
- Couplages cheval-jockey

Source de données PRINCIPALE pour l'application.
Format: Fichiers CSV avec délimiteur personnalisable.
"""

import logging
import csv
import io
from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field

import httpx
import aiofiles
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)


@dataclass
class AspiturfConfig:
    """Configuration pour le client Aspiturf"""
    csv_delimiter: str = ","
    csv_encoding: str = "utf-8"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 heure par défaut


@dataclass
class AspiturfHorse:
    """Représentation d'un cheval dans les données Aspiturf"""
    # Identifiants
    id_cheval: Optional[str] = None
    numero: Optional[int] = None

    # Informations de base
    sexe: Optional[str] = None
    age: Optional[int] = None
    coat: Optional[str] = None  # Robe
    country: Optional[str] = None

    # Généalogie
    pere: Optional[str] = None
    mere: Optional[str] = None

    # Statistiques générales
    courses_cheval: Optional[int] = None
    victoires_cheval: Optional[int] = None
    places_cheval: Optional[int] = None
    gains: Optional[float] = None
    gains_carriere: Optional[float] = None
    gains_victoires: Optional[float] = None
    gains_place: Optional[float] = None
    gains_annee_en_cours: Optional[float] = None
    gains_annee_precedente: Optional[float] = None

    # Performance
    musique_cheval: Optional[str] = None
    m1: Optional[str] = None  # 6 dernières musiques
    m2: Optional[str] = None
    m3: Optional[str] = None
    m4: Optional[str] = None
    m5: Optional[str] = None
    m6: Optional[str] = None
    recence: Optional[int] = None  # Jours depuis dernière course
    record_g: Optional[str] = None  # Record général

    # Course actuelle
    cote_direct: Optional[float] = None  # Rapport PMU
    cote_prob: Optional[float] = None  # Côte probable
    poids_mont: Optional[float] = None
    handicap_distance: Optional[int] = None
    handicap_poids: Optional[float] = None
    vha: Optional[int] = None  # Valeur handicap

    # Équipement et statut
    defoeil: Optional[str] = None  # Déferrage
    defoeil_prec: Optional[str] = None
    def_first_time: Optional[bool] = None
    oeil: Optional[str] = None  # Œillères
    oeil_first_time: Optional[bool] = None
    dernier_oeil: Optional[str] = None
    recul: Optional[int] = None  # Recul au départ
    est_supplemente: Optional[bool] = None
    indicateur_inedit: Optional[bool] = None
    jument_pleine: Optional[bool] = None

    # Statistiques sur l'hippodrome
    pourc_vict_cheval_hippo: Optional[float] = None
    pourc_place_cheval_hippo: Optional[float] = None
    nbr_course_cheval_hippo: Optional[int] = None
    appet_terrain: Optional[int] = None  # Appétence pour le terrain

    # Dernière course
    dernier_hippo: Optional[str] = None
    derniere_alloc: Optional[float] = None
    dernier_nb_partants: Optional[int] = None
    derniere_dist: Optional[int] = None
    derniere_place: Optional[int] = None
    derniere_cote: Optional[float] = None
    dernier_joc: Optional[str] = None
    dernier_ent: Optional[str] = None
    dernier_prop: Optional[str] = None
    dernier_red_km: Optional[str] = None
    dernier_tx_reclam: Optional[str] = None

    # Résultat (si course terminée)
    classement: Optional[int] = None  # cl
    ecart: Optional[str] = None  # ecar
    temps_tot: Optional[str] = None  # tempstot

    # Propriétaire
    proprietaire: Optional[str] = None

    # Écurie
    ecurie: Optional[str] = None

    # IDs relations
    id_jockey: Optional[str] = None
    id_entraineur: Optional[str] = None


@dataclass
class AspiturfJockey:
    """Représentation d'un jockey dans les données Aspiturf"""
    id_jockey: Optional[str] = None
    jockey: Optional[str] = None

    # Statistiques générales
    courses_jockey: Optional[int] = None
    victoires_jockey: Optional[int] = None
    place_jockey: Optional[int] = None
    pourc_vict_jock: Optional[float] = None
    pourc_place_jock: Optional[float] = None

    # Performance du jour
    montes_du_jockey_jour: Optional[int] = None
    courue_jockey_jour: Optional[int] = None
    victoire_jockey_jour: Optional[int] = None

    # Performance sur hippodrome
    pourc_vict_jock_hippo: Optional[float] = None
    pourc_place_jock_hippo: Optional[float] = None
    nbr_course_jock_hippo: Optional[int] = None

    # Musique
    musique_joc: Optional[str] = None


@dataclass
class AspiturfTrainer:
    """Représentation d'un entraineur dans les données Aspiturf"""
    id_entraineur: Optional[str] = None
    entraineur: Optional[str] = None

    # Statistiques générales
    courses_entraineur: Optional[int] = None
    victoires_entraineur: Optional[int] = None
    place_entraineur: Optional[int] = None

    # Performance du jour
    monte_entraineur_jour: Optional[int] = None
    courue_entraineur_jour: Optional[int] = None
    victoire_entraineur_jour: Optional[int] = None

    # Performance sur hippodrome
    pourc_vict_ent_hippo: Optional[float] = None
    pourc_place_ent_hippo: Optional[float] = None
    nbr_course_ent_hippo: Optional[int] = None

    # Musique
    musique_ent: Optional[str] = None


@dataclass
class AspiturfCourse:
    """Représentation d'une course dans les données Aspiturf"""
    # Identifiants
    numcourse: Optional[str] = None  # Numéro unique
    reun: Optional[str] = None  # Numéro réunion ou nom hippodrome
    prix: Optional[int] = None  # Numéro course
    jour: Optional[date] = None
    hippo: Optional[str] = None  # Hippodrome

    # Caractéristiques de la course
    typec: Optional[str] = None  # Type (plat, attelé, haies...)
    dist: Optional[int] = None  # Distance
    partant: Optional[int] = None  # Nombre de partants
    cheque: Optional[float] = None  # Allocation
    devise: Optional[str] = None

    # Partants
    partants: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AspiturfCouple:
    """Statistiques du couple cheval-jockey"""
    nb_course_couple: Optional[int] = None
    tx_vict_couple: Optional[float] = None
    tx_place_couple: Optional[float] = None
    nb_vict_couple: Optional[int] = None
    nb_place_couple: Optional[int] = None

    # Sur l'hippodrome
    nb_course_couple_hippo: Optional[int] = None
    tx_vict_couple_hippo: Optional[float] = None
    tx_place_couple_hippo: Optional[float] = None
    nb_vict_couple_hippo: Optional[int] = None
    nb_place_couple_hippo: Optional[int] = None


class AspiturfClient:
    """
    Client pour les données Aspiturf (format CSV)

    Contrairement aux autres clients (REST APIs), Aspiturf fournit
    des fichiers CSV ultra-détaillés avec 120+ colonnes.

    Usage:
        # Depuis fichier local
        async with AspiturfClient(csv_path="/path/to/data.csv") as client:
            courses = await client.get_courses_by_date(date_obj)
            partants = await client.get_partants_course(date_obj, "VINCENNES", 1)

        # Depuis URL
        async with AspiturfClient(csv_url="https://...") as client:
            courses = await client.get_courses_by_date(date_obj)
    """

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        csv_url: Optional[str] = None,
        config: Optional[AspiturfConfig] = None,
        timeout: float = 60.0
    ):
        """
        Initialise le client Aspiturf

        Args:
            csv_path: Chemin vers fichier CSV local (prioritaire sur csv_url)
            csv_url: URL pour télécharger le CSV
            config: Configuration Aspiturf
            timeout: Timeout pour téléchargement (si csv_url)
        """
        if not csv_path and not csv_url:
            raise ValueError("csv_path ou csv_url doit être fourni")

        self.csv_path = Path(csv_path) if csv_path else None
        self.csv_url = csv_url
        self.config = config or AspiturfConfig()
        self.timeout = timeout

        self._client: Optional[httpx.AsyncClient] = None
        self._data: List[Dict[str, Any]] = []
        self._data_loaded = False

    async def __aenter__(self):
        """Context manager entry"""
        if self.csv_url and not self.csv_path:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                headers={
                    "User-Agent": "PronoGold/1.0"
                }
            )

        # Charger les données au démarrage
        await self._load_data()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._client:
            await self._client.aclose()
        self._data.clear()
        self._data_loaded = False

    async def _load_data(self):
        """Charge les données CSV en mémoire"""
        if self._data_loaded:
            return

        try:
            if self.csv_path:
                # Lire depuis fichier local
                logger.info(f"Loading Aspiturf data from file: {self.csv_path}")
                csv_content = await self._read_file_async(self.csv_path)
            else:
                # Télécharger depuis URL
                logger.info(f"Downloading Aspiturf data from URL: {self.csv_url}")
                csv_content = await self._download_csv()

            # Parser le CSV
            self._data = self._parse_csv(csv_content)
            self._data_loaded = True

            logger.info(f"Loaded {len(self._data)} rows from Aspiturf CSV")

        except Exception as e:
            logger.error(f"Error loading Aspiturf data: {e}")
            raise

    async def _read_file_async(self, file_path: Path) -> str:
        """Lit un fichier CSV de manière asynchrone"""
        async with aiofiles.open(file_path, mode='r', encoding=self.config.csv_encoding) as f:
            return await f.read()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _download_csv(self) -> str:
        """Télécharge le fichier CSV depuis l'URL"""
        if not self._client:
            raise RuntimeError("HTTP client not initialized")

        response = await self._client.get(self.csv_url)
        response.raise_for_status()

        return response.text

    def _parse_csv(self, csv_content: str) -> List[Dict[str, Any]]:
        """
        Parse le contenu CSV en liste de dictionnaires

        Gère les 120+ colonnes de la procédure Aspiturf
        """
        reader = csv.DictReader(
            io.StringIO(csv_content),
            delimiter=self.config.csv_delimiter
        )

        data = []
        for row in reader:
            # Nettoyer et typer les données
            cleaned_row = self._clean_row(row)
            data.append(cleaned_row)

        return data

    def _clean_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """
        Nettoie et type une ligne du CSV

        Convertit les strings en types appropriés (int, float, date, bool)
        """
        cleaned = {}

        for key, value in row.items():
            # Gérer les valeurs vides
            if value == '' or value is None:
                cleaned[key] = None
                continue

            # Conversion selon le nom de la colonne
            try:
                # Dates
                if key == 'jour':
                    cleaned[key] = self._parse_date(value)

                # Entiers
                elif key in [
                    'prix', 'cl', 'dist', 'partant', 'numero', 'age', 'recence',
                    'recul', 'coursescheval', 'victoirescheval', 'placescheval',
                    'coursesentraineur', 'victoiresentraineur', 'placeentraineur',
                    'coursesjockey', 'victoiresjockey', 'placejockey',
                    'derniernbpartants', 'dernieredist', 'derniereplace',
                    'montesdujockeyjour', 'couruejockeyjour', 'victoirejockeyjour',
                    'monteentraineurjour', 'courueentraineurjour', 'victoireentraineurjour',
                    'nbrCourseJockHippo', 'nbrCourseEntHippo', 'nbrCourseChevalHippo',
                    'vha', 'handicapDistance', 'handicapPoids', 'appetTerrain',
                    'nbCourseCouple', 'nbVictCouple', 'nbPlaceCouple',
                    'nbCourseCoupleHippo', 'nbVictCoupleHippo', 'nbPlaceCoupleHippo'
                ]:
                    cleaned[key] = int(float(value)) if value else None

                # Flottants
                elif key in [
                    'cotedirect', 'coteprob', 'cheque', 'gains', 'dernierealloc',
                    'dernierecote', 'poidmont', 'gainsCarriere', 'gainsVictoires',
                    'gainsPlace', 'gainsAnneeEnCours', 'gainsAnneePrecedente',
                    'pourcVictEntHippo', 'pourcPlaceEntHippo', 'pourcPlaceJock',
                    'pourcVictJock', 'pourcVictJockHippo', 'pourcPlaceJockHippo',
                    'pourcVictChevalHippo', 'pourcPlaceChevalHippo',
                    'TxVictCouple', 'TxPlaceCouple', 'TxVictCoupleHippo', 'TxPlaceCoupleHippo'
                ]:
                    cleaned[key] = float(value) if value else None

                # Booléens
                elif key in [
                    'jumentPleine', 'indicateurInedit', 'estSupplemente',
                    'defFirstTime', 'oeilFirstTime'
                ]:
                    cleaned[key] = self._parse_bool(value)

                # Strings
                else:
                    cleaned[key] = value.strip()

            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing column {key}={value}: {e}")
                cleaned[key] = value  # Garder la valeur originale

        return cleaned

    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse une date depuis le CSV"""
        if not date_str:
            return None

        # Essayer différents formats
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%d%m%Y']

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return None

    def _parse_bool(self, value: str) -> bool:
        """Parse un booléen depuis le CSV"""
        if not value:
            return False

        value_lower = value.lower().strip()
        return value_lower in ['1', 'true', 'yes', 'oui', 't', 'y']

    async def get_courses_by_date(self, date_obj: date) -> List[AspiturfCourse]:
        """
        Récupère toutes les courses pour une date

        Args:
            date_obj: Date des courses

        Returns:
            Liste des courses avec leurs partants
        """
        if not self._data_loaded:
            await self._load_data()

        # Filtrer par date
        courses_data = [row for row in self._data if row.get('jour') == date_obj]

        # Grouper par course unique (numcourse)
        courses_dict: Dict[str, List[Dict[str, Any]]] = {}

        for row in courses_data:
            numcourse = row.get('numcourse')
            if numcourse:
                if numcourse not in courses_dict:
                    courses_dict[numcourse] = []
                courses_dict[numcourse].append(row)

        # Construire les objets AspiturfCourse
        courses = []
        for numcourse, partants_data in courses_dict.items():
            if not partants_data:
                continue

            # Prendre la première ligne pour les infos de course
            first_row = partants_data[0]

            course = AspiturfCourse(
                numcourse=numcourse,
                reun=first_row.get('reun'),
                prix=first_row.get('prix'),
                jour=first_row.get('jour'),
                hippo=first_row.get('hippo'),
                typec=first_row.get('typec'),
                dist=first_row.get('dist'),
                partant=first_row.get('partant'),
                cheque=first_row.get('cheque'),
                devise=first_row.get('devise'),
                partants=partants_data
            )

            courses.append(course)

        logger.info(f"Found {len(courses)} courses for {date_obj}")
        return courses

    async def get_partants_course(
        self,
        date_obj: date,
        hippodrome: str,
        course_num: int
    ) -> List[Dict[str, Any]]:
        """
        Récupère les partants d'une course spécifique

        Args:
            date_obj: Date de la course
            hippodrome: Nom de l'hippodrome
            course_num: Numéro de la course

        Returns:
            Liste des partants avec toutes leurs données
        """
        if not self._data_loaded:
            await self._load_data()

        # Filtrer les données
        partants = [
            row for row in self._data
            if (
                row.get('jour') == date_obj and
                row.get('hippo', '').upper() == hippodrome.upper() and
                row.get('prix') == course_num
            )
        ]

        logger.info(
            f"Found {len(partants)} partants for {hippodrome} "
            f"R{course_num} on {date_obj}"
        )

        return partants

    async def get_partant_details(
        self,
        date_obj: date,
        hippodrome: str,
        course_num: int,
        numero_cheval: int
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère les détails complets d'un partant

        Args:
            date_obj: Date de la course
            hippodrome: Nom de l'hippodrome
            course_num: Numéro de la course
            numero_cheval: Numéro du cheval

        Returns:
            Données complètes du partant ou None
        """
        partants = await self.get_partants_course(date_obj, hippodrome, course_num)

        for partant in partants:
            if partant.get('numero') == numero_cheval:
                return partants

        return None

    async def get_races(
        self,
        *,
        horse_id: Optional[str] = None,
        jockey_id: Optional[str] = None,
        trainer_id: Optional[str] = None,
        hippodrome: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retourne les lignes Aspiturf filtrées selon plusieurs critères."""

        if not self._data_loaded:
            await self._load_data()

        races = self._data

        if horse_id:
            races = [row for row in races if row.get('idChe') == horse_id]

        if jockey_id:
            races = [row for row in races if row.get('idJockey') == jockey_id]

        if trainer_id:
            races = [row for row in races if row.get('idEntraineur') == trainer_id]

        if hippodrome:
            hippo_upper = hippodrome.upper()
            races = [
                row for row in races
                if row.get('hippo') and row.get('hippo').upper() == hippo_upper
            ]

        return races

    async def leaderboard(
        self,
        entity_type: str,
        *,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        hippodrome: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Construit un classement agrégé par type d'entité."""

        if entity_type not in {"horse", "jockey", "trainer"}:
            raise ValueError(f"Type d'entité non supporté: {entity_type}")

        if not self._data_loaded:
            await self._load_data()

        hippo_upper = hippodrome.upper() if hippodrome else None

        aggregations: Dict[str, Dict[str, Any]] = {}

        for row in self._data:
            race_date = row.get("jour")

            if isinstance(race_date, date):
                if start_date and race_date < start_date:
                    continue
                if end_date and race_date > end_date:
                    continue
            elif start_date or end_date:
                # Sans information de date fiable on ignore la ligne si un filtre est demandé.
                continue

            if hippo_upper:
                hippo_value = row.get("hippo")
                if not hippo_value or hippo_value.upper() != hippo_upper:
                    continue

            if entity_type == "horse":
                entity_id = row.get("idChe")
                label = row.get("nom_cheval") or row.get("cheval")
            elif entity_type == "jockey":
                entity_id = row.get("idJockey")
                label = row.get("jockey")
            else:
                entity_id = row.get("idEntraineur")
                label = row.get("entraineur")

            if not entity_id:
                continue

            entry = aggregations.setdefault(
                str(entity_id),
                {
                    "entity_id": str(entity_id),
                    "label": label or str(entity_id),
                    "sample_size": 0,
                    "wins": 0,
                    "podiums": 0,
                    "positions": [],
                    "last_seen": None,
                },
            )

            if label:
                entry["label"] = label

            entry["sample_size"] += 1

            position = row.get("cl")
            if isinstance(position, int):
                entry["positions"].append(position)
                if position == 1:
                    entry["wins"] += 1
                if 1 <= position <= 3:
                    entry["podiums"] += 1

            if isinstance(race_date, date):
                current_last_seen = entry.get("last_seen")
                if current_last_seen is None or race_date > current_last_seen:
                    entry["last_seen"] = race_date

        leaderboard: List[Dict[str, Any]] = []

        for value in aggregations.values():
            sample_size = value["sample_size"] or 0
            if not sample_size:
                continue

            win_rate = value["wins"] / sample_size if sample_size else None
            podium_rate = value["podiums"] / sample_size if sample_size else None

            positions = [pos for pos in value.get("positions", []) if isinstance(pos, int)]
            average_finish = sum(positions) / len(positions) if positions else None

            leaderboard.append(
                {
                    "entity_id": value["entity_id"],
                    "label": value["label"],
                    "sample_size": sample_size,
                    "wins": value["wins"],
                    "podiums": value["podiums"],
                    "win_rate": round(win_rate, 4) if win_rate is not None else None,
                    "podium_rate": round(podium_rate, 4) if podium_rate is not None else None,
                    "average_finish": round(average_finish, 2) if average_finish is not None else None,
                    "last_seen": value.get("last_seen"),
                }
            )

        leaderboard.sort(
            key=lambda item: (
                -(item["win_rate"] or 0),
                -(item["podium_rate"] or 0),
                -item["sample_size"],
                item["label"],
            )
        )

        return leaderboard[: max(1, limit)]

    async def performance_trend(
        self,
        *,
        entity_type: str,
        entity_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        hippodrome: Optional[str] = None,
        granularity: str = "month",
    ) -> Dict[str, Any]:
        """Agrège les courses d'une entité par période temporelle."""

        if entity_type not in {"horse", "jockey", "trainer"}:
            raise ValueError(f"Type d'entité non supporté: {entity_type}")

        if not entity_id:
            raise ValueError("entity_id est requis pour calculer une tendance")

        if granularity not in {"month", "week"}:
            raise ValueError(f"Granularité inconnue: {granularity}")

        if not self._data_loaded:
            await self._load_data()

        hippo_upper = hippodrome.upper() if hippodrome else None

        if entity_type == "horse":
            key_field = "idChe"
            label_fields = ("nom_cheval", "cheval")
        elif entity_type == "jockey":
            key_field = "idJockey"
            label_fields = ("jockey",)
        else:
            key_field = "idEntraineur"
            label_fields = ("entraineur",)

        buckets: Dict[Any, Dict[str, Any]] = {}
        entity_label: Optional[str] = None
        first_date: Optional[date] = None
        last_date: Optional[date] = None

        def build_period_bounds(race_date: date) -> Dict[str, Any]:
            """Calcule les bornes et le libellé selon la granularité."""

            if granularity == "week":
                iso_year, iso_week, _ = race_date.isocalendar()
                start = race_date - timedelta(days=race_date.weekday())
                end = start + timedelta(days=6)
                label = f"{iso_year}-S{iso_week:02d}"
                return {
                    "key": (iso_year, iso_week),
                    "start": start,
                    "end": end,
                    "label": label,
                }

            month_last_day = monthrange(race_date.year, race_date.month)[1]
            start = race_date.replace(day=1)
            end = race_date.replace(day=month_last_day)
            label = f"{race_date.year}-{race_date.month:02d}"
            return {
                "key": (race_date.year, race_date.month),
                "start": start,
                "end": end,
                "label": label,
            }

        for row in self._data:
            if row.get(key_field) != entity_id:
                continue

            race_date = row.get("jour")
            if not isinstance(race_date, date):
                # Impossible de dater la course => on ignore pour un graphe temporel.
                continue

            if start_date and race_date < start_date:
                continue

            if end_date and race_date > end_date:
                continue

            if hippo_upper:
                hippo_value = row.get("hippo")
                if not hippo_value or hippo_value.upper() != hippo_upper:
                    continue

            if entity_label is None:
                for field in label_fields:
                    label_value = row.get(field)
                    if isinstance(label_value, str) and label_value.strip():
                        entity_label = label_value.strip()
                        break

            if first_date is None or race_date < first_date:
                first_date = race_date

            if last_date is None or race_date > last_date:
                last_date = race_date

            period_data = build_period_bounds(race_date)
            bucket = buckets.setdefault(
                period_data["key"],
                {
                    "period_start": period_data["start"],
                    "period_end": period_data["end"],
                    "label": period_data["label"],
                    "races": 0,
                    "wins": 0,
                    "podiums": 0,
                    "positions": [],
                    "odds": [],
                },
            )

            bucket["races"] += 1

            position = row.get("cl")
            if isinstance(position, int):
                bucket["positions"].append(position)
                if position == 1:
                    bucket["wins"] += 1
                if 1 <= position <= 3:
                    bucket["podiums"] += 1

            odds = row.get("cotedirect") or row.get("coteprob")
            if isinstance(odds, (int, float)):
                bucket["odds"].append(float(odds))

        points: List[Dict[str, Any]] = []

        for _, bucket in sorted(buckets.items(), key=lambda item: item[1]["period_start"]):
            races = bucket["races"]
            wins = bucket["wins"]
            podiums = bucket["podiums"]

            positions = bucket["positions"]
            average_finish = sum(positions) / len(positions) if positions else None

            odds_values = bucket["odds"]
            average_odds = sum(odds_values) / len(odds_values) if odds_values else None

            points.append(
                {
                    "period_start": bucket["period_start"],
                    "period_end": bucket["period_end"],
                    "label": bucket["label"],
                    "races": races,
                    "wins": wins,
                    "podiums": podiums,
                    "win_rate": round(wins / races, 4) if races else None,
                    "podium_rate": round(podiums / races, 4) if races else None,
                    "average_finish": round(average_finish, 2) if average_finish is not None else None,
                    "average_odds": round(average_odds, 2) if average_odds is not None else None,
                }
            )

        return {
            "entity_id": entity_id,
            "entity_label": entity_label,
            "date_start": first_date,
            "date_end": last_date,
            "points": points,
        }

    async def performance_distribution(
        self,
        *,
        entity_type: str,
        entity_id: str,
        dimension: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        hippodrome: Optional[str] = None,
        distance_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Construit une distribution des résultats d'une entité."""

        if entity_type not in {"horse", "jockey", "trainer"}:
            raise ValueError(f"Type d'entité non supporté: {entity_type}")

        if not entity_id:
            raise ValueError("entity_id est requis pour analyser une distribution")

        allowed_dimensions = {"distance", "draw", "hippodrome", "discipline"}
        if dimension not in allowed_dimensions:
            raise ValueError(f"Dimension inconnue: {dimension}")

        if dimension != "distance":
            distance_step = None
        else:
            step = distance_step or 200
            if step <= 0:
                raise ValueError("distance_step doit être strictement positif")
            distance_step = step

        if not self._data_loaded:
            await self._load_data()

        hippo_upper = hippodrome.upper() if hippodrome else None

        if entity_type == "horse":
            key_field = "idChe"
            label_fields = ("nom_cheval", "cheval")
        elif entity_type == "jockey":
            key_field = "idJockey"
            label_fields = ("jockey",)
        else:
            key_field = "idEntraineur"
            label_fields = ("entraineur",)

        def resolve_bucket(row: Dict[str, Any]) -> Optional[Tuple[Any, str]]:
            """Détermine le seau d'agrégation à partir de la dimension choisie."""

            if dimension == "distance":
                value = row.get("dist")
                if not isinstance(value, int):
                    return None
                assert distance_step is not None  # pour mypy
                bucket_start = (value // distance_step) * distance_step
                bucket_end = bucket_start + distance_step - 1
                label = f"{bucket_start}-{bucket_end} m"
                return (bucket_start, label)

            if dimension == "draw":
                value = row.get("numero")
                if value is None:
                    return None
                try:
                    numeric = int(value)
                except (TypeError, ValueError):
                    return None
                label = f"N° {numeric}"
                return (numeric, label)

            if dimension == "hippodrome":
                value = row.get("hippo")
                if not value:
                    return None
                label = str(value).upper()
                return (label, label)

            raw_discipline = (
                row.get("typec")
                or row.get("typecourse")
                or row.get("discipline")
                or "Inconnu"
            )
            label = str(raw_discipline).strip() or "Inconnu"
            return (label.lower(), label.title())

        buckets: Dict[Any, Dict[str, Any]] = {}
        entity_label: Optional[str] = None
        first_date: Optional[date] = None
        last_date: Optional[date] = None

        for row in self._data:
            if row.get(key_field) != entity_id:
                continue

            race_date = row.get("jour")
            if isinstance(race_date, date):
                if start_date and race_date < start_date:
                    continue
                if end_date and race_date > end_date:
                    continue
            elif start_date or end_date:
                # Les lignes sans date fiable ne permettent pas de respecter le filtre temporel.
                continue

            if hippo_upper:
                hippo_value = row.get("hippo")
                if not hippo_value or hippo_value.upper() != hippo_upper:
                    continue

            bucket_descriptor = resolve_bucket(row)
            if bucket_descriptor is None:
                continue

            bucket_key, bucket_label = bucket_descriptor

            if entity_label is None:
                for field in label_fields:
                    value = row.get(field)
                    if isinstance(value, str) and value.strip():
                        entity_label = value.strip()
                        break

            if isinstance(race_date, date):
                if first_date is None or race_date < first_date:
                    first_date = race_date
                if last_date is None or race_date > last_date:
                    last_date = race_date

            bucket = buckets.setdefault(
                bucket_key,
                {
                    "label": bucket_label,
                    "races": 0,
                    "wins": 0,
                    "podiums": 0,
                    "positions": [],
                    "odds": [],
                },
            )

            bucket["label"] = bucket_label
            bucket["races"] += 1

            position = row.get("cl")
            if isinstance(position, int):
                bucket["positions"].append(position)
                if position == 1:
                    bucket["wins"] += 1
                if 1 <= position <= 3:
                    bucket["podiums"] += 1

            odds_value = row.get("cotedirect") or row.get("coteprob")
            if isinstance(odds_value, (int, float)):
                bucket["odds"].append(float(odds_value))

        formatted: List[Dict[str, Any]] = []
        for key, bucket in buckets.items():
            races = bucket["races"]
            wins = bucket["wins"]
            podiums = bucket["podiums"]

            positions = [pos for pos in bucket["positions"] if isinstance(pos, (int, float))]
            odds_values = bucket["odds"]

            average_finish = sum(positions) / len(positions) if positions else None
            average_odds = sum(odds_values) / len(odds_values) if odds_values else None

            formatted.append(
                {
                    "key": key,
                    "label": bucket["label"],
                    "races": races,
                    "wins": wins,
                    "podiums": podiums,
                    "win_rate": round(wins / races, 4) if races else None,
                    "podium_rate": round(podiums / races, 4) if races else None,
                    "average_finish": round(average_finish, 2) if average_finish is not None else None,
                    "average_odds": round(average_odds, 2) if average_odds is not None else None,
                }
            )

        # Tri décroissant sur le volume de courses puis sur le nombre de victoires.
        formatted.sort(key=lambda item: (item["races"], item["wins"]), reverse=True)

        return {
            "entity_id": entity_id,
            "entity_label": entity_label,
            "date_start": first_date,
            "date_end": last_date,
            "dimension": dimension,
            "buckets": formatted,
        }

    async def performance_streaks(
        self,
        *,
        entity_type: str,
        entity_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        hippodrome: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyse les séries de résultats consécutifs pour une entité Aspiturf."""

        if entity_type not in {"horse", "jockey", "trainer"}:
            raise ValueError(f"Type d'entité non supporté: {entity_type}")

        if not entity_id:
            raise ValueError("entity_id est requis pour calculer des séries")

        if not self._data_loaded:
            await self._load_data()

        hippo_upper = hippodrome.upper() if hippodrome else None

        if entity_type == "horse":
            key_field = "idChe"
            label_fields = ("nom_cheval", "cheval")
        elif entity_type == "jockey":
            key_field = "idJockey"
            label_fields = ("jockey",)
        else:
            key_field = "idEntraineur"
            label_fields = ("entraineur",)

        filtered: List[Dict[str, Any]] = []
        entity_label: Optional[str] = None
        first_date: Optional[date] = None
        last_date: Optional[date] = None
        wins = 0
        podiums = 0

        for row in self._data:
            if row.get(key_field) != entity_id:
                continue

            race_date = row.get("jour")
            if not isinstance(race_date, date):
                # Les séries reposent sur l'ordre chronologique => on ignore les dates manquantes.
                continue

            if start_date and race_date < start_date:
                continue

            if end_date and race_date > end_date:
                continue

            if hippo_upper:
                hippo_value = row.get("hippo")
                if not hippo_value or hippo_value.upper() != hippo_upper:
                    continue

            if entity_label is None:
                for field in label_fields:
                    label = row.get(field)
                    if isinstance(label, str) and label.strip():
                        entity_label = label.strip()
                        break

            position = row.get("cl")
            if isinstance(position, int):
                if position == 1:
                    wins += 1
                if 1 <= position <= 3:
                    podiums += 1

            if first_date is None or race_date < first_date:
                first_date = race_date

            if last_date is None or race_date > last_date:
                last_date = race_date

            filtered.append(dict(row))

        filtered.sort(key=lambda item: item.get("jour") or date.min)

        total_races = len(filtered)

        if total_races == 0:
            return {
                "entity_id": str(entity_id),
                "entity_label": entity_label,
                "total_races": 0,
                "wins": 0,
                "podiums": 0,
                "date_start": None,
                "date_end": None,
                "best_win": None,
                "best_podium": None,
                "current_win": None,
                "current_podium": None,
                "history": [],
            }

        current = {
            "win": {"length": 0, "start": None, "end": None},
            "podium": {"length": 0, "start": None, "end": None},
        }
        best = {
            "win": {"length": 0, "start": None, "end": None},
            "podium": {"length": 0, "start": None, "end": None},
        }
        history: List[Dict[str, Any]] = []

        def update_best(kind: str) -> None:
            tracker = current[kind]
            if tracker["length"] > best[kind]["length"]:
                best[kind] = {
                    "length": tracker["length"],
                    "start": tracker["start"],
                    "end": tracker["end"],
                }

        def close_streak(kind: str) -> None:
            tracker = current[kind]
            if tracker["length"]:
                history.append(
                    {
                        "type": kind,
                        "length": tracker["length"],
                        "start_date": tracker["start"],
                        "end_date": tracker["end"],
                        "is_active": False,
                    }
                )
                tracker["length"] = 0
                tracker["start"] = None
                tracker["end"] = None

        for row in filtered:
            race_date = row.get("jour")
            position = row.get("cl")

            is_win = isinstance(position, int) and position == 1
            is_podium = isinstance(position, int) and 1 <= position <= 3

            if is_win:
                tracker = current["win"]
                if tracker["length"] == 0:
                    tracker["start"] = race_date
                tracker["length"] += 1
                tracker["end"] = race_date
                update_best("win")
            else:
                close_streak("win")

            if is_podium:
                tracker = current["podium"]
                if tracker["length"] == 0:
                    tracker["start"] = race_date
                tracker["length"] += 1
                tracker["end"] = race_date
                update_best("podium")
            else:
                close_streak("podium")

        for kind in ("win", "podium"):
            tracker = current[kind]
            if tracker["length"]:
                history.append(
                    {
                        "type": kind,
                        "length": tracker["length"],
                        "start_date": tracker["start"],
                        "end_date": tracker["end"],
                        "is_active": True,
                    }
                )
                update_best(kind)

        def build_summary(kind: str) -> Optional[Dict[str, Any]]:
            tracker = current[kind]
            if tracker["length"] <= 0:
                return None
            return {
                "type": kind,
                "length": tracker["length"],
                "start_date": tracker["start"],
                "end_date": tracker["end"],
                "is_active": True,
            }

        def build_best(kind: str) -> Optional[Dict[str, Any]]:
            data = best[kind]
            if data["length"] <= 0:
                return None
            tracker = current[kind]
            is_active = (
                tracker["length"] == data["length"]
                and tracker["start"] == data["start"]
                and tracker["end"] == data["end"]
            )
            return {
                "type": kind,
                "length": data["length"],
                "start_date": data["start"],
                "end_date": data["end"],
                "is_active": is_active,
            }

        history.sort(
            key=lambda item: (
                item.get("is_active", False),
                item.get("length", 0),
                item.get("end_date") or date.min,
            ),
            reverse=True,
        )

        history = history[:10]

        return {
            "entity_id": str(entity_id),
            "entity_label": entity_label,
            "total_races": total_races,
            "wins": wins,
            "podiums": podiums,
            "date_start": first_date,
            "date_end": last_date,
            "best_win": build_best("win"),
            "best_podium": build_best("podium"),
            "current_win": build_summary("win"),
            "current_podium": build_summary("podium"),
            "history": history,
        }

    async def search_entities(
        self,
        entity_type: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Recherche souple d'entités (cheval, jockey, entraineur, hippodrome)."""

        if not self._data_loaded:
            await self._load_data()

        normalized_query = query.strip().lower()
        if len(normalized_query) < 2:
            return []

        def matches(*values: Optional[str]) -> bool:
            for value in values:
                if value and normalized_query in value.lower():
                    return True
            return False

        results: Dict[str, Dict[str, Any]] = {}

        if entity_type == "horse":
            for row in self._data:
                horse_id = row.get("idChe")
                if not horse_id:
                    continue

                name = row.get("nom_cheval") or row.get("cheval")
                if not matches(horse_id, name):
                    continue

                entry = results.setdefault(
                    horse_id,
                    {
                        "id": horse_id,
                        "label": name or str(horse_id),
                        "metadata": {
                            "total_races": 0,
                            "hippodromes": set(),
                            "last_seen": None,
                        },
                    },
                )

                entry["label"] = name or entry["label"]
                entry["metadata"]["total_races"] += 1

                hippo = row.get("hippo")
                if hippo:
                    entry["metadata"]["hippodromes"].add(str(hippo))

                race_date = row.get("jour")
                if isinstance(race_date, date):
                    last_seen = entry["metadata"].get("last_seen")
                    if last_seen is None or race_date > last_seen:
                        entry["metadata"]["last_seen"] = race_date

        elif entity_type == "jockey":
            for row in self._data:
                jockey_id = row.get("idJockey")
                if not jockey_id:
                    continue

                name = row.get("jockey")
                if not matches(jockey_id, name):
                    continue

                entry = results.setdefault(
                    jockey_id,
                    {
                        "id": jockey_id,
                        "label": name or str(jockey_id),
                        "metadata": {
                            "total_races": 0,
                            "hippodromes": set(),
                            "last_seen": None,
                        },
                    },
                )

                entry["label"] = name or entry["label"]
                entry["metadata"]["total_races"] += 1

                hippo = row.get("hippo")
                if hippo:
                    entry["metadata"]["hippodromes"].add(str(hippo))

                race_date = row.get("jour")
                if isinstance(race_date, date):
                    last_seen = entry["metadata"].get("last_seen")
                    if last_seen is None or race_date > last_seen:
                        entry["metadata"]["last_seen"] = race_date

        elif entity_type == "trainer":
            for row in self._data:
                trainer_id = row.get("idEntraineur")
                if not trainer_id:
                    continue

                name = row.get("entraineur")
                if not matches(trainer_id, name):
                    continue

                entry = results.setdefault(
                    trainer_id,
                    {
                        "id": trainer_id,
                        "label": name or str(trainer_id),
                        "metadata": {
                            "total_races": 0,
                            "hippodromes": set(),
                            "last_seen": None,
                        },
                    },
                )

                entry["label"] = name or entry["label"]
                entry["metadata"]["total_races"] += 1

                hippo = row.get("hippo")
                if hippo:
                    entry["metadata"]["hippodromes"].add(str(hippo))

                race_date = row.get("jour")
                if isinstance(race_date, date):
                    last_seen = entry["metadata"].get("last_seen")
                    if last_seen is None or race_date > last_seen:
                        entry["metadata"]["last_seen"] = race_date

        elif entity_type == "hippodrome":
            for row in self._data:
                hippo = row.get("hippo")
                if not hippo:
                    continue

                hippo_str = str(hippo)
                hippo_id = hippo_str.upper()
                if not matches(hippo_id, hippo_str):
                    continue

                entry = results.setdefault(
                    hippo_id,
                    {
                        "id": hippo_id,
                        "label": hippo_str,
                        "metadata": {
                            "course_count": 0,
                            "disciplines": set(),
                            "last_meeting": None,
                        },
                    },
                )

                entry["metadata"]["course_count"] += 1

                discipline = row.get("typec")
                if discipline:
                    entry["metadata"]["disciplines"].add(str(discipline))

                race_date = row.get("jour")
                if isinstance(race_date, date):
                    last_meeting = entry["metadata"].get("last_meeting")
                    if last_meeting is None or race_date > last_meeting:
                        entry["metadata"]["last_meeting"] = race_date

        else:
            return []

        formatted_results: List[Dict[str, Any]] = []

        for entry in results.values():
            metadata = entry["metadata"]

            hippodromes: Optional[Set[str]] = metadata.get("hippodromes")
            if isinstance(hippodromes, set):
                metadata["hippodromes"] = sorted(hippodromes)[:3]

            disciplines: Optional[Set[str]] = metadata.get("disciplines")
            if isinstance(disciplines, set):
                metadata["disciplines"] = sorted(disciplines)

            formatted_results.append(
                {
                    "id": entry["id"],
                    "label": entry["label"],
                    "metadata": metadata,
                }
            )

        def sort_key(item: Dict[str, Any]):
            meta = item.get("metadata", {})
            primary = meta.get("total_races") or meta.get("course_count") or 0
            return (-int(primary), item.get("label") or "")

        formatted_results.sort(key=sort_key)

        return formatted_results[:limit]

    async def get_horse_statistics(
        self,
        horse_id: str,
        hippodrome: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Récupère les statistiques d'un cheval

        Args:
            horse_id: ID du cheval (idChe)
            hippodrome: Filtrer par hippodrome (optionnel)

        Returns:
            Statistiques agrégées du cheval
        """
        if not self._data_loaded:
            await self._load_data()

        # Filtrer les courses du cheval
        horse_races = [
            row for row in self._data
            if row.get('idChe') == horse_id
        ]

        if hippodrome:
            horse_races = [
                row for row in horse_races
                if row.get('hippo', '').upper() == hippodrome.upper()
            ]

        # Calculer les statistiques
        if not horse_races:
            return {}

        # Prendre la ligne la plus récente pour les stats
        most_recent = max(horse_races, key=lambda x: x.get('jour', date.min))

        return {
            'id_cheval': horse_id,
            'courses': most_recent.get('coursescheval'),
            'victoires': most_recent.get('victoirescheval'),
            'places': most_recent.get('placescheval'),
            'gains_carriere': most_recent.get('gainsCarriere'),
            'musique': most_recent.get('musiqueche'),
            'total_races_in_data': len(horse_races)
        }

    async def get_jockey_statistics(
        self,
        jockey_id: str,
        hippodrome: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Récupère les statistiques d'un jockey

        Args:
            jockey_id: ID du jockey (idJockey)
            hippodrome: Filtrer par hippodrome (optionnel)

        Returns:
            Statistiques agrégées du jockey
        """
        if not self._data_loaded:
            await self._load_data()

        # Filtrer les courses du jockey
        jockey_races = [
            row for row in self._data
            if row.get('idJockey') == jockey_id
        ]

        if hippodrome:
            jockey_races = [
                row for row in jockey_races
                if row.get('hippo', '').upper() == hippodrome.upper()
            ]

        if not jockey_races:
            return {}

        # Prendre la ligne la plus récente
        most_recent = max(jockey_races, key=lambda x: x.get('jour', date.min))

        return {
            'id_jockey': jockey_id,
            'jockey': most_recent.get('jockey'),
            'courses': most_recent.get('coursesjockey'),
            'victoires': most_recent.get('victoiresjockey'),
            'places': most_recent.get('placejockey'),
            'pourc_victoires': most_recent.get('pourcVictJock'),
            'musique': most_recent.get('musiquejoc'),
            'total_races_in_data': len(jockey_races)
        }

    async def get_trainer_statistics(
        self,
        trainer_id: str,
        hippodrome: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Récupère les statistiques d'un entraineur

        Args:
            trainer_id: ID de l'entraineur (idEntraineur)
            hippodrome: Filtrer par hippodrome (optionnel)

        Returns:
            Statistiques agrégées de l'entraineur
        """
        if not self._data_loaded:
            await self._load_data()

        # Filtrer les courses de l'entraineur
        trainer_races = [
            row for row in self._data
            if row.get('idEntraineur') == trainer_id
        ]

        if hippodrome:
            trainer_races = [
                row for row in trainer_races
                if row.get('hippo', '').upper() == hippodrome.upper()
            ]

        if not trainer_races:
            return {}

        # Prendre la ligne la plus récente
        most_recent = max(trainer_races, key=lambda x: x.get('jour', date.min))

        return {
            'id_entraineur': trainer_id,
            'entraineur': most_recent.get('entraineur'),
            'courses': most_recent.get('coursesentraineur'),
            'victoires': most_recent.get('victoiresentraineur'),
            'places': most_recent.get('placeentraineur'),
            'musique': most_recent.get('musiqueent'),
            'total_races_in_data': len(trainer_races)
        }

    async def get_couple_statistics(
        self,
        horse_id: str,
        jockey_id: str,
        hippodrome: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Récupère les statistiques du couple cheval-jockey

        Args:
            horse_id: ID du cheval
            jockey_id: ID du jockey
            hippodrome: Filtrer par hippodrome (optionnel)

        Returns:
            Statistiques du couple
        """
        if not self._data_loaded:
            await self._load_data()

        # Filtrer les courses du couple
        couple_races = [
            row for row in self._data
            if (
                row.get('idChe') == horse_id and
                row.get('idJockey') == jockey_id
            )
        ]

        if hippodrome:
            couple_races = [
                row for row in couple_races
                if row.get('hippo', '').upper() == hippodrome.upper()
            ]

        if not couple_races:
            return {}

        # Prendre la ligne la plus récente
        most_recent = max(couple_races, key=lambda x: x.get('jour', date.min))

        if hippodrome:
            return {
                'nb_courses': most_recent.get('nbCourseCoupleHippo'),
                'nb_victoires': most_recent.get('nbVictCoupleHippo'),
                'nb_places': most_recent.get('nbPlaceCoupleHippo'),
                'tx_victoires': most_recent.get('TxVictCoupleHippo'),
                'tx_places': most_recent.get('TxPlaceCoupleHippo'),
                'total_races_in_data': len(couple_races)
            }
        else:
            return {
                'nb_courses': most_recent.get('nbCourseCouple'),
                'nb_victoires': most_recent.get('nbVictCouple'),
                'nb_places': most_recent.get('nbPlaceCouple'),
                'tx_victoires': most_recent.get('TxVictCouple'),
                'tx_places': most_recent.get('TxPlaceCouple'),
                'total_races_in_data': len(couple_races)
            }


# Fonctions utilitaires

async def load_aspiturf_data(
    csv_path: Optional[Union[str, Path]] = None,
    csv_url: Optional[str] = None,
    date_filter: Optional[date] = None
) -> List[Dict[str, Any]]:
    """
    Fonction utilitaire pour charger les données Aspiturf

    Args:
        csv_path: Chemin vers fichier local
        csv_url: URL de téléchargement
        date_filter: Filtrer par date (optionnel)

    Returns:
        Liste des données
    """
    async with AspiturfClient(csv_path=csv_path, csv_url=csv_url) as client:
        if date_filter:
            courses = await client.get_courses_by_date(date_filter)
            # Aplatir en liste de partants
            data = []
            for course in courses:
                data.extend(course.partants)
            return data
        else:
            return client._data


async def get_course_complete_data(
    csv_path: Optional[Union[str, Path]] = None,
    csv_url: Optional[str] = None,
    date_obj: date = None,
    hippodrome: str = None,
    course_num: int = None
) -> Dict[str, Any]:
    """
    Fonction utilitaire pour récupérer toutes les données d'une course

    Args:
        csv_path: Chemin vers fichier local
        csv_url: URL de téléchargement
        date_obj: Date de la course
        hippodrome: Nom de l'hippodrome
        course_num: Numéro de la course

    Returns:
        Dict avec course et partants détaillés
    """
    async with AspiturfClient(csv_path=csv_path, csv_url=csv_url) as client:
        partants = await client.get_partants_course(date_obj, hippodrome, course_num)

        if not partants:
            return None

        # Info de la course depuis le premier partant
        first = partants[0]

        return {
            'course': {
                'date': first.get('jour'),
                'hippodrome': first.get('hippo'),
                'numero': first.get('prix'),
                'type': first.get('typec'),
                'distance': first.get('dist'),
                'allocation': first.get('cheque'),
                'nombre_partants': first.get('partant')
            },
            'partants': partants
        }
