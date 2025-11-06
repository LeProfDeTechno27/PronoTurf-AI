# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Service de récupération des programmes PMU
Gère la récupération des données de courses depuis différentes sources

Ordre de priorité des sources:
1. Aspiturf (données CSV ultra-détaillées) - SOURCE PRINCIPALE
2. TurfinfoClient (API gratuite) - FALLBACK
3. OpenPMUClient (résultats officiels) - RÉSULTATS

Architecture avec fallback automatique pour garantir la disponibilité.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime, time
from pathlib import Path
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import Hippodrome, Reunion, Course, Horse, Jockey, Trainer, Partant
from app.models.hippodrome import TrackType
from app.models.reunion import ReunionStatus
from app.models.course import Discipline, SurfaceType, StartType, CourseStatus
from app.models.horse import Gender
from app.services.aspiturf_client import AspiturfClient, AspiturfConfig
from app.services.turfinfo_client import TurfinfoClient, TurfinfoEndpoint
from app.services.open_pmu_client import OpenPMUClient

logger = logging.getLogger(__name__)


class DataSource:
    """Enum pour les sources de données"""
    ASPITURF = "aspiturf"
    TURFINFO = "turfinfo"
    OPENPMU = "openpmu"


class PMUService:
    """
    Service pour récupérer et synchroniser les programmes de courses PMU

    Architecture multi-source avec priorité:
    1. Aspiturf (si configuré) - Données complètes et détaillées
    2. TurfinfoClient - Fallback pour données avant-course
    3. OpenPMUClient - Résultats officiels après-course

    Le service bascule automatiquement sur la source suivante en cas d'erreur.
    """

    def __init__(
        self,
        db: AsyncSession,
        aspiturf_csv_path: Optional[Union[str, Path]] = None,
        aspiturf_csv_url: Optional[str] = None,
        aspiturf_config: Optional[AspiturfConfig] = None,
        endpoint_type: TurfinfoEndpoint = TurfinfoEndpoint.ONLINE,
        enable_fallback: bool = True
    ):
        """
        Initialise le service PMU

        Args:
            db: Session SQLAlchemy
            aspiturf_csv_path: Chemin vers fichier CSV Aspiturf (prioritaire)
            aspiturf_csv_url: URL pour télécharger CSV Aspiturf
            aspiturf_config: Configuration Aspiturf
            endpoint_type: Type d'endpoint TurfInfo (si fallback)
            enable_fallback: Activer le fallback automatique
        """
        self.db = db
        self.aspiturf_csv_path = aspiturf_csv_path
        self.aspiturf_csv_url = aspiturf_csv_url
        self.aspiturf_config = aspiturf_config
        self.endpoint_type = endpoint_type
        self.enable_fallback = enable_fallback

        # Déterminer la source principale
        self.primary_source = (
            DataSource.ASPITURF
            if (aspiturf_csv_path or aspiturf_csv_url)
            else DataSource.TURFINFO
        )

        logger.info(
            f"PMUService initialized with primary source: {self.primary_source}, "
            f"fallback: {enable_fallback}"
        )

    async def fetch_program_for_date(
        self,
        program_date: date,
        force_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Récupère le programme complet pour une date donnée

        Utilise la source principale (Aspiturf si configuré), avec fallback
        automatique sur TurfInfo en cas d'erreur.

        Args:
            program_date: Date du programme à récupérer
            force_source: Forcer une source spécifique (aspiturf, turfinfo)

        Returns:
            Dictionnaire avec:
            - source: Source de données utilisée
            - data: Données du programme (format unifié)
        """
        source = force_source or self.primary_source

        # Essayer Aspiturf en premier (si configuré)
        if source == DataSource.ASPITURF:
            try:
                data = await self._fetch_program_aspiturf(program_date)
                logger.info(
                    f"✅ Successfully fetched program from Aspiturf for {program_date}"
                )
                return {
                    'source': DataSource.ASPITURF,
                    'data': data
                }
            except Exception as e:
                logger.warning(
                    f"⚠️ Aspiturf failed for {program_date}: {e}"
                )
                if not self.enable_fallback:
                    raise

                logger.info("🔄 Falling back to TurfInfo...")

        # Fallback sur TurfInfo
        try:
            data = await self._fetch_program_turfinfo(program_date)
            logger.info(
                f"✅ Successfully fetched program from TurfInfo for {program_date}"
            )
            return {
                'source': DataSource.TURFINFO,
                'data': data
            }
        except Exception as e:
            logger.error(f"❌ All sources failed for {program_date}: {e}")
            raise

    async def _fetch_program_aspiturf(self, program_date: date) -> Dict[str, Any]:
        """
        Récupère le programme depuis Aspiturf (CSV)

        Returns:
            Format unifié compatible avec le reste du service
        """
        async with AspiturfClient(
            csv_path=self.aspiturf_csv_path,
            csv_url=self.aspiturf_csv_url,
            config=self.aspiturf_config
        ) as client:
            courses = await client.get_courses_by_date(program_date)

            # Convertir en format unifié (structure similaire à TurfInfo)
            return self._convert_aspiturf_to_unified(courses)

    async def _fetch_program_turfinfo(self, program_date: date) -> Dict[str, Any]:
        """Récupère le programme depuis TurfInfo"""
        async with TurfinfoClient(endpoint_type=self.endpoint_type) as client:
            return await client.get_programme_jour(program_date)

    def _convert_aspiturf_to_unified(
        self,
        aspiturf_courses: List[Any]
    ) -> Dict[str, Any]:
        """
        Convertit les données Aspiturf en format unifié

        Le format unifié ressemble à la structure TurfInfo pour compatibilité
        avec le reste du code existant.
        """
        # Grouper par hippodrome pour créer des réunions
        hippodromes: Dict[str, List[Any]] = {}

        for course in aspiturf_courses:
            hippo = course.hippo
            if hippo not in hippodromes:
                hippodromes[hippo] = []
            hippodromes[hippo].append(course)

        # Construire la structure unifiée
        reunions = []

        for hippo_name, courses_list in hippodromes.items():
            # Créer une réunion par hippodrome
            courses_data = []

            for course in courses_list:
                course_dict = {
                    'numero': course.prix,
                    'numcourse': course.numcourse,
                    'nom': f"Course {course.prix}",  # Aspiturf n'a pas le nom
                    'discipline': self._map_aspiturf_discipline(course.typec),
                    'distance': course.dist,
                    'allocation': course.cheque,
                    'nombre_partants': course.partant,
                    'heure_depart': '14:00',  # Valeur par défaut, Aspiturf n'a pas l'heure
                    'statut': 'scheduled',
                    'partants': self._convert_aspiturf_partants(course.partants)
                }
                courses_data.append(course_dict)

            reunion_dict = {
                'numero': 1,  # Aspiturf n'a pas de numéro de réunion explicite
                'hippodrome': {
                    'code': hippo_name[:3].upper(),
                    'nom': hippo_name,
                    'ville': '',
                    'pays': 'France'
                },
                'courses': courses_data,
                'statut': 'scheduled'
            }
            reunions.append(reunion_dict)

        return {'reunions': reunions}

    def _convert_aspiturf_partants(
        self,
        aspiturf_partants: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convertit les partants Aspiturf en format unifié"""
        partants = []

        for partant_data in aspiturf_partants:
            partant = {
                'numero': partant_data.get('numero'),
                'cheval': {
                    'id': partant_data.get('idChe'),
                    'nom': partant_data.get('nom_cheval', ''),
                    'sexe': partant_data.get('sexe'),
                    'age': partant_data.get('age'),
                    'annee_naissance': None,
                    'robe': partant_data.get('coat'),
                    'race': None,
                    'pere': partant_data.get('pere'),
                    'mere': partant_data.get('mere'),
                    'proprietaire': partant_data.get('proprietaire')
                },
                'jockey': {
                    'id': partant_data.get('idJockey'),
                    'nom': partant_data.get('jockey', ''),
                    'prenom': '',
                    'nationalite': None
                } if partant_data.get('jockey') else None,
                'entraineur': {
                    'id': partant_data.get('idEntraineur'),
                    'nom': partant_data.get('entraineur', ''),
                    'prenom': '',
                    'ecurie': partant_data.get('ecurie')
                },
                'poids': partant_data.get('poidmont'),
                'handicap': partant_data.get('vha'),
                'equipement': partant_data.get('defoeil'),
                'cote': partant_data.get('cotedirect') or partant_data.get('coteprob'),
                'musique': partant_data.get('musiqueche'),
                'jours_depuis_derniere_course': partant_data.get('recence'),

                # Données supplémentaires spécifiques Aspiturf
                'aspiturf_data': {
                    'gains_carriere': partant_data.get('gainsCarriere'),
                    'gains_annee': partant_data.get('gainsAnneeEnCours'),
                    'courses_total': partant_data.get('coursescheval'),
                    'victoires_total': partant_data.get('victoirescheval'),
                    'places_total': partant_data.get('placescheval'),
                    'pourc_vict_hippo': partant_data.get('pourcVictChevalHippo'),
                    'pourc_place_hippo': partant_data.get('pourcPlaceChevalHippo'),
                    'appet_terrain': partant_data.get('appetTerrain'),
                    'record_general': partant_data.get('recordG'),
                    'indicateur_inedit': partant_data.get('indicateurInedit'),

                    # Stats jockey
                    'jockey_courses': partant_data.get('coursesjockey'),
                    'jockey_victoires': partant_data.get('victoiresjockey'),
                    'jockey_pourc_vict': partant_data.get('pourcVictJock'),

                    # Stats entraineur
                    'entraineur_courses': partant_data.get('coursesentraineur'),
                    'entraineur_victoires': partant_data.get('victoiresentraineur'),
                    'entraineur_pourc_vict_hippo': partant_data.get('pourcVictEntHippo'),

                    # Stats couple
                    'couple_courses': partant_data.get('nbCourseCouple'),
                    'couple_victoires': partant_data.get('nbVictCouple'),
                    'couple_tx_vict': partant_data.get('TxVictCouple')
                }
            }
            partants.append(partant)

        return partants

    def _map_aspiturf_discipline(self, typec: Optional[str]) -> str:
        """
        Mappe le type de course Aspiturf vers discipline standard

        Args:
            typec: Type de course Aspiturf (ex: "plat", "attelé", "haies")

        Returns:
            Discipline standardisée
        """
        if not typec:
            return "PLAT"

        typec_lower = typec.lower()

        if "plat" in typec_lower:
            return "PLAT"
        elif "attelé" in typec_lower or "attele" in typec_lower:
            return "TROT_ATTELE"
        elif "monté" in typec_lower or "monte" in typec_lower:
            return "TROT_MONTE"
        elif "haies" in typec_lower:
            return "HAIES"
        elif "steeple" in typec_lower or "chase" in typec_lower:
            return "STEEPLE"
        elif "cross" in typec_lower:
            return "CROSS"
        else:
            return "PLAT"

    async def fetch_course_partants(
        self,
        program_date: date,
        reunion_num: int,
        course_num: int,
        hippodrome_name: Optional[str] = None,
        include_performances: bool = True,
        force_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Récupère les partants et performances d'une course spécifique

        Avec support Aspiturf et fallback automatique.

        Args:
            program_date: Date de la course
            reunion_num: Numéro de réunion
            course_num: Numéro de course
            hippodrome_name: Nom hippodrome (requis pour Aspiturf)
            include_performances: Inclure les performances détaillées
            force_source: Forcer une source spécifique

        Returns:
            Dictionnaire avec source, partants et performances
        """
        source = force_source or self.primary_source

        # Essayer Aspiturf en premier (si configuré et hippodrome fourni)
        if source == DataSource.ASPITURF and hippodrome_name:
            try:
                result = await self._fetch_partants_aspiturf(
                    program_date,
                    hippodrome_name,
                    course_num
                )
                logger.info(
                    f"✅ Fetched partants from Aspiturf for R{reunion_num}C{course_num}"
                )
                return {
                    'source': DataSource.ASPITURF,
                    **result
                }
            except Exception as e:
                logger.warning(
                    f"⚠️ Aspiturf failed for R{reunion_num}C{course_num}: {e}"
                )
                if not self.enable_fallback:
                    raise

                logger.info("🔄 Falling back to TurfInfo...")

        # Fallback sur TurfInfo
        try:
            async with TurfinfoClient(endpoint_type=self.endpoint_type) as client:
                result = {
                    "partants": await client.get_partants_course(
                        program_date, reunion_num, course_num
                    )
                }

                if include_performances:
                    result["performances"] = await client.get_performances_detaillees(
                        program_date, reunion_num, course_num
                    )

                logger.info(
                    f"✅ Fetched partants from TurfInfo for R{reunion_num}C{course_num}"
                )
                return {
                    'source': DataSource.TURFINFO,
                    **result
                }

        except Exception as e:
            logger.error(
                f"❌ All sources failed for R{reunion_num}C{course_num}: {e}"
            )
            raise

    async def _fetch_partants_aspiturf(
        self,
        program_date: date,
        hippodrome_name: str,
        course_num: int
    ) -> Dict[str, Any]:
        """Récupère les partants depuis Aspiturf"""
        async with AspiturfClient(
            csv_path=self.aspiturf_csv_path,
            csv_url=self.aspiturf_csv_url,
            config=self.aspiturf_config
        ) as client:
            partants_data = await client.get_partants_course(
                program_date,
                hippodrome_name,
                course_num
            )

            # Convertir en format unifié
            partants_unified = self._convert_aspiturf_partants(partants_data)

            return {
                "partants": {"participants": partants_unified},
                # Aspiturf n'a pas de performances détaillées séparées
                # Toutes les données sont déjà dans les partants
                "performances": None
            }

    async def fetch_course_results(
        self,
        program_date: date,
        hippodrome_name: str,
        course_num: int
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère les résultats officiels d'une course via Open-PMU

        Args:
            program_date: Date de la course
            hippodrome_name: Nom de l'hippodrome
            course_num: Numéro de course

        Returns:
            Résultats officiels avec arrivée et rapports PMU, ou None
        """
        try:
            async with OpenPMUClient() as client:
                arrivees = await client.get_arrivees_by_hippodrome(
                    program_date, hippodrome_name
                )

                for arrivee in arrivees:
                    if arrivee.get("numero_course") == course_num:
                        logger.info(
                            f"Successfully fetched results for {hippodrome_name} "
                            f"R{course_num} on {program_date}"
                        )
                        return arrivee

                logger.warning(
                    f"No results found for {hippodrome_name} R{course_num} "
                    f"on {program_date}"
                )
                return None

        except Exception as e:
            logger.error(
                f"Error fetching results for {hippodrome_name} R{course_num}: {e}"
            )
            raise

    async def fetch_daily_results(self, program_date: date) -> List[Dict[str, Any]]:
        """
        Récupère tous les résultats officiels d'une journée via Open-PMU

        Args:
            program_date: Date des résultats

        Returns:
            Liste de tous les résultats du jour
        """
        try:
            async with OpenPMUClient() as client:
                results = await client.get_arrivees_by_date(program_date)
                logger.info(
                    f"Successfully fetched {len(results)} results for {program_date}"
                )
                return results

        except Exception as e:
            logger.error(f"Error fetching daily results for {program_date}: {e}")
            raise

    async def get_or_create_hippodrome(self, hippodrome_data: Dict[str, Any]) -> Hippodrome:
        """
        Récupère ou crée un hippodrome depuis les données PMU

        Args:
            hippodrome_data: Données de l'hippodrome

        Returns:
            Instance de Hippodrome
        """
        code = hippodrome_data.get("code")

        # Chercher l'hippodrome existant
        stmt = select(Hippodrome).where(Hippodrome.code == code)
        result = await self.db.execute(stmt)
        hippodrome = result.scalar_one_or_none()

        if hippodrome:
            return hippodrome

        # Créer un nouvel hippodrome
        hippodrome = Hippodrome(
            code=code,
            name=hippodrome_data.get("nom", ""),
            city=hippodrome_data.get("ville"),
            country=hippodrome_data.get("pays", "France"),
            track_type=self._map_track_type(hippodrome_data.get("type")),
            latitude=hippodrome_data.get("latitude"),
            longitude=hippodrome_data.get("longitude"),
        )

        self.db.add(hippodrome)
        await self.db.flush()

        logger.info(f"Created new hippodrome: {hippodrome.name} ({hippodrome.code})")
        return hippodrome

    async def get_or_create_horse(self, horse_data: Dict[str, Any]) -> Horse:
        """
        Récupère ou crée un cheval depuis les données PMU

        Args:
            horse_data: Données du cheval

        Returns:
            Instance de Horse
        """
        official_id = horse_data.get("id")

        if official_id:
            stmt = select(Horse).where(Horse.official_id == official_id)
            result = await self.db.execute(stmt)
            horse = result.scalar_one_or_none()

            if horse:
                return horse

        # Créer un nouveau cheval
        horse = Horse(
            official_id=official_id,
            name=horse_data.get("nom", ""),
            birth_year=horse_data.get("annee_naissance"),
            gender=self._map_gender(horse_data.get("sexe")),
            coat_color=horse_data.get("robe"),
            breed=horse_data.get("race"),
            sire=horse_data.get("pere"),
            dam=horse_data.get("mere"),
            owner=horse_data.get("proprietaire"),
        )

        self.db.add(horse)
        await self.db.flush()

        logger.info(f"Created new horse: {horse.name}")
        return horse

    async def get_or_create_jockey(self, jockey_data: Dict[str, Any]) -> Optional[Jockey]:
        """
        Récupère ou crée un jockey depuis les données PMU

        Args:
            jockey_data: Données du jockey

        Returns:
            Instance de Jockey ou None
        """
        if not jockey_data:
            return None

        official_id = jockey_data.get("id")

        if official_id:
            stmt = select(Jockey).where(Jockey.official_id == official_id)
            result = await self.db.execute(stmt)
            jockey = result.scalar_one_or_none()

            if jockey:
                return jockey

        # Créer un nouveau jockey
        jockey = Jockey(
            official_id=official_id,
            first_name=jockey_data.get("prenom", ""),
            last_name=jockey_data.get("nom", ""),
            nationality=jockey_data.get("nationalite"),
            weight=jockey_data.get("poids"),
        )

        self.db.add(jockey)
        await self.db.flush()

        logger.info(f"Created new jockey: {jockey.full_name}")
        return jockey

    async def get_or_create_trainer(self, trainer_data: Dict[str, Any]) -> Trainer:
        """
        Récupère ou crée un entraîneur depuis les données PMU

        Args:
            trainer_data: Données de l'entraîneur

        Returns:
            Instance de Trainer
        """
        official_id = trainer_data.get("id")

        if official_id:
            stmt = select(Trainer).where(Trainer.official_id == official_id)
            result = await self.db.execute(stmt)
            trainer = result.scalar_one_or_none()

            if trainer:
                return trainer

        # Créer un nouvel entraîneur
        trainer = Trainer(
            official_id=official_id,
            first_name=trainer_data.get("prenom", ""),
            last_name=trainer_data.get("nom", ""),
            stable_name=trainer_data.get("ecurie"),
            nationality=trainer_data.get("nationalite"),
        )

        self.db.add(trainer)
        await self.db.flush()

        logger.info(f"Created new trainer: {trainer.full_name}")
        return trainer

    async def sync_program_for_date(
        self,
        program_date: date,
        force_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronise le programme complet pour une date donnée

        Utilise la source principale avec fallback automatique.

        Args:
            program_date: Date du programme à synchroniser
            force_source: Forcer une source spécifique (optionnel)

        Returns:
            Statistiques de synchronisation avec source utilisée
        """
        stats = {
            "reunions_created": 0,
            "reunions_updated": 0,
            "courses_created": 0,
            "courses_updated": 0,
            "partants_created": 0,
            "source_used": None
        }

        try:
            # Récupérer le programme (avec fallback automatique)
            program_response = await self.fetch_program_for_date(
                program_date,
                force_source=force_source
            )

            stats["source_used"] = program_response.get('source')
            program_data = program_response.get('data')

            logger.info(
                f"📊 Syncing program for {program_date} from {stats['source_used']}"
            )

            # Traiter chaque réunion
            reunions = program_data.get("reunions", [])

            for reunion_data in reunions:
                hippodrome = await self.get_or_create_hippodrome(
                    reunion_data.get("hippodrome", {})
                )

                # Créer ou mettre à jour la réunion
                reunion = await self._sync_reunion(
                    reunion_data, hippodrome, program_date, stats
                )

                # Traiter les courses de la réunion
                courses = reunion_data.get("courses", [])
                for course_data in courses:
                    course = await self._sync_course(course_data, reunion, stats)

                    # Traiter les partants de la course
                    partants = course_data.get("partants", [])
                    for partant_data in partants:
                        await self._sync_partant(partant_data, course, stats)

            await self.db.commit()
            logger.info(f"Successfully synced program for {program_date}: {stats}")
            return stats

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error syncing program for {program_date}: {e}")
            raise

    async def _sync_reunion(
        self,
        reunion_data: Dict[str, Any],
        hippodrome: Hippodrome,
        program_date: date,
        stats: Dict[str, int]
    ) -> Reunion:
        """Synchronise une réunion"""
        reunion_number = reunion_data.get("numero", 1)

        # Chercher réunion existante
        stmt = select(Reunion).where(
            Reunion.hippodrome_id == hippodrome.hippodrome_id,
            Reunion.reunion_date == program_date,
            Reunion.reunion_number == reunion_number
        )
        result = await self.db.execute(stmt)
        reunion = result.scalar_one_or_none()

        if reunion:
            # Mettre à jour
            reunion.status = self._map_reunion_status(reunion_data.get("statut"))
            reunion.weather_conditions = reunion_data.get("meteo")
            stats["reunions_updated"] += 1
        else:
            # Créer
            reunion = Reunion(
                hippodrome_id=hippodrome.hippodrome_id,
                reunion_date=program_date,
                reunion_number=reunion_number,
                status=self._map_reunion_status(reunion_data.get("statut")),
                api_source="PMU",
                weather_conditions=reunion_data.get("meteo"),
            )
            self.db.add(reunion)
            stats["reunions_created"] += 1

        await self.db.flush()
        return reunion

    async def _sync_course(
        self,
        course_data: Dict[str, Any],
        reunion: Reunion,
        stats: Dict[str, int]
    ) -> Course:
        """Synchronise une course"""
        course_number = course_data.get("numero", 1)

        # Chercher course existante
        stmt = select(Course).where(
            Course.reunion_id == reunion.reunion_id,
            Course.course_number == course_number
        )
        result = await self.db.execute(stmt)
        course = result.scalar_one_or_none()

        scheduled_time_str = course_data.get("heure_depart", "14:00")
        scheduled_time = datetime.strptime(scheduled_time_str, "%H:%M").time()

        if course:
            # Mettre à jour
            course.course_name = course_data.get("nom")
            course.discipline = self._map_discipline(course_data.get("discipline"))
            course.distance = course_data.get("distance", 2000)
            course.prize_money = course_data.get("allocation")
            course.surface_type = self._map_surface(course_data.get("surface"))
            course.scheduled_time = scheduled_time
            course.number_of_runners = course_data.get("nombre_partants")
            course.status = self._map_course_status(course_data.get("statut"))
            stats["courses_updated"] += 1
        else:
            # Créer
            course = Course(
                reunion_id=reunion.reunion_id,
                course_number=course_number,
                course_name=course_data.get("nom"),
                discipline=self._map_discipline(course_data.get("discipline")),
                distance=course_data.get("distance", 2000),
                prize_money=course_data.get("allocation"),
                race_category=course_data.get("categorie"),
                race_class=course_data.get("classe"),
                surface_type=self._map_surface(course_data.get("surface")),
                start_type=self._map_start_type(course_data.get("type_depart")),
                scheduled_time=scheduled_time,
                number_of_runners=course_data.get("nombre_partants"),
                status=self._map_course_status(course_data.get("statut")),
            )
            self.db.add(course)
            stats["courses_created"] += 1

        await self.db.flush()
        return course

    async def _sync_partant(
        self,
        partant_data: Dict[str, Any],
        course: Course,
        stats: Dict[str, int]
    ) -> Partant:
        """Synchronise un partant"""
        # Récupérer ou créer les entités liées
        horse = await self.get_or_create_horse(partant_data.get("cheval", {}))
        jockey = await self.get_or_create_jockey(partant_data.get("jockey"))
        trainer = await self.get_or_create_trainer(partant_data.get("entraineur", {}))

        numero_corde = partant_data.get("numero", 1)

        # Chercher partant existant
        stmt = select(Partant).where(
            Partant.course_id == course.course_id,
            Partant.numero_corde == numero_corde
        )
        result = await self.db.execute(stmt)
        partant = result.scalar_one_or_none()

        if partant:
            # Mettre à jour
            partant.horse_id = horse.horse_id
            partant.jockey_id = jockey.jockey_id if jockey else None
            partant.trainer_id = trainer.trainer_id
            partant.poids_porte = partant_data.get("poids")
            partant.handicap_value = partant_data.get("handicap")
            partant.equipment = partant_data.get("equipement")
            partant.odds_pmu = partant_data.get("cote")
            partant.recent_form = partant_data.get("musique")
            partant.days_since_last_race = partant_data.get("jours_depuis_derniere_course")
            partant.aspiturf_stats = partant_data.get("aspiturf_data")
        else:
            # Créer
            partant = Partant(
                course_id=course.course_id,
                horse_id=horse.horse_id,
                jockey_id=jockey.jockey_id if jockey else None,
                trainer_id=trainer.trainer_id,
                numero_corde=numero_corde,
                poids_porte=partant_data.get("poids"),
                handicap_value=partant_data.get("handicap"),
                equipment=partant_data.get("equipement"),
                days_since_last_race=partant_data.get("jours_depuis_derniere_course"),
                recent_form=partant_data.get("musique"),
                odds_pmu=partant_data.get("cote"),
                aspiturf_stats=partant_data.get("aspiturf_data"),
            )
            self.db.add(partant)
            stats["partants_created"] += 1

        await self.db.flush()
        return partant

    # Méthodes de mapping

    def _map_track_type(self, track_type: Optional[str]) -> TrackType:
        """Mappe le type de piste"""
        mapping = {
            "plat": TrackType.PLAT,
            "trot": TrackType.TROT,
            "obstacles": TrackType.OBSTACLES,
            "mixte": TrackType.MIXTE,
        }
        return mapping.get(track_type.lower() if track_type else "", TrackType.MIXTE)

    def _map_gender(self, gender: Optional[str]) -> Gender:
        """Mappe le genre du cheval"""
        if not gender:
            return Gender.MALE

        gender_lower = gender.lower()
        if gender_lower in ["m", "male", "mâle"]:
            return Gender.MALE
        elif gender_lower in ["f", "female", "femelle", "jument"]:
            return Gender.FEMALE
        elif gender_lower in ["h", "hongre", "castré"]:
            return Gender.HONGRE
        return Gender.MALE

    def _map_discipline(self, discipline: Optional[str]) -> Discipline:
        """Mappe la discipline"""
        if not discipline:
            return Discipline.PLAT

        mapping = {
            "plat": Discipline.PLAT,
            "trot_monte": Discipline.TROT_MONTE,
            "trot_attele": Discipline.TROT_ATTELE,
            "haies": Discipline.HAIES,
            "steeple": Discipline.STEEPLE,
            "cross": Discipline.CROSS,
        }
        return mapping.get(discipline.lower(), Discipline.PLAT)

    def _map_surface(self, surface: Optional[str]) -> SurfaceType:
        """Mappe le type de surface"""
        if not surface:
            return SurfaceType.PELOUSE

        mapping = {
            "pelouse": SurfaceType.PELOUSE,
            "piste": SurfaceType.PISTE,
            "sable": SurfaceType.SABLE,
            "fibre": SurfaceType.FIBRE,
        }
        return mapping.get(surface.lower(), SurfaceType.PELOUSE)

    def _map_start_type(self, start_type: Optional[str]) -> StartType:
        """Mappe le type de départ"""
        if not start_type:
            return StartType.STALLE

        mapping = {
            "autostart": StartType.AUTOSTART,
            "volte": StartType.VOLTE,
            "elastique": StartType.ELASTIQUE,
            "stalle": StartType.STALLE,
            "corde": StartType.CORDE,
        }
        return mapping.get(start_type.lower(), StartType.STALLE)

    def _map_reunion_status(self, status: Optional[str]) -> ReunionStatus:
        """Mappe le statut de réunion"""
        if not status:
            return ReunionStatus.SCHEDULED

        mapping = {
            "scheduled": ReunionStatus.SCHEDULED,
            "ongoing": ReunionStatus.ONGOING,
            "completed": ReunionStatus.COMPLETED,
            "cancelled": ReunionStatus.CANCELLED,
        }
        return mapping.get(status.lower(), ReunionStatus.SCHEDULED)

    def _map_course_status(self, status: Optional[str]) -> CourseStatus:
        """Mappe le statut de course"""
        if not status:
            return CourseStatus.SCHEDULED

        mapping = {
            "scheduled": CourseStatus.SCHEDULED,
            "running": CourseStatus.RUNNING,
            "finished": CourseStatus.FINISHED,
            "cancelled": CourseStatus.CANCELLED,
        }
        return mapping.get(status.lower(), CourseStatus.SCHEDULED)