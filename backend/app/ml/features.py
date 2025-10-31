"""
Feature Engineering pour le modèle ML de prédiction hippique

Ce module extrait et calcule toutes les features nécessaires pour le modèle
de prédiction basé sur les données disponibles (chevaux, jockeys, entraîneurs, courses).
"""

import logging
import re
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
from sqlalchemy import func, and_
from sqlalchemy.orm import Session

from app.models.partant import Partant
from app.models.horse import Horse, Gender
from app.models.jockey import Jockey
from app.models.trainer import Trainer
from app.models.course import Course, Discipline, SurfaceType
from app.models.reunion import Reunion
from app.models.hippodrome import Hippodrome

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Classe responsable de l'extraction et du calcul des features ML
    pour les prédictions hippiques
    """

    def __init__(self, db: Session):
        """
        Initialise le feature engineer

        Args:
            db: Session de base de données SQLAlchemy
        """
        self.db = db

    def extract_features_for_partant(
        self,
        partant: Partant,
        include_odds: bool = False
    ) -> Dict[str, Any]:
        """
        Extrait toutes les features pour un partant donné

        Args:
            partant: Instance du partant
            include_odds: Inclure les cotes PMU (False pour l'entraînement si pas disponibles)

        Returns:
            Dictionnaire contenant toutes les features
        """
        features = {}

        # Features du cheval
        features.update(self._extract_horse_features(partant))

        # Features du jockey
        features.update(self._extract_jockey_features(partant))

        # Features de l'entraîneur
        features.update(self._extract_trainer_features(partant))

        # Features de la course
        features.update(self._extract_course_features(partant))

        # Features du partant lui-même
        features.update(self._extract_partant_features(partant, include_odds))

        # Features de l'hippodrome
        features.update(self._extract_hippodrome_features(partant))

        # Features enrichies Aspiturf (si disponibles)
        features.update(self._extract_aspiturf_features(partant))

        return features

    def _extract_horse_features(self, partant: Partant) -> Dict[str, Any]:
        """
        Extrait les features liées au cheval

        Args:
            partant: Instance du partant

        Returns:
            Dictionnaire des features du cheval
        """
        horse = partant.horse
        if not horse:
            return self._get_default_horse_features()

        features = {
            "horse_age": horse.age or 0,
            "horse_gender_male": 1 if horse.gender == Gender.MALE else 0,
            "horse_gender_female": 1 if horse.gender == Gender.FEMALE else 0,
            "horse_gender_hongre": 1 if horse.gender == Gender.HONGRE else 0,
        }

        # Statistiques de forme récente
        form_stats = self._calculate_recent_form(horse, partant.course)
        features.update(form_stats)

        # Statistiques sur distance similaire
        distance_stats = self._calculate_distance_stats(horse, partant.course)
        features.update(distance_stats)

        # Statistiques sur surface similaire
        surface_stats = self._calculate_surface_stats(horse, partant.course)
        features.update(surface_stats)

        return features

    def _extract_jockey_features(self, partant: Partant) -> Dict[str, Any]:
        """
        Extrait les features liées au jockey

        Args:
            partant: Instance du partant

        Returns:
            Dictionnaire des features du jockey
        """
        jockey = partant.jockey
        if not jockey:
            return self._get_default_jockey_features()

        # Calculer les stats récentes (30 derniers jours)
        cutoff_date = datetime.now().date() - timedelta(days=30)

        # Stats récentes du jockey
        recent_runs = (
            self.db.query(Partant)
            .join(Course)
            .join(Reunion)
            .filter(
                Partant.jockey_id == jockey.jockey_id,
                Reunion.reunion_date >= cutoff_date,
                Partant.final_position.isnot(None)
            )
            .all()
        )

        if recent_runs:
            wins = sum(1 for r in recent_runs if r.final_position == 1)
            places = sum(1 for r in recent_runs if r.final_position <= 3)
            total = len(recent_runs)

            win_rate = wins / total if total > 0 else 0
            place_rate = places / total if total > 0 else 0
        else:
            win_rate = 0
            place_rate = 0
            total = 0

        # Affinité jockey-cheval
        affinity = self._calculate_jockey_horse_affinity(jockey, partant.horse)

        return {
            "jockey_win_rate": win_rate,
            "jockey_place_rate": place_rate,
            "jockey_recent_runs": min(total, 50),  # Cap à 50 pour normalisation
            "jockey_horse_affinity": affinity,
        }

    def _extract_trainer_features(self, partant: Partant) -> Dict[str, Any]:
        """
        Extrait les features liées à l'entraîneur

        Args:
            partant: Instance du partant

        Returns:
            Dictionnaire des features de l'entraîneur
        """
        trainer = partant.trainer
        if not trainer:
            return self._get_default_trainer_features()

        # Calculer les stats récentes (30 derniers jours)
        cutoff_date = datetime.now().date() - timedelta(days=30)

        recent_runs = (
            self.db.query(Partant)
            .join(Course)
            .join(Reunion)
            .filter(
                Partant.trainer_id == trainer.trainer_id,
                Reunion.reunion_date >= cutoff_date,
                Partant.final_position.isnot(None)
            )
            .all()
        )

        if recent_runs:
            wins = sum(1 for r in recent_runs if r.final_position == 1)
            places = sum(1 for r in recent_runs if r.final_position <= 3)
            total = len(recent_runs)

            win_rate = wins / total if total > 0 else 0
            place_rate = places / total if total > 0 else 0
        else:
            win_rate = 0
            place_rate = 0
            total = 0

        return {
            "trainer_win_rate": win_rate,
            "trainer_place_rate": place_rate,
            "trainer_recent_runs": min(total, 100),  # Cap à 100
        }

    def _extract_course_features(self, partant: Partant) -> Dict[str, Any]:
        """
        Extrait les features liées à la course

        Args:
            partant: Instance du partant

        Returns:
            Dictionnaire des features de la course
        """
        course = partant.course

        features = {
            "course_distance": course.distance,
            "course_number_of_runners": course.number_of_runners or 0,
            "course_prize_money": float(course.prize_money or 0),

            # Encodage discipline (one-hot)
            "discipline_plat": 1 if course.discipline == Discipline.PLAT else 0,
            "discipline_trot_monte": 1 if course.discipline == Discipline.TROT_MONTE else 0,
            "discipline_trot_attele": 1 if course.discipline == Discipline.TROT_ATTELE else 0,
            "discipline_haies": 1 if course.discipline == Discipline.HAIES else 0,
            "discipline_steeple": 1 if course.discipline == Discipline.STEEPLE else 0,
            "discipline_cross": 1 if course.discipline == Discipline.CROSS else 0,

            # Encodage surface (one-hot)
            "surface_pelouse": 1 if course.surface_type == SurfaceType.PELOUSE else 0,
            "surface_piste": 1 if course.surface_type == SurfaceType.PISTE else 0,
            "surface_sable": 1 if course.surface_type == SurfaceType.SABLE else 0,
            "surface_fibre": 1 if course.surface_type == SurfaceType.FIBRE else 0,
        }

        return features

    def _extract_partant_features(
        self,
        partant: Partant,
        include_odds: bool = False
    ) -> Dict[str, Any]:
        """
        Extrait les features liées au partant lui-même

        Args:
            partant: Instance du partant
            include_odds: Inclure les cotes PMU

        Returns:
            Dictionnaire des features du partant
        """
        features = {
            "numero_corde": partant.numero_corde,
            "poids_porte": float(partant.poids_porte or 0),
            "handicap_value": partant.handicap_value or 0,
            "days_since_last_race": partant.days_since_last_race or 0,
            "has_oeilleres": 1 if partant.has_oeilleres else 0,
        }

        # Forme récente du partant
        if partant.recent_form_list:
            form_list = partant.recent_form_list
            features["recent_avg_position"] = np.mean(form_list)
            features["recent_best_position"] = min(form_list)
            features["recent_form_length"] = len(form_list)
            features["has_won_recently"] = 1 if partant.has_won_recently else 0
        else:
            features["recent_avg_position"] = 0
            features["recent_best_position"] = 0
            features["recent_form_length"] = 0
            features["has_won_recently"] = 0

        # Cotes PMU (optionnel)
        if include_odds and partant.odds_pmu:
            features["odds_pmu"] = float(partant.odds_pmu)
        else:
            features["odds_pmu"] = 0

        return features

    def _extract_hippodrome_features(self, partant: Partant) -> Dict[str, Any]:
        """
        Extrait les features liées à l'hippodrome

        Args:
            partant: Instance du partant

        Returns:
            Dictionnaire des features de l'hippodrome
        """
        hippodrome = partant.course.reunion.hippodrome

        # Calculer l'affinité du cheval avec l'hippodrome
        affinity = self._calculate_hippodrome_affinity(partant.horse, hippodrome)

        return {
            "hippodrome_affinity_win_rate": affinity.get("win_rate", 0),
            "hippodrome_affinity_runs": affinity.get("total_runs", 0),
        }

    def _extract_aspiturf_features(self, partant: Partant) -> Dict[str, Any]:
        """Extrait les features enrichies issues des données Aspiturf."""

        stats = getattr(partant, "aspiturf_data", None)
        features = self._get_default_aspiturf_features()

        if not stats:
            return features

        # Statistiques cheval
        total_runs = self._parse_numeric(stats.get("courses_total"))
        total_wins = self._parse_numeric(stats.get("victoires_total"))
        total_places = self._parse_numeric(stats.get("places_total"))

        features.update({
            "aspiturf_horse_gains_total": self._parse_numeric(stats.get("gains_carriere")),
            "aspiturf_horse_gains_year": self._parse_numeric(stats.get("gains_annee")),
            "aspiturf_horse_total_runs": total_runs,
            "aspiturf_horse_total_wins": total_wins,
            "aspiturf_horse_total_places": total_places,
            "aspiturf_horse_win_rate": (total_wins / total_runs) if total_runs else 0,
            "aspiturf_horse_place_rate": (total_places / total_runs) if total_runs else 0,
            "aspiturf_horse_win_rate_hippodrome": self._parse_percentage(stats.get("pourc_vict_hippo")),
            "aspiturf_horse_place_rate_hippodrome": self._parse_percentage(stats.get("pourc_place_hippo")),
            "aspiturf_horse_record_seconds": self._parse_time_to_seconds(stats.get("record_general")),
            "aspiturf_horse_is_inedit": self._to_bool_flag(stats.get("indicateur_inedit")),
        })

        # Statistiques jockey
        jockey_runs = self._parse_numeric(stats.get("jockey_courses"))
        jockey_wins = self._parse_numeric(stats.get("jockey_victoires"))
        features.update({
            "aspiturf_jockey_total_runs": jockey_runs,
            "aspiturf_jockey_total_wins": jockey_wins,
            "aspiturf_jockey_win_rate": self._parse_percentage(stats.get("jockey_pourc_vict")) or ((jockey_wins / jockey_runs) if jockey_runs else 0),
        })

        # Statistiques entraîneur
        trainer_runs = self._parse_numeric(stats.get("entraineur_courses"))
        trainer_wins = self._parse_numeric(stats.get("entraineur_victoires"))
        features.update({
            "aspiturf_trainer_total_runs": trainer_runs,
            "aspiturf_trainer_total_wins": trainer_wins,
            "aspiturf_trainer_win_rate_hippodrome": self._parse_percentage(stats.get("entraineur_pourc_vict_hippo")),
        })

        # Statistiques couple cheval/jockey
        couple_runs = self._parse_numeric(stats.get("couple_courses"))
        couple_wins = self._parse_numeric(stats.get("couple_victoires"))
        features.update({
            "aspiturf_couple_total_runs": couple_runs,
            "aspiturf_couple_total_wins": couple_wins,
            "aspiturf_couple_win_rate": self._parse_percentage(stats.get("couple_tx_vict")) or ((couple_wins / couple_runs) if couple_runs else 0),
        })

        # Préférences de terrain (mots clés)
        terrain_pref = stats.get("appet_terrain")
        if terrain_pref:
            terrain_text = str(terrain_pref).lower()
            if "bon" in terrain_text:
                features["aspiturf_terrain_prefers_bon"] = 1
            if any(keyword in terrain_text for keyword in ["lourd", "collant", "souple"]):
                features["aspiturf_terrain_prefers_lourd"] = 1
            if any(keyword in terrain_text for keyword in ["psf", "sable", "fibre"]):
                features["aspiturf_terrain_prefers_psf"] = 1

        return features

    def _calculate_recent_form(
        self,
        horse: Horse,
        current_course: Course
    ) -> Dict[str, float]:
        """
        Calcule les statistiques de forme récente du cheval

        Args:
            horse: Instance du cheval
            current_course: Course actuelle (pour éviter de l'inclure)

        Returns:
            Dict avec stats de forme récente
        """
        # Récupérer les 5 dernières courses (excluant la course actuelle)
        recent_partants = (
            self.db.query(Partant)
            .join(Course)
            .join(Reunion)
            .filter(
                Partant.horse_id == horse.horse_id,
                Partant.final_position.isnot(None),
                Course.course_id != current_course.course_id,
                Reunion.reunion_date < current_course.reunion.reunion_date
            )
            .order_by(Reunion.reunion_date.desc())
            .limit(5)
            .all()
        )

        if not recent_partants:
            return {
                "horse_recent_avg_position": 0,
                "horse_recent_win_rate": 0,
                "horse_recent_place_rate": 0,
                "horse_recent_runs": 0,
            }

        positions = [p.final_position for p in recent_partants if p.final_position]
        wins = sum(1 for p in positions if p == 1)
        places = sum(1 for p in positions if p <= 3)
        total = len(positions)

        return {
            "horse_recent_avg_position": np.mean(positions) if positions else 0,
            "horse_recent_win_rate": wins / total if total > 0 else 0,
            "horse_recent_place_rate": places / total if total > 0 else 0,
            "horse_recent_runs": total,
        }

    def _calculate_distance_stats(
        self,
        horse: Horse,
        current_course: Course
    ) -> Dict[str, float]:
        """
        Calcule les stats du cheval sur distances similaires (+/- 200m)

        Args:
            horse: Instance du cheval
            current_course: Course actuelle

        Returns:
            Dict avec stats sur distance similaire
        """
        min_distance = current_course.distance - 200
        max_distance = current_course.distance + 200

        similar_runs = (
            self.db.query(Partant)
            .join(Course)
            .join(Reunion)
            .filter(
                Partant.horse_id == horse.horse_id,
                Partant.final_position.isnot(None),
                Course.distance >= min_distance,
                Course.distance <= max_distance,
                Course.course_id != current_course.course_id,
                Reunion.reunion_date < current_course.reunion.reunion_date
            )
            .all()
        )

        if not similar_runs:
            return {
                "horse_distance_win_rate": 0,
                "horse_distance_runs": 0,
            }

        wins = sum(1 for r in similar_runs if r.final_position == 1)
        total = len(similar_runs)

        return {
            "horse_distance_win_rate": wins / total if total > 0 else 0,
            "horse_distance_runs": min(total, 20),  # Cap à 20
        }

    def _calculate_surface_stats(
        self,
        horse: Horse,
        current_course: Course
    ) -> Dict[str, float]:
        """
        Calcule les stats du cheval sur le même type de surface

        Args:
            horse: Instance du cheval
            current_course: Course actuelle

        Returns:
            Dict avec stats sur surface similaire
        """
        surface_runs = (
            self.db.query(Partant)
            .join(Course)
            .join(Reunion)
            .filter(
                Partant.horse_id == horse.horse_id,
                Partant.final_position.isnot(None),
                Course.surface_type == current_course.surface_type,
                Course.course_id != current_course.course_id,
                Reunion.reunion_date < current_course.reunion.reunion_date
            )
            .all()
        )

        if not surface_runs:
            return {
                "horse_surface_win_rate": 0,
                "horse_surface_runs": 0,
            }

        wins = sum(1 for r in surface_runs if r.final_position == 1)
        total = len(surface_runs)

        return {
            "horse_surface_win_rate": wins / total if total > 0 else 0,
            "horse_surface_runs": min(total, 20),  # Cap à 20
        }

    def _calculate_jockey_horse_affinity(
        self,
        jockey: Jockey,
        horse: Horse
    ) -> float:
        """
        Calcule l'affinité entre un jockey et un cheval

        Args:
            jockey: Instance du jockey
            horse: Instance du cheval

        Returns:
            Win rate du jockey avec ce cheval
        """
        if not jockey or not horse:
            return 0

        combo_runs = (
            self.db.query(Partant)
            .filter(
                Partant.jockey_id == jockey.jockey_id,
                Partant.horse_id == horse.horse_id,
                Partant.final_position.isnot(None)
            )
            .all()
        )

        if not combo_runs:
            return 0

        wins = sum(1 for r in combo_runs if r.final_position == 1)
        total = len(combo_runs)

        return wins / total if total > 0 else 0

    def _calculate_hippodrome_affinity(
        self,
        horse: Horse,
        hippodrome: Hippodrome
    ) -> Dict[str, float]:
        """
        Calcule l'affinité du cheval avec un hippodrome

        Args:
            horse: Instance du cheval
            hippodrome: Instance de l'hippodrome

        Returns:
            Dict avec stats d'affinité hippodrome
        """
        if not horse or not hippodrome:
            return {"win_rate": 0, "total_runs": 0}

        hippo_runs = (
            self.db.query(Partant)
            .join(Course)
            .join(Reunion)
            .filter(
                Partant.horse_id == horse.horse_id,
                Partant.final_position.isnot(None),
                Reunion.hippodrome_id == hippodrome.hippodrome_id
            )
            .all()
        )

        if not hippo_runs:
            return {"win_rate": 0, "total_runs": 0}

        wins = sum(1 for r in hippo_runs if r.final_position == 1)
        total = len(hippo_runs)

        return {
            "win_rate": wins / total if total > 0 else 0,
            "total_runs": min(total, 10),  # Cap à 10
        }

    def _get_default_horse_features(self) -> Dict[str, Any]:
        """Retourne les features par défaut pour un cheval manquant"""
        return {
            "horse_age": 0,
            "horse_gender_male": 0,
            "horse_gender_female": 0,
            "horse_gender_hongre": 0,
            "horse_recent_avg_position": 0,
            "horse_recent_win_rate": 0,
            "horse_recent_place_rate": 0,
            "horse_recent_runs": 0,
            "horse_distance_win_rate": 0,
            "horse_distance_runs": 0,
            "horse_surface_win_rate": 0,
            "horse_surface_runs": 0,
        }

    def _get_default_jockey_features(self) -> Dict[str, Any]:
        """Retourne les features par défaut pour un jockey manquant"""
        return {
            "jockey_win_rate": 0,
            "jockey_place_rate": 0,
            "jockey_recent_runs": 0,
            "jockey_horse_affinity": 0,
        }

    def _get_default_trainer_features(self) -> Dict[str, Any]:
        """Retourne les features par défaut pour un entraîneur manquant"""
        return {
            "trainer_win_rate": 0,
            "trainer_place_rate": 0,
            "trainer_recent_runs": 0,
        }

    def _get_default_aspiturf_features(self) -> Dict[str, Any]:
        """Retourne les features Aspiturf par défaut."""
        return {
            "aspiturf_horse_gains_total": 0.0,
            "aspiturf_horse_gains_year": 0.0,
            "aspiturf_horse_total_runs": 0.0,
            "aspiturf_horse_total_wins": 0.0,
            "aspiturf_horse_total_places": 0.0,
            "aspiturf_horse_win_rate": 0.0,
            "aspiturf_horse_place_rate": 0.0,
            "aspiturf_horse_win_rate_hippodrome": 0.0,
            "aspiturf_horse_place_rate_hippodrome": 0.0,
            "aspiturf_horse_record_seconds": 0.0,
            "aspiturf_horse_is_inedit": 0,
            "aspiturf_jockey_total_runs": 0.0,
            "aspiturf_jockey_total_wins": 0.0,
            "aspiturf_jockey_win_rate": 0.0,
            "aspiturf_trainer_total_runs": 0.0,
            "aspiturf_trainer_total_wins": 0.0,
            "aspiturf_trainer_win_rate_hippodrome": 0.0,
            "aspiturf_couple_total_runs": 0.0,
            "aspiturf_couple_total_wins": 0.0,
            "aspiturf_couple_win_rate": 0.0,
            "aspiturf_terrain_prefers_bon": 0,
            "aspiturf_terrain_prefers_lourd": 0,
            "aspiturf_terrain_prefers_psf": 0,
        }

    def get_feature_names(self) -> List[str]:
        """
        Retourne la liste ordonnée de tous les noms de features

        Returns:
            Liste des noms de features
        """
        return [
            # Horse features
            "horse_age",
            "horse_gender_male",
            "horse_gender_female",
            "horse_gender_hongre",
            "horse_recent_avg_position",
            "horse_recent_win_rate",
            "horse_recent_place_rate",
            "horse_recent_runs",
            "horse_distance_win_rate",
            "horse_distance_runs",
            "horse_surface_win_rate",
            "horse_surface_runs",

            # Jockey features
            "jockey_win_rate",
            "jockey_place_rate",
            "jockey_recent_runs",
            "jockey_horse_affinity",

            # Trainer features
            "trainer_win_rate",
            "trainer_place_rate",
            "trainer_recent_runs",

            # Course features
            "course_distance",
            "course_number_of_runners",
            "course_prize_money",
            "discipline_plat",
            "discipline_trot_monte",
            "discipline_trot_attele",
            "discipline_haies",
            "discipline_steeple",
            "discipline_cross",
            "surface_pelouse",
            "surface_piste",
            "surface_sable",
            "surface_fibre",

            # Partant features
            "numero_corde",
            "poids_porte",
            "handicap_value",
            "days_since_last_race",
            "has_oeilleres",
            "recent_avg_position",
            "recent_best_position",
            "recent_form_length",
            "has_won_recently",
            "odds_pmu",

            # Hippodrome features
            "hippodrome_affinity_win_rate",
            "hippodrome_affinity_runs",

            # Aspiturf enriched features
            "aspiturf_horse_gains_total",
            "aspiturf_horse_gains_year",
            "aspiturf_horse_total_runs",
            "aspiturf_horse_total_wins",
            "aspiturf_horse_total_places",
            "aspiturf_horse_win_rate",
            "aspiturf_horse_place_rate",
            "aspiturf_horse_win_rate_hippodrome",
            "aspiturf_horse_place_rate_hippodrome",
            "aspiturf_horse_record_seconds",
            "aspiturf_horse_is_inedit",
            "aspiturf_jockey_total_runs",
            "aspiturf_jockey_total_wins",
            "aspiturf_jockey_win_rate",
            "aspiturf_trainer_total_runs",
            "aspiturf_trainer_total_wins",
            "aspiturf_trainer_win_rate_hippodrome",
            "aspiturf_couple_total_runs",
            "aspiturf_couple_total_wins",
            "aspiturf_couple_win_rate",
            "aspiturf_terrain_prefers_bon",
            "aspiturf_terrain_prefers_lourd",
            "aspiturf_terrain_prefers_psf",
        ]

    @staticmethod
    def _parse_numeric(value: Any) -> float:
        """Convertit une valeur Aspiturf en float robuste."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        if not text:
            return 0.0

        normalized = text.replace("\u202f", "").replace(" ", "").replace(",", ".")

        try:
            return float(normalized)
        except ValueError:
            match = re.search(r"-?\d+(?:\.\d+)?", normalized)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    return 0.0
            return 0.0

    @classmethod
    def _parse_percentage(cls, value: Any) -> float:
        """Normalise un pourcentage Aspiturf en valeur entre 0 et 1."""
        if value is None:
            return 0.0

        numeric = cls._parse_numeric(value)
        if isinstance(value, str) and "%" in value:
            return numeric / 100 if numeric > 1 else numeric

        if numeric > 1:
            return numeric / 100
        return numeric

    @staticmethod
    def _parse_time_to_seconds(value: Any) -> float:
        """Convertit un chrono Aspiturf en secondes."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        if not text:
            return 0.0

        cleaned = text.replace(" ", "")
        pattern = re.match(r"(?:(\d+)[\'’])?(\d{1,2})(?:[\"”](\d+))?", cleaned)
        if pattern:
            minutes = int(pattern.group(1)) if pattern.group(1) else 0
            seconds = int(pattern.group(2))
            hundredths = pattern.group(3)
            fraction = 0.0
            if hundredths:
                try:
                    fraction = float(f"0.{hundredths}")
                except ValueError:
                    fraction = 0.0
            return minutes * 60 + seconds + fraction

        fallback = cleaned.replace("'", ":").replace("\"", ".").replace("”", ".").replace("’", ":")
        try:
            return float(fallback.replace(",", "."))
        except ValueError:
            return 0.0

    @staticmethod
    def _to_bool_flag(value: Any) -> int:
        """Convertit une valeur Aspiturf en indicateur binaire."""
        if value is None:
            return 0
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return 1 if value != 0 else 0

        text = str(value).strip().lower()
        return 1 if text in {"1", "true", "vrai", "oui", "o", "y"} else 0
