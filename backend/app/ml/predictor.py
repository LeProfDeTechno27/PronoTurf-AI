# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Service de prédiction principal pour les courses hippiques

Ce module combine le feature engineering, le modèle ML et l'explicabilité SHAP
pour générer des pronostics complets avec explications.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, selectinload

from app.ml.features import FeatureEngineer
from app.ml.model import HorseRacingModel
from app.ml.explainer import SHAPExplainer
from app.models.course import Course, CourseStatus
from app.models.partant import Partant
from app.models.reunion import Reunion
from app.core.config import settings

logger = logging.getLogger(__name__)


class RacePredictionService:
    """
    Service principal pour générer des prédictions de courses hippiques
    avec explicabilité SHAP et détection de value bets
    """

    def __init__(
        self,
        db: Session,
        model_path: Optional[Path] = None
    ):
        """
        Initialise le service de prédiction

        Args:
            db: Session de base de données
            model_path: Chemin vers le modèle ML (optionnel, utilise le chemin par défaut)
        """
        self.db = db
        self.feature_engineer = FeatureEngineer(db)

        # Charger le modèle
        if model_path is None:
            model_path = Path(settings.ML_MODEL_PATH) if hasattr(settings, 'ML_MODEL_PATH') else Path("models/horse_racing_model.pkl")

        self.model = HorseRacingModel(model_path=model_path if model_path.exists() else None)

        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Please train a model first.")
            self.explainer = None
        else:
            # Initialiser l'explainer SHAP
            try:
                self.explainer = SHAPExplainer(self.model)
                logger.info("Prediction service initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing SHAP explainer: {e}")
                self.explainer = None

    def predict_course(
        self,
        course_id: int,
        include_explanations: bool = True,
        detect_value_bets: bool = True
    ) -> Dict[str, Any]:
        """
        Génère les prédictions pour une course complète

        Args:
            course_id: ID de la course
            include_explanations: Inclure les explications SHAP
            detect_value_bets: Détecter les value bets

        Returns:
            Dictionnaire contenant les prédictions et recommandations
        """
        # Récupérer la course avec ses partants
        course = (
            self.db.query(Course)
            .options(
                selectinload(Course.partants)
                .selectinload(Partant.horse),
                selectinload(Course.partants)
                .selectinload(Partant.jockey),
                selectinload(Course.partants)
                .selectinload(Partant.trainer),
                selectinload(Course.reunion)
                .selectinload(Reunion.hippodrome)
            )
            .filter(Course.course_id == course_id)
            .first()
        )

        if not course:
            raise ValueError(f"Course {course_id} not found")

        if not course.partants:
            raise ValueError(f"No partants found for course {course_id}")

        logger.info(f"Predicting for course {course.full_name} with {len(course.partants)} partants")

        # Préparer les prédictions
        predictions = []

        for partant in course.partants:
            try:
                prediction = self._predict_partant(
                    partant,
                    include_explanations=include_explanations
                )
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting for partant {partant.partant_id}: {e}")
                continue

        if not predictions:
            raise ValueError("No predictions could be generated")

        # Trier par probabilité décroissante
        predictions.sort(key=lambda x: x['probability'], reverse=True)

        # Générer les recommandations
        recommendations = self._generate_recommendations(predictions, course)

        # Détecter les value bets si demandé
        value_bets = []
        if detect_value_bets:
            value_bets = self._detect_value_bets(predictions)

        result = {
            "course_id": course_id,
            "course_name": course.full_name,
            "course_distance": course.distance,
            "course_discipline": course.discipline.value,
            "number_of_runners": len(predictions),
            "predictions": predictions,
            "recommendations": recommendations,
            "value_bets": value_bets,
            "generated_at": datetime.now().isoformat(),
            "model_version": self.model.model_version if self.model else "unknown",
        }

        return result

    def predict_multiple_courses(
        self,
        course_ids: List[int],
        include_explanations: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Génère les prédictions pour plusieurs courses

        Args:
            course_ids: Liste des IDs de courses
            include_explanations: Inclure les explications SHAP

        Returns:
            Liste des prédictions pour chaque course
        """
        results = []

        for course_id in course_ids:
            try:
                prediction = self.predict_course(
                    course_id,
                    include_explanations=include_explanations
                )
                results.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting course {course_id}: {e}")
                continue

        return results

    def predict_reunion(
        self,
        reunion_id: int,
        include_explanations: bool = False
    ) -> Dict[str, Any]:
        """
        Génère les prédictions pour toutes les courses d'une réunion

        Args:
            reunion_id: ID de la réunion
            include_explanations: Inclure les explications SHAP

        Returns:
            Prédictions pour toutes les courses de la réunion
        """
        courses = (
            self.db.query(Course)
            .filter(Course.reunion_id == reunion_id)
            .order_by(Course.course_number)
            .all()
        )

        if not courses:
            raise ValueError(f"No courses found for reunion {reunion_id}")

        course_ids = [c.course_id for c in courses]
        course_predictions = self.predict_multiple_courses(course_ids, include_explanations)

        return {
            "reunion_id": reunion_id,
            "number_of_races": len(course_predictions),
            "races": course_predictions,
            "generated_at": datetime.now().isoformat(),
        }

    def predict_daily_program(
        self,
        target_date: Optional[date] = None,
        include_explanations: bool = False
    ) -> Dict[str, Any]:
        """
        Génère les prédictions pour tout le programme d'une journée

        Args:
            target_date: Date cible (None = aujourd'hui)
            include_explanations: Inclure les explications SHAP

        Returns:
            Prédictions pour toutes les courses de la journée
        """
        if target_date is None:
            target_date = date.today()

        # Récupérer toutes les courses du jour
        courses = (
            self.db.query(Course)
            .join(Reunion)
            .filter(Reunion.reunion_date == target_date)
            .order_by(Reunion.reunion_number, Course.course_number)
            .all()
        )

        if not courses:
            raise ValueError(f"No courses found for date {target_date}")

        logger.info(f"Predicting daily program for {target_date}: {len(courses)} courses")

        course_ids = [c.course_id for c in courses]
        course_predictions = self.predict_multiple_courses(course_ids, include_explanations)

        return {
            "date": target_date.isoformat(),
            "number_of_races": len(course_predictions),
            "races": course_predictions,
            "generated_at": datetime.now().isoformat(),
        }

    def _predict_partant(
        self,
        partant: Partant,
        include_explanations: bool = True
    ) -> Dict[str, Any]:
        """
        Génère la prédiction pour un partant

        Args:
            partant: Instance du partant
            include_explanations: Inclure les explications SHAP

        Returns:
            Dictionnaire contenant la prédiction et les explications
        """
        # Extraire les features
        features_dict = self.feature_engineer.extract_features_for_partant(
            partant,
            include_odds=True
        )

        # Convertir en DataFrame
        features_df = pd.DataFrame([features_dict])

        # Assurer que toutes les features sont présentes
        expected_features = self.feature_engineer.get_feature_names()
        for feature in expected_features:
            if feature not in features_df.columns:
                features_df[feature] = 0

        # Réordonner les colonnes
        features_df = features_df[expected_features]
        features_df = features_df.fillna(0)

        # Faire la prédiction
        probability = self.model.predict_proba(features_df)[0]

        # Préparer le résultat de base
        result = {
            "partant_id": partant.partant_id,
            "numero_corde": partant.numero_corde,
            "horse_name": partant.horse.name if partant.horse else "Unknown",
            "horse_age": partant.horse.age if partant.horse else None,
            "jockey_name": partant.jockey_name,
            "trainer_name": partant.trainer_name,
            "odds_pmu": float(partant.odds_pmu) if partant.odds_pmu else None,
            "probability": float(probability),
            "confidence_level": self._get_confidence_level(probability),
        }

        # Ajouter les explications SHAP si demandé
        if include_explanations and self.explainer:
            try:
                partant_info = {
                    "horse_name": result["horse_name"],
                    "numero_corde": result["numero_corde"],
                }
                explanation = self.explainer.explain_prediction(features_df, partant_info)
                result["explanation"] = explanation
            except Exception as e:
                logger.error(f"Error generating explanation for partant {partant.partant_id}: {e}")
                result["explanation"] = None

        return result

    def _generate_recommendations(
        self,
        predictions: List[Dict[str, Any]],
        course: Course
    ) -> Dict[str, Any]:
        """
        Génère les recommandations de paris

        Args:
            predictions: Liste des prédictions triées
            course: Instance de la course

        Returns:
            Dictionnaire avec les recommandations
        """
        # Top 5 pour le quinté
        top5 = predictions[:5]

        # Gagnant (meilleure probabilité)
        gagnant = predictions[0] if predictions else None

        # Placé (top 3)
        place = predictions[:3]

        # Tiercé (top 3 dans l'ordre)
        tierce = predictions[:3]

        # Quarté (top 4)
        quarte = predictions[:4]

        # Quinté (top 5)
        quinte = top5

        return {
            "gagnant": {
                "numero": gagnant["numero_corde"],
                "horse_name": gagnant["horse_name"],
                "probability": gagnant["probability"],
            } if gagnant else None,
            "place": [
                {
                    "numero": p["numero_corde"],
                    "horse_name": p["horse_name"],
                    "probability": p["probability"],
                }
                for p in place
            ],
            "tierce": [p["numero_corde"] for p in tierce],
            "quarte": [p["numero_corde"] for p in quarte],
            "quinte": [p["numero_corde"] for p in quinte],
        }

    def _detect_value_bets(
        self,
        predictions: List[Dict[str, Any]],
        min_edge: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Détecte les value bets (paris à valeur)

        Args:
            predictions: Liste des prédictions
            min_edge: Edge minimum pour considérer un value bet (10% par défaut)

        Returns:
            Liste des value bets détectés
        """
        value_bets = []

        for pred in predictions:
            if pred["odds_pmu"] is None:
                continue

            # Calculer la probabilité implicite de la cote
            implied_probability = 1.0 / pred["odds_pmu"]

            # Calculer l'edge (avantage)
            edge = pred["probability"] - implied_probability

            # Si edge > min_edge, c'est un value bet
            if edge > min_edge:
                value_bets.append({
                    "partant_id": pred["partant_id"],
                    "numero_corde": pred["numero_corde"],
                    "horse_name": pred["horse_name"],
                    "odds_pmu": pred["odds_pmu"],
                    "model_probability": pred["probability"],
                    "implied_probability": implied_probability,
                    "edge": edge,
                    "edge_percentage": edge / implied_probability if implied_probability > 0 else 0,
                    "value_level": self._get_value_level(edge),
                })

        # Trier par edge décroissant
        value_bets.sort(key=lambda x: x["edge"], reverse=True)

        return value_bets

    def _get_confidence_level(self, probability: float) -> str:
        """
        Détermine le niveau de confiance

        Args:
            probability: Probabilité prédite

        Returns:
            Niveau de confiance (low/medium/high/very_high)
        """
        if probability >= 0.7:
            return "very_high"
        elif probability >= 0.5:
            return "high"
        elif probability >= 0.3:
            return "medium"
        else:
            return "low"

    def _get_value_level(self, edge: float) -> str:
        """
        Détermine le niveau de value

        Args:
            edge: Edge calculé

        Returns:
            Niveau de value (low/medium/high)
        """
        if edge >= 0.3:
            return "high"
        elif edge >= 0.2:
            return "medium"
        else:
            return "low"