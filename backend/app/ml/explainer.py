"""
Module d'explicabilité SHAP pour les prédictions hippiques

Ce module calcule les SHAP values pour expliquer les prédictions
du modèle ML et identifie les facteurs clés influençant chaque pronostic.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap

from app.ml.model import HorseRacingModel

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Explainer SHAP pour le modèle de prédiction hippique

    Calcule et interprète les SHAP values pour rendre transparent
    le processus de décision du modèle ML.
    """

    def __init__(self, model: HorseRacingModel, background_data: Optional[pd.DataFrame] = None):
        """
        Initialise l'explainer SHAP

        Args:
            model: Instance du modèle entraîné
            background_data: Données de référence pour SHAP (optionnel)
                           Si None, utilisera un échantillon aléatoire
        """
        self.model = model
        self.background_data = background_data

        # Créer l'explainer Tree pour Gradient Boosting
        try:
            self.explainer = shap.TreeExplainer(
                model.model,
                feature_perturbation="tree_path_dependent"
            )
            logger.info("SHAP TreeExplainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            self.explainer = None

    def explain_prediction(
        self,
        features: pd.DataFrame,
        partant_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Explique une prédiction en calculant les SHAP values

        Args:
            features: Features du partant (DataFrame à 1 ligne)
            partant_info: Informations supplémentaires du partant (optionnel)

        Returns:
            Dictionnaire contenant les SHAP values et leur interprétation
        """
        if self.explainer is None:
            logger.error("SHAP explainer not initialized")
            return self._get_empty_explanation()

        try:
            # Calculer les SHAP values
            shap_values = self.explainer.shap_values(features)

            # Si binaire, shap_values peut être un array 2D
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Classe positive

            # Extraire les valeurs pour cette instance
            shap_values_instance = shap_values[0] if len(shap_values.shape) > 1 else shap_values

            # Obtenir la valeur de base (expected value)
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1]  # Classe positive

            # Créer le dictionnaire d'explication
            explanation = {
                "base_value": float(base_value),
                "shap_values": {
                    feature: float(shap_values_instance[i])
                    for i, feature in enumerate(features.columns)
                },
                "top_positive_features": self._get_top_features(
                    features, shap_values_instance, top_n=5, positive=True
                ),
                "top_negative_features": self._get_top_features(
                    features, shap_values_instance, top_n=5, positive=False
                ),
                "prediction_impact_summary": self._generate_impact_summary(
                    features, shap_values_instance, partant_info
                ),
            }

            return explanation

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return self._get_empty_explanation()

    def explain_race(
        self,
        partants_features: List[pd.DataFrame],
        partants_info: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Explique les prédictions pour tous les partants d'une course

        Args:
            partants_features: Liste des features de chaque partant
            partants_info: Liste des informations de chaque partant (optionnel)

        Returns:
            Liste des explications pour chaque partant
        """
        if partants_info is None:
            partants_info = [None] * len(partants_features)

        explanations = []
        for features, info in zip(partants_features, partants_info):
            explanation = self.explain_prediction(features, info)
            explanations.append(explanation)

        return explanations

    def _get_top_features(
        self,
        features: pd.DataFrame,
        shap_values: np.ndarray,
        top_n: int = 5,
        positive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Récupère les top N features ayant le plus d'impact

        Args:
            features: Features du partant
            shap_values: SHAP values calculées
            top_n: Nombre de features à retourner
            positive: True pour impact positif, False pour négatif

        Returns:
            Liste des top features avec leur impact
        """
        # Créer un DataFrame avec features et SHAP values
        feature_impacts = pd.DataFrame({
            'feature': features.columns,
            'value': features.iloc[0].values,
            'shap_value': shap_values
        })

        # Trier selon l'impact (positif ou négatif)
        if positive:
            feature_impacts = feature_impacts.sort_values('shap_value', ascending=False)
        else:
            feature_impacts = feature_impacts.sort_values('shap_value', ascending=True)

        # Prendre les top N
        top_features = feature_impacts.head(top_n)

        # Formater le résultat
        result = []
        for _, row in top_features.iterrows():
            if (positive and row['shap_value'] > 0) or (not positive and row['shap_value'] < 0):
                result.append({
                    'feature': row['feature'],
                    'feature_value': float(row['value']),
                    'shap_value': float(row['shap_value']),
                    'impact': 'positive' if row['shap_value'] > 0 else 'negative',
                    'explanation': self._explain_feature(
                        row['feature'],
                        float(row['value']),
                        float(row['shap_value'])
                    )
                })

        return result

    def _explain_feature(
        self,
        feature_name: str,
        feature_value: float,
        shap_value: float
    ) -> str:
        """
        Génère une explication textuelle pour une feature

        Args:
            feature_name: Nom de la feature
            feature_value: Valeur de la feature
            shap_value: SHAP value (impact)

        Returns:
            Explication en français
        """
        impact = "favorise" if shap_value > 0 else "défavorise"
        impact_strength = "fortement" if abs(shap_value) > 0.1 else "légèrement"

        # Explications personnalisées par type de feature
        if "win_rate" in feature_name:
            entity = feature_name.split("_")[0]
            return f"Le taux de victoire du {entity} ({feature_value:.1%}) {impact} {impact_strength} ce partant"

        elif "place_rate" in feature_name:
            entity = feature_name.split("_")[0]
            return f"Le taux de placement du {entity} ({feature_value:.1%}) {impact} {impact_strength} ce partant"

        elif "age" in feature_name:
            return f"L'âge du cheval ({int(feature_value)} ans) {impact} {impact_strength} ses chances"

        elif "distance" in feature_name and "course_distance" in feature_name:
            return f"La distance de la course ({int(feature_value)}m) {impact} {impact_strength} ce partant"

        elif "numero_corde" in feature_name:
            return f"Le numéro de corde {int(feature_value)} {impact} {impact_strength} ses chances"

        elif "poids_porte" in feature_name:
            return f"Le poids porté ({feature_value:.1f}kg) {impact} {impact_strength} la performance"

        elif "days_since_last_race" in feature_name:
            return f"Le repos depuis la dernière course ({int(feature_value)} jours) {impact} {impact_strength}"

        elif "recent_avg_position" in feature_name:
            return f"La position moyenne récente ({feature_value:.1f}) {impact} {impact_strength} ce partant"

        elif "has_oeilleres" in feature_name:
            if feature_value > 0:
                return f"Le port d'œillères {impact} {impact_strength} ce partant"
            else:
                return f"L'absence d'œillères {impact} {impact_strength} ce partant"

        elif "affinity" in feature_name:
            return f"L'affinité historique ({feature_value:.1%}) {impact} {impact_strength} ce partant"

        elif "discipline_" in feature_name or "surface_" in feature_name:
            discipline_or_surface = feature_name.split("_")[1]
            if feature_value > 0:
                return f"La discipline/surface {discipline_or_surface} {impact} {impact_strength} ce partant"

        elif "odds_pmu" in feature_name:
            return f"La cote PMU ({feature_value:.2f}) {impact} {impact_strength} l'évaluation"

        # Explication générique
        return f"{feature_name} (valeur: {feature_value:.2f}) {impact} {impact_strength} ce partant"

    def _generate_impact_summary(
        self,
        features: pd.DataFrame,
        shap_values: np.ndarray,
        partant_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Génère un résumé textuel de l'impact global

        Args:
            features: Features du partant
            shap_values: SHAP values calculées
            partant_info: Informations du partant (optionnel)

        Returns:
            Résumé textuel en français
        """
        # Calculer l'impact total positif et négatif
        positive_impact = np.sum(shap_values[shap_values > 0])
        negative_impact = np.sum(shap_values[shap_values < 0])
        net_impact = positive_impact + negative_impact

        # Identifier les principaux facteurs
        top_positive = self._get_top_features(features, shap_values, top_n=3, positive=True)
        top_negative = self._get_top_features(features, shap_values, top_n=3, positive=False)

        # Construire le résumé
        summary_parts = []

        if partant_info:
            horse_name = partant_info.get('horse_name', 'Ce cheval')
            numero = partant_info.get('numero_corde', '')
            summary_parts.append(f"**Analyse de {horse_name} (N°{numero})**\n")

        if net_impact > 0:
            summary_parts.append(
                f"Les facteurs favorables l'emportent sur les défavorables " +
                f"(impact net: +{net_impact:.3f})."
            )
        else:
            summary_parts.append(
                f"Les facteurs défavorables l'emportent sur les favorables " +
                f"(impact net: {net_impact:.3f})."
            )

        if top_positive:
            summary_parts.append("\n**Principaux atouts:**")
            for i, feature in enumerate(top_positive[:3], 1):
                summary_parts.append(f"{i}. {feature['explanation']}")

        if top_negative:
            summary_parts.append("\n**Principaux handicaps:**")
            for i, feature in enumerate(top_negative[:3], 1):
                summary_parts.append(f"{i}. {feature['explanation']}")

        return "\n".join(summary_parts)

    def _get_empty_explanation(self) -> Dict[str, Any]:
        """
        Retourne une explication vide en cas d'erreur

        Returns:
            Dictionnaire d'explication vide
        """
        return {
            "base_value": 0.0,
            "shap_values": {},
            "top_positive_features": [],
            "top_negative_features": [],
            "prediction_impact_summary": "Explication non disponible",
        }

    def get_global_feature_importance(
        self,
        data: pd.DataFrame,
        max_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Calcule l'importance globale des features sur un dataset

        Args:
            data: Dataset de features
            max_samples: Nombre maximum d'échantillons à analyser

        Returns:
            DataFrame avec l'importance moyenne absolue de chaque feature
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")

        # Limiter le nombre d'échantillons pour les performances
        if len(data) > max_samples:
            data_sample = data.sample(n=max_samples, random_state=42)
        else:
            data_sample = data

        logger.info(f"Calculating global feature importance on {len(data_sample)} samples")

        # Calculer les SHAP values
        shap_values = self.explainer.shap_values(data_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Classe positive

        # Calculer l'importance moyenne absolue
        feature_importance = pd.DataFrame({
            'feature': data.columns,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)

        return feature_importance
