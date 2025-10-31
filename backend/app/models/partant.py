"""
Modèle SQLAlchemy pour la table partants (participants/runners dans une course)
"""

from typing import Optional, Dict, List, Any

from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, DECIMAL, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Partant(Base):
    """
    Modèle pour les partants (chevaux participant à une course)
    """
    __tablename__ = "partants"

    partant_id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(Integer, ForeignKey("courses.course_id", ondelete="CASCADE"), nullable=False, index=True)
    horse_id = Column(Integer, ForeignKey("horses.horse_id", ondelete="CASCADE"), nullable=False, index=True)
    jockey_id = Column(Integer, ForeignKey("jockeys.jockey_id", ondelete="SET NULL"), nullable=True)
    trainer_id = Column(Integer, ForeignKey("trainers.trainer_id", ondelete="CASCADE"), nullable=False)

    numero_corde = Column(Integer, nullable=False)  # Numéro de départ
    poids_porte = Column(DECIMAL(4, 1), nullable=True)  # Poids porté en kg
    handicap_value = Column(Integer, nullable=True)  # Valeur de handicap
    equipment = Column(JSON, nullable=True)  # Équipement (œillères, etc.)
    aspiturf_stats = Column(JSON, nullable=True)  # Statistiques enrichies Aspiturf
    days_since_last_race = Column(Integer, nullable=True)  # Jours depuis dernière course
    recent_form = Column(String(50), nullable=True)  # Forme récente (ex: "1-3-2-5")
    odds_pmu = Column(DECIMAL(6, 2), nullable=True)  # Cote PMU

    # Résultats (rempli après la course)
    final_position = Column(Integer, nullable=True)
    disqualified = Column(Boolean, default=False)

    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationships
    course = relationship("Course", back_populates="partants")
    horse = relationship("Horse", back_populates="partants")
    jockey = relationship("Jockey", back_populates="partants")
    trainer = relationship("Trainer", back_populates="partants")
    predictions = relationship("PartantPrediction", back_populates="partant", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Partant(id={self.partant_id}, course_id={self.course_id}, numero={self.numero_corde}, horse={self.horse.name if self.horse else 'N/A'})>"

    @property
    def display_name(self) -> str:
        """Retourne le nom d'affichage du partant"""
        if self.horse:
            return f"N°{self.numero_corde} - {self.horse.name}"
        return f"N°{self.numero_corde}"

    @property
    def equipment_list(self) -> Optional[List[str]]:
        """Retourne la liste d'équipement"""
        if self.equipment and isinstance(self.equipment, dict):
            return self.equipment.get("items", [])
        return None

    @property
    def aspiturf_data(self) -> Dict[str, Any]:
        """Retourne les statistiques Aspiturf (toujours un dictionnaire)."""
        if isinstance(self.aspiturf_stats, dict):
            return self.aspiturf_stats
        return {}

    @property
    def has_oeilleres(self) -> bool:
        """Vérifie si le cheval porte des œillères"""
        equipment_list = self.equipment_list
        if equipment_list:
            return any("oeillere" in item.lower() or "œillère" in item.lower() for item in equipment_list)
        return False

    @property
    def is_favorite(self) -> bool:
        """Détermine si c'est un favori (cote < 5)"""
        if self.odds_pmu:
            return float(self.odds_pmu) < 5.0
        return False

    @property
    def odds_category(self) -> str:
        """Catégorise la cote"""
        if not self.odds_pmu:
            return "Inconnue"
        odds = float(self.odds_pmu)
        if odds < 3:
            return "Très favori"
        elif odds < 5:
            return "Favori"
        elif odds < 10:
            return "Moyen"
        elif odds < 20:
            return "Outsider"
        else:
            return "Gros outsider"

    @property
    def recent_form_list(self) -> Optional[List[int]]:
        """Convertit la forme récente en liste d'entiers"""
        if self.recent_form:
            try:
                return [int(x) for x in self.recent_form.split("-") if x.isdigit()]
            except ValueError:
                return None
        return None

    @property
    def average_recent_position(self) -> Optional[float]:
        """Calcule la position moyenne récente"""
        form_list = self.recent_form_list
        if form_list:
            return sum(form_list) / len(form_list)
        return None

    @property
    def has_won_recently(self) -> bool:
        """Vérifie si le cheval a gagné récemment"""
        form_list = self.recent_form_list
        if form_list:
            return 1 in form_list
        return False

    @property
    def weight_display(self) -> Optional[str]:
        """Retourne le poids formaté"""
        if self.poids_porte:
            return f"{float(self.poids_porte)} kg"
        return None

    @property
    def rest_days_category(self) -> str:
        """Catégorise le nombre de jours de repos"""
        if self.days_since_last_race is None:
            return "Inconnue"
        days = self.days_since_last_race
        if days < 14:
            return "Très frais (< 2 semaines)"
        elif days < 30:
            return "Frais (2-4 semaines)"
        elif days < 60:
            return "Normal (1-2 mois)"
        elif days < 180:
            return "Long (2-6 mois)"
        else:
            return "Très long (> 6 mois)"

    @property
    def is_finished(self) -> bool:
        """Vérifie si le partant a fini la course"""
        return self.final_position is not None and not self.disqualified

    @property
    def result_display(self) -> str:
        """Affiche le résultat de la course"""
        if self.disqualified:
            return "Disqualifié"
        elif self.final_position:
            return f"{self.final_position}ème"
        else:
            return "En attente"

    @property
    def jockey_name(self) -> str:
        """Retourne le nom du jockey"""
        if self.jockey:
            return self.jockey.full_name
        return "Non assigné"

    @property
    def trainer_name(self) -> str:
        """Retourne le nom de l'entraîneur"""
        if self.trainer:
            return self.trainer.full_name
        return "Inconnu"
