"""Tests unitaires pour la tâche d'évaluation des performances ML."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import Column, ForeignKey, Integer, create_engine
from sqlalchemy.orm import Session, relationship, sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("CORS_ORIGINS", "[]")

from app.core.database import Base
from app.core import database as core_database

if not hasattr(core_database, "SessionLocal"):
    core_database.SessionLocal = lambda: None  # type: ignore[attr-defined]


class PerformanceHistorique(Base):
    """Stub minimal pour satisfaire la relation déclarée sur ``Horse``."""

    __tablename__ = "performance_historique"

    performance_id = Column(Integer, primary_key=True, autoincrement=True)
    horse_id = Column(Integer, ForeignKey("horses.horse_id"), nullable=False)

    horse = relationship("Horse", back_populates="performances")
from app.models.course import Course, CourseStatus, Discipline, SurfaceType, StartType
from app.models.hippodrome import Hippodrome, TrackType
from app.models.horse import Gender, Horse
from app.models.jockey import Jockey
from app.models.ml_model import MLModel
from app.models.partant import Partant
from app.models.partant_prediction import PartantPrediction
from app.models.pronostic import Pronostic
from app.models.reunion import Reunion, ReunionStatus
from app.models.trainer import Trainer
from app.tasks import ml_tasks


@pytest.fixture()
def in_memory_session(monkeypatch: pytest.MonkeyPatch) -> sessionmaker:
    """Configure une base SQLite en mémoire et remplace ``SessionLocal``."""

    engine = create_engine("sqlite:///:memory:", echo=False)
    TestingSessionLocal = sessionmaker(bind=engine)

    Base.metadata.create_all(bind=engine)

    def _session_local() -> Session:
        return TestingSessionLocal()

    monkeypatch.setattr(ml_tasks, "SessionLocal", _session_local)

    yield TestingSessionLocal

    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def _seed_reference_data(db: Session) -> None:
    """Insère les entités de base nécessaires aux tests."""

    hippodrome_flat = Hippodrome(
        code="TEST",
        name="Hippodrome Test",
        city="Paris",
        country="France",
        track_type=TrackType.PLAT,
    )
    hippodrome_trot = Hippodrome(
        code="TROT",
        name="Hippodrome Trot",
        city="Mons",
        country="Belgique",
        track_type=TrackType.TROT,
    )
    db.add_all([hippodrome_flat, hippodrome_trot])
    db.flush()

    today = date.today()
    reunion = Reunion(
        hippodrome_id=hippodrome_flat.hippodrome_id,
        reunion_date=today - timedelta(days=1),
        reunion_number=1,
        status=ReunionStatus.COMPLETED,
        weather_conditions={"condition": "Ensoleillé", "temperature": 18},
    )
    reunion_evening = Reunion(
        hippodrome_id=hippodrome_trot.hippodrome_id,
        reunion_date=today,
        reunion_number=4,
        status=ReunionStatus.COMPLETED,
        weather_conditions={"condition": "Pluie battante", "temperature": 9},
    )
    db.add_all([reunion, reunion_evening])
    db.flush()

    trainer_fr = Trainer(first_name="Anne", last_name="Durand", nationality="FR")
    trainer_be = Trainer(first_name="Marc", last_name="Dupont", nationality="BE")
    jockey_fr = Jockey(first_name="Leo", last_name="Martin", nationality="FR")
    jockey_be = Jockey(first_name="Noah", last_name="Verbeeck", nationality="BE")
    db.add_all([trainer_fr, trainer_be, jockey_fr, jockey_be])
    db.flush()

    current_year = date.today().year
    horses = [
        Horse(
            name="Cheval A",
            gender=Gender.MALE,
            birth_year=current_year - 4,
            owner="Ecurie Horizon",
        ),
        Horse(
            name="Cheval B",
            gender=Gender.FEMALE,
            birth_year=current_year - 3,
            owner="Ecurie Horizon",
        ),
        Horse(
            name="Cheval C",
            gender=Gender.MALE,
            birth_year=current_year - 5,
            owner="Ecurie Equinoxe",
        ),
        Horse(
            name="Cheval D",
            gender=Gender.MALE,
            birth_year=current_year - 7,
            owner="Ecurie Equinoxe",
        ),
        Horse(
            name="Cheval E",
            gender=Gender.FEMALE,
            birth_year=current_year - 9,
            owner="Ecurie Boreale",
        ),
        Horse(name="Cheval F", gender=Gender.MALE),
    ]
    db.add_all(horses)
    db.flush()

    course1 = Course(
        reunion_id=reunion.reunion_id,
        course_number=1,
        course_name="R1C1",
        discipline=Discipline.PLAT,
        distance=1400,
        prize_money=Decimal("10000"),
        race_category="Groupe",
        race_class="A",
        surface_type=SurfaceType.PELOUSE,
        start_type=StartType.STALLE,
        scheduled_time=time(14, 0),
        status=CourseStatus.FINISHED,
        number_of_runners=8,
    )
    course2 = Course(
        reunion_id=reunion_evening.reunion_id,
        course_number=7,
        course_name="R1C2",
        discipline=Discipline.TROT_ATTELE,
        distance=3000,
        prize_money=Decimal("8000"),
        race_category="Classe",
        race_class="B",
        surface_type=SurfaceType.SABLE,
        start_type=StartType.AUTOSTART,
        scheduled_time=time(20, 30),
        status=CourseStatus.FINISHED,
        number_of_runners=14,
    )
    db.add_all([course1, course2])
    db.flush()

    pronostic1 = Pronostic(
        course_id=course1.course_id,
        model_version="v1.0",
        gagnant_predicted=None,
        place_predicted=None,
        tierce_predicted=None,
        quarte_predicted=None,
        quinte_predicted=None,
        confidence_score=Decimal("0.55"),
        value_bet_detected=True,
        generated_at=datetime.combine(date.today() - timedelta(days=1), time(11, 0)),
    )
    pronostic2 = Pronostic(
        course_id=course2.course_id,
        model_version="v2.0",
        gagnant_predicted=None,
        place_predicted=None,
        tierce_predicted=None,
        quarte_predicted=None,
        quinte_predicted=None,
        confidence_score=Decimal("0.40"),
        value_bet_detected=False,
        generated_at=datetime.combine(date.today(), time(10, 30)),
    )
    db.add_all([pronostic1, pronostic2])
    db.flush()

    partants = [
        Partant(
            course_id=course1.course_id,
            horse_id=horses[0].horse_id,
            jockey_id=jockey_fr.jockey_id,
            trainer_id=trainer_fr.trainer_id,
            numero_corde=1,
            poids_porte=Decimal("52.5"),
            final_position=1,
            odds_pmu=Decimal("3.0"),
            days_since_last_race=10,
            handicap_value=8,
            equipment={"items": ["Oeillères"]},
            recent_form="1-1-2",
        ),
        Partant(
            course_id=course1.course_id,
            horse_id=horses[1].horse_id,
            jockey_id=jockey_fr.jockey_id,
            trainer_id=trainer_fr.trainer_id,
            numero_corde=5,
            poids_porte=Decimal("55.0"),
            final_position=2,
            odds_pmu=Decimal("4.0"),
            days_since_last_race=25,
            handicap_value=16,
            equipment={"items": ["Licol"]},
            recent_form="3-4-5",
        ),
        Partant(
            course_id=course1.course_id,
            horse_id=horses[2].horse_id,
            jockey_id=jockey_fr.jockey_id,
            trainer_id=trainer_fr.trainer_id,
            numero_corde=8,
            poids_porte=Decimal("58.5"),
            final_position=4,
            odds_pmu=Decimal("12.0"),
            days_since_last_race=75,
            handicap_value=26,
            equipment={"items": ["Bonnet", "Mors"]},
            recent_form="6-7-8",
        ),
        Partant(
            course_id=course2.course_id,
            horse_id=horses[3].horse_id,
            jockey_id=jockey_be.jockey_id,
            trainer_id=trainer_be.trainer_id,
            numero_corde=2,
            poids_porte=Decimal("60.0"),
            final_position=1,
            odds_pmu=Decimal("5.5"),
            days_since_last_race=45,
            handicap_value=34,
            equipment={"items": []},
            recent_form="2-3-2",
        ),
        Partant(
            course_id=course2.course_id,
            horse_id=horses[4].horse_id,
            jockey_id=jockey_be.jockey_id,
            trainer_id=trainer_be.trainer_id,
            numero_corde=8,
            poids_porte=Decimal("62.0"),
            final_position=4,
            odds_pmu=Decimal("7.0"),
            days_since_last_race=210,
            handicap_value=12,
            equipment=None,
            recent_form="9-10-11",
        ),
        Partant(
            course_id=course2.course_id,
            horse_id=horses[5].horse_id,
            jockey_id=jockey_be.jockey_id,
            trainer_id=trainer_be.trainer_id,
            numero_corde=13,
            final_position=2,
            odds_pmu=Decimal("6.0"),
            equipment={"items": ["Œillères australiennes"]},
        ),
    ]
    db.add_all(partants)
    db.flush()

    predictions = [
        PartantPrediction(
            pronostic_id=pronostic1.pronostic_id,
            partant_id=partants[0].partant_id,
            win_probability=Decimal("0.55"),
            confidence_level="high",
        ),
        PartantPrediction(
            pronostic_id=pronostic1.pronostic_id,
            partant_id=partants[1].partant_id,
            win_probability=Decimal("0.30"),
            confidence_level="medium",
        ),
        PartantPrediction(
            pronostic_id=pronostic1.pronostic_id,
            partant_id=partants[2].partant_id,
            win_probability=Decimal("0.15"),
            confidence_level="low",
        ),
        PartantPrediction(
            pronostic_id=pronostic2.pronostic_id,
            partant_id=partants[3].partant_id,
            win_probability=Decimal("0.25"),
            confidence_level="low",
        ),
        PartantPrediction(
            pronostic_id=pronostic2.pronostic_id,
            partant_id=partants[4].partant_id,
            win_probability=Decimal("0.40"),
            confidence_level="medium",
        ),
        PartantPrediction(
            pronostic_id=pronostic2.pronostic_id,
            partant_id=partants[5].partant_id,
            win_probability=Decimal("0.35"),
            confidence_level="medium",
        ),
    ]
    db.add_all(predictions)

    model = MLModel(
        model_name="horse_racing_gradient_boosting",
        version="20250101",
        algorithm="GradientBoosting",
        file_path="model.pkl",
        is_active=True,
    )
    db.add(model)


def test_update_model_performance_with_results(in_memory_session: sessionmaker) -> None:
    """Vérifie le calcul des métriques et la mise à jour du modèle actif."""

    session = in_memory_session()
    _seed_reference_data(session)
    session.commit()
    session.close()

    result = ml_tasks.update_model_performance.run(days_back=2, probability_threshold=0.3)

    assert result["status"] == "success"
    assert result["evaluated_samples"] == 6
    assert result["courses_evaluated"] == 2
    assert result["value_bet_courses"] == 1

    metrics = result["metrics"]
    assert metrics["accuracy"] == pytest.approx(2 / 3, rel=1e-3)
    assert metrics["precision"] == pytest.approx(0.75, rel=1e-3)
    assert metrics["recall"] == pytest.approx(0.75, rel=1e-3)
    assert metrics["f1"] == pytest.approx(0.75, rel=1e-3)
    assert metrics["roc_auc"] == pytest.approx(0.625, rel=1e-3)
    assert metrics["brier_score"] == pytest.approx(0.31, rel=1e-2)
    assert metrics["top1_accuracy"] == pytest.approx(0.5, rel=1e-3)
    assert metrics["course_top3_hit_rate"] == pytest.approx(1.0, rel=1e-3)

    calibration_table = metrics["calibration_table"]
    assert len(calibration_table) == 5
    assert calibration_table[0]["empirical_rate"] == pytest.approx(0.0, abs=1e-6)
    assert calibration_table[-1]["count"] == 2
    assert calibration_table[-1]["empirical_rate"] == pytest.approx(0.5, rel=1e-3)

    calibration_diagnostics = metrics["calibration_diagnostics"]
    assert calibration_diagnostics["expected_calibration_error"] == pytest.approx(0.38333, rel=1e-3)
    assert calibration_diagnostics["maximum_calibration_gap"] == pytest.approx(0.75, rel=1e-3)
    assert calibration_diagnostics["weighted_bias"] == pytest.approx(1 / 3, rel=1e-3)
    assert len(calibration_diagnostics["bins"]) == len(calibration_table)
    assert calibration_diagnostics["bins"][0]["calibration_gap"] == pytest.approx(-0.15, rel=1e-3)
    assert calibration_diagnostics["bins"][1]["weight"] == pytest.approx(1 / 6, rel=1e-3)

    threshold_grid = metrics["threshold_sensitivity"]
    assert threshold_grid["0.20"]["recall"] == pytest.approx(1.0, rel=1e-3)
    assert threshold_grid["0.20"]["precision"] == pytest.approx(0.8, rel=1e-3)
    assert threshold_grid["0.40"]["accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    assert threshold_grid["0.50"]["precision"] == pytest.approx(1.0, rel=1e-3)

    threshold_recommendations = metrics["threshold_recommendations"]
    sweep = [(float(threshold), values) for threshold, values in threshold_grid.items()]
    assert len(threshold_recommendations["grid"]) == len(sweep)

    best_f1_threshold, best_f1_metrics = max(
        (
            (threshold, values)
            for threshold, values in sweep
            if values.get("f1") is not None
        ),
        key=lambda item: (item[1]["f1"], -item[0]),
    )
    assert threshold_recommendations["best_f1"]["threshold"] == pytest.approx(best_f1_threshold, rel=1e-3)
    assert threshold_recommendations["best_f1"]["f1"] == pytest.approx(best_f1_metrics["f1"], rel=1e-3)
    assert threshold_recommendations["best_f1"]["precision"] == pytest.approx(best_f1_metrics["precision"], rel=1e-3)
    assert threshold_recommendations["best_f1"]["recall"] == pytest.approx(best_f1_metrics["recall"], rel=1e-3)

    best_precision_threshold, best_precision_metrics = max(
        (
            (threshold, values)
            for threshold, values in sweep
            if values.get("precision") is not None
        ),
        key=lambda item: (item[1]["precision"], -item[0]),
    )
    assert threshold_recommendations["maximize_precision"]["threshold"] == pytest.approx(best_precision_threshold, rel=1e-3)
    assert threshold_recommendations["maximize_precision"]["precision"] == pytest.approx(best_precision_metrics["precision"], rel=1e-3)

    best_recall_threshold, best_recall_metrics = max(
        (
            (threshold, values)
            for threshold, values in sweep
            if values.get("recall") is not None
        ),
        key=lambda item: (item[1]["recall"], -item[0]),
    )
    assert threshold_recommendations["maximize_recall"]["threshold"] == pytest.approx(best_recall_threshold, rel=1e-3)
    assert threshold_recommendations["maximize_recall"]["recall"] == pytest.approx(best_recall_metrics["recall"], rel=1e-3)

    gain_curve = metrics["gain_curve"]
    assert len(gain_curve) == 5
    assert gain_curve[0]["observations"] == 2
    assert gain_curve[0]["coverage"] == pytest.approx(2 / 6, rel=1e-3)
    assert gain_curve[0]["capture_rate"] == pytest.approx(0.25, rel=1e-3)
    assert gain_curve[-1]["cumulative_hit_rate"] == pytest.approx(4 / 6, rel=1e-3)
    assert gain_curve[-1]["capture_rate"] == pytest.approx(1.0, rel=1e-3)

    lift_analysis = metrics["lift_analysis"]
    assert lift_analysis["baseline_rate"] == pytest.approx(4 / 6, rel=1e-3)
    assert len(lift_analysis["buckets"]) == 5
    assert lift_analysis["buckets"][0]["lift"] == pytest.approx(1.5, rel=1e-3)
    assert lift_analysis["buckets"][1]["lift"] == pytest.approx(0.0, abs=1e-6)
    assert lift_analysis["buckets"][0]["cumulative_capture"] == pytest.approx(0.25, rel=1e-3)
    assert lift_analysis["buckets"][-1]["cumulative_coverage"] == pytest.approx(1.0, rel=1e-6)

    assert metrics["average_precision"] == pytest.approx(0.8041666, rel=1e-3)

    betting_value = metrics["betting_value_analysis"]
    assert betting_value["priced_samples"] == 6
    assert betting_value["bets_considered"] == 4
    assert betting_value["actual_win_rate"] == pytest.approx(0.25, rel=1e-3)
    assert betting_value["realized_roi"] == pytest.approx(-0.25, rel=1e-3)
    assert betting_value["expected_value_per_bet"] == pytest.approx(0.9375, rel=1e-3)
    assert betting_value["average_edge"] == pytest.approx(0.1767857, rel=1e-3)
    assert betting_value["average_implied_probability"] == pytest.approx(0.223214, rel=1e-3)
    assert betting_value["average_predicted_probability"] == pytest.approx(0.4, rel=1e-3)
    assert len(betting_value["best_value_candidates"]) == 3
    assert betting_value["best_value_candidates"][0]["edge"] == pytest.approx(0.2571428, rel=1e-3)
    assert betting_value["best_value_candidates"][0]["won"] is False

    odds_alignment = metrics["odds_alignment"]
    assert odds_alignment["priced_samples"] == 6
    assert odds_alignment["usable_samples"] == 6
    assert odds_alignment["pearson_correlation"] == pytest.approx(0.765364, rel=1e-3)
    assert odds_alignment["mean_probability_gap"] == pytest.approx(0.140331, rel=1e-3)
    assert odds_alignment["mean_absolute_error"] == pytest.approx(0.140331, rel=1e-3)
    assert odds_alignment["root_mean_squared_error"] == pytest.approx(0.1624147, rel=1e-3)
    assert odds_alignment["average_predicted_probability"] == pytest.approx(1 / 3, rel=1e-3)
    assert odds_alignment["average_implied_probability"] == pytest.approx(0.1930014, rel=1e-3)
    assert odds_alignment["average_overround"] == pytest.approx(-0.4209956, rel=1e-3)
    assert odds_alignment["median_overround"] == pytest.approx(-0.4209956, rel=1e-3)
    assert odds_alignment["courses_with_overlay"] == 2
    assert len(odds_alignment["course_overrounds"]) == 2
    assert odds_alignment["course_overrounds"][0]["overround"] == pytest.approx(-1 / 3, rel=1e-3)
    assert odds_alignment["course_overrounds"][0]["runner_count"] == 3

    jockey_nat_metrics = metrics["jockey_nationality_performance"]
    assert set(jockey_nat_metrics.keys()) == {"france", "belgique"}
    assert jockey_nat_metrics["france"]["actors"] == 1
    assert jockey_nat_metrics["belgique"]["actors"] == 1

    trainer_nat_metrics = metrics["trainer_nationality_performance"]
    assert set(trainer_nat_metrics.keys()) == {"france", "belgique"}
    assert trainer_nat_metrics["france"]["courses"] == 1
    assert trainer_nat_metrics["belgique"]["courses"] == 1

    pr_curve = metrics["precision_recall_curve"]
    assert len(pr_curve) == 7
    assert pr_curve[0]["threshold"] == pytest.approx(0.15, rel=1e-3)
    assert pr_curve[0]["precision"] == pytest.approx(0.8, rel=1e-3)
    assert pr_curve[0]["recall"] == pytest.approx(1.0, rel=1e-3)
    assert pr_curve[2]["f1"] == pytest.approx(0.571428, rel=1e-3)
    assert pr_curve[-1]["threshold"] == pytest.approx(0.0, abs=1e-6)

    roc_curve_points = metrics["roc_curve"]
    assert len(roc_curve_points) == 5
    assert roc_curve_points[0]["threshold"] is None
    assert roc_curve_points[0]["true_positive_rate"] == pytest.approx(0.0, abs=1e-6)
    assert roc_curve_points[1]["threshold"] == pytest.approx(0.55, rel=1e-3)
    assert roc_curve_points[1]["youden_j"] == pytest.approx(0.25, rel=1e-3)
    assert roc_curve_points[3]["false_positive_rate"] == pytest.approx(0.5, rel=1e-3)
    assert roc_curve_points[3]["specificity"] == pytest.approx(0.5, rel=1e-3)

    ks_analysis = metrics["ks_analysis"]
    assert ks_analysis["ks_statistic"] == pytest.approx(0.5, rel=1e-3)
    assert ks_analysis["ks_threshold"] == pytest.approx(0.25, rel=1e-3)
    assert len(ks_analysis["curve"]) == 6
    assert ks_analysis["curve"][0]["threshold"] == pytest.approx(0.55, rel=1e-3)
    assert ks_analysis["curve"][4]["distance"] == pytest.approx(0.5, rel=1e-3)

    confidence = result["confidence_distribution"]
    assert confidence == {"high": 1, "low": 2, "medium": 3}

    level_metrics = metrics["confidence_level_metrics"]
    assert set(level_metrics.keys()) == {"high", "low", "medium"}

    assert level_metrics["high"]["samples"] == 1
    assert level_metrics["high"]["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert level_metrics["high"]["positive_rate"] == pytest.approx(1.0, rel=1e-3)

    assert level_metrics["medium"]["samples"] == 3
    assert level_metrics["medium"]["precision"] == pytest.approx(2 / 3, rel=1e-3)
    assert level_metrics["medium"]["recall"] == pytest.approx(1.0, rel=1e-3)

    assert level_metrics["low"]["samples"] == 2
    assert level_metrics["low"]["precision"] == pytest.approx(0.0, abs=1e-6)
    assert level_metrics["low"]["positive_rate"] == pytest.approx(0.0, abs=1e-6)

    confidence_score_performance = metrics["confidence_score_performance"]
    assert set(confidence_score_performance.keys()) == {"low", "medium"}

    medium_confidence = confidence_score_performance["medium"]
    assert medium_confidence["label"] == "Confiance moyenne (50-70%)"
    assert medium_confidence["samples"] == 3
    assert medium_confidence["courses"] == 1
    assert medium_confidence["share"] == pytest.approx(0.5, rel=1e-3)
    assert medium_confidence["average_confidence"] == pytest.approx(55.0, rel=1e-3)
    assert medium_confidence["min_confidence"] == pytest.approx(55.0, rel=1e-3)
    assert medium_confidence["max_confidence"] == pytest.approx(55.0, rel=1e-3)
    assert medium_confidence["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert medium_confidence["positive_rate"] == pytest.approx(2 / 3, rel=1e-3)

    low_confidence = confidence_score_performance["low"]
    assert low_confidence["label"] == "Confiance faible (30-50%)"
    assert low_confidence["samples"] == 3
    assert low_confidence["courses"] == 1
    assert low_confidence["share"] == pytest.approx(0.5, rel=1e-3)
    assert low_confidence["average_confidence"] == pytest.approx(40.0, rel=1e-3)
    assert low_confidence["min_confidence"] == pytest.approx(40.0, rel=1e-3)
    assert low_confidence["max_confidence"] == pytest.approx(40.0, rel=1e-3)
    assert low_confidence["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert low_confidence["accuracy"] == pytest.approx(1 / 3, rel=1e-3)

    daily_performance = metrics["daily_performance"]
    assert len(daily_performance) == 2

    day_minus_one = (date.today() - timedelta(days=1)).isoformat()
    today = date.today().isoformat()

    assert daily_performance[0]["day"] == day_minus_one
    assert daily_performance[0]["samples"] == 3
    assert daily_performance[0]["courses"] == 1
    assert daily_performance[0]["value_bet_courses"] == 1
    assert daily_performance[0]["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert daily_performance[0]["positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert daily_performance[0]["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)

    assert daily_performance[1]["day"] == today
    assert daily_performance[1]["samples"] == 3
    assert daily_performance[1]["courses"] == 1
    assert daily_performance[1]["value_bet_courses"] == 0
    assert daily_performance[1]["accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    assert daily_performance[1]["precision"] == pytest.approx(0.5, rel=1e-3)
    assert daily_performance[1]["recall"] == pytest.approx(0.5, rel=1e-3)
    assert daily_performance[1]["positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert daily_performance[1]["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)

    day_part_performance = metrics["day_part_performance"]
    assert set(day_part_performance.keys()) == {"afternoon", "evening"}

    afternoon_metrics = day_part_performance["afternoon"]
    assert afternoon_metrics["label"] == "Après-midi"
    assert afternoon_metrics["samples"] == 3
    assert afternoon_metrics["courses"] == 1
    assert afternoon_metrics["share"] == pytest.approx(0.5, rel=1e-3)
    assert afternoon_metrics["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert afternoon_metrics["precision"] == pytest.approx(1.0, rel=1e-3)
    assert afternoon_metrics["recall"] == pytest.approx(1.0, rel=1e-3)
    assert afternoon_metrics["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert afternoon_metrics["average_post_time"] == "14:00"
    assert afternoon_metrics["earliest_post_time"] == "14:00"
    assert afternoon_metrics["latest_post_time"] == "14:00"

    evening_metrics = day_part_performance["evening"]
    assert evening_metrics["label"] == "Soir"
    assert evening_metrics["samples"] == 3
    assert evening_metrics["courses"] == 1
    assert evening_metrics["share"] == pytest.approx(0.5, rel=1e-3)
    assert evening_metrics["accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    assert evening_metrics["precision"] == pytest.approx(0.5, rel=1e-3)
    assert evening_metrics["recall"] == pytest.approx(0.5, rel=1e-3)
    assert evening_metrics["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert evening_metrics["average_post_time"] == "20:30"
    assert evening_metrics["earliest_post_time"] == "20:30"
    assert evening_metrics["latest_post_time"] == "20:30"

    month_performance = metrics["month_performance"]
    current_month_key = date.today().strftime("%Y-%m")
    assert current_month_key in month_performance
    assert month_performance[current_month_key]["samples"] >= 3
    assert month_performance[current_month_key]["reunions"] >= 1
    assert sum(entry["samples"] for entry in month_performance.values()) == 6
    assert str(date.today().year) in month_performance[current_month_key]["label"]

    race_order_performance = metrics["race_order_performance"]
    assert set(race_order_performance.keys()) == {"early_card", "late_card"}

    early_segment = race_order_performance["early_card"]
    assert early_segment["label"] == "Début de réunion (courses 1-3)"
    assert early_segment["samples"] == 3
    assert early_segment["courses"] == 1
    assert early_segment["reunions"] == 1
    assert early_segment["share"] == pytest.approx(0.5, rel=1e-3)
    assert early_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert early_segment["precision"] == pytest.approx(1.0, rel=1e-3)
    assert early_segment["recall"] == pytest.approx(1.0, rel=1e-3)
    assert early_segment["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert early_segment["average_course_number"] == pytest.approx(1.0, abs=1e-6)
    assert early_segment["min_course_number"] == 1
    assert early_segment["max_course_number"] == 1

    late_segment = race_order_performance["late_card"]
    assert late_segment["label"] == "Fin de réunion (courses 7+)"
    assert late_segment["samples"] == 3
    assert late_segment["courses"] == 1
    assert late_segment["reunions"] == 1
    assert late_segment["share"] == pytest.approx(0.5, rel=1e-3)
    assert late_segment["accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    assert late_segment["precision"] == pytest.approx(0.5, rel=1e-3)
    assert late_segment["recall"] == pytest.approx(0.5, rel=1e-3)
    assert late_segment["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert late_segment["average_course_number"] == pytest.approx(7.0, abs=1e-6)
    assert late_segment["min_course_number"] == 7
    assert late_segment["max_course_number"] == 7

    weekday_performance = metrics["weekday_performance"]
    weekday_labels = {
        0: ("monday", "Lundi"),
        1: ("tuesday", "Mardi"),
        2: ("wednesday", "Mercredi"),
        3: ("thursday", "Jeudi"),
        4: ("friday", "Vendredi"),
        5: ("saturday", "Samedi"),
        6: ("sunday", "Dimanche"),
    }

    previous_day = date.today() - timedelta(days=1)
    previous_key, previous_label = weekday_labels[previous_day.weekday()]
    today_key, today_label = weekday_labels[date.today().weekday()]

    assert set(weekday_performance.keys()) == {previous_key, today_key}

    previous_metrics = weekday_performance[previous_key]
    assert previous_metrics["label"] == previous_label
    assert previous_metrics["samples"] == 3
    assert previous_metrics["courses"] == 1
    assert previous_metrics["reunions"] == 1
    assert previous_metrics["share"] == pytest.approx(0.5, rel=1e-3)
    assert previous_metrics["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert previous_metrics["precision"] == pytest.approx(1.0, rel=1e-3)
    assert previous_metrics["recall"] == pytest.approx(1.0, rel=1e-3)
    assert previous_metrics["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert previous_metrics["weekday_index"] == previous_day.weekday()
    assert previous_metrics["first_date"] == previous_day.isoformat()
    assert previous_metrics["last_date"] == previous_day.isoformat()

    today_metrics = weekday_performance[today_key]
    assert today_metrics["label"] == today_label
    assert today_metrics["samples"] == 3
    assert today_metrics["courses"] == 1
    assert today_metrics["reunions"] == 1
    assert today_metrics["share"] == pytest.approx(0.5, rel=1e-3)
    assert today_metrics["accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    assert today_metrics["precision"] == pytest.approx(0.5, rel=1e-3)
    assert today_metrics["recall"] == pytest.approx(0.5, rel=1e-3)
    assert today_metrics["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert today_metrics["weekday_index"] == date.today().weekday()
    assert today_metrics["first_date"] == date.today().isoformat()
    assert today_metrics["last_date"] == date.today().isoformat()

    hippodrome_performance = metrics["hippodrome_performance"]
    assert len(hippodrome_performance) == 2
    hippodrome_map = {entry["label"]: entry for entry in hippodrome_performance}
    assert set(hippodrome_map.keys()) == {"Hippodrome Test", "Hippodrome Trot"}

    venue_entry = hippodrome_map["Hippodrome Test"]
    assert venue_entry["samples"] == 3
    assert venue_entry["courses"] == 1
    assert venue_entry["reunions"] == 1
    assert venue_entry["horses"] == 3
    assert venue_entry["accuracy"] == pytest.approx(1.0, rel=1e-3)

    trot_entry = hippodrome_map["Hippodrome Trot"]
    assert trot_entry["samples"] == 3
    assert trot_entry["courses"] == 1
    assert trot_entry["reunions"] == 1
    assert trot_entry["horses"] == 3
    assert trot_entry["accuracy"] == pytest.approx(1 / 3, rel=1e-3)

    country_performance = metrics["country_performance"]
    assert set(country_performance.keys()) == {"belgique", "france"}

    france_metrics = country_performance["france"]
    assert france_metrics["label"] == "France"
    assert france_metrics["samples"] == 3
    assert france_metrics["courses"] == 1
    assert france_metrics["reunions"] == 1
    assert france_metrics["hippodromes"] == 1
    assert france_metrics["share"] == pytest.approx(0.5, rel=1e-3)
    assert france_metrics["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert france_metrics["cities"] == ["Paris"]

    belgium_metrics = country_performance["belgique"]
    assert belgium_metrics["label"] == "Belgique"
    assert belgium_metrics["samples"] == 3
    assert belgium_metrics["courses"] == 1
    assert belgium_metrics["reunions"] == 1
    assert belgium_metrics["hippodromes"] == 1
    assert belgium_metrics["share"] == pytest.approx(0.5, rel=1e-3)
    assert belgium_metrics["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert belgium_metrics["cities"] == ["Mons"]

    city_performance = metrics["city_performance"]
    assert set(city_performance.keys()) == {"paris", "mons"}

    paris_metrics = city_performance["paris"]
    assert paris_metrics["label"] == "Paris"
    assert paris_metrics["samples"] == 3
    assert paris_metrics["courses"] == 1
    assert paris_metrics["reunions"] == 1
    assert paris_metrics["hippodromes"] == 1
    assert paris_metrics["share"] == pytest.approx(0.5, rel=1e-3)
    assert paris_metrics["countries"] == ["France"]

    mons_metrics = city_performance["mons"]
    assert mons_metrics["label"] == "Mons"
    assert mons_metrics["samples"] == 3
    assert mons_metrics["courses"] == 1
    assert mons_metrics["reunions"] == 1
    assert mons_metrics["hippodromes"] == 1
    assert mons_metrics["share"] == pytest.approx(0.5, rel=1e-3)
    assert mons_metrics["countries"] == ["Belgique"]

    discipline_performance = metrics["discipline_performance"]
    assert set(discipline_performance.keys()) == {"plat", "trot_attele"}
    assert discipline_performance["plat"]["samples"] == 3
    assert discipline_performance["plat"]["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert discipline_performance["plat"]["courses"] == 1
    assert discipline_performance["trot_attele"]["samples"] == 3
    assert discipline_performance["trot_attele"]["accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    assert discipline_performance["trot_attele"]["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)

    distance_performance = metrics["distance_performance"]
    assert set(distance_performance.keys()) == {"long_distance", "short_distance"}
    assert distance_performance["short_distance"]["samples"] == 3
    assert distance_performance["short_distance"]["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert distance_performance["short_distance"]["average_distance"] == pytest.approx(1400.0, abs=1e-6)
    assert distance_performance["long_distance"]["samples"] == 3
    assert distance_performance["long_distance"]["precision"] == pytest.approx(0.5, rel=1e-3)
    assert distance_performance["long_distance"]["average_distance"] == pytest.approx(3000.0, abs=1e-6)

    surface_performance = metrics["surface_performance"]
    assert set(surface_performance.keys()) == {"pelouse", "sable"}
    assert surface_performance["pelouse"]["samples"] == 3
    assert surface_performance["pelouse"]["positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert surface_performance["sable"]["samples"] == 3
    assert surface_performance["sable"]["precision"] == pytest.approx(0.5, rel=1e-3)

    weather_performance = metrics["weather_performance"]
    assert set(weather_performance.keys()) == {"clear", "rain"}
    assert weather_performance["clear"]["label"] == "Conditions claires"
    assert weather_performance["clear"]["courses"] == 1
    assert weather_performance["clear"]["reunions"] == 1
    assert weather_performance["clear"]["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert weather_performance["clear"]["average_temperature"] == pytest.approx(18.0, rel=1e-3)
    assert weather_performance["rain"]["label"] == "Pluie / Averses"
    assert weather_performance["rain"]["courses"] == 1
    assert weather_performance["rain"]["reunions"] == 1
    assert weather_performance["rain"]["average_temperature"] == pytest.approx(9.0, rel=1e-3)

    track_type_performance = metrics["track_type_performance"]
    assert set(track_type_performance.keys()) == {"flat", "trot"}
    assert track_type_performance["flat"]["label"] == "Piste plate"
    assert track_type_performance["flat"]["samples"] == 3
    assert track_type_performance["flat"]["share"] == pytest.approx(0.5, rel=1e-3)
    assert track_type_performance["trot"]["label"] == "Piste de trot"
    assert track_type_performance["trot"]["samples"] == 3
    assert track_type_performance["trot"]["reunions"] == 1
    assert track_type_performance["trot"]["hippodromes"] == 1

    prize_money_performance = metrics["prize_money_performance"]
    assert set(prize_money_performance.keys()) == {"low_prize", "medium_prize"}
    assert prize_money_performance["low_prize"]["samples"] == 3
    assert prize_money_performance["low_prize"]["accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    assert prize_money_performance["low_prize"]["average_prize_eur"] == pytest.approx(
        8000.0, abs=1e-6
    )
    assert prize_money_performance["medium_prize"]["samples"] == 3
    assert prize_money_performance["medium_prize"]["accuracy"] == pytest.approx(
        1.0, rel=1e-3
    )
    assert prize_money_performance["medium_prize"]["average_prize_eur"] == pytest.approx(
        10000.0, abs=1e-6
    )

    handicap_performance = metrics["handicap_performance"]
    assert set(handicap_performance.keys()) == {
        "competitive_handicap",
        "high_handicap",
        "light_handicap",
        "medium_handicap",
        "unknown",
    }

    light_segment = handicap_performance["light_handicap"]
    assert light_segment["label"] == "Handicap léger (≤10)"
    assert light_segment["samples"] == 1
    assert light_segment["courses"] == 1
    assert light_segment["horses"] == 1
    assert light_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert light_segment["precision"] == pytest.approx(1.0, rel=1e-3)
    assert light_segment["recall"] == pytest.approx(1.0, rel=1e-3)
    assert light_segment["observed_positive_rate"] == pytest.approx(1.0, rel=1e-3)
    assert light_segment["average_handicap_value"] == pytest.approx(8.0, rel=1e-3)
    assert light_segment["share"] == pytest.approx(1 / 6, rel=1e-3)

    medium_segment = handicap_performance["medium_handicap"]
    assert medium_segment["label"] == "Handicap moyen (11-20)"
    assert medium_segment["samples"] == 2
    assert medium_segment["courses"] == 2
    assert medium_segment["horses"] == 2
    assert medium_segment["accuracy"] == pytest.approx(0.5, rel=1e-3)
    assert medium_segment["precision"] == pytest.approx(0.5, rel=1e-3)
    assert medium_segment["recall"] == pytest.approx(1.0, rel=1e-3)
    assert medium_segment["observed_positive_rate"] == pytest.approx(0.5, rel=1e-3)
    assert medium_segment["average_handicap_value"] == pytest.approx(14.0, rel=1e-3)
    assert medium_segment["min_handicap_value"] == pytest.approx(12.0, rel=1e-3)
    assert medium_segment["max_handicap_value"] == pytest.approx(16.0, rel=1e-3)
    assert medium_segment["share"] == pytest.approx(2 / 6, rel=1e-3)

    competitive_segment = handicap_performance["competitive_handicap"]
    assert competitive_segment["label"] == "Handicap relevé (21-30)"
    assert competitive_segment["samples"] == 1
    assert competitive_segment["courses"] == 1
    assert competitive_segment["horses"] == 1
    assert competitive_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert competitive_segment["precision"] == pytest.approx(0.0, abs=1e-6)
    assert competitive_segment["recall"] == pytest.approx(0.0, abs=1e-6)
    assert competitive_segment["observed_positive_rate"] == pytest.approx(0.0, abs=1e-6)
    assert competitive_segment["average_handicap_value"] == pytest.approx(26.0, rel=1e-3)
    assert competitive_segment["share"] == pytest.approx(1 / 6, rel=1e-3)

    high_segment = handicap_performance["high_handicap"]
    assert high_segment["label"] == "Handicap très élevé (>30)"
    assert high_segment["samples"] == 1
    assert high_segment["courses"] == 1
    assert high_segment["horses"] == 1
    assert high_segment["accuracy"] == pytest.approx(0.0, abs=1e-6)
    assert high_segment["precision"] == pytest.approx(0.0, abs=1e-6)
    assert high_segment["recall"] == pytest.approx(0.0, abs=1e-6)
    assert high_segment["observed_positive_rate"] == pytest.approx(1.0, rel=1e-3)
    assert high_segment["average_handicap_value"] == pytest.approx(34.0, rel=1e-3)
    assert high_segment["share"] == pytest.approx(1 / 6, rel=1e-3)

    unknown_segment = handicap_performance["unknown"]
    assert unknown_segment["label"] == "Handicap inconnu"
    assert unknown_segment["samples"] == 1
    assert unknown_segment["courses"] == 1
    assert unknown_segment["horses"] == 1
    assert unknown_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert unknown_segment["precision"] == pytest.approx(1.0, rel=1e-3)
    assert unknown_segment["recall"] == pytest.approx(1.0, rel=1e-3)
    assert unknown_segment["observed_positive_rate"] == pytest.approx(1.0, rel=1e-3)
    assert unknown_segment["average_handicap_value"] is None
    assert unknown_segment["share"] == pytest.approx(1 / 6, rel=1e-3)

    weight_performance = metrics["weight_performance"]
    assert set(weight_performance.keys()) == {
        "heavy",
        "light",
        "medium",
        "unknown",
        "very_light",
    }

    heavy_weight_segment = weight_performance["heavy"]
    assert heavy_weight_segment["label"] == "Lourd (≥60 kg)"
    assert heavy_weight_segment["samples"] == 2
    assert heavy_weight_segment["horses"] == 2
    assert heavy_weight_segment["average_weight"] == pytest.approx(61.0, rel=1e-3)
    assert heavy_weight_segment["min_weight"] == pytest.approx(60.0, rel=1e-3)
    assert heavy_weight_segment["max_weight"] == pytest.approx(62.0, rel=1e-3)

    light_weight_segment = weight_performance["light"]
    assert light_weight_segment["label"] == "Léger (54-57 kg)"
    assert light_weight_segment["samples"] == 1
    assert light_weight_segment["average_weight"] == pytest.approx(55.0, rel=1e-3)

    very_light_weight_segment = weight_performance["very_light"]
    assert very_light_weight_segment["label"] == "Très léger (<54 kg)"
    assert very_light_weight_segment["samples"] == 1
    assert very_light_weight_segment["average_weight"] == pytest.approx(52.5, rel=1e-3)

    medium_weight_segment = weight_performance["medium"]
    assert medium_weight_segment["label"] == "Moyen (57-60 kg)"
    assert medium_weight_segment["samples"] == 1
    assert medium_weight_segment["average_weight"] == pytest.approx(58.5, rel=1e-3)

    unknown_weight_segment = weight_performance["unknown"]
    assert unknown_weight_segment["label"] == "Poids inconnu"
    assert unknown_weight_segment["samples"] == 1
    assert unknown_weight_segment["average_weight"] is None

    equipment_performance = metrics["equipment_performance"]
    assert set(equipment_performance.keys()) == {
        "blinkers",
        "multi_gear",
        "no_equipment",
        "single_gear",
        "unknown",
    }

    blinkers_segment = equipment_performance["blinkers"]
    assert blinkers_segment["label"] == "Œillères déclarées"
    assert blinkers_segment["samples"] == 2
    assert blinkers_segment["courses"] == 2
    assert blinkers_segment["horses"] == 2
    assert blinkers_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert blinkers_segment["observed_positive_rate"] == pytest.approx(1.0, rel=1e-3)
    assert blinkers_segment["share"] == pytest.approx(2 / 6, rel=1e-3)
    assert blinkers_segment["average_equipment_items"] == pytest.approx(1.0, rel=1e-3)
    assert blinkers_segment["blinkers_rate"] == pytest.approx(1.0, rel=1e-3)

    single_gear_segment = equipment_performance["single_gear"]
    assert single_gear_segment["label"] == "Équipement isolé"
    assert single_gear_segment["samples"] == 1
    assert single_gear_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert single_gear_segment["average_equipment_items"] == pytest.approx(1.0, rel=1e-3)
    assert single_gear_segment["blinkers_rate"] == pytest.approx(0.0, abs=1e-6)

    multi_gear_segment = equipment_performance["multi_gear"]
    assert multi_gear_segment["label"] == "Équipement combiné (≥2)"
    assert multi_gear_segment["samples"] == 1
    assert multi_gear_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert multi_gear_segment["average_equipment_items"] == pytest.approx(2.0, rel=1e-3)
    assert multi_gear_segment["blinkers_rate"] == pytest.approx(0.0, abs=1e-6)

    no_equipment_segment = equipment_performance["no_equipment"]
    assert no_equipment_segment["label"] == "Aucun équipement déclaré"
    assert no_equipment_segment["samples"] == 1
    assert no_equipment_segment["accuracy"] == pytest.approx(0.0, abs=1e-6)
    assert no_equipment_segment["average_equipment_items"] == pytest.approx(0.0, abs=1e-6)
    assert no_equipment_segment["blinkers_rate"] == pytest.approx(0.0, abs=1e-6)

    unknown_equipment_segment = equipment_performance["unknown"]
    assert unknown_equipment_segment["label"] == "Équipement inconnu"
    assert unknown_equipment_segment["samples"] == 1
    assert unknown_equipment_segment["accuracy"] == pytest.approx(0.0, abs=1e-6)
    assert unknown_equipment_segment["average_equipment_items"] is None
    assert unknown_equipment_segment["blinkers_rate"] is None

    odds_band_performance = metrics["odds_band_performance"]
    assert set(odds_band_performance.keys()) == {"challenger", "favorite", "outsider"}

    favorite_segment = odds_band_performance["favorite"]
    assert favorite_segment["label"] == "Favori (≤4/1)"
    assert favorite_segment["samples"] == 2
    assert favorite_segment["courses"] == 1
    assert favorite_segment["horses"] == 2
    assert favorite_segment["share"] == pytest.approx(2 / 6, rel=1e-3)
    assert favorite_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert favorite_segment["precision"] == pytest.approx(1.0, rel=1e-3)
    assert favorite_segment["recall"] == pytest.approx(1.0, rel=1e-3)
    assert favorite_segment["average_odds"] == pytest.approx(3.5, rel=1e-3)
    assert favorite_segment["average_implied_probability"] == pytest.approx(0.291666, rel=1e-3)

    challenger_segment = odds_band_performance["challenger"]
    assert challenger_segment["label"] == "Challenger (4-8/1)"
    assert challenger_segment["samples"] == 3
    assert challenger_segment["courses"] == 1
    assert challenger_segment["horses"] == 3
    assert challenger_segment["share"] == pytest.approx(0.5, rel=1e-3)
    assert challenger_segment["accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    assert challenger_segment["precision"] == pytest.approx(0.5, rel=1e-3)
    assert challenger_segment["recall"] == pytest.approx(0.5, rel=1e-3)
    assert challenger_segment["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert challenger_segment["average_odds"] == pytest.approx(6.166666, rel=1e-3)
    assert challenger_segment["min_odds"] == pytest.approx(5.5, rel=1e-3)
    assert challenger_segment["max_odds"] == pytest.approx(7.0, rel=1e-3)
    assert challenger_segment["average_implied_probability"] == pytest.approx(0.163780, rel=1e-3)

    outsider_segment = odds_band_performance["outsider"]
    assert outsider_segment["label"] == "Outsider (8-15/1)"
    assert outsider_segment["samples"] == 1
    assert outsider_segment["courses"] == 1
    assert outsider_segment["horses"] == 1
    assert outsider_segment["share"] == pytest.approx(1 / 6, rel=1e-3)
    assert outsider_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert outsider_segment["precision"] == pytest.approx(0.0, abs=1e-6)
    assert outsider_segment["recall"] == pytest.approx(0.0, abs=1e-6)
    assert outsider_segment["observed_positive_rate"] == pytest.approx(0.0, abs=1e-6)
    assert outsider_segment["average_odds"] == pytest.approx(12.0, rel=1e-3)
    assert outsider_segment["average_implied_probability"] == pytest.approx(1 / 12, rel=1e-3)

    horse_age_performance = metrics["horse_age_performance"]
    assert set(horse_age_performance.keys()) == {
        "experienced",
        "juvenile",
        "prime",
        "senior",
        "unknown",
    }
    assert horse_age_performance["prime"]["samples"] == 2
    assert horse_age_performance["prime"]["courses"] == 1
    assert horse_age_performance["prime"]["horses"] == 2
    assert horse_age_performance["prime"]["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert horse_age_performance["prime"]["average_age"] == pytest.approx(4.5, rel=1e-3)
    assert horse_age_performance["juvenile"]["samples"] == 1
    assert horse_age_performance["juvenile"]["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert horse_age_performance["experienced"]["samples"] == 1
    assert horse_age_performance["experienced"]["recall"] == pytest.approx(0.0, abs=1e-6)
    assert horse_age_performance["senior"]["samples"] == 1
    assert horse_age_performance["senior"]["precision"] == pytest.approx(0.0, abs=1e-6)
    assert horse_age_performance["unknown"]["samples"] == 1
    assert horse_age_performance["unknown"]["average_age"] is None

    horse_gender_performance = metrics["horse_gender_performance"]
    assert set(horse_gender_performance.keys()) == {"female", "male"}
    assert horse_gender_performance["male"]["label"] == "Mâle"
    assert horse_gender_performance["male"]["samples"] == 4
    assert horse_gender_performance["male"]["horses"] == 4
    assert horse_gender_performance["male"]["courses"] == 2
    assert horse_gender_performance["male"]["accuracy"] == pytest.approx(0.75, rel=1e-3)
    assert horse_gender_performance["male"]["recall"] == pytest.approx(2 / 3, rel=1e-3)
    assert horse_gender_performance["male"]["observed_positive_rate"] == pytest.approx(3 / 4, rel=1e-3)
    assert horse_gender_performance["male"]["share"] == pytest.approx(4 / 6, rel=1e-3)
    assert horse_gender_performance["female"]["label"] == "Femelle"
    assert horse_gender_performance["female"]["samples"] == 2
    assert horse_gender_performance["female"]["horses"] == 2
    assert horse_gender_performance["female"]["courses"] == 2
    assert horse_gender_performance["female"]["precision"] == pytest.approx(0.5, rel=1e-3)
    assert horse_gender_performance["female"]["recall"] == pytest.approx(1.0, rel=1e-3)
    assert horse_gender_performance["female"]["observed_positive_rate"] == pytest.approx(0.5, rel=1e-3)
    assert horse_gender_performance["female"]["share"] == pytest.approx(2 / 6, rel=1e-3)

    owner_performance = metrics["owner_performance"]
    assert set(owner_performance.keys()) == {
        "ecurie_boreale",
        "ecurie_equinoxe",
        "ecurie_horizon",
        "unknown",
    }

    horizon_segment = owner_performance["ecurie_horizon"]
    assert horizon_segment["label"] == "Ecurie Horizon"
    assert horizon_segment["samples"] == 2
    assert horizon_segment["courses"] == 1
    assert horizon_segment["horses"] == 2
    assert horizon_segment["trainers"] == 1
    assert horizon_segment["jockeys"] == 1
    assert horizon_segment["hippodromes"] == 1
    assert horizon_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert horizon_segment["observed_positive_rate"] == pytest.approx(1.0, rel=1e-3)

    equinoxe_segment = owner_performance["ecurie_equinoxe"]
    assert equinoxe_segment["label"] == "Ecurie Equinoxe"
    assert equinoxe_segment["samples"] == 2
    assert equinoxe_segment["courses"] == 2
    assert equinoxe_segment["horses"] == 2
    assert equinoxe_segment["trainers"] == 2
    assert equinoxe_segment["jockeys"] == 2
    assert equinoxe_segment["hippodromes"] == 2
    assert equinoxe_segment["accuracy"] == pytest.approx(0.5, rel=1e-3)
    assert equinoxe_segment["observed_positive_rate"] == pytest.approx(0.5, rel=1e-3)

    boreale_segment = owner_performance["ecurie_boreale"]
    assert boreale_segment["label"] == "Ecurie Boreale"
    assert boreale_segment["samples"] == 1
    assert boreale_segment["courses"] == 1
    assert boreale_segment["accuracy"] == pytest.approx(0.0, abs=1e-6)
    assert boreale_segment["observed_positive_rate"] == pytest.approx(0.0, abs=1e-6)

    unknown_segment = owner_performance["unknown"]
    assert unknown_segment["label"] == "Propriétaire inconnu"
    assert unknown_segment["samples"] == 1
    assert unknown_segment["courses"] == 1
    assert unknown_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert unknown_segment["observed_positive_rate"] == pytest.approx(1.0, rel=1e-3)

    race_category_performance = metrics["race_category_performance"]
    assert set(race_category_performance.keys()) == {"classe", "groupe"}
    assert race_category_performance["groupe"]["samples"] == 3
    assert race_category_performance["groupe"]["courses"] == 1
    assert race_category_performance["groupe"]["label"] == "Groupe"
    assert race_category_performance["classe"]["samples"] == 3
    assert race_category_performance["classe"]["precision"] == pytest.approx(0.5, rel=1e-3)

    race_class_performance = metrics["race_class_performance"]
    assert set(race_class_performance.keys()) == {"class_a", "class_b"}
    assert race_class_performance["class_a"]["label"] == "Classe A"
    assert race_class_performance["class_a"]["courses"] == 1
    assert race_class_performance["class_b"]["samples"] == 3
    assert race_class_performance["class_b"]["observed_positive_rate"] == pytest.approx(
        2 / 3, rel=1e-3
    )

    start_type_performance = metrics["start_type_performance"]
    assert set(start_type_performance.keys()) == {"autostart", "stalle"}
    assert start_type_performance["stalle"]["samples"] == 3
    assert start_type_performance["stalle"]["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert start_type_performance["stalle"]["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert start_type_performance["autostart"]["samples"] == 3
    assert start_type_performance["autostart"]["accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    assert start_type_performance["autostart"]["precision"] == pytest.approx(0.5, rel=1e-3)
    assert start_type_performance["autostart"]["courses"] == 1

    value_bet_performance = metrics["value_bet_performance"]
    assert set(value_bet_performance.keys()) == {"standard", "value_bet"}
    assert value_bet_performance["value_bet"]["samples"] == 3
    assert value_bet_performance["value_bet"]["courses"] == 1
    assert value_bet_performance["standard"]["samples"] == 3
    assert value_bet_performance["standard"]["observed_positive_rate"] == pytest.approx(2 / 3, rel=1e-3)

    field_size_performance = metrics["field_size_performance"]
    assert set(field_size_performance.keys()) == {"large_field", "small_field"}
    assert field_size_performance["small_field"]["samples"] == 3
    assert field_size_performance["small_field"]["average_field_size"] == pytest.approx(8.0, abs=1e-6)
    assert field_size_performance["large_field"]["samples"] == 3
    assert field_size_performance["large_field"]["max_field_size"] == 14

    draw_performance = metrics["draw_performance"]
    assert set(draw_performance.keys()) == {"inside", "middle", "outside"}
    assert draw_performance["inside"]["samples"] == 2
    assert draw_performance["inside"]["accuracy"] == pytest.approx(0.5, rel=1e-3)
    assert draw_performance["inside"]["average_draw"] == pytest.approx(1.5, abs=1e-6)
    assert draw_performance["inside"]["average_field_size"] == pytest.approx(11.0, abs=1e-6)
    assert draw_performance["middle"]["courses"] == 2
    assert draw_performance["middle"]["precision"] == pytest.approx(0.5, rel=1e-3)
    assert draw_performance["outside"]["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert draw_performance["outside"]["observed_positive_rate"] == pytest.approx(0.5, rel=1e-3)

    rest_period_performance = metrics["rest_period_performance"]
    assert set(rest_period_performance.keys()) == {
        "extended_break",
        "fresh",
        "normal_cycle",
        "unknown",
        "very_fresh",
    }
    assert rest_period_performance["very_fresh"]["samples"] == 1
    assert rest_period_performance["very_fresh"]["average_rest_days"] == pytest.approx(10.0, abs=1e-6)
    assert rest_period_performance["fresh"]["samples"] == 1
    assert rest_period_performance["fresh"]["precision"] == pytest.approx(1.0, rel=1e-3)
    assert rest_period_performance["normal_cycle"]["samples"] == 2
    assert rest_period_performance["normal_cycle"]["average_rest_days"] == pytest.approx(60.0, abs=1e-6)
    assert rest_period_performance["normal_cycle"]["min_rest_days"] == 45
    assert rest_period_performance["extended_break"]["samples"] == 1
    assert rest_period_performance["extended_break"]["average_rest_days"] == pytest.approx(210.0, abs=1e-6)
    assert rest_period_performance["unknown"]["samples"] == 1
    assert rest_period_performance["unknown"]["average_rest_days"] is None

    version_performance = metrics["model_version_performance"]
    assert set(version_performance.keys()) == {"v1.0", "v2.0"}

    assert version_performance["v1.0"]["samples"] == 3
    assert version_performance["v1.0"]["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert version_performance["v1.0"]["share"] == pytest.approx(0.5, rel=1e-3)
    assert version_performance["v1.0"]["confidence_distribution"] == {
        "high": 1,
        "medium": 1,
        "low": 1,
    }

    assert version_performance["v2.0"]["samples"] == 3
    assert version_performance["v2.0"]["precision"] == pytest.approx(0.5, rel=1e-3)
    assert version_performance["v2.0"]["recall"] == pytest.approx(0.5, rel=1e-3)
    assert version_performance["v2.0"]["share"] == pytest.approx(0.5, rel=1e-3)
    assert version_performance["v2.0"]["confidence_distribution"] == {
        "low": 1,
        "medium": 2,
    }

    rank_performance = metrics["prediction_rank_performance"]
    assert set(rank_performance.keys()) == {"rank_1", "rank_2", "rank_3"}

    top_rank_segment = rank_performance["rank_1"]
    assert top_rank_segment["label"] == "Sélection prioritaire (rang 1)"
    assert top_rank_segment["samples"] == 2
    assert top_rank_segment["courses"] == 2
    assert top_rank_segment["share"] == pytest.approx(1 / 3, rel=1e-3)
    assert top_rank_segment["accuracy"] == pytest.approx(0.5, rel=1e-3)
    assert top_rank_segment["observed_positive_rate"] == pytest.approx(0.5, rel=1e-3)
    assert top_rank_segment["average_final_position"] == pytest.approx(2.5, rel=1e-3)
    assert top_rank_segment["average_rank"] == pytest.approx(1.0, rel=1e-3)

    second_rank_segment = rank_performance["rank_2"]
    assert second_rank_segment["samples"] == 2
    assert second_rank_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert second_rank_segment["observed_positive_rate"] == pytest.approx(1.0, rel=1e-3)
    assert second_rank_segment["average_final_position"] == pytest.approx(2.0, rel=1e-3)
    assert second_rank_segment["average_rank"] == pytest.approx(2.0, rel=1e-3)

    third_rank_segment = rank_performance["rank_3"]
    assert third_rank_segment["samples"] == 2
    assert third_rank_segment["accuracy"] == pytest.approx(0.5, rel=1e-3)
    assert third_rank_segment["observed_positive_rate"] == pytest.approx(0.5, rel=1e-3)
    assert third_rank_segment["average_rank"] == pytest.approx(3.0, rel=1e-3)
    assert third_rank_segment["best_final_position"] == 1
    assert third_rank_segment["worst_final_position"] == 4

    jockey_leaderboard = metrics["jockey_performance"]
    assert len(jockey_leaderboard) == 2
    assert {entry["label"] for entry in jockey_leaderboard} == {
        "Leo Martin",
        "Noah Verbeeck",
    }
    top_jockey = jockey_leaderboard[0]
    assert top_jockey["label"] == "Leo Martin"
    assert top_jockey["samples"] == 3
    assert top_jockey["courses"] == 1
    assert top_jockey["horses"] == 3
    assert top_jockey["share"] == pytest.approx(0.5, rel=1e-3)
    assert top_jockey["accuracy"] == pytest.approx(1.0, rel=1e-3)

    second_jockey = jockey_leaderboard[1]
    assert second_jockey["label"] == "Noah Verbeeck"
    assert second_jockey["samples"] == 3
    assert second_jockey["courses"] == 1
    assert second_jockey["horses"] == 3
    assert second_jockey["share"] == pytest.approx(0.5, rel=1e-3)
    assert second_jockey["precision"] == pytest.approx(0.5, rel=1e-3)

    trainer_leaderboard = metrics["trainer_performance"]
    assert len(trainer_leaderboard) == 2
    assert {entry["label"] for entry in trainer_leaderboard} == {
        "Anne Durand",
        "Marc Dupont",
    }
    top_trainer = trainer_leaderboard[0]
    assert top_trainer["label"] == "Anne Durand"
    assert top_trainer["samples"] == 3
    assert top_trainer["courses"] == 1
    assert top_trainer["horses"] == 3
    assert top_trainer["share"] == pytest.approx(0.5, rel=1e-3)
    assert top_trainer["precision"] == pytest.approx(1.0, rel=1e-3)

    second_trainer = trainer_leaderboard[1]
    assert second_trainer["label"] == "Marc Dupont"
    assert second_trainer["samples"] == 3
    assert second_trainer["courses"] == 1
    assert second_trainer["horses"] == 3
    assert second_trainer["share"] == pytest.approx(0.5, rel=1e-3)
    assert second_trainer["recall"] == pytest.approx(0.5, rel=1e-3)

    assert set(result["prediction_rank_performance"].keys()) == {"rank_1", "rank_2", "rank_3"}
    assert result["prediction_rank_performance"]["rank_2"]["average_rank"] == pytest.approx(2.0, rel=1e-3)

    assert result["jockey_performance"][0]["label"] == "Leo Martin"
    assert result["jockey_performance"][1]["label"] == "Noah Verbeeck"
    assert result["trainer_performance"][0]["label"] == "Anne Durand"
    assert result["trainer_performance"][1]["label"] == "Marc Dupont"
    assert set(result["jockey_nationality_performance"].keys()) == {"france", "belgique"}
    assert set(result["trainer_nationality_performance"].keys()) == {"france", "belgique"}
    hippodrome_labels = {entry["label"] for entry in result["hippodrome_performance"]}
    assert hippodrome_labels == {"Hippodrome Test", "Hippodrome Trot"}
    assert set(result["country_performance"].keys()) == {"belgique", "france"}
    assert result["country_performance"]["france"]["cities"] == ["Paris"]
    assert set(result["city_performance"].keys()) == {"paris", "mons"}
    assert result["city_performance"]["paris"]["countries"] == ["France"]
    assert result["track_type_performance"]["flat"]["label"] == "Piste plate"
    assert result["weather_performance"]["clear"]["label"] == "Conditions claires"
    assert result["day_part_performance"]["afternoon"]["samples"] == 3
    assert date.today().strftime("%Y-%m") in result["month_performance"]
    assert result["horse_age_performance"]["prime"]["samples"] == 2
    assert result["horse_gender_performance"]["male"]["samples"] == 4
    assert result["recent_form_performance"]["recent_winner"]["label"] == "Gagnant récent"
    assert result["recent_form_performance"]["recent_winner"]["samples"] == 1
    assert result["owner_performance"]["ecurie_horizon"]["samples"] == 2
    recent_form_performance = metrics["recent_form_performance"]
    assert set(recent_form_performance.keys()) == {
        "recent_winner",
        "strong_form",
        "steady_form",
        "inconsistent_form",
        "poor_form",
        "unknown",
    }

    winner_segment = recent_form_performance["recent_winner"]
    assert winner_segment["label"] == "Gagnant récent"
    assert winner_segment["samples"] == 1
    assert winner_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert winner_segment["recent_win_rate"] == pytest.approx(1.0, rel=1e-3)
    assert winner_segment["average_recent_position"] == pytest.approx(4 / 3, rel=1e-3)
    assert winner_segment["best_recent_position"] == 1
    assert winner_segment["average_recent_history_size"] == pytest.approx(3.0, rel=1e-3)

    strong_segment = recent_form_performance["strong_form"]
    assert strong_segment["label"] == "Forme solide (moyenne ≤3)"
    assert strong_segment["samples"] == 1
    assert strong_segment["accuracy"] == pytest.approx(0.0, rel=1e-3)
    assert strong_segment["recent_win_rate"] == pytest.approx(0.0, rel=1e-3)
    assert strong_segment["average_recent_position"] == pytest.approx(7 / 3, rel=1e-3)
    assert strong_segment["share"] == pytest.approx(1 / 6, rel=1e-3)

    steady_segment = recent_form_performance["steady_form"]
    assert steady_segment["label"] == "Forme régulière (moyenne 3-5)"
    assert steady_segment["samples"] == 1
    assert steady_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert steady_segment["recent_win_rate"] == pytest.approx(0.0, rel=1e-3)
    assert steady_segment["average_recent_position"] == pytest.approx(4.0, rel=1e-3)

    inconsistent_segment = recent_form_performance["inconsistent_form"]
    assert inconsistent_segment["label"] == "Forme irrégulière (moyenne 5-8)"
    assert inconsistent_segment["samples"] == 1
    assert inconsistent_segment["accuracy"] == pytest.approx(1.0, rel=1e-3)
    assert inconsistent_segment["observed_positive_rate"] == pytest.approx(0.0, rel=1e-3)
    assert inconsistent_segment["average_recent_position"] == pytest.approx(7.0, rel=1e-3)

    poor_segment = recent_form_performance["poor_form"]
    assert poor_segment["label"] == "Forme en difficulté (>8)"
    assert poor_segment["samples"] == 1
    assert poor_segment["accuracy"] == pytest.approx(0.0, rel=1e-3)
    assert poor_segment["observed_positive_rate"] == pytest.approx(0.0, rel=1e-3)
    assert poor_segment["average_recent_position"] == pytest.approx(10.0, rel=1e-3)

    unknown_segment = recent_form_performance["unknown"]
    assert unknown_segment["label"] == "Forme inconnue"
    assert unknown_segment["samples"] == 1
    assert unknown_segment["average_recent_position"] is None
    assert unknown_segment["recent_win_rate"] is None
    assert unknown_segment["share"] == pytest.approx(1 / 6, rel=1e-3)

    assert result["handicap_performance"]["medium_handicap"]["samples"] == 2
    assert result["odds_band_performance"]["favorite"]["samples"] == 2
    assert result["equipment_performance"]["blinkers"]["samples"] == 2
    assert result["race_order_performance"]["early_card"]["samples"] == 3
    assert result["race_order_performance"]["late_card"]["average_course_number"] == pytest.approx(7.0, abs=1e-6)
    assert result["confidence_score_performance"]["medium"]["label"] == "Confiance moyenne (50-70%)"
    with in_memory_session() as check_session:
        stored_model = check_session.query(MLModel).filter(MLModel.is_active.is_(True)).one()
        assert float(stored_model.accuracy) == pytest.approx(2 / 3, rel=1e-3)

        stored_metrics = stored_model.performance_metrics
        if isinstance(stored_metrics, str):
            stored_metrics = json.loads(stored_metrics)

        assert stored_metrics["last_evaluation"]["metrics"]["accuracy"] == pytest.approx(2 / 3, rel=1e-3)
        assert "confidence_level_metrics" in stored_metrics["last_evaluation"]
        assert "confidence_score_performance" in stored_metrics["last_evaluation"]
        assert "gain_curve" in stored_metrics["last_evaluation"]["metrics"]
        assert "lift_analysis" in stored_metrics["last_evaluation"]["metrics"]
        assert "betting_value_analysis" in stored_metrics["last_evaluation"]["metrics"]
        assert "owner_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "odds_alignment" in stored_metrics["last_evaluation"]["metrics"]
        assert "precision_recall_curve" in stored_metrics["last_evaluation"]["metrics"]
        assert "roc_curve" in stored_metrics["last_evaluation"]["metrics"]
        assert "ks_analysis" in stored_metrics["last_evaluation"]["metrics"]
        assert "calibration_diagnostics" in stored_metrics["last_evaluation"]
        assert "calibration_diagnostics" in stored_metrics["last_evaluation"]["metrics"]
        assert "threshold_recommendations" in stored_metrics["last_evaluation"]
        assert "threshold_recommendations" in stored_metrics["last_evaluation"]["metrics"]
        assert stored_metrics["last_evaluation"]["threshold_recommendations"]["best_f1"]["threshold"] == pytest.approx(best_f1_threshold, rel=1e-3)
        assert "daily_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "day_part_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "confidence_score_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "month_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "recent_form_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "weekday_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "discipline_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "distance_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "surface_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "weather_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "hippodrome_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "track_type_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "country_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "city_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "odds_band_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "race_order_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "prize_money_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "handicap_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "weight_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "equipment_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "horse_age_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "horse_gender_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "race_category_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "race_class_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "value_bet_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "field_size_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "draw_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "start_type_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "rest_period_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "model_version_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "prediction_rank_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "jockey_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "trainer_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "jockey_nationality_performance" in stored_metrics["last_evaluation"]["metrics"]
        assert "trainer_nationality_performance" in stored_metrics["last_evaluation"]["metrics"]


def test_update_model_performance_without_predictions(in_memory_session: sessionmaker) -> None:
    """Retourne un statut explicite lorsqu'aucune donnée n'est disponible."""

    session = in_memory_session()
    session.add(
        MLModel(
            model_name="horse_racing_gradient_boosting",
            version="20250101",
            algorithm="GradientBoosting",
            file_path="model.pkl",
            is_active=True,
        )
    )
    session.commit()
    session.close()

    result = ml_tasks.update_model_performance.run(days_back=7)

    assert result == {
        "status": "no_data",
        "days_evaluated": 7,
        "cutoff_date": (date.today() - timedelta(days=7)).isoformat(),
        "evaluated_samples": 0,
        "message": "No predictions with associated race results in the given window",
    }
