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

    hippodrome = Hippodrome(
        code="TEST",
        name="Hippodrome Test",
        track_type=TrackType.PLAT,
    )
    db.add(hippodrome)
    db.flush()

    reunion = Reunion(
        hippodrome_id=hippodrome.hippodrome_id,
        reunion_date=date.today(),
        reunion_number=1,
        status=ReunionStatus.COMPLETED,
    )
    db.add(reunion)
    db.flush()

    trainer = Trainer(first_name="Anne", last_name="Durand")
    jockey = Jockey(first_name="Leo", last_name="Martin")
    db.add_all([trainer, jockey])
    db.flush()

    horses = [
        Horse(name="Cheval A", gender=Gender.MALE),
        Horse(name="Cheval B", gender=Gender.FEMALE),
        Horse(name="Cheval C", gender=Gender.MALE),
        Horse(name="Cheval D", gender=Gender.MALE),
        Horse(name="Cheval E", gender=Gender.FEMALE),
        Horse(name="Cheval F", gender=Gender.MALE),
    ]
    db.add_all(horses)
    db.flush()

    course1 = Course(
        reunion_id=reunion.reunion_id,
        course_number=1,
        course_name="R1C1",
        discipline=Discipline.PLAT,
        distance=2000,
        prize_money=Decimal("10000"),
        race_category="Groupe",
        race_class="A",
        surface_type=SurfaceType.PELOUSE,
        start_type=StartType.STALLE,
        scheduled_time=time(14, 0),
        status=CourseStatus.FINISHED,
        number_of_runners=3,
    )
    course2 = Course(
        reunion_id=reunion.reunion_id,
        course_number=2,
        course_name="R1C2",
        discipline=Discipline.PLAT,
        distance=1800,
        prize_money=Decimal("8000"),
        race_category="Classe",
        race_class="B",
        surface_type=SurfaceType.PELOUSE,
        start_type=StartType.STALLE,
        scheduled_time=time(15, 0),
        status=CourseStatus.FINISHED,
        number_of_runners=3,
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
            jockey_id=jockey.jockey_id,
            trainer_id=trainer.trainer_id,
            numero_corde=1,
            final_position=1,
        ),
        Partant(
            course_id=course1.course_id,
            horse_id=horses[1].horse_id,
            jockey_id=jockey.jockey_id,
            trainer_id=trainer.trainer_id,
            numero_corde=2,
            final_position=2,
        ),
        Partant(
            course_id=course1.course_id,
            horse_id=horses[2].horse_id,
            jockey_id=jockey.jockey_id,
            trainer_id=trainer.trainer_id,
            numero_corde=3,
            final_position=4,
        ),
        Partant(
            course_id=course2.course_id,
            horse_id=horses[3].horse_id,
            jockey_id=jockey.jockey_id,
            trainer_id=trainer.trainer_id,
            numero_corde=1,
            final_position=1,
        ),
        Partant(
            course_id=course2.course_id,
            horse_id=horses[4].horse_id,
            jockey_id=jockey.jockey_id,
            trainer_id=trainer.trainer_id,
            numero_corde=2,
            final_position=4,
        ),
        Partant(
            course_id=course2.course_id,
            horse_id=horses[5].horse_id,
            jockey_id=jockey.jockey_id,
            trainer_id=trainer.trainer_id,
            numero_corde=3,
            final_position=2,
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

    with in_memory_session() as check_session:
        stored_model = check_session.query(MLModel).filter(MLModel.is_active.is_(True)).one()
        assert float(stored_model.accuracy) == pytest.approx(2 / 3, rel=1e-3)

        stored_metrics = stored_model.performance_metrics
        if isinstance(stored_metrics, str):
            stored_metrics = json.loads(stored_metrics)

        assert stored_metrics["last_evaluation"]["metrics"]["accuracy"] == pytest.approx(2 / 3, rel=1e-3)
        assert "confidence_level_metrics" in stored_metrics["last_evaluation"]
        assert "gain_curve" in stored_metrics["last_evaluation"]["metrics"]
        assert "lift_analysis" in stored_metrics["last_evaluation"]["metrics"]
        assert "precision_recall_curve" in stored_metrics["last_evaluation"]["metrics"]
        assert "roc_curve" in stored_metrics["last_evaluation"]["metrics"]
        assert "ks_analysis" in stored_metrics["last_evaluation"]["metrics"]
        assert "calibration_diagnostics" in stored_metrics["last_evaluation"]
        assert "calibration_diagnostics" in stored_metrics["last_evaluation"]["metrics"]
        assert "daily_performance" in stored_metrics["last_evaluation"]["metrics"]


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
