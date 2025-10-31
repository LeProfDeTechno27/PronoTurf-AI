#!/usr/bin/env python3
"""
Script de test pour vérifier la configuration de Sprint 1-2
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test que tous les modules s'importent correctement"""
    print("🔍 Test des imports...")

    try:
        # Core
        from app.core.config import settings
        print("✅ app.core.config")

        from app.core.database import Base, get_db, init_db
        print("✅ app.core.database")

        from app.core.deps import get_current_user, get_current_admin
        print("✅ app.core.deps")

        from app.core.security import create_access_token, verify_password
        print("✅ app.core.security")

        # Models
        from app.models import (
            User, Hippodrome, Reunion, Course,
            Horse, Jockey, Trainer, Partant
        )
        print("✅ app.models")

        # Schemas
        from app.schemas import (
            HippodromeResponse, ReunionResponse, CourseResponse,
            HorseResponse, JockeyResponse, TrainerResponse, PartantResponse
        )
        print("✅ app.schemas")

        # Services
        from app.services import PMUService, WeatherService
        print("✅ app.services")

        # Tasks
        from app.tasks import celery_app, sync_tasks, ml_tasks, notification_tasks
        print("✅ app.tasks")

        # API endpoints
        from app.api.endpoints import auth, health, hippodromes, reunions, courses
        print("✅ app.api.endpoints")

        print("\n✅ Tous les imports sont OK!\n")
        return True

    except Exception as e:
        print(f"\n❌ Erreur d'import: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test que la configuration est valide"""
    print("🔍 Test de la configuration...")

    try:
        from app.core.config import settings

        # Test variables critiques
        assert settings.DATABASE_URL, "DATABASE_URL manquante"
        print(f"✅ DATABASE_URL: {settings.DATABASE_URL[:30]}...")

        assert settings.SECRET_KEY, "SECRET_KEY manquante"
        print(f"✅ SECRET_KEY: configurée ({len(settings.SECRET_KEY)} chars)")

        assert settings.REDIS_URL, "REDIS_URL manquante"
        print(f"✅ REDIS_URL: {settings.REDIS_URL}")

        assert settings.CELERY_BROKER_URL, "CELERY_BROKER_URL manquante"
        print(f"✅ CELERY_BROKER_URL: {settings.CELERY_BROKER_URL}")

        assert settings.CELERY_RESULT_BACKEND, "CELERY_RESULT_BACKEND manquante"
        print(f"✅ CELERY_RESULT_BACKEND: {settings.CELERY_RESULT_BACKEND}")

        print(f"✅ CORS_ORIGINS: {len(settings.CORS_ORIGINS)} origines configurées")
        print(f"✅ APP_NAME: {settings.APP_NAME}")
        print(f"✅ API_V1_PREFIX: {settings.API_V1_PREFIX}")

        print("\n✅ Configuration valide!\n")
        return True

    except Exception as e:
        print(f"\n❌ Erreur de configuration: {e}\n")
        return False


def test_models():
    """Test que les modèles sont bien définis"""
    print("🔍 Test des modèles SQLAlchemy...")

    try:
        from app.models import (
            User, Hippodrome, Reunion, Course,
            Horse, Jockey, Trainer, Partant
        )
        from app.models.hippodrome import TrackType
        from app.models.reunion import ReunionStatus
        from app.models.course import Discipline, SurfaceType, StartType, CourseStatus
        from app.models.horse import Gender

        # Vérifier que les modèles ont les attributs attendus
        assert hasattr(User, 'user_id'), "User.user_id manquant"
        assert hasattr(Hippodrome, 'hippodrome_id'), "Hippodrome.hippodrome_id manquant"
        assert hasattr(Reunion, 'reunion_id'), "Reunion.reunion_id manquant"
        assert hasattr(Course, 'course_id'), "Course.course_id manquant"
        assert hasattr(Horse, 'horse_id'), "Horse.horse_id manquant"
        assert hasattr(Jockey, 'jockey_id'), "Jockey.jockey_id manquant"
        assert hasattr(Trainer, 'trainer_id'), "Trainer.trainer_id manquant"
        assert hasattr(Partant, 'partant_id'), "Partant.partant_id manquant"

        print("✅ User")
        print("✅ Hippodrome")
        print("✅ Reunion")
        print("✅ Course")
        print("✅ Horse")
        print("✅ Jockey")
        print("✅ Trainer")
        print("✅ Partant")

        # Vérifier les enums
        assert len(list(TrackType)) > 0, "TrackType enum vide"
        print(f"✅ TrackType enum: {len(list(TrackType))} valeurs")

        assert len(list(Discipline)) > 0, "Discipline enum vide"
        print(f"✅ Discipline enum: {len(list(Discipline))} valeurs")

        assert len(list(Gender)) > 0, "Gender enum vide"
        print(f"✅ Gender enum: {len(list(Gender))} valeurs")

        print("\n✅ Tous les modèles sont OK!\n")
        return True

    except Exception as e:
        print(f"\n❌ Erreur de modèles: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_schemas():
    """Test que les schémas Pydantic sont bien définis"""
    print("🔍 Test des schémas Pydantic...")

    try:
        from app.schemas import (
            HippodromeCreate, HippodromeResponse,
            ReunionCreate, ReunionResponse,
            CourseCreate, CourseResponse,
            HorseCreate, HorseResponse,
            JockeyCreate, JockeyResponse,
            TrainerCreate, TrainerResponse,
            PartantCreate, PartantResponse,
        )

        print("✅ HippodromeCreate, HippodromeResponse")
        print("✅ ReunionCreate, ReunionResponse")
        print("✅ CourseCreate, CourseResponse")
        print("✅ HorseCreate, HorseResponse")
        print("✅ JockeyCreate, JockeyResponse")
        print("✅ TrainerCreate, TrainerResponse")
        print("✅ PartantCreate, PartantResponse")

        print("\n✅ Tous les schémas sont OK!\n")
        return True

    except Exception as e:
        print(f"\n❌ Erreur de schémas: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_services():
    """Test que les services sont disponibles"""
    print("🔍 Test des services...")

    try:
        from app.services import PMUService, WeatherService

        # Vérifier que les classes ont les méthodes attendues
        assert hasattr(PMUService, 'fetch_program_for_date'), "PMUService.fetch_program_for_date manquante"
        assert hasattr(PMUService, 'sync_program_for_date'), "PMUService.sync_program_for_date manquante"
        print("✅ PMUService")

        assert hasattr(WeatherService, 'get_weather'), "WeatherService.get_weather manquante"
        assert hasattr(WeatherService, 'get_weather_for_hippodrome'), "WeatherService.get_weather_for_hippodrome manquante"
        print("✅ WeatherService")

        print("\n✅ Tous les services sont OK!\n")
        return True

    except Exception as e:
        print(f"\n❌ Erreur de services: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_tasks():
    """Test que les tâches Celery sont disponibles"""
    print("🔍 Test des tâches Celery...")

    try:
        from app.tasks import celery_app
        from app.tasks.sync_tasks import (
            sync_daily_programs,
            update_odds,
            check_race_results,
            sync_specific_date
        )

        print("✅ celery_app")
        print("✅ sync_daily_programs")
        print("✅ update_odds")
        print("✅ check_race_results")
        print("✅ sync_specific_date")

        # Vérifier la configuration Celery
        assert celery_app.conf.broker_url, "broker_url non configuré"
        print(f"✅ Celery broker: {celery_app.conf.broker_url}")

        assert celery_app.conf.result_backend, "result_backend non configuré"
        print(f"✅ Celery backend: {celery_app.conf.result_backend}")

        print("\n✅ Toutes les tâches sont OK!\n")
        return True

    except Exception as e:
        print(f"\n❌ Erreur de tâches: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """Test que les endpoints API sont disponibles"""
    print("🔍 Test des endpoints API...")

    try:
        from app.api.endpoints import auth, health, hippodromes, reunions, courses

        # Vérifier que chaque module a un router
        assert hasattr(auth, 'router'), "auth.router manquant"
        print("✅ auth.router")

        assert hasattr(health, 'router'), "health.router manquant"
        print("✅ health.router")

        assert hasattr(hippodromes, 'router'), "hippodromes.router manquant"
        print("✅ hippodromes.router")

        assert hasattr(reunions, 'router'), "reunions.router manquant"
        print("✅ reunions.router")

        assert hasattr(courses, 'router'), "courses.router manquant"
        print("✅ courses.router")

        print("\n✅ Tous les endpoints sont OK!\n")
        return True

    except Exception as e:
        print(f"\n❌ Erreur d'endpoints: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Exécute tous les tests"""
    print("\n" + "=" * 60)
    print("🚀 TEST DE CONFIGURATION SPRINT 1-2")
    print("=" * 60 + "\n")

    results = []

    # Exécuter tous les tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Modèles", test_models()))
    results.append(("Schémas", test_schemas()))
    results.append(("Services", test_services()))
    results.append(("Tâches Celery", test_tasks()))
    results.append(("Endpoints API", test_api_endpoints()))

    # Afficher le résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60 + "\n")

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    total_pass = sum(1 for _, success in results if success)
    total_tests = len(results)

    print(f"\n{total_pass}/{total_tests} tests réussis")

    if total_pass == total_tests:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_pass} test(s) en échec\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
