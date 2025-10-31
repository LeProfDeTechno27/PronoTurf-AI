"""
Script de test pour le client Aspiturf

Test les fonctionnalités principales:
- Lecture CSV local
- Parsing des données
- Récupération des courses par date
- Récupération des partants
- Statistiques chevaux/jockeys/entraineurs

Usage:
    python backend/scripts/test_aspiturf_client.py --csv-path /path/to/aspiturf.csv
    python backend/scripts/test_aspiturf_client.py --csv-url https://example.com/data.csv
"""

import asyncio
import argparse
import sys
from datetime import date, datetime
from pathlib import Path
import logging

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.aspiturf_client import (
    AspiturfClient,
    AspiturfConfig,
    load_aspiturf_data,
    get_course_complete_data
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_basic_loading(csv_path: str = None, csv_url: str = None):
    """Test le chargement basique des données"""
    logger.info("=" * 60)
    logger.info("TEST 1: Chargement basique des données")
    logger.info("=" * 60)

    try:
        async with AspiturfClient(csv_path=csv_path, csv_url=csv_url) as client:
            logger.info(f"✅ Client initialisé avec succès")
            logger.info(f"📊 Nombre de lignes chargées: {len(client._data)}")

            if client._data:
                first_row = client._data[0]
                logger.info(f"🔍 Première ligne (échantillon):")
                logger.info(f"   - Date: {first_row.get('jour')}")
                logger.info(f"   - Hippodrome: {first_row.get('hippo')}")
                logger.info(f"   - Course: {first_row.get('prix')}")
                logger.info(f"   - Numéro cheval: {first_row.get('numero')}")

        return True

    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement: {e}")
        return False


async def test_courses_by_date(
    csv_path: str = None,
    csv_url: str = None,
    test_date: date = None
):
    """Test la récupération des courses par date"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Récupération des courses par date")
    logger.info("=" * 60)

    if not test_date:
        test_date = date.today()

    logger.info(f"📅 Date de test: {test_date}")

    try:
        async with AspiturfClient(csv_path=csv_path, csv_url=csv_url) as client:
            courses = await client.get_courses_by_date(test_date)

            logger.info(f"✅ {len(courses)} courses trouvées pour {test_date}")

            for i, course in enumerate(courses[:3], 1):  # Afficher max 3 courses
                logger.info(f"\n🏇 Course {i}:")
                logger.info(f"   - Hippodrome: {course.hippo}")
                logger.info(f"   - Numéro: R{course.reun}C{course.prix}")
                logger.info(f"   - Type: {course.typec}")
                logger.info(f"   - Distance: {course.dist}m")
                logger.info(f"   - Partants: {len(course.partants)}")
                logger.info(f"   - Allocation: {course.cheque}€")

            return True

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return False


async def test_partants_course(
    csv_path: str = None,
    csv_url: str = None,
    test_date: date = None,
    hippodrome: str = None,
    course_num: int = 1
):
    """Test la récupération des partants d'une course"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Récupération des partants d'une course")
    logger.info("=" * 60)

    if not test_date:
        test_date = date.today()

    logger.info(f"📅 Date: {test_date}")
    logger.info(f"🏇 Hippodrome: {hippodrome or 'AUTO (premier trouvé)'}")
    logger.info(f"🎯 Course: {course_num}")

    try:
        async with AspiturfClient(csv_path=csv_path, csv_url=csv_url) as client:
            # Si pas d'hippodrome spécifié, prendre le premier disponible
            if not hippodrome:
                courses = await client.get_courses_by_date(test_date)
                if not courses:
                    logger.warning("⚠️ Aucune course trouvée pour cette date")
                    return False
                hippodrome = courses[0].hippo
                logger.info(f"🔄 Hippodrome auto-détecté: {hippodrome}")

            partants = await client.get_partants_course(
                test_date,
                hippodrome,
                course_num
            )

            logger.info(f"✅ {len(partants)} partants trouvés")

            for i, partant in enumerate(partants[:5], 1):  # Max 5 partants
                logger.info(f"\n🐴 Partant {i} (Numéro {partant.get('numero')}):")
                logger.info(f"   - Sexe/Age: {partant.get('sexe')}/{partant.get('age')} ans")
                logger.info(f"   - Jockey: {partant.get('jockey')}")
                logger.info(f"   - Entraineur: {partant.get('entraineur')}")
                logger.info(f"   - Côte directe: {partant.get('cotedirect')}")
                logger.info(f"   - Côte probable: {partant.get('coteprob')}")
                logger.info(f"   - Musique: {partant.get('musiqueche')}")
                logger.info(f"   - Récence: {partant.get('recence')} jours")

                # Statistiques enrichies
                logger.info(f"   - Courses/Victoires/Places: "
                           f"{partant.get('coursescheval')}/"
                           f"{partant.get('victoirescheval')}/"
                           f"{partant.get('placescheval')}")

                if partant.get('gainsCarriere'):
                    logger.info(f"   - Gains carrière: {partant.get('gainsCarriere')}€")

            return True

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return False


async def test_statistics(
    csv_path: str = None,
    csv_url: str = None
):
    """Test les statistiques agrégées"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Statistiques agrégées")
    logger.info("=" * 60)

    try:
        async with AspiturfClient(csv_path=csv_path, csv_url=csv_url) as client:
            # Prendre un cheval au hasard dans les données
            if not client._data:
                logger.warning("⚠️ Pas de données disponibles")
                return False

            sample_row = client._data[0]
            horse_id = sample_row.get('idChe')
            jockey_id = sample_row.get('idJockey')
            trainer_id = sample_row.get('idEntraineur')

            # Statistiques cheval
            if horse_id:
                horse_stats = await client.get_horse_statistics(horse_id)
                logger.info(f"\n🐴 Statistiques cheval (ID: {horse_id}):")
                logger.info(f"   - Courses: {horse_stats.get('courses')}")
                logger.info(f"   - Victoires: {horse_stats.get('victoires')}")
                logger.info(f"   - Places: {horse_stats.get('places')}")
                logger.info(f"   - Gains: {horse_stats.get('gains_carriere')}€")
                logger.info(f"   - Musique: {horse_stats.get('musique')}")

            # Statistiques jockey
            if jockey_id:
                jockey_stats = await client.get_jockey_statistics(jockey_id)
                logger.info(f"\n👤 Statistiques jockey (ID: {jockey_id}):")
                logger.info(f"   - Nom: {jockey_stats.get('jockey')}")
                logger.info(f"   - Courses: {jockey_stats.get('courses')}")
                logger.info(f"   - Victoires: {jockey_stats.get('victoires')}")
                logger.info(f"   - % Victoires: {jockey_stats.get('pourc_victoires')}%")

            # Statistiques entraineur
            if trainer_id:
                trainer_stats = await client.get_trainer_statistics(trainer_id)
                logger.info(f"\n🎓 Statistiques entraineur (ID: {trainer_id}):")
                logger.info(f"   - Nom: {trainer_stats.get('entraineur')}")
                logger.info(f"   - Courses: {trainer_stats.get('courses')}")
                logger.info(f"   - Victoires: {trainer_stats.get('victoires')}")

            # Statistiques couple
            if horse_id and jockey_id:
                couple_stats = await client.get_couple_statistics(horse_id, jockey_id)
                logger.info(f"\n💑 Statistiques couple cheval-jockey:")
                logger.info(f"   - Courses ensemble: {couple_stats.get('nb_courses')}")
                logger.info(f"   - Victoires: {couple_stats.get('nb_victoires')}")
                logger.info(f"   - % Victoires: {couple_stats.get('tx_victoires')}%")

            return True

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return False


async def run_all_tests(
    csv_path: str = None,
    csv_url: str = None,
    test_date: date = None
):
    """Exécute tous les tests"""
    logger.info("\n" + "🧪" * 30)
    logger.info("DÉBUT DES TESTS ASPITURF CLIENT")
    logger.info("🧪" * 30 + "\n")

    results = []

    # Test 1: Chargement basique
    results.append(await test_basic_loading(csv_path, csv_url))

    # Test 2: Courses par date
    results.append(await test_courses_by_date(csv_path, csv_url, test_date))

    # Test 3: Partants d'une course
    results.append(await test_partants_course(csv_path, csv_url, test_date))

    # Test 4: Statistiques
    results.append(await test_statistics(csv_path, csv_url))

    # Résumé
    logger.info("\n" + "=" * 60)
    logger.info("RÉSUMÉ DES TESTS")
    logger.info("=" * 60)

    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0

    logger.info(f"✅ Tests réussis: {passed}/{total} ({success_rate:.1f}%)")

    if passed == total:
        logger.info("🎉 Tous les tests ont réussi!")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) ont échoué")

    return success_rate == 100


def main():
    """Point d'entrée du script"""
    parser = argparse.ArgumentParser(
        description="Test du client Aspiturf"
    )

    parser.add_argument(
        '--csv-path',
        type=str,
        help='Chemin vers le fichier CSV local'
    )

    parser.add_argument(
        '--csv-url',
        type=str,
        help='URL pour télécharger le CSV'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Date de test au format YYYY-MM-DD (défaut: aujourd\'hui)'
    )

    args = parser.parse_args()

    # Validation
    if not args.csv_path and not args.csv_url:
        parser.error("--csv-path ou --csv-url doit être fourni")

    # Parse date
    test_date = None
    if args.date:
        try:
            test_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"❌ Format de date invalide: {args.date}")
            logger.error("   Format attendu: YYYY-MM-DD (ex: 2025-01-15)")
            sys.exit(1)

    # Lancer les tests
    success = asyncio.run(
        run_all_tests(
            csv_path=args.csv_path,
            csv_url=args.csv_url,
            test_date=test_date
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
