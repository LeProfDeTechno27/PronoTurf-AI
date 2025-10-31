#!/usr/bin/env python3
"""
Script de test des clients API TurfInfo et Open-PMU

Usage:
    python test_api_clients.py --test turfinfo
    python test_api_clients.py --test openpmu
    python test_api_clients.py --test both (défaut)
"""

import asyncio
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
import importlib.util

# Import direct des modules clients pour éviter les dépendances
backend_dir = Path(__file__).resolve().parent.parent

# Importer TurfInfo client
turfinfo_path = backend_dir / "app" / "services" / "turfinfo_client.py"
spec = importlib.util.spec_from_file_location("turfinfo_client", turfinfo_path)
turfinfo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(turfinfo_module)
TurfinfoClient = turfinfo_module.TurfinfoClient
TurfinfoEndpoint = turfinfo_module.TurfinfoEndpoint

# Importer Open-PMU client
openpmu_path = backend_dir / "app" / "services" / "open_pmu_client.py"
spec = importlib.util.spec_from_file_location("open_pmu_client", openpmu_path)
openpmu_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(openpmu_module)
OpenPMUClient = openpmu_module.OpenPMUClient


async def test_turfinfo_client():
    """Test du client TurfInfo"""
    print("\n" + "=" * 60)
    print("TEST TURFINFO CLIENT")
    print("=" * 60)

    today = date.today()
    print(f"\nDate de test: {today}")

    try:
        async with TurfinfoClient(endpoint_type=TurfinfoEndpoint.ONLINE) as client:
            # Test 1: Programme du jour
            print("\n1. Test get_programme_jour()...")
            programme = await client.get_programme_jour(today)

            if programme and "programme" in programme:
                reunions = programme["programme"].get("reunions", [])
                print(f"   ✓ Programme récupéré: {len(reunions)} réunions trouvées")

                if reunions:
                    # Afficher la première réunion
                    first_reunion = reunions[0]
                    hippodrome = first_reunion.get("hippodrome", {})
                    courses = first_reunion.get("courses", [])
                    print(f"   - Hippodrome: {hippodrome.get('libelleCourt', 'N/A')}")
                    print(f"   - Nombre de courses: {len(courses)}")

                    if courses:
                        # Test 2: Partants d'une course
                        reunion_num = first_reunion.get("numOfficiel", 1)
                        course_num = courses[0].get("numOrdre", 1)

                        print(f"\n2. Test get_partants_course(R{reunion_num}C{course_num})...")
                        partants_data = await client.get_partants_course(
                            today, reunion_num, course_num
                        )

                        if partants_data and "participants" in partants_data:
                            partants = partants_data["participants"]
                            print(f"   ✓ Partants récupérés: {len(partants)} chevaux")

                            if partants:
                                first_partant = partants[0]
                                cheval = first_partant.get("cheval", {})
                                print(f"   - Premier partant: {cheval.get('nom', 'N/A')}")

                        # Test 3: Performances détaillées
                        print(f"\n3. Test get_performances_detaillees(R{reunion_num}C{course_num})...")
                        perfs_data = await client.get_performances_detaillees(
                            today, reunion_num, course_num
                        )

                        if perfs_data and "participants" in perfs_data:
                            print(f"   ✓ Performances récupérées")
                            first_perf = perfs_data["participants"][0]
                            perfs = first_perf.get("performances", [])
                            print(f"   - Historique premier cheval: {len(perfs)} courses")

                        print("\n✓ TurfInfo Client: TOUS LES TESTS RÉUSSIS")
                        return True
                else:
                    print("   ⚠ Aucune réunion trouvée pour aujourd'hui")
                    return True  # Pas une erreur si pas de courses
            else:
                print("   ✗ Format de réponse invalide")
                return False

    except Exception as e:
        print(f"\n✗ ERREUR TurfInfo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_open_pmu_client():
    """Test du client Open-PMU"""
    print("\n" + "=" * 60)
    print("TEST OPEN-PMU CLIENT")
    print("=" * 60)

    # Tester avec une date récente (hier ou avant-hier pour avoir des résultats)
    test_date = date.today() - timedelta(days=1)
    print(f"\nDate de test: {test_date}")

    try:
        async with OpenPMUClient() as client:
            # Test 1: Arrivées du jour
            print("\n1. Test get_arrivees_by_date()...")
            arrivees = await client.get_arrivees_by_date(test_date)

            if arrivees and len(arrivees) > 0:
                print(f"   ✓ Arrivées récupérées: {len(arrivees)} courses")

                # Afficher la première arrivée
                first_arrivee = arrivees[0]
                hippodrome = first_arrivee.get("hippodrome", "N/A")
                numero_course = first_arrivee.get("numero_course", "N/A")
                arrivee_ordre = first_arrivee.get("arrivee", [])

                print(f"   - Hippodrome: {hippodrome}")
                print(f"   - Course n°: {numero_course}")
                print(f"   - Arrivée: {arrivee_ordre}")

                # Test 2: Arrivées par hippodrome
                if hippodrome != "N/A":
                    print(f"\n2. Test get_arrivees_by_hippodrome('{hippodrome}')...")
                    hippo_arrivees = await client.get_arrivees_by_hippodrome(
                        test_date, hippodrome
                    )
                    print(f"   ✓ {len(hippo_arrivees)} courses trouvées pour {hippodrome}")

                # Test 3: Rapports d'une course
                if hippodrome != "N/A" and numero_course != "N/A":
                    print(f"\n3. Test get_rapports_course(R{numero_course})...")
                    rapports = await client.get_rapports_course(
                        test_date, hippodrome, numero_course
                    )

                    if rapports:
                        print(f"   ✓ Rapports récupérés")

                        # Afficher quelques rapports
                        if "simple_gagnant" in rapports:
                            sg = rapports["simple_gagnant"]
                            if sg:
                                print(f"   - Simple gagnant: {sg[0]}")

                        if "trio" in rapports:
                            trio = rapports["trio"]
                            if trio:
                                print(f"   - Trio: {trio[0]}")
                    else:
                        print("   ⚠ Aucun rapport trouvé")

                # Test 4: Non-partants
                if hippodrome != "N/A" and numero_course != "N/A":
                    print(f"\n4. Test check_non_partants()...")
                    non_partants = await client.check_non_partants(
                        test_date, hippodrome, numero_course
                    )
                    print(f"   ✓ Non-partants: {non_partants if non_partants else 'Aucun'}")

                print("\n✓ Open-PMU Client: TOUS LES TESTS RÉUSSIS")
                return True

            else:
                print(f"   ⚠ Aucune arrivée trouvée pour {test_date}")
                print("   (Essayez avec une date plus ancienne)")
                return True  # Pas une erreur si pas de résultats

    except Exception as e:
        print(f"\n✗ ERREUR Open-PMU: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="Test des clients API TurfInfo et Open-PMU"
    )
    parser.add_argument(
        "--test",
        choices=["turfinfo", "openpmu", "both"],
        default="both",
        help="Quel client tester (défaut: both)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TESTS DES CLIENTS API")
    print("=" * 60)

    results = []

    if args.test in ["turfinfo", "both"]:
        result = await test_turfinfo_client()
        results.append(("TurfInfo", result))

    if args.test in ["openpmu", "both"]:
        result = await test_open_pmu_client()
        results.append(("Open-PMU", result))

    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)

    for name, result in results:
        status = "✓ RÉUSSI" if result else "✗ ÉCHEC"
        print(f"{name}: {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✓ TOUS LES TESTS ONT RÉUSSI")
        return 0
    else:
        print("\n✗ CERTAINS TESTS ONT ÉCHOUÉ")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
