#!/usr/bin/env python3
# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
PronoTurf - Script de test de communication inter-services
Vérifie que tous les services peuvent communiquer entre eux
"""

import asyncio
import sys
from typing import Dict, Any

# Add parent directory to path to import app modules
sys.path.insert(0, '/app')

try:
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    import redis.asyncio as aioredis
    import httpx
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("   Assurez-vous d'exécuter ce script depuis le container backend")
    sys.exit(1)


class ServiceTester:
    """Classe pour tester la connectivité des services"""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.db_url = "mysql+aiomysql://pronoturf_user:PronoTurf_DB_2025!@mysql:3306/pronoturf"
        self.redis_url = "redis://redis:6379/0"

    async def test_mysql(self) -> Dict[str, Any]:
        """Test de la connexion MySQL"""
        print("🔍 Test de connexion à MySQL...", end=" ")
        result = {
            "service": "MySQL",
            "connected": False,
            "details": {}
        }

        try:
            engine = create_async_engine(self.db_url, echo=False)
            async with engine.begin() as conn:
                # Test simple query
                await conn.execute(text("SELECT 1"))

                # Get MySQL version
                version_result = await conn.execute(text("SELECT VERSION()"))
                mysql_version = version_result.scalar()

                # Count tables
                tables_result = await conn.execute(text("SHOW TABLES"))
                table_count = len(tables_result.fetchall())

                result["connected"] = True
                result["details"] = {
                    "version": mysql_version,
                    "tables_count": table_count
                }

            await engine.dispose()
            print("✅")

        except Exception as e:
            result["error"] = str(e)
            print(f"❌ {e}")

        return result

    async def test_redis(self) -> Dict[str, Any]:
        """Test de la connexion Redis"""
        print("🔍 Test de connexion à Redis...", end=" ")
        result = {
            "service": "Redis",
            "connected": False,
            "details": {}
        }

        try:
            redis_client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )

            # Test PING
            pong = await redis_client.ping()

            # Test SET/GET
            test_key = "connectivity_test"
            test_value = "test_value_12345"
            await redis_client.set(test_key, test_value, ex=10)
            retrieved = await redis_client.get(test_key)

            # Get Redis info
            info = await redis_client.info()

            result["connected"] = True
            result["details"] = {
                "ping": pong,
                "test_passed": retrieved == test_value,
                "version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human")
            }

            await redis_client.close()
            print("✅")

        except Exception as e:
            result["error"] = str(e)
            print(f"❌ {e}")

        return result

    async def test_backend_api(self) -> Dict[str, Any]:
        """Test de l'API Backend"""
        print("🔍 Test de l'API Backend...", end=" ")
        result = {
            "service": "Backend API",
            "connected": False,
            "details": {}
        }

        try:
            async with httpx.AsyncClient() as client:
                # Test root endpoint
                response = await client.get("http://backend:8000/")
                root_data = response.json()

                # Test health endpoint
                health_response = await client.get("http://backend:8000/health")
                health_data = health_response.json()

                result["connected"] = response.status_code == 200
                result["details"] = {
                    "root_status": response.status_code,
                    "health_status": health_response.status_code,
                    "app_name": root_data.get("app"),
                    "version": root_data.get("version")
                }

            print("✅")

        except Exception as e:
            result["error"] = str(e)
            print(f"❌ {e}")

        return result

    async def test_frontend(self) -> Dict[str, Any]:
        """Test du Frontend React"""
        print("🔍 Test du Frontend...", end=" ")
        result = {
            "service": "Frontend",
            "connected": False,
            "details": {}
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://frontend:3000/")

                result["connected"] = response.status_code == 200
                result["details"] = {
                    "status_code": response.status_code,
                    "content_length": len(response.content)
                }

            print("✅")

        except Exception as e:
            result["error"] = str(e)
            print(f"❌ {e}")

        return result

    async def test_streamlit(self) -> Dict[str, Any]:
        """Test de Streamlit"""
        print("🔍 Test de Streamlit...", end=" ")
        result = {
            "service": "Streamlit",
            "connected": False,
            "details": {}
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://streamlit:8501/")

                result["connected"] = response.status_code == 200
                result["details"] = {
                    "status_code": response.status_code
                }

            print("✅")

        except Exception as e:
            result["error"] = str(e)
            print(f"❌ {e}")

        return result

    async def run_all_tests(self):
        """Exécute tous les tests"""
        print("=" * 60)
        print("🏇 PronoTurf - Test de communication inter-services")
        print("=" * 60)
        print()

        # Run all tests
        self.results["mysql"] = await self.test_mysql()
        self.results["redis"] = await self.test_redis()
        self.results["backend"] = await self.test_backend_api()
        self.results["frontend"] = await self.test_frontend()
        self.results["streamlit"] = await self.test_streamlit()

        print()
        print("=" * 60)
        print("📊 Résumé des tests")
        print("=" * 60)

        all_connected = True
        for service_key, service_result in self.results.items():
            status = "✅" if service_result["connected"] else "❌"
            service_name = service_result["service"]
            print(f"{status} {service_name}")

            if not service_result["connected"]:
                all_connected = False
                if "error" in service_result:
                    print(f"   Erreur: {service_result['error']}")

        print()
        if all_connected:
            print("✅ Tous les services communiquent correctement!")
            return 0
        else:
            print("❌ Certains services ne communiquent pas correctement")
            return 1


async def main():
    """Fonction principale"""
    tester = ServiceTester()
    exit_code = await tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())