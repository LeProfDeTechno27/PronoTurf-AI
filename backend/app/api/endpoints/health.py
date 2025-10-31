"""
Router de healthcheck et test de connectivité
Vérifie la santé de tous les services (API, MySQL, Redis)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as aioredis
from datetime import datetime

from app.core.config import settings
from app.core.database import get_db

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check simple de l'API
    """
    return {
        "status": "healthy",
        "service": "backend",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/db")
async def health_check_database(db: AsyncSession = Depends(get_db)):
    """
    Vérifie la connectivité à la base de données MySQL
    """
    try:
        # Test simple query
        result = await db.execute(text("SELECT 1"))
        result.scalar()

        # Test version MySQL
        version_result = await db.execute(text("SELECT VERSION()"))
        mysql_version = version_result.scalar()

        return {
            "status": "healthy",
            "service": "mysql",
            "connected": True,
            "version": mysql_version,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "service": "mysql",
                "connected": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/health/redis")
async def health_check_redis():
    """
    Vérifie la connectivité à Redis
    """
    try:
        # Connect to Redis
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

        # Test ping
        await redis_client.ping()

        # Test set/get
        test_key = "health_check_test"
        test_value = f"test_{datetime.utcnow().timestamp()}"
        await redis_client.set(test_key, test_value, ex=10)  # expire in 10 seconds
        retrieved_value = await redis_client.get(test_key)

        # Get Redis info
        info = await redis_client.info()

        await redis_client.close()

        return {
            "status": "healthy",
            "service": "redis",
            "connected": True,
            "test_passed": retrieved_value == test_value,
            "redis_version": info.get("redis_version"),
            "used_memory_human": info.get("used_memory_human"),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "service": "redis",
                "connected": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/health/all")
async def health_check_all(db: AsyncSession = Depends(get_db)):
    """
    Vérifie la santé de tous les services
    """
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }

    # Check API
    results["services"]["api"] = {
        "status": "healthy",
        "connected": True
    }

    # Check MySQL
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        version_result = await db.execute(text("SELECT VERSION()"))
        mysql_version = version_result.scalar()

        results["services"]["mysql"] = {
            "status": "healthy",
            "connected": True,
            "version": mysql_version
        }
    except Exception as e:
        results["services"]["mysql"] = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }

    # Check Redis
    try:
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
        info = await redis_client.info()
        await redis_client.close()

        results["services"]["redis"] = {
            "status": "healthy",
            "connected": True,
            "version": info.get("redis_version")
        }
    except Exception as e:
        results["services"]["redis"] = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e)
        }

    # Overall status
    all_healthy = all(
        service["status"] == "healthy"
        for service in results["services"].values()
    )
    results["overall_status"] = "healthy" if all_healthy else "unhealthy"

    if not all_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=results
        )

    return results


@router.get("/health/celery")
async def health_check_celery():
    """
    Vérifie l'état de Celery (worker et beat)
    Note: Nécessite que Celery soit configuré avec Redis
    """
    try:
        redis_client = aioredis.from_url(
            settings.CELERY_BROKER_URL,
            encoding="utf-8",
            decode_responses=True
        )

        # Check if Celery broker is accessible
        await redis_client.ping()

        # Note: Pour un check complet de Celery, il faudrait utiliser
        # celery.app.control.inspect() mais cela nécessite l'import de l'app Celery
        # Pour l'instant, on vérifie juste que le broker est accessible

        await redis_client.close()

        return {
            "status": "healthy",
            "service": "celery",
            "broker_connected": True,
            "note": "Broker accessible. Pour un check complet des workers, voir les logs Celery.",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "service": "celery",
                "broker_connected": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
