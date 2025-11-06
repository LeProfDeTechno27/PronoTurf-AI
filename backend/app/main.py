# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Point d'entrée principal de l'application FastAPI PronoTurf
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import init_db
from app.api.endpoints import (
    analytics,
    auth,
    health,
    hippodromes,
    reunions,
    courses,
    pronostics,
    paris_simules,
    bankroll,
    favoris,
    notifications,
)

# Import des routers (à compléter au fur et à mesure)
# from app.api.endpoints import users, analytics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie de l'application
    Exécuté au démarrage et à l'arrêt
    """
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug mode: {settings.DEBUG}")

    # Initialize database (commented out for now, use Alembic migrations in production)
    # await init_db()

    yield

    # Shutdown
    print(f"Shutting down {settings.APP_NAME}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Application de pronostics hippiques intelligents",
    docs_url=f"{settings.API_V1_PREFIX}/docs" if settings.DEBUG else None,
    redoc_url=f"{settings.API_V1_PREFIX}/redoc" if settings.DEBUG else None,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/")
async def root():
    """
    Endpoint racine - Health check
    """
    return JSONResponse(
        content={
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "status": "running",
            "docs": f"{settings.API_V1_PREFIX}/docs" if settings.DEBUG else None,
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return JSONResponse(
        content={
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
        }
    )


# Include routers
# Health checks
app.include_router(
    health.router,
    prefix=f"{settings.API_V1_PREFIX}",
    tags=["Health"]
)

# Authentication
app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_PREFIX}/auth",
    tags=["Authentication"]
)

# Users (à implémenter)
# app.include_router(
#     users.router,
#     prefix=f"{settings.API_V1_PREFIX}/users",
#     tags=["Users"]
# )

# Hippodromes
app.include_router(
    hippodromes.router,
    prefix=f"{settings.API_V1_PREFIX}/hippodromes",
    tags=["Hippodromes"]
)

# Reunions
app.include_router(
    reunions.router,
    prefix=f"{settings.API_V1_PREFIX}/reunions",
    tags=["Reunions"]
)

# Courses
app.include_router(
    courses.router,
    prefix=f"{settings.API_V1_PREFIX}/courses",
    tags=["Courses"]
)

# Pronostics
app.include_router(
    pronostics.router,
    prefix=f"{settings.API_V1_PREFIX}/pronostics",
    tags=["Pronostics"]
)

# Analytics Aspiturf
app.include_router(
    analytics.router,
    prefix=f"{settings.API_V1_PREFIX}/analytics",
    tags=["Analytics"]
)

# Paris Simulés
app.include_router(
    paris_simules.router,
    prefix=f"{settings.API_V1_PREFIX}/paris-simules",
    tags=["Paris Simulés"]
)

# Bankroll
app.include_router(
    bankroll.router,
    prefix=f"{settings.API_V1_PREFIX}/bankroll",
    tags=["Bankroll"]
)

# Favoris
app.include_router(
    favoris.router,
    prefix=f"{settings.API_V1_PREFIX}/favoris",
    tags=["Favoris"]
)

# Notifications
app.include_router(
    notifications.router,
    prefix=f"{settings.API_V1_PREFIX}/notifications",
    tags=["Notifications"]
)

# Analytics (à implémenter)
# app.include_router(
#     analytics.router,
#     prefix=f"{settings.API_V1_PREFIX}/analytics",
#     tags=["Analytics"]
# )


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """
    Handler pour les erreurs 404
    """
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "Resource not found",
            "error": str(exc)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """
    Handler pour les erreurs 500
    """
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )