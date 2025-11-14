# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Point d'entrée principal de l'application FastAPI PronoTurf
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

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


def _resolve_frontend_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Retourne les chemins vers le build frontend si disponible."""

    if not settings.SERVE_FRONTEND:
        return None, None

    if settings.FRONTEND_DIST_PATH:
        candidate = Path(settings.FRONTEND_DIST_PATH).expanduser()
        if not candidate.is_absolute():
            base_dir = Path(__file__).resolve().parents[2]
            candidate = base_dir / candidate
    else:
        candidate = Path(__file__).resolve().parents[2] / "frontend" / "dist"

    candidate = candidate.resolve()
    index_file = candidate / "index.html"

    if candidate.is_dir() and index_file.is_file():
        print(f"[frontend] Serving static files from: {candidate}")
        return candidate, index_file

    print("[frontend] Build directory not found, SPA serving disabled.")
    return None, None


FRONTEND_DIST_PATH, FRONTEND_INDEX_FILE = _resolve_frontend_paths()


def _should_serve_spa() -> bool:
    return FRONTEND_DIST_PATH is not None and FRONTEND_INDEX_FILE is not None


def _build_frontend_redirect(path: Optional[str] = None) -> Optional[RedirectResponse]:
    """Construit une redirection vers FRONTEND_URL si configuré."""

    if not settings.FRONTEND_URL:
        return None

    base_url = settings.FRONTEND_URL.rstrip("/")
    if not base_url:
        return None

    if path:
        target = f"{base_url}/{path.lstrip('/')}"
    else:
        target = base_url

    return RedirectResponse(url=target, status_code=307)


def _frontend_response(path: Optional[str] = None):
    """Retourne soit un fichier SPA soit une redirection vers FRONTEND_URL."""

    if _should_serve_spa():
        return _spa_file_response(path)

    redirect = _build_frontend_redirect(path)
    if redirect is not None:
        return redirect

    return None


def _spa_file_response(path: Optional[str] = None) -> FileResponse:
    assert FRONTEND_DIST_PATH is not None
    assert FRONTEND_INDEX_FILE is not None

    if not path:
        return FileResponse(FRONTEND_INDEX_FILE)

    requested_path = (FRONTEND_DIST_PATH / path).resolve()

    try:
        requested_path.relative_to(FRONTEND_DIST_PATH)
    except ValueError:
        # Tentative d'accès en dehors du dossier frontend
        return FileResponse(FRONTEND_INDEX_FILE)

    if requested_path.is_file():
        return FileResponse(requested_path)

    return FileResponse(FRONTEND_INDEX_FILE)


# Root endpoint / SPA entrypoint
@app.get("/", include_in_schema=False)
async def root():
    """Health check JSON ou SPA selon la configuration."""

    spa_response = _frontend_response()
    if spa_response is not None:
        return spa_response

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


# SPA fallback route (doit être enregistré après les routes API)
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    api_prefix = settings.API_V1_PREFIX.strip("/")

    if api_prefix and full_path.startswith(api_prefix):
        raise HTTPException(status_code=404)

    spa_response = _frontend_response(full_path)
    if spa_response is not None:
        return spa_response

    raise HTTPException(status_code=404)

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