# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""
Configuration de la base de données
SQLAlchemy avec support asynchrone (aiomysql)
"""

from typing import AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

from .config import settings


def _convert_to_sync_url(async_url: str) -> str:
    """Convertit une URL SQLAlchemy asynchrone vers son équivalent synchrone."""

    url = make_url(async_url)

    if "+aiomysql" in url.drivername:
        driver = url.drivername.replace("+aiomysql", "+pymysql")
    elif "+asyncpg" in url.drivername:
        driver = url.drivername.replace("+asyncpg", "+psycopg")
    elif "+sqlite+aiosqlite" in url.drivername:
        driver = url.drivername.replace("+aiosqlite", "")
    else:
        driver = url.drivername

    return url.set(drivername=driver).render_as_string(hide_password=False)

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    poolclass=NullPool,  # For async, use NullPool or custom pool
    pool_pre_ping=True,
)

# Create async session maker
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Create sync engine + session maker for Celery / background tasks
sync_engine = create_engine(
    _convert_to_sync_url(settings.DATABASE_URL),
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine,
)

# Base class for ORM models
Base = declarative_base()


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency pour obtenir une session de base de données
    À utiliser dans les endpoints FastAPI
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Alias conservé pour compatibilité avec les endpoints asynchrones."""

    async for session in get_db():
        yield session


async def init_db() -> None:
    """
    Initialize database (create tables if they don't exist)
    Note: In production, use Alembic migrations instead
    """
    async with engine.begin() as conn:
        # Import all models to ensure they are registered
        from app.models import user, course, pronostic, pari  # noqa
        # Create tables
        await conn.run_sync(Base.metadata.create_all)