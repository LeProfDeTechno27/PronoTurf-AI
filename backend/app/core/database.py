"""
Configuration de la base de données
SQLAlchemy avec support asynchrone (aiomysql)
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from .config import settings

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
