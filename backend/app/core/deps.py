"""
Dépendances FastAPI réutilisables
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .database import get_db
from .security import decode_token, validate_token_type
from app.models.user import User

# Security scheme pour JWT
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Récupère l'utilisateur actuellement authentifié à partir du token JWT
    Raise HTTPException si le token est invalide ou l'utilisateur n'existe pas
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Decode token
    token = credentials.credentials
    payload = decode_token(token)

    if payload is None:
        raise credentials_exception

    # Validate token type
    if not validate_token_type(payload, "access"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user_id from payload
    user_id: Optional[int] = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    # Fetch user from database
    result = await db.execute(select(User).where(User.user_id == int(user_id)))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Vérifie que l'utilisateur est actif
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_subscriber(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Vérifie que l'utilisateur est au moins Abonné (subscriber ou admin)
    """
    if current_user.role not in ["subscriber", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Subscriber role required."
        )
    return current_user


async def get_current_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Vérifie que l'utilisateur est Administrateur
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Admin role required."
        )
    return current_user


# Alias pour rétrocompatibilité
get_current_superuser = get_current_admin
