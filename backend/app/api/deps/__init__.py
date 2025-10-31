"""
Dépendances FastAPI communes
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.models.user import User

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Récupère l'utilisateur actuel à partir du token JWT

    Args:
        credentials: Credentials HTTP Bearer
        db: Session de base de données

    Returns:
        User: Utilisateur authentifié

    Raises:
        HTTPException: Si le token est invalide ou l'utilisateur n'existe pas
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: Optional[int] = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.user_id == int(user_id)).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Vérifie que l'utilisateur est actif

    Args:
        current_user: Utilisateur actuel

    Returns:
        User: Utilisateur actif

    Raises:
        HTTPException: Si l'utilisateur est inactif
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    Vérifie que l'utilisateur est administrateur

    Args:
        current_user: Utilisateur actuel

    Returns:
        User: Utilisateur admin

    Raises:
        HTTPException: Si l'utilisateur n'est pas admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


def get_current_subscriber(current_user: User = Depends(get_current_user)) -> User:
    """
    Vérifie que l'utilisateur est abonné (ou admin)

    Args:
        current_user: Utilisateur actuel

    Returns:
        User: Utilisateur abonné

    Raises:
        HTTPException: Si l'utilisateur n'est pas abonné
    """
    if not current_user.is_subscriber:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Subscription required"
        )
    return current_user
