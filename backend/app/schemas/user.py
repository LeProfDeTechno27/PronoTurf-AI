"""
Schémas Pydantic pour les utilisateurs
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator

from app.models.user import UserRole, BankrollStrategy


# ============================
# Schémas de base
# ============================

class UserBase(BaseModel):
    """Schéma de base pour un utilisateur"""
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserCreate(UserBase):
    """Schéma pour la création d'un utilisateur"""
    password: str = Field(..., min_length=8, max_length=100)
    role: UserRole = UserRole.GUEST

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Valide la force du mot de passe"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.islower() for char in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseModel):
    """Schéma pour la mise à jour d'un utilisateur"""
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    telegram_id: Optional[str] = None
    profile_picture_url: Optional[str] = None
    preferred_strategy: Optional[BankrollStrategy] = None


class UserRead(UserBase):
    """Schéma pour la lecture d'un utilisateur"""
    user_id: int
    role: UserRole
    telegram_id: Optional[str] = None
    profile_picture_url: Optional[str] = None
    initial_bankroll: Decimal
    current_bankroll: Decimal
    preferred_strategy: BankrollStrategy
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool

    model_config = {"from_attributes": True}


class UserInDB(UserRead):
    """Schéma pour un utilisateur en base de données (avec password_hash)"""
    password_hash: str


# ============================
# Schémas d'authentification
# ============================

class Token(BaseModel):
    """Schéma pour les tokens JWT"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """Schéma pour le payload d'un token JWT"""
    sub: Optional[int] = None
    type: str = "access"


class LoginRequest(BaseModel):
    """Schéma pour la requête de connexion"""
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    """Schéma pour la requête de rafraîchissement de token"""
    refresh_token: str


class PasswordChange(BaseModel):
    """Schéma pour le changement de mot de passe"""
    old_password: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Valide la force du nouveau mot de passe"""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(char.isupper() for char in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(char.islower() for char in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(char.isdigit() for char in v):
            raise ValueError("Password must contain at least one digit")
        return v


# ============================
# Schémas de réponse
# ============================

class UserResponse(BaseModel):
    """Schéma de réponse générique pour un utilisateur"""
    success: bool
    message: str
    data: Optional[UserRead] = None


class UsersListResponse(BaseModel):
    """Schéma de réponse pour une liste d'utilisateurs"""
    success: bool
    message: str
    data: list[UserRead]
    total: int
    page: int
    page_size: int
