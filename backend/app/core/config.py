"""
Configuration de l'application PronoTurf
Utilise pydantic-settings pour la gestion des variables d'environnement
"""

from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration principale de l'application"""

    # Application
    APP_NAME: str = "PronoTurf"
    APP_VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str
    MYSQL_ROOT_PASSWORD: str = "root_password"
    MYSQL_DATABASE: str = "pronoturf"
    MYSQL_USER: str = "pronoturf_user"
    MYSQL_PASSWORD: str = "pronoturf_password"

    # JWT Authentication
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8501"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # External APIs
    # AspiTurf - API PRINCIPALE (données complètes)
    ASPITURF_API_KEY: Optional[str] = None
    ASPITURF_API_URL: str = "https://api.aspiturf.com"
    ASPITURF_ENABLED: bool = True

    # TurfInfo - COMPLÉMENTAIRE (sans clé API)
    TURFINFO_OFFLINE_URL: str = "https://offline.turfinfo.api.pmu.fr/rest/client/7"
    TURFINFO_ONLINE_URL: str = "https://online.turfinfo.api.pmu.fr/rest/client/61"
    TURFINFO_ENABLED: bool = True

    # Open-PMU - COMPLÉMENTAIRE pour résultats (sans clé API)
    OPENPMU_API_URL: str = "https://open-pmu-api.vercel.app/api"
    OPENPMU_ENABLED: bool = True

    # Open-Meteo - Météo
    OPENMETEO_API_URL: str = "https://api.open-meteo.com/v1/forecast"

    # Telegram
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_ENABLED: bool = False

    # Machine Learning
    ML_MODELS_PATH: str = "/app/models"
    ML_MODEL_PATH: str = "/app/models/horse_racing_model.pkl"
    ML_MODEL_VERSION: str = "v1"
    ML_RETRAIN_SCHEDULE: str = "0 2 * * 1"  # Every Monday at 2 AM

    # Celery Tasks Schedule
    TASK_SYNC_PROGRAMME_SCHEDULE: str = "0 6 * * *"  # Every day at 6 AM
    TASK_GENERATE_PRONOSTICS_SCHEDULE: str = "0 7 * * *"  # Every day at 7 AM
    TASK_CHECK_RESULTS_SCHEDULE: str = "0 12-23 * * *"  # Every hour from 12 to 23

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "/app/logs/app.log"

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000

    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 5
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "gif"]

    # Email (optional)
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_EMAIL: str = "noreply@pronoturf.ai"
    EMAIL_ENABLED: bool = False

    # Frontend URL for email links
    FRONTEND_URL: str = "http://localhost:3000"

    # Sentry (optional)
    SENTRY_DSN: Optional[str] = None
    SENTRY_ENVIRONMENT: str = "development"
    SENTRY_ENABLED: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


# Instance globale des settings
settings = Settings()
