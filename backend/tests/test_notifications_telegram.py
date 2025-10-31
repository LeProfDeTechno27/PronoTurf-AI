from unittest.mock import AsyncMock

import os

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("CORS_ORIGINS", "[\"http://localhost:3000\"]")

from app.api.endpoints import notifications
from app.api import deps
from app.core import database
from app.core.config import settings
from app.services.telegram_service import telegram_service


class DummyUser:
    """Utilisateur minimal pour tester les endpoints Telegram."""

    def __init__(self) -> None:
        self.user_id = 1
        self.email = "john@example.com"
        self.first_name = "John"
        self.last_name = "Doe"
        self.telegram_chat_id = None
        self.telegram_notifications_enabled = False
        self.telegram_linked_at = None
        self.is_active = True

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def has_telegram_notifications(self) -> bool:
        return bool(self.telegram_notifications_enabled and self.telegram_chat_id)


class FakeSession:
    """Session asynchrone factice qui trace les commits/refresh."""

    def __init__(self) -> None:
        self.committed = False
        self.refreshed = False

    async def commit(self) -> None:
        self.committed = True

    async def refresh(self, obj) -> None:  # pragma: no cover - simple stub
        self.refreshed = True


@pytest.fixture
def test_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Construit une application FastAPI isolée avec les overrides nécessaires."""

    previous_flag = settings.TELEGRAM_ENABLED
    settings.TELEGRAM_ENABLED = True

    app = FastAPI()
    app.include_router(notifications.router, prefix="/api/v1/notifications")

    user = DummyUser()
    session = FakeSession()

    async def override_user():
        return user

    async def override_db():
        yield session

    app.dependency_overrides[deps.get_current_user] = override_user
    app.dependency_overrides[database.get_async_db] = override_db

    try:
        yield app
    finally:
        settings.TELEGRAM_ENABLED = previous_flag
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_register_telegram_chat_success(monkeypatch: pytest.MonkeyPatch, test_app: FastAPI) -> None:
    """Vérifie qu'un utilisateur peut lier un chat Telegram et recevoir un message de bienvenue."""

    send_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(telegram_service, "send_link_confirmation", send_mock)
    monkeypatch.setattr(telegram_service, "bot", object())

    async with AsyncClient(app=test_app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/notifications/telegram/register",
            json={"chat_id": "123456"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "chat_id": "123456",
        "telegram_enabled": True,
        "test_message_sent": True,
    }

    assert send_mock.await_count == 1


@pytest.mark.asyncio
async def test_register_telegram_chat_disabled(monkeypatch: pytest.MonkeyPatch, test_app: FastAPI) -> None:
    """La route doit refuser l'enregistrement lorsque Telegram est désactivé côté serveur."""

    previous_flag = settings.TELEGRAM_ENABLED
    settings.TELEGRAM_ENABLED = False

    async with AsyncClient(app=test_app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/notifications/telegram/register",
            json={"chat_id": "123456"},
        )

    assert response.status_code == 503
    assert response.json()["detail"]

    settings.TELEGRAM_ENABLED = previous_flag


@pytest.mark.asyncio
async def test_get_telegram_status(monkeypatch: pytest.MonkeyPatch, test_app: FastAPI) -> None:
    """Le statut expose la configuration du bot et l'état de l'utilisateur."""

    monkeypatch.setattr(telegram_service, "bot", object())

    async with AsyncClient(app=test_app, base_url="http://test") as client:
        response = await client.get("/api/v1/notifications/telegram/status")

    assert response.status_code == 200
    assert response.json() == {
        "chat_id": None,
        "telegram_enabled": False,
        "bot_configured": True,
    }


@pytest.mark.asyncio
async def test_unlink_telegram_chat(monkeypatch: pytest.MonkeyPatch, test_app: FastAPI) -> None:
    """Délier doit désactiver le flag et effacer le chat_id."""

    # Pré-configurer le mock pour ne pas envoyer de message lors du register
    send_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(telegram_service, "send_link_confirmation", send_mock)
    monkeypatch.setattr(telegram_service, "bot", object())

    async with AsyncClient(app=test_app, base_url="http://test") as client:
        await client.post("/api/v1/notifications/telegram/register", json={"chat_id": "999"})
        response = await client.delete("/api/v1/notifications/telegram/unlink")

    assert response.status_code == 200
    assert response.json() == {
        "chat_id": None,
        "telegram_enabled": False,
        "test_message_sent": None,
    }
