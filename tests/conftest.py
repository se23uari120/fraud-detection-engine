"""
Shared pytest fixtures for the Real-Time Fraud Detection Engine test suite.

The FastAPI lifespan (startup/shutdown) attempts real network connections to Redis
and MongoDB, and loads the model from disk. These fixtures patch those side-effects
so every test controls its own state and is fully isolated from external services.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app, state


@pytest.fixture(autouse=False)
def mock_lifespan_services(monkeypatch):
    """
    Patch all three lifespan side-effects before TestClient starts the app:
      - load_ml_model: no-op (does NOT overwrite state["model"])
      - init_redis:    no-op (does NOT overwrite state["redis"])
      - init_mongo:    no-op (does NOT overwrite state["mongo_client"] / state["mongo_db"])

    Individual tests set state["model"], state["redis"], etc. themselves.
    """
    async def _noop_init_redis():
        pass

    async def _noop_init_mongo():
        pass

    def _noop_load_model():
        pass

    monkeypatch.setattr("api.main.init_redis", _noop_init_redis)
    monkeypatch.setattr("api.main.init_mongo", _noop_init_mongo)
    monkeypatch.setattr("api.main.load_ml_model", _noop_load_model)

