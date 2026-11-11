"""
Integration & unit tests for the Real-Time Fraud Detection Engine.

All tests that exercise the FastAPI app use the `mock_lifespan_services` fixture
(defined in conftest.py) to suppress real Redis/Mongo/disk connections during startup,
so each test can inject its own state["model"], state["redis"], state["mongo_db"].
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

from config import settings
from common import (
    compute_mock_user_id,
    get_user_feature_key,
    get_user_history_key,
    MODEL_FEATURE_NAMES,
    BASE_FEATURES,
    ROLLING_FEATURES,
)
from api.schemas import TransactionInput, PredictionResponse
from api.main import app, state


# ---------------------------------------------------------------------------
# Pure unit tests (no HTTP, no external services)
# ---------------------------------------------------------------------------

class TestFeatureConstants:
    def test_feature_list_length_and_order(self):
        """Verify feature ordering: [V1..V28, Amount, count_1h, avg_amount_1h] = 31 features."""
        assert len(MODEL_FEATURE_NAMES) == 31
        assert MODEL_FEATURE_NAMES[:28] == [f"V{i}" for i in range(1, 29)]
        assert MODEL_FEATURE_NAMES[28] == "Amount"
        assert MODEL_FEATURE_NAMES[29] == "transaction_count_1h"
        assert MODEL_FEATURE_NAMES[30] == "avg_amount_1h"

    def test_base_features(self):
        assert len(BASE_FEATURES) == 29  # 28 PCA + Amount
        assert "Amount" in BASE_FEATURES

    def test_rolling_features(self):
        assert ROLLING_FEATURES == ["transaction_count_1h", "avg_amount_1h"]


class TestUserIdDeriving:
    def test_determinism_same_inputs(self):
        """Same inputs must always produce the same user_id."""
        uid1 = compute_mock_user_id(-1.3598, -0.0727)
        uid2 = compute_mock_user_id(-1.3598, -0.0727)
        assert uid1 == uid2

    def test_format(self):
        uid = compute_mock_user_id(1.0, 2.0)
        assert uid.startswith("user_")

    def test_different_inputs_different_ids(self):
        uid_a = compute_mock_user_id(0.0, 0.0)
        uid_b = compute_mock_user_id(10.5, 20.3)
        assert uid_a != uid_b

    def test_redis_keys(self):
        uid = compute_mock_user_id(1.5, -0.5)
        assert get_user_feature_key(uid) == f"user_features:{uid}"
        assert get_user_history_key(uid) == f"user_history:{uid}"


class TestTransactionInputSchema:
    def test_valid_payload(self):
        data = {
            "transaction_id": "test_tx_001",
            "Amount": 100.50,
            **{f"V{i}": 0.1 * i for i in range(1, 29)}
        }
        tx = TransactionInput(**data)
        assert tx.Amount == 100.50
        assert tx.V1 == pytest.approx(0.1)
        assert tx.V28 == pytest.approx(2.8)
        assert tx.transaction_id == "test_tx_001"

    def test_optional_transaction_id(self):
        """transaction_id must be optional (defaults to None, API auto-generates)."""
        data = {
            "Amount": 50.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        tx = TransactionInput(**data)
        assert tx.transaction_id is None

    def test_invalid_negative_amount(self):
        """Amount must be >= 0."""
        import pydantic
        data = {
            "Amount": -10.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        with pytest.raises(pydantic.ValidationError):
            TransactionInput(**data)


# ---------------------------------------------------------------------------
# API endpoint tests (use mock_lifespan_services fixture)
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def _make_payload(self, amount: float = 55.0, v_val: float = -0.5, tx_id: str = None):
        payload = {
            "Amount": amount,
            **{f"V{i}": v_val for i in range(1, 29)}
        }
        if tx_id:
            payload["transaction_id"] = tx_id
        return payload

    def test_allow_action_with_redis_features(self, mock_lifespan_services):
        """
        Full predict flow: 15% fraud probability with Redis historical features
        should return ALLOW + correct enriched feature values.
        """
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {
            "transaction_count_1h": "3",
            "avg_amount_1h": "75.25",
        }

        mock_mongo_db = MagicMock()

        state["model"] = mock_model
        state["redis"] = mock_redis
        state["mongo_db"] = mock_mongo_db

        with TestClient(app) as client:
            response = client.post("/predict", json=self._make_payload(tx_id="tx_mock_999"))

        assert response.status_code == 200
        data = response.json()
        assert data["transaction_id"] == "tx_mock_999"
        assert data["fraud_probability"] == 0.15
        assert data["action"] == "ALLOW"
        assert data["historical_count_1h"] == 3
        assert data["historical_avg_amount_1h"] == 75.25

        # Verify model received 31-feature vector
        call_args = mock_model.predict_proba.call_args[0][0]
        assert call_args.shape == (1, 31)
        assert call_args[0][29] == pytest.approx(3.0)
        assert call_args[0][30] == pytest.approx(75.25)

    def test_block_action_no_redis(self, mock_lifespan_services):
        """92% fraud probability with no Redis should return BLOCK and default rolling features."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.08, 0.92]])

        state["model"] = mock_model
        state["redis"] = None
        state["mongo_db"] = None

        with TestClient(app) as client:
            response = client.post("/predict", json=self._make_payload(amount=1000.0, v_val=2.5))

        assert response.status_code == 200
        data = response.json()
        assert data["fraud_probability"] == 0.92
        assert data["action"] == "BLOCK"
        assert data["historical_count_1h"] == 0
        assert data["historical_avg_amount_1h"] == 0.0

    def test_review_action_threshold(self, mock_lifespan_services):
        """50% probability should map to REVIEW (between ALLOW and BLOCK thresholds)."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.50, 0.50]])

        state["model"] = mock_model
        state["redis"] = None
        state["mongo_db"] = None

        with TestClient(app) as client:
            response = client.post("/predict", json=self._make_payload())

        assert response.status_code == 200
        assert response.json()["action"] == "REVIEW"

    def test_auto_generates_transaction_id(self, mock_lifespan_services):
        """When transaction_id is omitted, API must auto-generate a non-empty string."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.99, 0.01]])
        state["model"] = mock_model
        state["redis"] = None
        state["mongo_db"] = None

        with TestClient(app) as client:
            response = client.post("/predict", json=self._make_payload())

        assert response.status_code == 200
        tx_id = response.json()["transaction_id"]
        assert isinstance(tx_id, str)
        assert len(tx_id) > 0

    def test_model_not_loaded_returns_503(self, mock_lifespan_services):
        """When model is absent, /predict must return 503 Service Unavailable."""
        state["model"] = None
        state["redis"] = None
        state["mongo_db"] = None

        with TestClient(app) as client:
            response = client.post("/predict", json=self._make_payload())

        assert response.status_code == 503


class TestHealthEndpoint:
    def test_degraded_when_all_disconnected(self, mock_lifespan_services):
        """All services disconnected → status must be DEGRADED and model_loaded False."""
        state["model"] = None
        state["redis"] = None
        state["mongo_client"] = None
        state["mongo_db"] = None

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "DEGRADED"
        assert data["model_loaded"] is False
        assert data["redis_connected"] is False
        assert data["mongo_connected"] is False

    def test_healthy_when_all_connected(self, mock_lifespan_services):
        """All mocked services healthy → status must be HEALTHY."""
        mock_model = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_mongo_client = MagicMock()
        mock_mongo_client.admin.command = AsyncMock(return_value={"ok": 1})

        state["model"] = mock_model
        state["redis"] = mock_redis
        state["mongo_client"] = mock_mongo_client
        state["mongo_db"] = MagicMock()

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "HEALTHY"
        assert data["model_loaded"] is True
