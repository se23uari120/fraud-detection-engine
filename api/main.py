import os
import sys
import time
import uuid
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Query
from fastapi.responses import JSONResponse, HTMLResponse
import redis.asyncio as aioredis
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

from config import settings
from common import (
    compute_mock_user_id,
    get_user_feature_key,
    get_user_history_key,
    MODEL_FEATURE_NAMES,
    PCA_FEATURES
)
from api.schemas import TransactionInput, PredictionResponse

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fraud_api")

# Global runtime state containers
state: Dict[str, Any] = {
    "model": None,
    "redis": None,
    "mongo_client": None,
    "mongo_db": None,
    "model_loaded_at": None
}

# In-memory recent predictions ring buffer for instantaneous UI updates
recent_predictions_cache: List[Dict[str, Any]] = []
MAX_CACHE_SIZE = 50

# In-memory sample cache for sub-millisecond stream simulations
sim_cache_df: Optional[pd.DataFrame] = None

def get_sim_sample(n: int = 10) -> pd.DataFrame:
    global sim_cache_df
    if sim_cache_df is None:
        if os.path.exists(settings.CSV_PATH):
            sim_cache_df = pd.read_csv(settings.CSV_PATH, nrows=200)
        else:
            sim_cache_df = pd.DataFrame()
    if not sim_cache_df.empty:
        return sim_cache_df.sample(n=min(n, len(sim_cache_df))).reset_index(drop=True)
    return pd.DataFrame()


async def init_redis():
    """Establish connection to Redis feature store with multi-host fallback."""
    urls = [
        settings.redis_url,
        f"redis://127.0.0.1:{settings.REDIS_PORT}/{settings.REDIS_DB}",
        f"redis://172.22.80.82:{settings.REDIS_PORT}/{settings.REDIS_DB}",
    ]
    for url in urls:
        try:
            r = aioredis.from_url(
                url,
                decode_responses=True,
                socket_timeout=1.5,
                socket_connect_timeout=1.5
            )
            await r.ping()
            state["redis"] = r
            logger.info(f"Connected to Redis feature store at {url}")
            return
        except Exception:
            continue
    logger.warning("Could not connect to Redis at startup. Historical features will fallback to 0.")
    state["redis"] = None


async def init_mongo():
    """Establish connection to MongoDB audit collection."""
    try:
        client = AsyncIOMotorClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=3000
        )
        # Test connection
        await client.admin.command("ping")
        state["mongo_client"] = client
        state["mongo_db"] = client[settings.MONGO_DB]
        logger.info(f"Connected to MongoDB at {settings.MONGO_URI}, DB: {settings.MONGO_DB}")
    except Exception as e:
        logger.warning(f"Could not connect to MongoDB at startup: {e}. Predictions will be served without persistence.")
        state["mongo_client"] = None
        state["mongo_db"] = None


def load_ml_model():
    """Load serialized XGBoost fraud classifier."""
    model_path = settings.MODEL_PATH
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            state["model"] = model
            state["model_loaded_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"Loaded XGBoost model from '{model_path}' successfully.")
        except Exception as e:
            logger.error(f"Error loading model from '{model_path}': {e}", exc_info=True)
            if state.get("model") is None:
                state["model"] = None
    elif state.get("model") is not None:
        logger.info("Using pre-configured in-memory model in state['model'].")
    else:
        logger.warning(
            f"Model file '{model_path}' not found. Please train model using 'py train_model.py'. "
            "Predictions will return HTTP 503 until model is available."
        )
        state["model"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown."""
    logger.info("Initializing Real-Time Fraud Detection Engine...")
    load_ml_model()
    await init_redis()
    await init_mongo()
    yield
    logger.info("Shutting down resources...")
    if state["redis"]:
        await state["redis"].close()
        logger.info("Closed Redis connection.")
    if state["mongo_client"]:
        state["mongo_client"].close()
        logger.info("Closed MongoDB connection.")


app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Low-latency credit card transaction risk assessment and audit pipeline.",
    version="1.0.0",
    lifespan=lifespan
)


async def log_prediction_to_mongo(audit_record: Dict[str, Any]):
    """
    Asynchronously persist prediction audit log to MongoDB cold storage.
    Runs in background task to ensure zero impact on inference latency.
    """
    db = state.get("mongo_db")
    if db is None:
        return
    try:
        collection = db[settings.MONGO_COLLECTION]
        await collection.insert_one(audit_record)
    except PyMongoError as pe:
        logger.error(f"MongoDB write failed for tx {audit_record.get('transaction_id')}: {pe}")
    except Exception as e:
        logger.error(f"Unexpected error writing audit log to MongoDB: {e}")


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Check connectivity to ML model, Redis feature store, and MongoDB."""
    if state["redis"] is None:
        await init_redis()
    if state["mongo_client"] is None:
        await init_mongo()

    redis_healthy = False
    if state["redis"]:
        try:
            await state["redis"].ping()
            redis_healthy = True
        except Exception:
            redis_healthy = False

    mongo_healthy = False
    if state["mongo_client"]:
        try:
            await state["mongo_client"].admin.command("ping")
            mongo_healthy = True
        except Exception:
            mongo_healthy = False

    model_ready = state["model"] is not None

    overall = "HEALTHY" if (model_ready and redis_healthy and mongo_healthy) else "DEGRADED"

    return {
        "status": overall,
        "model_loaded": model_ready,
        "model_path": settings.MODEL_PATH,
        "redis_connected": redis_healthy,
        "mongo_connected": mongo_healthy,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/features/{user_id}", tags=["Feature Store"])
async def get_user_features(user_id: str):
    """Inspect real-time rolling 1-hour feature values for a user from Redis."""
    redis_client = state.get("redis")
    if not redis_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis feature store not connected"
        )
    try:
        feature_key = get_user_feature_key(user_id)
        data = await redis_client.hgetall(feature_key)
        ttl = await redis_client.ttl(feature_key)
        return {
            "user_id": user_id,
            "key": feature_key,
            "ttl_seconds": ttl,
            "features": data
        }
    except Exception as e:
        logger.error(f"Error querying Redis features for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict_transaction(
    tx: TransactionInput,
    background_tasks: BackgroundTasks
):
    """
    Real-time fraud scoring pipeline:
      1. Validate input payload.
      2. Derive deterministic mock user_id = hash(V1 + V2).
      3. Query Redis for rolling 1h historical features (tx count, avg amount).
      4. Assemble ordered feature matrix [V1..V28, Amount, count_1h, avg_amount_1h].
      5. Evaluate XGBoost classifier for fraud probability.
      6. Dispatch asynchronous audit logging to MongoDB predictions_log.
      7. Return risk action: ALLOW (prob < 0.2), REVIEW (0.2 <= prob < 0.8), BLOCK (prob >= 0.8).
    """
    model = state.get("model")
    if model is None:
        # Attempt just-in-time reload if model file was recently written
        load_ml_model()
        model = state.get("model")
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Fraud detection model is not loaded. Ensure 'py train_model.py' has executed."
            )

    # 1. Transaction ID
    tx_id = tx.transaction_id or f"tx_{uuid.uuid4().hex[:12]}"

    # 2. Derive mock user_id from hash(V1 + V2)
    user_id = compute_mock_user_id(tx.V1, tx.V2)

    # 3. Query Redis feature store for user's historical rolling features
    count_1h = 0
    avg_amount_1h = 0.0
    redis_client = state.get("redis")
    if redis_client:
        try:
            feature_key = get_user_feature_key(user_id)
            historical = await redis_client.hgetall(feature_key)
            if historical:
                count_1h = int(historical.get("transaction_count_1h", 0))
                avg_amount_1h = float(historical.get("avg_amount_1h", 0.0))
        except Exception as re:
            logger.warning(f"Redis feature store lookup failed for {user_id}: {re}. Defaulting features to 0.")

    # 4. Construct feature vector matching exact MODEL_FEATURE_NAMES order:
    #    [V1, V2, ..., V28, Amount, transaction_count_1h, avg_amount_1h]
    feature_values: List[float] = [
        getattr(tx, f"V{i}") for i in range(1, 29)
    ]
    feature_values.append(float(tx.Amount))
    feature_values.append(float(count_1h))
    feature_values.append(float(avg_amount_1h))

    features_array = np.array(feature_values, dtype=np.float32).reshape(1, -1)

    # 5. Model Inference
    try:
        probabilities = model.predict_proba(features_array)[0]
        fraud_prob = float(probabilities[1])
    except Exception as me:
        logger.error(f"Inference execution failed for tx {tx_id}: {me}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {me}"
        )

    # Determine automated decision action
    if fraud_prob < settings.THRESHOLD_ALLOW:
        action = "ALLOW"
    elif fraud_prob < settings.THRESHOLD_BLOCK:
        action = "REVIEW"
    else:
        action = "BLOCK"

    # 6. Asynchronously log to MongoDB collection 'predictions_log'
    audit_document = {
        "transaction_id": tx_id,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_transaction": tx.model_dump(),
        "feature_store_features": {
            "transaction_count_1h": count_1h,
            "avg_amount_1h": avg_amount_1h,
        },
        "fraud_probability": round(fraud_prob, 6),
        "action": action,
    }
    recent_predictions_cache.insert(0, audit_document)
    if len(recent_predictions_cache) > MAX_CACHE_SIZE:
        recent_predictions_cache.pop()

    background_tasks.add_task(log_prediction_to_mongo, audit_document)

    # 7. Return PredictionResponse
    return PredictionResponse(
        transaction_id=tx_id,
        fraud_probability=round(fraud_prob, 6),
        action=action,
        user_id=user_id,
        historical_count_1h=count_1h,
        historical_avg_amount_1h=round(avg_amount_1h, 4)
    )


# ---------------------------------------------------------------------------
# Visual UI & Simulator Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Dashboard UI"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard UI"])
async def serve_dashboard():
    """Serves the Sentinel Real-Time Fraud Detection interactive web interface."""
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Dashboard UI template not found")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/api/audit-logs", tags=["Audit Trail"])
async def get_audit_logs(limit: int = Query(default=15, ge=1, le=50)):
    """Retrieve recent fraud prediction audit logs from MongoDB or local cache."""
    db = state.get("mongo_db")
    if db is not None:
        try:
            cursor = db[settings.MONGO_COLLECTION].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
            logs = await cursor.to_list(length=limit)
            if logs:
                return logs
        except Exception as e:
            logger.warning(f"Failed to query MongoDB audit logs: {e}. Falling back to memory cache.")
    
    # Fallback to in-memory recent cache
    return recent_predictions_cache[:limit]


@app.post("/api/simulate-batch", tags=["Simulator"])
async def simulate_stream_batch(count: int = Query(default=10, ge=1, le=50)):
    """
    Simulates a burst of real POS transactions from creditcard.csv,
    pushes them into Redis Stream, updates rolling feature store,
    and runs predictions with live results.
    """
    # Sample random slice from in-memory cache
    df_sample = get_sim_sample(count)
    if df_sample.empty:
        raise HTTPException(status_code=404, detail="Dataset sample not available for simulation")
    results = []

    redis_client = state.get("redis")
    model = state.get("model")

    for idx, row in df_sample.iterrows():
        row_dict = row.to_dict()
        tx_id = f"tx_sim_{int(row_dict.get('Time', 0))}_{uuid.uuid4().hex[:6]}"
        row_dict["transaction_id"] = tx_id

        v1 = float(row_dict.get("V1", 0.0))
        v2 = float(row_dict.get("V2", 0.0))
        amount = float(row_dict.get("Amount", 0.0))
        user_id = compute_mock_user_id(v1, v2)

        count_1h = 0
        avg_amount_1h = 0.0

        if redis_client:
            try:
                # Push to Redis Stream
                await redis_client.xadd(settings.REDIS_STREAM, {"data": json.dumps(row_dict), "transaction_id": tx_id})
                
                # Update user features in Redis
                now = time.time()
                cutoff = now - settings.FEATURE_WINDOW_SECONDS
                history_key = get_user_history_key(user_id)
                feature_key = get_user_feature_key(user_id)

                await redis_client.zremrangebyscore(history_key, "-inf", cutoff)
                await redis_client.zadd(history_key, {f"{tx_id}:{amount}": now})
                await redis_client.expire(history_key, settings.FEATURE_WINDOW_SECONDS)

                members = await redis_client.zrange(history_key, 0, -1)
                count_1h = len(members)
                total_amt = sum(float(m.rsplit(":", 1)[1]) for m in members if ":" in m)
                avg_amount_1h = (total_amt / count_1h) if count_1h > 0 else 0.0

                await redis_client.hset(feature_key, mapping={
                    "transaction_count_1h": str(count_1h),
                    "avg_amount_1h": f"{avg_amount_1h:.4f}",
                    "last_updated": str(now),
                    "user_id": user_id
                })
                await redis_client.expire(feature_key, settings.FEATURE_WINDOW_SECONDS)
            except Exception as re:
                logger.warning(f"Simulate Redis update failed: {re}")

        # Predict
        if model:
            feature_values = [float(row_dict.get(f"V{i}", 0.0)) for i in range(1, 29)]
            feature_values.append(amount)
            feature_values.append(float(count_1h))
            feature_values.append(float(avg_amount_1h))

            arr = np.array(feature_values, dtype=np.float32).reshape(1, -1)
            prob = float(model.predict_proba(arr)[0][1])
            action = "ALLOW" if prob < settings.THRESHOLD_ALLOW else ("REVIEW" if prob < settings.THRESHOLD_BLOCK else "BLOCK")
        else:
            prob = 0.05
            action = "ALLOW"

        item = {
            "transaction_id": tx_id,
            "user_id": user_id,
            "amount": amount,
            "fraud_probability": round(prob, 4),
            "action": action,
            "count_1h": count_1h,
            "avg_amount_1h": round(avg_amount_1h, 2)
        }
        results.append(item)
        
        # Add to recent cache
        recent_predictions_cache.insert(0, {
            "transaction_id": tx_id,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_transaction": {"Amount": amount},
            "feature_store_features": {"transaction_count_1h": count_1h, "avg_amount_1h": avg_amount_1h},
            "fraud_probability": round(prob, 4),
            "action": action
        })

    return {"simulated_count": len(results), "results": results}
