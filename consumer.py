import os
import sys
import json
import time
import signal
import logging
from typing import Optional, Dict, Any, List
import redis
from redis.exceptions import ConnectionError as RedisConnectionError, ResponseError, RedisError

from config import settings
from common import (
    compute_mock_user_id,
    get_user_feature_key,
    get_user_history_key,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fraud_consumer")

running = True


def handle_shutdown(signum, frame):
    global running
    logger.info("Shutdown signal received. Exiting consumer loop...")
    running = False


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def get_redis_client() -> redis.Redis:
    """Initialize Redis connection with retry logic."""
    client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD or None,
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5
    )
    client.ping()
    logger.info(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}")
    return client


def setup_consumer_group(client: redis.Redis, stream: str, group: str):
    """
    Ensure the consumer group exists on the stream.
    Creates the stream if it does not exist.
    """
    try:
        client.xgroup_create(stream, group, id="0", mkstream=True)
        logger.info(f"Created consumer group '{group}' on stream '{stream}'")
    except ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"Consumer group '{group}' already exists on stream '{stream}'")
        else:
            logger.warning(f"Consumer group creation returned: {e}")


def update_user_rolling_features(
    client: redis.Redis,
    user_id: str,
    tx_id: str,
    amount: float,
    window_seconds: int = 3600
) -> Dict[str, Any]:
    """
    Maintains rolling 1-hour window features for a user using Redis Sorted Sets and Hashes.
    
    Logic:
      1. Key 'user_history:{user_id}' stores sorted set of (tx_id:amount, timestamp).
      2. Expire old records outside the rolling 1-hour window.
      3. Add current transaction.
      4. Compute rolling transaction count and average amount.
      5. Store rolling metrics in Redis Hash 'user_features:{user_id}'.
      6. Set TTL (3600s) on both keys so inactive users automatically expire.
    """
    now = time.time()
    cutoff = now - window_seconds
    history_key = get_user_history_key(user_id)
    feature_key = get_user_feature_key(user_id)

    # Use pipeline for atomic transaction updates
    pipe = client.pipeline()
    # 1. Prune events older than the 1-hour rolling window
    pipe.zremrangebyscore(history_key, "-inf", cutoff)
    # 2. Add current transaction to the sorted set: member="<tx_id>:<amount>", score=now
    member = f"{tx_id}:{amount}"
    pipe.zadd(history_key, {member: now})
    # 3. Refresh TTL on the history sorted set
    pipe.expire(history_key, window_seconds)
    # 4. Fetch all remaining elements in the window
    pipe.zrange(history_key, 0, -1)
    
    results = pipe.execute()
    current_entries: List[str] = results[3]

    # Calculate rolling metrics
    count_1h = len(current_entries)
    total_amount_1h = 0.0
    for entry in current_entries:
        try:
            parts = entry.rsplit(":", 1)
            total_amount_1h += float(parts[1])
        except (ValueError, IndexError):
            continue

    avg_amount_1h = (total_amount_1h / count_1h) if count_1h > 0 else 0.0

    # 5. Update Redis Hash with calculated rolling features and TTL
    hash_data = {
        "transaction_count_1h": str(count_1h),
        "avg_amount_1h": f"{avg_amount_1h:.4f}",
        "last_updated": str(now),
        "user_id": user_id
    }

    pipe = client.pipeline()
    pipe.hset(feature_key, mapping=hash_data)
    pipe.expire(feature_key, window_seconds)
    pipe.execute()

    return {
        "user_id": user_id,
        "transaction_count_1h": count_1h,
        "avg_amount_1h": round(avg_amount_1h, 4)
    }


def process_stream_message(client: redis.Redis, msg_id: str, fields: Dict[str, str]):
    """
    Parses a single stream entry, extracts transaction fields,
    computes mock user_id, and updates the Redis feature store.
    """
    raw_data = fields.get("data")
    if not raw_data:
        logger.warning(f"Message {msg_id} missing 'data' field. Raw fields: {fields}")
        return

    try:
        tx_data = json.loads(raw_data)
    except json.JSONDecodeError as e:
        logger.error(f"Malformed JSON in message {msg_id}: {e}")
        return

    v1 = tx_data.get("V1")
    v2 = tx_data.get("V2")
    amount = float(tx_data.get("Amount", 0.0))
    tx_id = fields.get("transaction_id") or tx_data.get("transaction_id") or msg_id

    if v1 is None or v2 is None:
        logger.warning(f"Transaction {tx_id} missing V1 or V2. Cannot derive user_id.")
        return

    # Derive mock user ID deterministically: hash(V1 + V2)
    user_id = compute_mock_user_id(float(v1), float(v2))

    # Update rolling feature store
    features = update_user_rolling_features(
        client=client,
        user_id=user_id,
        tx_id=tx_id,
        amount=amount,
        window_seconds=settings.FEATURE_WINDOW_SECONDS
    )

    return features


def run_consumer(max_messages: Optional[int] = None):
    """
    Main loop consuming from Redis Stream and maintaining rolling features.
    Uses consumer groups with fallback to direct XREAD.
    """
    client = get_redis_client()
    stream_name = settings.REDIS_STREAM
    group_name = settings.REDIS_CONSUMER_GROUP
    consumer_name = settings.REDIS_CONSUMER_NAME

    setup_consumer_group(client, stream_name, group_name)

    logger.info(
        f"Fraud Feature Consumer started. Listening to stream '{stream_name}' "
        f"as group '{group_name}', worker '{consumer_name}'..."
    )

    processed_count = 0
    start_time = time.time()

    while running:
        if max_messages is not None and processed_count >= max_messages:
            logger.info(f"Reached max messages limit ({max_messages}). Exiting consumer.")
            break
        try:
            # Read new messages for this consumer group
            response = client.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={stream_name: ">"},
                count=50,
                block=2000
            )

            if not response:
                continue

            for stream_key, messages in response:
                for msg_id, fields in messages:
                    if not running:
                        break

                    try:
                        features = process_stream_message(client, msg_id, fields)
                        # Acknowledge processed message in the stream
                        client.xack(stream_name, group_name, msg_id)
                        processed_count += 1

                        if processed_count % 500 == 0 or processed_count == 1:
                            rate = processed_count / max(time.time() - start_time, 0.001)
                            logger.info(
                                f"Processed {processed_count:,} messages. "
                                f"Last user {features.get('user_id') if features else 'N/A'}: "
                                f"count_1h={features.get('transaction_count_1h') if features else 0}, "
                                f"avg_amt_1h={features.get('avg_amount_1h') if features else 0.0} "
                                f"({rate:.1f} msg/s)"
                            )
                    except Exception as ex:
                        logger.error(f"Error processing message {msg_id}: {ex}", exc_info=True)

                    if max_messages is not None and processed_count >= max_messages:
                        break

        except RedisConnectionError as ce:
            logger.error(f"Redis connection lost: {ce}. Retrying in 3 seconds...")
            time.sleep(3)
            try:
                client = get_redis_client()
                setup_consumer_group(client, stream_name, group_name)
            except Exception:
                pass
        except RedisError as re:
            logger.error(f"Redis error: {re}. Retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Unexpected error in consumer loop: {e}", exc_info=True)
            time.sleep(1)

    logger.info(f"Consumer shutdown complete. Total processed: {processed_count:,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Fraud Feature Store Consumer")
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Optional message processing limit (default: run indefinitely)"
    )
    args = parser.parse_args()
    run_consumer(max_messages=args.max_messages)

