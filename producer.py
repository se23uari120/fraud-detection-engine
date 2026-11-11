import os
import sys
import json
import time
import signal
import logging
import argparse
from typing import Optional
import pandas as pd
import redis
from redis.exceptions import ConnectionError as RedisConnectionError, RedisError

from config import settings

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fraud_producer")

# Global graceful shutdown flag
running = True


def handle_shutdown(signum, frame):
    global running
    logger.info("Shutdown signal received. Exiting producer loop...")
    running = False


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def get_redis_client() -> redis.Redis:
    """Initialize and test Redis connection with authentication support."""
    try:
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
    except RedisConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise


def produce_transactions(
    csv_path: str,
    sleep_interval: float,
    max_records: Optional[int] = None
):
    """
    Reads transactions from CSV and streams them into Redis Stream 'fraud_stream'.
    """
    if not os.path.exists(csv_path):
        err_msg = (
            f"Dataset not found at '{csv_path}'. Please ensure the Kaggle dataset is downloaded:\n"
            f"  kaggle datasets download -d mlg-ulb/creditcardfraud\n"
            f"  unzip creditcardfraud.zip"
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"Loading transactions from {csv_path}...")
    try:
        # Load CSV efficiently
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(df):,} transactions from {csv_path}")
    except Exception as e:
        logger.error(f"Failed to read CSV file {csv_path}: {e}")
        raise

    client = get_redis_client()
    stream_name = settings.REDIS_STREAM

    total_rows = len(df) if max_records is None else min(len(df), max_records)
    logger.info(
        f"Starting transaction firehose -> Redis Stream: '{stream_name}' "
        f"({total_rows:,} records, delay: {sleep_interval}s)..."
    )

    published_count = 0
    start_time = time.time()

    for idx, row in df.iterrows():
        if not running:
            break
        if max_records and published_count >= max_records:
            logger.info(f"Reached max records limit ({max_records}). Stopping stream.")
            break

        row_dict = row.to_dict()

        # Construct unique transaction ID if not in dataset
        tx_id = f"tx_{int(row_dict.get('Time', 0))}_{idx:07d}"
        row_dict["transaction_id"] = tx_id

        # Package payload as JSON
        payload = json.dumps(row_dict)

        # Publish to Redis Stream with retry logic
        pushed = False
        retry_delay = 1.0
        while running and not pushed:
            try:
                msg_id = client.xadd(
                    name=stream_name,
                    fields={"data": payload, "transaction_id": tx_id}
                )
                published_count += 1
                pushed = True

                if published_count % 500 == 0 or published_count == 1:
                    rate = published_count / max(time.time() - start_time, 0.001)
                    logger.info(
                        f"Published tx #{published_count:,}/{total_rows:,} "
                        f"[ID: {tx_id}] -> Stream Msg: {msg_id} ({rate:.1f} tx/s)"
                    )
            except (RedisConnectionError, RedisError) as re:
                logger.warning(f"Redis write failed: {re}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 10.0)

        # Sleep to simulate real-world POS traffic rate
        if sleep_interval > 0:
            time.sleep(sleep_interval)

    duration = time.time() - start_time
    logger.info(
        f"Producer completed. Published {published_count:,} transactions "
        f"in {duration:.2f}s ({published_count / max(duration, 0.001):.1f} tx/s)."
    )


def main():
    parser = argparse.ArgumentParser(description="Real-Time Fraud Transaction Stream Producer")
    parser.add_argument(
        "--csv",
        type=str,
        default=settings.CSV_PATH,
        help="Path to creditcard.csv (default: from config/env)"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=settings.PRODUCER_SLEEP_SEC,
        help="Sleep interval in seconds between transactions (default: 0.05)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of transactions to stream (default: unlimited)"
    )
    args = parser.parse_args()

    try:
        produce_transactions(
            csv_path=args.csv,
            sleep_interval=args.sleep,
            max_records=args.limit
        )
    except Exception as e:
        logger.error(f"Fatal error in producer: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

