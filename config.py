import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables and .env file.
    """
    # Redis configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_STREAM: str = "fraud_stream"
    REDIS_CONSUMER_GROUP: str = "fraud_feature_group"
    REDIS_CONSUMER_NAME: str = "feature_worker_1"

    # MongoDB configuration
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "fraud_detection"
    MONGO_COLLECTION: str = "predictions_log"

    # Model and Data Paths
    MODEL_PATH: str = "fraud_xgboost.joblib"
    CSV_PATH: str = "creditcard.csv"

    # Stream Producer settings
    PRODUCER_SLEEP_SEC: float = 0.05

    # Decision Thresholds
    THRESHOLD_ALLOW: float = 0.20
    THRESHOLD_BLOCK: float = 0.80

    # Feature engineering parameters
    FEATURE_WINDOW_SECONDS: int = 3600  # 1 hour rolling window

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


# Singleton instance
settings = Settings()

