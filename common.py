import hashlib
from typing import List

# Base features from the Kaggle dataset used as model inputs
PCA_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)]
BASE_FEATURES: List[str] = PCA_FEATURES + ["Amount"]

# Feature store rolling features
ROLLING_FEATURES: List[str] = ["transaction_count_1h", "avg_amount_1h"]

# Full ordered list of features expected by the XGBoost model
MODEL_FEATURE_NAMES: List[str] = BASE_FEATURES + ROLLING_FEATURES


def compute_mock_user_id(v1: float, v2: float, num_users: int = 5000) -> str:
    """
    Derives a deterministic user_id from hash(V1 + V2).
    
    The ULB credit card dataset lacks a natural cardholder ID. As specified,
    a mock user_id is derived from hash(V1 + V2). By taking a deterministic hash
    of the rounded sum modulo 5,000, we map the 284,807 transactions across 48 hours
    into realistic cardholder accounts. This generates authentic repeat transaction
    velocity and rolling 1-hour statistics for both ML training and real-time serving.
    """
    combined_val = round(float(v1) + float(v2), 2)
    digest = hashlib.md5(str(combined_val).encode("utf-8")).hexdigest()
    user_num = int(digest, 16) % num_users
    return f"user_{user_num:04d}"


def get_user_feature_key(user_id: str) -> str:
    """Redis Hash key for user's computed rolling features."""
    return f"user_features:{user_id}"


def get_user_history_key(user_id: str) -> str:
    """Redis Sorted Set key for user's rolling transaction timestamps & amounts."""
    return f"user_history:{user_id}"
