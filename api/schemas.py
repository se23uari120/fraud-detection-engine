import uuid
from typing import Optional, Literal
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """
    Schema for incoming transaction payload.
    Features V1 through V28 correspond to PCA transformations.
    Amount is the transaction value.
    """
    transaction_id: Optional[str] = Field(
        default=None,
        description="Optional unique transaction identifier. If not provided, a UUID is generated."
    )
    Time: Optional[float] = Field(
        default=None,
        description="Seconds elapsed since the reference transaction in the dataset."
    )
    Amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction amount in currency units."
    )
    V1: float = Field(..., description="PCA transformed feature V1")
    V2: float = Field(..., description="PCA transformed feature V2")
    V3: float = Field(..., description="PCA transformed feature V3")
    V4: float = Field(..., description="PCA transformed feature V4")
    V5: float = Field(..., description="PCA transformed feature V5")
    V6: float = Field(..., description="PCA transformed feature V6")
    V7: float = Field(..., description="PCA transformed feature V7")
    V8: float = Field(..., description="PCA transformed feature V8")
    V9: float = Field(..., description="PCA transformed feature V9")
    V10: float = Field(..., description="PCA transformed feature V10")
    V11: float = Field(..., description="PCA transformed feature V11")
    V12: float = Field(..., description="PCA transformed feature V12")
    V13: float = Field(..., description="PCA transformed feature V13")
    V14: float = Field(..., description="PCA transformed feature V14")
    V15: float = Field(..., description="PCA transformed feature V15")
    V16: float = Field(..., description="PCA transformed feature V16")
    V17: float = Field(..., description="PCA transformed feature V17")
    V18: float = Field(..., description="PCA transformed feature V18")
    V19: float = Field(..., description="PCA transformed feature V19")
    V20: float = Field(..., description="PCA transformed feature V20")
    V21: float = Field(..., description="PCA transformed feature V21")
    V22: float = Field(..., description="PCA transformed feature V22")
    V23: float = Field(..., description="PCA transformed feature V23")
    V24: float = Field(..., description="PCA transformed feature V24")
    V25: float = Field(..., description="PCA transformed feature V25")
    V26: float = Field(..., description="PCA transformed feature V26")
    V27: float = Field(..., description="PCA transformed feature V27")
    V28: float = Field(..., description="PCA transformed feature V28")

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "tx_sample_001",
                "Amount": 149.62,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Response schema returning fraud evaluation results.
    """
    transaction_id: str = Field(..., description="Unique transaction ID")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted fraud probability (0.0 to 1.0)")
    action: Literal["ALLOW", "REVIEW", "BLOCK"] = Field(..., description="Automated risk decision")
    user_id: Optional[str] = Field(default=None, description="Derived mock user ID")
    historical_count_1h: Optional[int] = Field(default=0, description="Transactions by this user in past 1 hour")
    historical_avg_amount_1h: Optional[float] = Field(default=0.0, description="Average amount in past 1 hour")

