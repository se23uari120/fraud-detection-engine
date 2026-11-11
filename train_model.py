import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    auc,
    confusion_matrix
)

from config import settings
from common import (
    compute_mock_user_id,
    MODEL_FEATURE_NAMES,
    BASE_FEATURES,
    ROLLING_FEATURES
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fraud_trainer")


def engineer_rolling_features(df: pd.DataFrame, window_seconds: int = 3600) -> pd.DataFrame:
    """
    Computes rolling 1-hour transaction statistics per mock user.
    Uses closed='left' so that the rolling window only considers transactions
    strictly BEFORE the current transaction, preventing data leakage and matching
    the production feature store behavior.
    """
    logger.info("Deriving mock user IDs from hash(V1 + V2)...")
    t0 = time.time()
    df["user_id"] = [
        compute_mock_user_id(v1, v2) for v1, v2 in zip(df["V1"], df["V2"])
    ]
    logger.info(f"Generated user IDs in {time.time() - t0:.2f}s across {df['user_id'].nunique():,} unique users.")

    logger.info(f"Calculating rolling {window_seconds}s historical features (count, avg amount)...")
    t1 = time.time()
    
    # Store original row order to restore after groupby
    df["_orig_index"] = df.index
    df["_datetime"] = pd.to_datetime(df["Time"], unit="s")
    
    # Sort strictly by user and time
    sorted_df = df.sort_values(["user_id", "_datetime", "_orig_index"])
    
    # Group by user and compute rolling window on Time index
    time_indexed = sorted_df.set_index("_datetime")
    rolling_group = (
        time_indexed.groupby("user_id")["Amount"]
        .rolling(f"{window_seconds}s", closed="left")
        .agg(["count", "mean"])
        .fillna(0.0)
        .reset_index()
    )

    # Attach computed values back onto sorted_df
    sorted_df["transaction_count_1h"] = rolling_group["count"].values
    sorted_df["avg_amount_1h"] = rolling_group["mean"].values

    # Restore original row ordering and clean up temporary columns
    df_engineered = sorted_df.sort_values("_orig_index").drop(
        columns=["_orig_index", "_datetime"]
    ).reset_index(drop=True)

    logger.info(
        f"Rolling features engineered in {time.time() - t1:.2f}s. "
        f"Mean tx_count_1h: {df_engineered['transaction_count_1h'].mean():.2f}, "
        f"Mean avg_amount_1h: {df_engineered['avg_amount_1h'].mean():.2f}"
    )

    return df_engineered


def train_fraud_model(
    csv_path: str = settings.CSV_PATH,
    model_output_path: str = settings.MODEL_PATH,
    test_size: float = 0.20,
    random_state: int = 42
) -> XGBClassifier:
    """
    Loads creditcard.csv, engineers rolling features, trains XGBoost, evaluates PR-AUC,
    and serializes the trained model artifact.
    """
    if not os.path.exists(csv_path):
        err_msg = (
            f"Dataset file '{csv_path}' not found.\n"
            f"Please download the Kaggle dataset:\n"
            f"  kaggle datasets download -d mlg-ulb/creditcardfraud\n"
            f"  unzip creditcardfraud.zip"
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"Loading dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} transactions. Columns: {list(df.columns)}")

    # Verify expected columns
    for col in ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {csv_path}")

    # Engineer exact feature store features
    df_features = engineer_rolling_features(
        df,
        window_seconds=settings.FEATURE_WINDOW_SECONDS
    )

    # Prepare training matrices
    X = df_features[MODEL_FEATURE_NAMES].copy()
    y = df_features["Class"].astype(int).values

    num_total = len(y)
    num_fraud = int(np.sum(y))
    num_legit = num_total - num_fraud
    imbalance_ratio = num_legit / max(num_fraud, 1)

    logger.info(
        f"Dataset distribution: {num_total:,} total | "
        f"Legit (0): {num_legit:,} ({num_legit/num_total*100:.2f}%) | "
        f"Fraud (1): {num_fraud:,} ({num_fraud/num_total*100:.3f}%) | "
        f"Imbalance ratio: {imbalance_ratio:.1f}:1"
    )

    # Stratified Train/Test split
    logger.info(f"Splitting dataset (test_size={test_size}, random_state={random_state}, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Train set: {len(X_train):,} samples ({np.sum(y_train)} fraud)")
    logger.info(f"Test set:  {len(X_test):,} samples ({np.sum(y_test)} fraud)")

    # Configure XGBoost with scale_pos_weight for extreme class imbalance
    # Using square root of imbalance ratio or tuned scale to balance precision & recall
    scale_weight = min(np.sqrt(imbalance_ratio), 50.0)

    model = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="aucpr",
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1
    )

    logger.info("Training XGBoost classifier...")
    t_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    logger.info(f"Model training completed in {time.time() - t_start:.2f}s")

    # Evaluation
    logger.info("Evaluating model performance on test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = (y_pred_proba >= 0.50).astype(int)

    # Precision-Recall AUC (Primary metric for highly imbalanced fraud datasets)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_curve_auc = auc(recall, precision)

    cm = confusion_matrix(y_test, y_pred_class)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 65)
    print("           REAL-TIME FRAUD DETECTION MODEL EVALUATION")
    print("=" * 65)
    print(f" Test Set Size            : {len(y_test):,} transactions")
    print(f" Test Set Frauds          : {np.sum(y_test)} positive cases")
    print("-" * 65)
    print(f" Precision-Recall AUC     : {pr_auc:.4f} (PR-AUC / Average Precision)")
    print(f" Area Under ROC (ROC-AUC) : {roc_auc:.4f}")
    print("-" * 65)
    print(" Confusion Matrix (Threshold = 0.50):")
    print(f"   True Negatives  (Legit -> Legit) : {tn:,}")
    print(f"   False Positives (Legit -> Fraud) : {fp:,}")
    print(f"   False Negatives (Fraud -> Legit) : {fn:,}")
    print(f"   True Positives  (Fraud -> Fraud) : {tp:,}")
    print("-" * 65)
    print(" Detailed Classification Report:")
    print(classification_report(y_test, y_pred_class, target_names=["Legit", "Fraud"], digits=4))
    print("=" * 65 + "\n")

    # Feature Importance analysis
    feature_importances = pd.Series(model.feature_importances_, index=MODEL_FEATURE_NAMES).sort_values(ascending=False)
    logger.info("Top 10 Most Predictive Features:")
    for feat, imp in feature_importances.head(10).items():
        logger.info(f"   - {feat:25s}: {imp:.4f}")

    # Serialize trained model artifact
    logger.info(f"Serializing trained XGBoost model to '{model_output_path}'...")
    joblib.dump(model, model_output_path)
    logger.info(f"Model successfully saved to '{model_output_path}' ({os.path.getsize(model_output_path) / 1024:.1f} KB)")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Real-Time Fraud Detection Model")
    parser.add_argument(
        "--csv",
        type=str,
        default=settings.CSV_PATH,
        help="Path to creditcard.csv dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=settings.MODEL_PATH,
        help="Output path for fraud_xgboost.joblib"
    )
    args = parser.parse_args()

    try:
        train_fraud_model(
            csv_path=args.csv,
            model_output_path=args.output
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

