import json
from pathlib import Path

import joblib
import pandas as pd
from kafka import KafkaConsumer
import os

KAFKA_SERVER = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "xgboost_fraud_optuna.pkl"
)

PREPROCESSOR_PATH = (
    BASE_DIR
    / "models"
    / "preprocessor.pkl"
)

THRESHOLD_PATH = (
    BASE_DIR
    / "models"
    / "threshold.json"
)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_PATH = LOG_DIR / "predictions.jsonl"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading fraud detection model...")

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

with open(THRESHOLD_PATH, "r") as f:
    threshold_config = json.load(f)

THRESHOLD = threshold_config["threshold"]

print(
    f"Model loaded successfully | "
    f"Threshold = {THRESHOLD}"
)


# ============================================================
# KAFKA CONSUMER
# ============================================================

consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers=KAFKA_SERVER,
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="fraud-detection-consumer",
    value_deserializer=lambda value: json.loads(
        value.decode("utf-8")
    ),
)


print("Kafka consumer started...")
print("Waiting for transactions...\n")


# ============================================================
# REAL-TIME INFERENCE
# ============================================================

for message in consumer:

    transaction = message.value

    try:

        # Convert Kafka message to DataFrame
        data = pd.DataFrame(
            [transaction]
        )

        # Apply training-time preprocessing
        X_processed = preprocessor.transform(
            data
        )

        # Get fraud probability
        fraud_probability = model.predict_proba(
            X_processed
        )[0, 1]

        # Apply optimized threshold
        is_fraud = (
            fraud_probability >= THRESHOLD
        )

        action = (
            "BLOCK"
            if is_fraud
            else "ALLOW"
        )
        # --------------------------------------------------
        # Fraud alert
        # --------------------------------------------------

        if is_fraud:
            
            transaction_id = transaction.get(
                "Transaction_ID",
                "STREAM_TX"
            )

            print("\n" + "!" * 60)
            print("🚨 FRAUD ALERT")
            print("!" * 60)

            print(
                f"Transaction: {transaction_id}"
            )

            print(
                f"Amount: ₹"
                f"{transaction['Transaction_Amount']:,.2f}"
            )

            print(
                f"Probability: "
                f"{fraud_probability:.4f}"
            )

            print(
                f"Decision: {action}"
            )

            print("!" * 60 + "\n")

        # --------------------------------------------------
        # Console output
        # --------------------------------------------------

        print("=" * 60)

        print(
            f"Transaction Amount : "
            f"{transaction['Transaction_Amount']:.2f}"
        )

        print(
            f"Risk Score         : "
            f"{transaction['Risk_Score']:.2f}"
        )

        print(
            f"Fraud Probability  : "
            f"{fraud_probability:.4f}"
        )

        print(
            f"Threshold          : "
            f"{THRESHOLD:.2f}"
        )

        print(
            f"Prediction         : "
            f"{'FRAUD' if is_fraud else 'NORMAL'}"
        )

        print(
            f"Action             : "
            f"{action}"
        )

        # --------------------------------------------------
        # Prediction log
        # --------------------------------------------------

        prediction_record = {
            "transaction": transaction,
            "fraud_probability": float(
                fraud_probability
            ),
            "threshold": float(
                THRESHOLD
            ),
            "is_fraud": bool(
                is_fraud
            ),
            "action": action,
        }

        with open(
            LOG_PATH,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    prediction_record
                )
                + "\n"
            )

    except Exception as e:

        print(
            f"Error processing transaction: {e}"
        )