from pathlib import Path
import json

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


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


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

with open(THRESHOLD_PATH, "r") as f:
    threshold_config = json.load(f)

THRESHOLD = threshold_config["threshold"]


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description=(
        "XGBoost-based fraud detection inference service"
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class Transaction(BaseModel):

    Transaction_Amount: float
    Transaction_Type: str

    Account_Balance: float

    Device_Type: str
    Location: str
    Merchant_Category: str

    IP_Address_Flag: int

    Previous_Fraudulent_Activity: int

    Daily_Transaction_Count: int

    Avg_Transaction_Amount_7d: float

    Failed_Transaction_Count_7d: int

    Card_Type: str

    Card_Age: int

    Transaction_Distance: float

    Authentication_Method: str

    Risk_Score: float

    Is_Weekend: int

    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_dayofweek: int

    amount_vs_avg_7d: float
    failed_transaction_ratio: float
    amount_balance_ratio: float


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert request to DataFrame
    data = pd.DataFrame(
        [transaction.model_dump()]
    )

    # Apply exactly the same preprocessing
    # used during model training
    X_processed = preprocessor.transform(
        data
    )

    # Fraud probability
    fraud_probability = model.predict_proba(
        X_processed
    )[0, 1]

    # Apply optimized threshold
    is_fraud = (
        fraud_probability >= THRESHOLD
    )

    # Business action
    action = (
        "BLOCK"
        if is_fraud
        else "ALLOW"
    )

    return {
        "fraud_probability": round(
            float(fraud_probability),
            4
        ),
        "threshold": round(
            float(THRESHOLD),
            4
        ),
        "is_fraud": bool(is_fraud),
        "action": action,
    }