import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from preprocessing import prepare_data


def evaluate(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main(args):
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    data = prepare_data(args.data)

    X_train = data["X_train_processed"]
    X_val = data["X_val_processed"]
    X_test = data["X_test_processed"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    # ---------------------------------------------------------
    # Baseline 1: always predict NORMAL
    # ---------------------------------------------------------
    normal_model = DummyClassifier(strategy="most_frequent")
    normal_model.fit(X_train, y_train)

    normal_proba = normal_model.predict_proba(X_val)[:, 1]
    normal_metrics = evaluate(y_val, normal_proba, threshold=0.5)

    # ---------------------------------------------------------
    # Baseline 2: Logistic Regression
    # ---------------------------------------------------------
    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=42,
    )

    logistic.fit(X_train, y_train)

    logistic_proba = logistic.predict_proba(X_val)[:, 1]
    logistic_metrics = evaluate(y_val, logistic_proba, threshold=0.5)

    # ---------------------------------------------------------
    # XGBoost baseline
    # ---------------------------------------------------------
    n_normal = int((y_train == 0).sum())
    n_fraud = int((y_train == 1).sum())
    scale_pos_weight = n_normal / n_fraud

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    xgb.fit(X_train, y_train)

    xgb_val_proba = xgb.predict_proba(X_val)[:, 1]
    xgb_metrics = evaluate(y_val, xgb_val_proba, threshold=0.5)

    # ---------------------------------------------------------
    # Final untouched test evaluation at default threshold.
    # Threshold optimization is done separately in optimize.py.
    # ---------------------------------------------------------
    xgb_test_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_test_metrics = evaluate(y_test, xgb_test_proba, threshold=0.5)

    print("\n=== DATA ===")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    print("\n=== BASELINE 1: ALWAYS NORMAL ===")
    print(json.dumps(normal_metrics, indent=2))

    print("\n=== BASELINE 2: LOGISTIC REGRESSION ===")
    print(json.dumps(logistic_metrics, indent=2))

    print("\n=== XGBOOST VALIDATION ===")
    print(json.dumps(xgb_metrics, indent=2))

    print("\n=== XGBOOST TEST @ 0.50 ===")
    print(json.dumps(xgb_test_metrics, indent=2))

    joblib.dump(
        xgb,
        Path(args.model_dir) / "xgboost_fraud_baseline.pkl",
    )

    joblib.dump(
        logistic,
        Path(args.model_dir) / "logistic_regression.pkl",
    )

    joblib.dump(
        data["preprocessor"],
        Path(args.model_dir) / "preprocessor.pkl",
    )

    with open(Path(args.model_dir) / "features.json", "w") as f:
        json.dump(
            {
                "categorical_columns": data["categorical_cols"],
                "numerical_columns": data["numerical_cols"],
                "model_feature_count": int(X_train.shape[1]),
            },
            f,
            indent=2,
        )

    with open(Path(args.model_dir) / "baseline_metrics.json", "w") as f:
        json.dump(
            {
                "always_normal": normal_metrics,
                "logistic_regression": logistic_metrics,
                "xgboost_validation": xgb_metrics,
                "xgboost_test_default_threshold": xgb_test_metrics,
                "scale_pos_weight": scale_pos_weight,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/raw/fraud_transactions.csv",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
    )
    args = parser.parse_args()
    main(args)
