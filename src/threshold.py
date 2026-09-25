import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from preprocessing import prepare_data


def evaluate_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main(args):
    Path(args.model_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # 1. PREPARE DATA
    # --------------------------------------------------

    data = prepare_data(args.data)

    # --------------------------------------------------
    # 2. LOAD OPTUNA MODEL
    # --------------------------------------------------

    model = joblib.load(
        Path(args.model_dir)
        / "xgboost_fraud_optuna.pkl"
    )

    # --------------------------------------------------
    # 3. VALIDATION PREDICTIONS
    # --------------------------------------------------

    val_proba = model.predict_proba(
        data["X_val_processed"]
    )[:, 1]

    # --------------------------------------------------
    # 4. THRESHOLD SEARCH
    # --------------------------------------------------

    thresholds = np.arange(
        0.05,
        0.91,
        0.05
    )

    results = [
        evaluate_threshold(
            data["y_val"],
            val_proba,
            float(threshold),
        )
        for threshold in thresholds
    ]

    results_df = pd.DataFrame(results)

    print("\n=== THRESHOLD RESULTS ===")

    print(
        results_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # 5. SELECT THRESHOLD USING VALIDATION F1
    # --------------------------------------------------

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    print("\n=== SELECTED THRESHOLD ===")
    print(
        json.dumps(
            best_row.to_dict(),
            indent=2
        )
    )

    # --------------------------------------------------
    # 6. SAVE THRESHOLD
    # --------------------------------------------------

    threshold_path = (
        Path(args.model_dir)
        / "threshold.json"
    )

    with open(threshold_path, "w") as f:
        json.dump(
            {
                "threshold": best_threshold,
                "selection_metric": "F1",
                "selection_split": "validation",
                "note": (
                    "Threshold chosen by maximum "
                    "validation F1. A production threshold "
                    "should ultimately be chosen using "
                    "business costs for false positives "
                    "and false negatives."
                ),
            },
            f,
            indent=2,
        )

    # --------------------------------------------------
    # 7. SAVE THRESHOLD SEARCH RESULTS
    # --------------------------------------------------

    results_df.to_csv(
        Path(args.model_dir)
        / "threshold_results.csv",
        index=False,
    )

    # --------------------------------------------------
    # 8. FINAL TEST EVALUATION
    # --------------------------------------------------

    test_proba = model.predict_proba(
        data["X_test_processed"]
    )[:, 1]

    test_metrics = evaluate_threshold(
        data["y_test"],
        test_proba,
        best_threshold,
    )

    # Add threshold-independent metrics
    test_metrics["roc_auc"] = float(
        roc_auc_score(
            data["y_test"],
            test_proba
        )
    )

    test_metrics["pr_auc"] = float(
        average_precision_score(
            data["y_test"],
            test_proba
        )
    )

    print("\n=== FINAL TEST RESULTS ===")

    print(
        json.dumps(
            test_metrics,
            indent=2
        )
    )

    # --------------------------------------------------
    # 9. SAVE FINAL TEST METRICS
    # --------------------------------------------------

    metrics_path = (
        Path(args.model_dir)
        / "final_test_metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(
            test_metrics,
            f,
            indent=2
        )

    print(
        f"\nSaved final test metrics to: "
        f"{metrics_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/raw/fraud_transactions.csv"
    )

    parser.add_argument(
        "--model-dir",
        default="models"
    )

    args = parser.parse_args()

    main(args)