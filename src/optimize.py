import argparse
import json
from pathlib import Path

import joblib
import optuna
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from preprocessing import prepare_data


def main(args):
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    data = prepare_data(args.data)

    X_train = data["X_train_processed"]
    X_val = data["X_val_processed"]
    y_train = data["y_train"]
    y_val = data["y_val"]

    n_normal = int((y_train == 0).sum())
    n_fraud = int((y_train == 1).sum())
    scale_pos_weight = n_normal / n_fraud

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 10
            ),
        }

        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]

        return average_precision_score(y_val, val_proba)

    study = optuna.create_study(
        direction="maximize",
        study_name="fraud_xgboost_pr_auc",
    )

    study.optimize(
        objective,
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print("\n=== BEST OPTUNA RESULT ===")
    print("Best PR-AUC:", study.best_value)
    print("Best parameters:")
    print(json.dumps(study.best_params, indent=2))

    best_model = XGBClassifier(
        **study.best_params,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    best_model.fit(X_train, y_train)

    joblib.dump(
        best_model,
        Path(args.model_dir) / "xgboost_fraud_optuna.pkl",
    )

    with open(Path(args.model_dir) / "optuna_results.json", "w") as f:
        json.dump(
            {
                "best_pr_auc": float(study.best_value),
                "best_params": study.best_params,
                "scale_pos_weight": scale_pos_weight,
                "n_trials": args.trials,
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
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
    )
    args = parser.parse_args()
    main(args)
