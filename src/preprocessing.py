import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


TARGET = "Fraud_Label"

DROP_COLS = [
    "Transaction_ID",
    "User_ID",
    "Timestamp",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time and transaction-behavior features."""
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df["transaction_hour"] = df["Timestamp"].dt.hour
    df["transaction_day"] = df["Timestamp"].dt.day
    df["transaction_month"] = df["Timestamp"].dt.month
    df["transaction_dayofweek"] = df["Timestamp"].dt.dayofweek

    df["amount_vs_avg_7d"] = (
        df["Transaction_Amount"]
        / (df["Avg_Transaction_Amount_7d"] + 1e-6)
    )

    df["failed_transaction_ratio"] = (
        df["Failed_Transaction_Count_7d"]
        / (df["Daily_Transaction_Count"] + 1)
    )

    df["amount_balance_ratio"] = (
        df["Transaction_Amount"]
        / (df["Account_Balance"] + 1e-6)
    )

    return df


def temporal_split(df: pd.DataFrame, train_q=0.70, val_q=0.85):
    """Chronological 70/15/15 split."""
    df = df.sort_values("Timestamp").reset_index(drop=True)

    train_end = df["Timestamp"].quantile(train_q)
    val_end = df["Timestamp"].quantile(val_q)

    train_df = df[df["Timestamp"] < train_end].copy()
    val_df = df[
        (df["Timestamp"] >= train_end)
        & (df["Timestamp"] < val_end)
    ].copy()
    test_df = df[df["Timestamp"] >= val_end].copy()

    return train_df, val_df, test_df


def make_xy(data: pd.DataFrame):
    """Remove target and identifiers/raw timestamp."""
    X = data.drop(columns=[TARGET] + DROP_COLS)
    y = data[TARGET].astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype="float32",
                ),
                categorical_cols,
            )
        ],
        remainder="passthrough",
    )

    return preprocessor, categorical_cols, numerical_cols


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df = add_features(df)

    train_df, val_df, test_df = temporal_split(df)

    X_train, y_train = make_xy(train_df)
    X_val, y_val = make_xy(val_df)
    X_test, y_test = make_xy(test_df)

    preprocessor, categorical_cols, numerical_cols = build_preprocessor(
        X_train
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return {
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_val_processed": X_val_processed,
        "X_test_processed": X_test_processed,
        "preprocessor": preprocessor,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
    }


def save_preprocessor(preprocessor, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
