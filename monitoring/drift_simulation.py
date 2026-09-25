import argparse
from pathlib import Path

import pandas as pd


def simulate_drift(
    input_path,
    output_path,
    drift_fraction=0.5
):
    """
    Create a simulated production dataset where
    transaction behavior changes.

    The original dataset is kept untouched.
    """

    df = pd.read_csv(input_path)

    # Convert timestamp
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )

    # Number of rows to modify
    n_drift = int(
        len(df) * drift_fraction
    )

    # Use the final portion as simulated
    # production traffic
    drift_df = df.tail(
        n_drift
    ).copy()

    # --------------------------------------------------
    # Simulate amount drift
    # --------------------------------------------------

    drift_df["Transaction_Amount"] *= 3

    # --------------------------------------------------
    # Simulate hour drift
    # --------------------------------------------------

    drift_df["Timestamp"] = (
        drift_df["Timestamp"]
        .apply(
            lambda x: x.replace(
                hour=18
            )
        )
    )

    # --------------------------------------------------
    # Recalculate related features
    # --------------------------------------------------

    drift_df["transaction_hour"] = (
        drift_df["Timestamp"].dt.hour
    )

    drift_df["amount_vs_avg_7d"] = (
        drift_df["Transaction_Amount"]
        /
        (
            drift_df["Avg_Transaction_Amount_7d"]
            + 1e-6
        )
    )

    drift_df["amount_balance_ratio"] = (
        drift_df["Transaction_Amount"]
        /
        (
            drift_df["Account_Balance"]
            + 1e-6
        )
    )

    # Save
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    drift_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Created drift dataset: "
        f"{output_path}"
    )

    print(
        f"Rows: {len(drift_df)}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/raw/fraud_transactions.csv"
    )

    parser.add_argument(
        "--output",
        default="data/drift/"
                "drift_transactions.csv"
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    simulate_drift(
        args.input,
        args.output,
        args.fraction
    )