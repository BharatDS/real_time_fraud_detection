import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PSI
# ============================================================

def calculate_psi(
    reference,
    current,
    bins=10
):
    """
    Calculate Population Stability Index.

    reference = training/reference distribution
    current   = production/current distribution
    """

    reference = np.asarray(
        reference,
        dtype=float
    )

    current = np.asarray(
        current,
        dtype=float
    )

    # Remove NaN / infinity
    reference = reference[
        np.isfinite(reference)
    ]

    current = current[
        np.isfinite(current)
    ]

    # Create bins using reference distribution
    breakpoints = np.percentile(
        reference,
        np.linspace(
            0,
            100,
            bins + 1
        )
    )

    # Remove duplicate breakpoints
    breakpoints = np.unique(
        breakpoints
    )

    if len(breakpoints) < 3:
        return 0.0

    reference_counts = np.histogram(
        reference,
        bins=breakpoints
    )[0]

    current_counts = np.histogram(
        current,
        bins=breakpoints
    )[0]

    # Convert to proportions
    reference_pct = (
        reference_counts
        /
        len(reference)
    )

    current_pct = (
        current_counts
        /
        len(current)
    )

    # Avoid log(0)
    epsilon = 1e-6

    reference_pct = np.clip(
        reference_pct,
        epsilon,
        None
    )

    current_pct = np.clip(
        current_pct,
        epsilon,
        None
    )

    psi = np.sum(
        (
            current_pct
            -
            reference_pct
        )
        *
        np.log(
            current_pct
            /
            reference_pct
        )
    )

    return float(psi)


# ============================================================
# DRIFT STATUS
# ============================================================

def drift_status(psi):

    if psi < 0.10:
        return "NORMAL"

    elif psi < 0.25:
        return "WARNING"

    return "DRIFT"


# ============================================================
# MONITOR DATA
# ============================================================

def monitor(
    reference_path,
    current_path
):

    reference = pd.read_csv(
        reference_path
    )

    current = pd.read_csv(
        current_path
    )

    features = [
        "Transaction_Amount",
        "Daily_Transaction_Count",
        "Avg_Transaction_Amount_7d",
        "Failed_Transaction_Count_7d",
        "Risk_Score",
        "Transaction_Distance",
    ]

    results = {}

    for feature in features:

        if (
            feature not in reference.columns
            or
            feature not in current.columns
        ):
            continue

        psi = calculate_psi(
            reference[feature],
            current[feature]
        )

        results[feature] = {
            "psi": round(
                psi,
                4
            ),
            "status": drift_status(
                psi
            )
        }

    # --------------------------------------------------
    # Overall drift
    # --------------------------------------------------

    psi_values = [
        value["psi"]
        for value in results.values()
    ]

    max_psi = (
        max(psi_values)
        if psi_values
        else 0.0
    )

    if max_psi >= 0.25:
        overall_status = "DRIFT"

    elif max_psi >= 0.10:
        overall_status = "WARNING"

    else:
        overall_status = "NORMAL"

    output = {
        "overall_status": overall_status,
        "max_psi": round(
            max_psi,
            4
        ),
        "features": results
    }

    return output


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = monitor(
        "data/raw/fraud_transactions.csv",
        "data/drift/drift_transactions.csv"
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    Path(
        "models/drift_metrics.json"
    ).write_text(
        json.dumps(
            result,
            indent=2
        )
    )