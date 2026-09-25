import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

LOG_PATH = (
    BASE_DIR
    / "logs"
    / "predictions.jsonl"
)

THRESHOLD_PATH = (
    BASE_DIR
    / "models"
    / "threshold.json"
)

DRIFT_PATH = (
    BASE_DIR
    / "models"
    / "drift_metrics.json"
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Fraud Detection Monitor",
    layout="wide"
)

st.title(
    "🚨 Real-Time Fraud Detection Monitor"
)

st.caption(
    "Kafka + XGBoost fraud detection monitoring"
)


# ============================================================
# LOAD THRESHOLD
# ============================================================

with open(
    THRESHOLD_PATH,
    "r"
) as f:

    threshold = json.load(f)[
        "threshold"
    ]


# ============================================================
# LOAD PREDICTIONS
# ============================================================

records = []

if LOG_PATH.exists():

    with open(
        LOG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                continue


# ============================================================
# METRICS
# ============================================================

if records:

    df = pd.DataFrame(
        records
    )

    total_transactions = len(
        df
    )

    fraud_count = int(
        df["is_fraud"].sum()
    )

    fraud_rate = (
        fraud_count
        /
        total_transactions
        if total_transactions
        else 0
    )

    avg_probability = (
        df["fraud_probability"]
        .mean()
    )

else:

    total_transactions = 0
    fraud_count = 0
    fraud_rate = 0
    avg_probability = 0


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Transactions Processed",
        f"{total_transactions:,}"
    )

with col2:

    st.metric(
        "Fraud Detected",
        f"{fraud_count:,}"
    )

with col3:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2%}"
    )

with col4:

    st.metric(
        "Average Probability",
        f"{avg_probability:.4f}"
    )


st.divider()


# ============================================================
# MODEL INFO
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "Model Configuration"
    )

    st.write(
        f"**Threshold:** {threshold:.2f}"
    )

    st.write(
        "**Model:** Optuna-tuned XGBoost"
    )

with col2:

    st.subheader(
        "Prediction Distribution"
    )

    if records:

        st.bar_chart(
            df["is_fraud"]
            .value_counts()
        )


st.divider()


# ============================================================
# DRIFT
# ============================================================

st.subheader(
    "📊 Concept Drift"
)

if DRIFT_PATH.exists():

    with open(
        DRIFT_PATH,
        "r"
    ) as f:

        drift = json.load(f)

    status = drift[
        "overall_status"
    ]

    max_psi = drift[
        "max_psi"
    ]

    st.metric(
        "Maximum PSI",
        f"{max_psi:.4f}"
    )

    if status == "NORMAL":

        st.success(
            "✅ No significant drift detected"
        )

    elif status == "WARNING":

        st.warning(
            "⚠️ Possible distribution shift"
        )

    else:

        st.error(
            "🚨 DATA DRIFT DETECTED"
        )

    drift_rows = []

    for feature, values in (
        drift["features"].items()
    ):

        drift_rows.append(
            {
                "Feature": feature,
                "PSI": values["psi"],
                "Status": values["status"],
            }
        )

    st.dataframe(
        pd.DataFrame(
            drift_rows
        ),
        use_container_width=True
    )

else:

    st.info(
        "Run monitoring/drift_simulation.py "
        "and monitoring/monitor.py first."
    )


# ============================================================
# RECENT PREDICTIONS
# ============================================================

st.divider()

st.subheader(
    "Recent Predictions"
)

if records:

    display_df = df[
        [
            "fraud_probability",
            "threshold",
            "is_fraud",
            "action",
        ]
    ].tail(20)

    st.dataframe(
        display_df,
        use_container_width=True
    )

else:

    st.info(
        "No prediction records yet."
    )