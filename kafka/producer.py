import json
import random
import time
import uuid

from kafka import KafkaProducer


import os

KAFKA_SERVER = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092"
)
TOPIC = "transactions"


producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)


def generate_transaction():
    """
    Generate a synthetic transaction for streaming demonstration.

    Most transactions are normal.
    A small percentage are generated with suspicious
    behavioral characteristics.
    """

    is_suspicious = random.random() < 0.05

    # --------------------------------------------------
    # Normal transaction
    # --------------------------------------------------
    transaction_id = (
    f"TX{uuid.uuid4().hex[:8].upper()}"
    )
    if not is_suspicious:

        amount = round(
            random.uniform(200, 3000),
            2
        )

        avg_amount = round(
            random.uniform(500, 2500),
            2
        )

        account_balance = round(
            random.uniform(10000, 100000),
            2
        )

        daily_count = random.randint(
            1,
            10
        )

        failed_count = random.randint(
            0,
            1
        )

        risk_score = round(
            random.uniform(0.05, 0.45),
            4
        )

        transaction_type = random.choice(
            [
                "Online",
                "POS",
                "ATM",
            ]
        )

        device_type = random.choice(
            [
                "Mobile",
                "Desktop",
            ]
        )

        previous_fraud = 0

    # --------------------------------------------------
    # Suspicious transaction
    # --------------------------------------------------

    else:

        amount = round(
            random.uniform(20000, 100000),
            2
        )

        avg_amount = round(
            random.uniform(500, 3000),
            2
        )

        account_balance = round(
            random.uniform(5000, 30000),
            2
        )

        daily_count = random.randint(
            10,
            30
        )

        failed_count = random.randint(
            4,
            10
        )

        risk_score = round(
            random.uniform(0.75, 1.0),
            4
        )

        transaction_type = random.choice(
            [
                "Online",
                "ATM",
            ]
        )

        device_type = random.choice(
            [
                "Mobile",
                "Desktop",
            ]
        )

        previous_fraud = random.choice(
            [
                0,
                1,
            ]
        )

    # --------------------------------------------------
    # Derived features
    # --------------------------------------------------

    amount_vs_avg = (
        amount /
        (avg_amount + 1e-6)
    )

    failed_ratio = (
        failed_count /
        (daily_count + 1)
    )

    amount_balance_ratio = (
        amount /
        (account_balance + 1e-6)
    )

    # Random timestamp-related features
    hour = random.randint(0, 23)
    day = random.randint(1, 28)
    month = random.randint(1, 12)

    # Monday = 0, Sunday = 6
    dayofweek = random.randint(0, 6)

    is_weekend = int(
        dayofweek >= 5
    )

    transaction = {
        "Transaction_ID": transaction_id,
        "Transaction_Amount": amount,
        "Transaction_Type": transaction_type,

        "Account_Balance": account_balance,

        "Device_Type": device_type,
        "Location": random.choice(
            [
                "Hyderabad",
                "Mumbai",
                "Delhi",
                "Bangalore",
                "Chennai",
            ]
        ),

        "Merchant_Category": random.choice(
            [
                "Grocery",
                "Electronics",
                "Travel",
                "Restaurant",
                "Clothing",
            ]
        ),

        "IP_Address_Flag": random.choice(
            [
                0,
                1,
            ]
        ),

        "Previous_Fraudulent_Activity": previous_fraud,

        "Daily_Transaction_Count": daily_count,

        "Avg_Transaction_Amount_7d": avg_amount,

        "Failed_Transaction_Count_7d": failed_count,

        "Card_Type": random.choice(
            [
                "Visa",
                "Mastercard",
                "Amex",
            ]
        ),

        "Card_Age": random.randint(
            30,
            1500
        ),

        "Transaction_Distance": round(
            random.uniform(0.5, 100),
            2
        ),

        "Authentication_Method": random.choice(
            [
                "OTP",
                "Biometric",
                "Password",
            ]
        ),

        "Risk_Score": risk_score,

        "Is_Weekend": is_weekend,

        "transaction_hour": hour,
        "transaction_day": day,
        "transaction_month": month,
        "transaction_dayofweek": dayofweek,

        "amount_vs_avg_7d": round(
            amount_vs_avg,
            4
        ),

        "failed_transaction_ratio": round(
            failed_ratio,
            4
        ),

        "amount_balance_ratio": round(
            amount_balance_ratio,
            4
        ),
    }

    return transaction


def main():

    print(
        "Starting Kafka producer..."
    )

    print(
        f"Sending transactions to "
        f"{TOPIC}..."
    )

    while True:

        transaction = generate_transaction()

        producer.send(
            TOPIC,
            value=transaction
        )

        producer.flush()

        print(
            "Transaction sent | "
            f"amount="
            f"{transaction['Transaction_Amount']:.2f} | "
            f"risk_score="
            f"{transaction['Risk_Score']:.2f}"
        )

        time.sleep(1)


if __name__ == "__main__":
    main()