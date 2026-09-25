# Real-Time Fraud Detection System

An end-to-end **real-time fraud detection platform** built with **XGBoost, FastAPI, Apache Kafka, Streamlit, PSI-based drift monitoring, and Docker**.

The system simulates financial transactions, streams them through Kafka, performs real-time fraud prediction using an optimized XGBoost model, blocks suspicious transactions using a configurable decision threshold, generates fraud alerts, and monitors feature distribution drift.

---

## Architecture

```text
                  ┌─────────────────────┐
                  │ Transaction Producer│
                  └──────────┬──────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Apache Kafka    │
                    │ transactions    │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Fraud Consumer      │
                  │                     │
                  │ XGBoost Prediction  │
                  │ Threshold Decision  │
                  └──────────┬──────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
          ┌──────────────┐       ┌────────────────┐
          │ Fraud Alert  │       │ predictions    │
          │ BLOCK/ALLOW  │       │ .jsonl         │
          └──────────────┘       └───────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │ Streamlit       │
                                │ Dashboard       │
                                └─────────────────┘

                  ┌─────────────────────┐
                  │ FastAPI Prediction  │
                  │ API                 │
                  └─────────────────────┘

                  ┌─────────────────────┐
                  │ PSI Drift Monitor   │
                  └─────────────────────┘
```

---

## Features

* **Temporal train/validation/test split** to simulate production deployment.
* Feature engineering for transaction time, amount ratios, failed transaction ratios, and account balance relationships.
* **XGBoost fraud classifier**.
* **Optuna hyperparameter optimization**.
* Validation-based decision threshold optimization.
* **FastAPI inference API**.
* **Apache Kafka real-time transaction streaming**.
* Real-time `ALLOW` / `BLOCK` decisions.
* Fraud alert generation.
* JSONL prediction logging.
* **PSI-based concept/data drift monitoring**.
* Simulated production drift.
* **Streamlit monitoring dashboard**.
* Dockerized deployment using Docker Compose.

---

# Machine Learning

## Dataset

The project uses a synthetic financial transaction dataset containing:

* Transaction information
* Account information
* Device information
* Location
* Merchant category
* Transaction history
* Risk-related features
* Fraud labels

The dataset contains approximately **50,000 transactions**.

The target variable is:

```text
Fraud_Label
```

where:

```text
0 → Normal transaction
1 → Fraudulent transaction
```

### Important note

The dataset contains highly predictive precomputed behavioral/risk features. As a result, the offline benchmark performance is unusually high.

Therefore, the reported metrics should **not be interpreted as representative of real-world fraud detection performance**.

The primary goal of this project is to demonstrate an end-to-end production-style ML system including:

* temporal validation
* model optimization
* threshold selection
* API serving
* streaming inference
* alerting
* monitoring
* containerization

---

# Feature Engineering

The following features were created:

```text
transaction_hour
transaction_day
transaction_month
transaction_dayofweek
amount_vs_avg_7d
failed_transaction_ratio
amount_balance_ratio
```

For example:

```text
amount_vs_avg_7d =
Transaction_Amount / Avg_Transaction_Amount_7d
```

and:

```text
failed_transaction_ratio =
Failed_Transaction_Count_7d /
(Daily_Transaction_Count + 1)
```

---

# Data Splitting

Instead of randomly splitting the dataset, transactions are sorted chronologically.

The dataset is divided into:

```text
Train      70%
Validation 15%
Test       15%
```

This prevents future transactions from leaking into the training process and better approximates a real deployment scenario.

---

# Model

The fraud classifier uses **XGBoost**.

Hyperparameters were optimized using Optuna.

The optimized configuration includes:

```text
n_estimators      = 588
max_depth         = 6
learning_rate     ≈ 0.046
subsample         ≈ 0.889
colsample_bytree  ≈ 0.710
min_child_weight  = 10
```

The optimization objective was validation **PR-AUC**.

---

# Threshold Optimization

Instead of simply using the default probability threshold of `0.5`, multiple thresholds were evaluated on the validation set.

The selected threshold was:

```text
0.85
```

This threshold was selected using validation F1.

The production threshold is configurable through:

```text
models/threshold.json
```

In a real financial system, threshold selection would additionally incorporate the business costs of:

* false positives
* false negatives
* manual review
* customer friction
* financial losses

---

# Test Results

The final model was evaluated on an untouched temporal test set.

| Metric              |   Result |
| ------------------- | -------: |
| Precision           |   1.0000 |
| Recall              |   0.9984 |
| F1 Score            |   0.9992 |
| ROC-AUC             | 0.999999 |
| PR-AUC              | 0.999997 |
| False Positive Rate |   0.0000 |

Confusion matrix:

```text
TN = 5060
FP = 0
FN = 4
TP = 2436
```

Again, these results are specific to the synthetic dataset and should not be treated as real-world fraud detection benchmarks.

---

# Real-Time Pipeline

## 1. Transaction Producer

The producer continuously generates synthetic transactions and sends them to Kafka.

Example:

```text
Transaction
    ↓
Kafka topic: transactions
```

The producer simulates both normal and suspicious transactions.

---

## 2. Kafka

Apache Kafka acts as the real-time transaction streaming layer.

The main topic is:

```text
transactions
```

Kafka allows the transaction producer and fraud detection consumer to operate independently.

---

## 3. Fraud Consumer

The consumer:

1. Reads transactions from Kafka.
2. Converts them into model features.
3. Applies the saved preprocessing pipeline.
4. Generates a fraud probability using XGBoost.
5. Compares the probability against the configured threshold.
6. Produces an `ALLOW` or `BLOCK` decision.
7. Logs the prediction.
8. Generates an alert for detected fraud.

Example:

```text
Fraud probability: 0.97
Threshold:         0.85
Decision:          BLOCK
```

---

# FastAPI

The trained model is also exposed through a REST API.

Start the API locally:

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

Available endpoints:

```text
GET  /health
POST /predict
```

The Swagger interface can be used to test individual transactions.

---

# Monitoring

The project includes a simple feature drift monitoring system using **Population Stability Index (PSI)**.

The monitored features include:

```text
Transaction_Amount
Daily_Transaction_Count
Avg_Transaction_Amount_7d
Failed_Transaction_Count_7d
Risk_Score
Transaction_Distance
```

The current implementation uses:

```text
PSI < 0.10       → NORMAL
0.10–0.25        → WARNING
PSI > 0.25       → DRIFT
```

A drift simulation is included to artificially change transaction distributions.

Run:

```bash
python monitoring/drift_simulation.py
```

Then:

```bash
python monitoring/monitor.py
```

The resulting drift metrics are saved to:

```text
models/drift_metrics.json
```

---

# Streamlit Dashboard

The dashboard provides monitoring information such as:

* Transactions processed
* Fraud count
* Fraud rate
* Average fraud probability
* Current threshold
* Prediction distribution
* Recent predictions
* Feature drift
* PSI values
* Drift status

Start the dashboard:

```bash
streamlit run dashboard/app.py
```

Then open:

```text
http://localhost:8501
```

---

# Docker Deployment

The application can be run using Docker Compose.

### Build the images

```bash
docker compose build
```

### Start the system

```bash
docker compose up -d
```

Check the services:

```bash
docker compose ps
```

Expected services:

```text
fraud-api
fraud-kafka
fraud-producer
fraud-consumer
```

Kafka topic initialization is handled automatically by:

```text
kafka-init
```

---

# Docker Logs

View fraud detection predictions:

```bash
docker compose logs -f consumer
```

View transaction generation:

```bash
docker compose logs -f producer
```

View API logs:

```bash
docker compose logs -f fraud-api
```

View Kafka logs:

```bash
docker compose logs -f kafka
```

---

# Testing Kafka

List Kafka topics:

```bash
docker exec fraud-kafka \
/opt/kafka/bin/kafka-topics.sh \
--bootstrap-server localhost:9092 \
--list
```

Consume transactions directly:

```bash
docker exec -it fraud-kafka \
/opt/kafka/bin/kafka-console-consumer.sh \
--bootstrap-server localhost:9092 \
--topic transactions \
--from-beginning
```

---

# Tech Stack

| Component                   | Technology     |
| --------------------------- | -------------- |
| Language                    | Python         |
| ML                          | XGBoost        |
| Hyperparameter Optimization | Optuna         |
| Data Processing             | Pandas, NumPy  |
| ML Preprocessing            | Scikit-learn   |
| API                         | FastAPI        |
| Streaming                   | Apache Kafka   |
| Dashboard                   | Streamlit      |
| Monitoring                  | PSI            |
| Containerization            | Docker         |
| Orchestration               | Docker Compose |

---

# Running the Complete System

### Option 1 — Docker

```bash
docker compose build
docker compose up -d
```

Check:

```bash
docker compose ps
```

Then monitor predictions:

```bash
docker compose logs -f consumer
```

Open the API:

```text
http://localhost:8000/docs
```

Start the dashboard separately:

```bash
streamlit run dashboard/app.py
```

Open:

```text
http://localhost:8501
```

---

# Model Training

To retrain the model:

```bash
python src/train.py
```

Run hyperparameter optimization:

```bash
python src/optimize.py
```

Optimize the decision threshold:

```bash
python src/threshold.py
```

---

# Key Engineering Decisions

### Temporal validation

Fraud detection is a time-dependent problem. A chronological split better represents deployment than a purely random split.

### Threshold optimization

The probability threshold is separated from model training so the business decision layer can be adjusted without retraining the classifier.

### Kafka

Kafka provides a scalable streaming layer between transaction generation and fraud inference.

### FastAPI

FastAPI provides a lightweight REST inference interface for applications that need synchronous predictions.

### Monitoring

PSI provides a simple mechanism for detecting changes in incoming feature distributions.

### Docker

Docker packages the inference components and Kafka infrastructure into a reproducible local deployment.

---

# Limitations

This is a portfolio/engineering project rather than a production financial fraud platform.

Important limitations include:

* The dataset is synthetic.
* Several features are already highly predictive of fraud.
* The benchmark therefore produces unusually high offline metrics.
* The drift simulation is artificially generated.
* PSI detects feature-distribution changes but does not directly measure model performance degradation.
* Real production systems would require authentication, authorization, encryption, schema validation, model versioning, observability, alert routing, and more sophisticated drift/performance monitoring.
* Real fraud systems would also require cost-sensitive threshold optimization and human review workflows.

---

# Future Improvements

Potential extensions include:

* Kafka Schema Registry
* Redis feature store
* MLflow model registry
* Prometheus/Grafana monitoring
* SHAP-based fraud explanations
* Real-time alert service
* PostgreSQL transaction storage
* Model versioning
* Automated retraining
* Online feature engineering
* Cost-sensitive threshold optimization
* Kubernetes deployment
* Authentication and API rate limiting

---

## Author

**Bharat Goel**

M.Tech — Artificial Intelligence
Indian Institute of Technology Hyderabad
