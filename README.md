# Marketing Mix Modeling (MMM) + ETL Pipeline Project

An end-to-end Marketing Mix Modeling (MMM) system built in Python that simulates marketing data, applies feature transformations (Adstock + Hill Saturation), estimates channel impact using OLS regression and operationalises the workflow through an automated ETL pipeline with DuckDB storage and model retraining.



---

# Business Problem

Marketing teams spend across multiple channels such as TV, Search, and Social Media, but struggle to answer:

> Which marketing channels actually drive revenue, and how should we allocate budget efficiently?

This project solves that by:

- Measuring channel-level contribution to revenue
- Capturing delayed marketing effects
- Accounting for diminishing returns
- Estimating ROI per channel
- Automating model retraining as new data arrives

---

# System Overview

The system has two main components:

### 1. MMM Model (model.py)
- Builds transformation + regression-based MMM
- Performs grid search for optimal parameters
- Estimates channel contribution and ROI

### 2. ETL + Pipeline (pipeline.py)
- Simulates weekly data ingestion
- Validates and transforms data
- Loads into DuckDB warehouse
- Re-trains MMM automatically
- Logs each run for monitoring

---

# Architecture
graph TD

A[1_generate_data.py<br>Generate Synthetic Data] --> B[data.csv]

B --> C[pipeline.py<br>ETL Orchestration]

C --> C1[EXTRACT<br>Read CSV + Simulate New Week]
C1 --> C2[TRANSFORM<br>Validation + Adstock + Saturation]
C2 --> C3[LOAD<br>DuckDB Warehouse]

C3 --> D[mmm_warehouse.duckdb]

D --> E[Refit MMM Model<br>OLS Regression]

E --> F[Channel Attribution]
E --> G[ROI Analysis]
E --> H[Model Diagnostics]

E --> I[pipeline_log.csv<br>Run Monitoring]

# Project Structure

graph TD

A[mmm-project] --> B[1_generate_data.py]
A --> C[2_model.py]
A --> D[pipeline.py]
A --> E[transforms.py]

A --> F[data.csv]
A --> G[contributions.csv]
A --> H[model_params.csv]
A --> I[mmm_results.png]
A --> J[pipeline_log.csv]
A --> K[mmm_warehouse.duckdb]
A --> L[requirements.txt]
A --> M[README.md]
A --> N[.gitignore]

---

# Marketing Mix Model (MMM)

## Model Equation

After transformation:

Revenue is modeled as:

Revenue = Intercept + TV + Search + Social + Controls

Where:

- TV, Search, Social → Adstock + Saturation transformed
- Controls → price, holiday, seasonality

---

## Feature Engineering

### 1. Adstock Transformation

Captures delayed advertising effects:
- Marketing impact persists over time
- Each week's spend decays gradually

---

### 2. Hill Saturation

Captures diminishing returns:
- Initial spend has high impact
- Additional spend yields lower incremental returns

---

## Parameter Optimisation

For each channel, the model performs grid search over:

- Decay
- Alpha
- Gamma

The combination that minimises residual error is selected.

---

## Regression Model

A standard OLS regression (Statsmodels) is used to estimate:

- Channel coefficients
- Statistical significance
- Model fit (R², MAPE)

---

# ETL Pipeline (Production Style)

The pipeline (pipeline.py) simulates a real-world marketing data system.

---

## 1. Extract

- Reads data.csv
- Simulates arrival of a new weekly record
- Mimics real marketing data ingestion

In production, this could connect to:

- Google Ads API
- Meta Ads API
- GA4
- CRM systems

---

## 2. Transform

### Data Validation
Checks for:
- Missing dates
- Negative spend values
- Duplicate records
- Invalid revenue values

### Feature Engineering
Applies:
- Adstock transformation
- Hill saturation transformation
(using parameters from model_params.csv if available)

---

## 3. Load

Data is stored in a local DuckDB warehouse:

mmm_warehouse.duckdb

Features:
- Fast analytical queries
- Lightweight deployment
- SQL-based storage
- No external infrastructure required

---

## 4. Model Retraining

After each pipeline run:

- Full dataset is reloaded from DuckDB
- MMM model is refitted
- Metrics are recalculated

Outputs:
- R²
- MAPE
- Observation count

---

## 5. Monitoring & Logging

Every run is logged in:

pipeline_log.csv

Example:

| run_at | status | r2 | mape | n_obs |
|--------|--------|----|------|------|
| 2026-06-09T09:00 | SUCCESS | 0.95 | 4.8 | 157 |

This provides a lightweight model monitoring system.

---

# Running the Project

## 1. Install dependencies

bash pip install -r requirements.txt 

---

## 2. Generate synthetic data

bash python 1_generate_data.py 

---

## 3. Train MMM model

bash python 2_model.py 

Outputs:
- contributions.csv
- model_params.csv
- mmm_results.png

---

## 4. Run ETL pipeline

bash python pipeline.py 

This will:
- ingest new data
- transform features
- load into DuckDB
- retrain MMM
- log results

---

# Outputs

## Channel Contributions

Breakdown of revenue by channel over time.

## ROI per Channel

ROI = Revenue Attributed / Spend

Used to compare marketing efficiency.

## Model Diagnostics

- Actual vs Predicted revenue
- Attribution breakdown
- Channel performance comparison

---


# Future Improvements

- Bayesian MMM (PyMC)
- Time-varying coefficients
- Budget optimisation engine
- Streamlit dashboard
- Cloud deployment (AWS/GCP)
- API-based data ingestion
