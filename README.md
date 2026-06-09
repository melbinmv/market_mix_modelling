# Marketing Mix Modeling (MMM) + ETL Pipeline Project

An end-to-end Marketing Mix Modeling (MMM) system built in Python that simulates marketing data, applies feature transformations (Adstock + Hill Saturation), estimates channel impact using OLS regression, and operationalises the workflow through an automated ETL pipeline with DuckDB storage and model retraining.

This project demonstrates both data science (MMM modelling) and data engineering (ETL pipeline + warehouse + automation) capabilities.

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

### 1. MMM Model (2_model.py)
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

text id="0kq9xm"                         ┌────────────────────┐                         │ 1_generate_data.py │                         │ Synthetic Data      │                         └──────────┬─────────┘                                    │                                    ▼                               ┌─────────┐                               │ data.csv│                               └────┬────┘                                    │          ┌─────────────────────────┼─────────────────────────┐          │                         │                         │          ▼                         ▼                         ▼   ┌────────────────┐     ┌────────────────┐     ┌────────────────────┐  │ 2_model.py     │     │ pipeline.py     │     │ transforms.py      │  │ MMM Training   │     │ ETL Pipeline    │     │ Feature Engineering │  └──────┬─────────┘     └──────┬─────────┘     └─────────┬──────────┘         │                      │                        │         ▼                      ▼                        ▼   ┌────────────────┐   ┌────────────────────┐   ┌────────────────────┐  │ OLS Regression │   │ DuckDB Warehouse   │   │ Adstock + Saturation│  │ Attribution    │   │ mmm_data table     │   │ Transformations     │  └──────┬─────────┘   └──────────┬─────────┘   └────────────────────┘         │                        │         ▼                        ▼   ┌────────────────┐     ┌────────────────────┐  │ ROI Analysis   │     │ pipeline_log.csv   │  │ Contribution   │     │ Monitoring Log     │  └────────────────┘     └────────────────────┘ 

---

# Project Structure

text id="w2d8qs" mmm-project/ │ ├── 1_generate_data.py     # Synthetic marketing data generator ├── 2_model.py             # MMM training + ROI + attribution ├── pipeline.py            # ETL + automation + retraining ├── transforms.py          # Adstock + Hill saturation functions │ ├── data.csv               # Input dataset ├── contributions.csv      # Channel revenue attribution ├── model_params.csv       # Learned MMM parameters ├── mmm_results.png        # Diagnostic plots ├── pipeline_log.csv       # ETL run logs ├── mmm_warehouse.duckdb   # Local analytics warehouse │ ├── requirements.txt ├── README.md └── .gitignore 

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

# Skills Demonstrated

## Data Science
- Marketing Mix Modeling (MMM)
- OLS Regression (Statsmodels)
- Feature Engineering
- Hyperparameter Search
- Attribution Modeling
- ROI Analysis

## Data Engineering
- ETL pipeline design
- Data validation framework
- DuckDB analytical warehouse
- Automated retraining loop
- Pipeline logging & monitoring

## Analytics Engineering
- Business metric modelling
- Revenue decomposition
- Marketing performance analysis

---

# Future Improvements

- Bayesian MMM (PyMC)
- Time-varying coefficients
- Budget optimisation engine
- Streamlit dashboard
- Cloud deployment (AWS/GCP)
- API-based data ingestion
