# Marketing Mix Model

A from-scratch MMM built in pure Python. No black-box libraries.

## Files

| File | Purpose |
|------|---------|
| `1_generate_data.py` | Creates synthetic weekly dataset (`data.csv`) with known ground truth |
| `transforms.py` | Adstock + Hill saturation transforms (reusable module) |
| `2_model.py` | Fits the MMM: grid search → OLS → decomposition → ROI → plots |
| `3_pipeline.py` | ETL pipeline: Extract → Transform → Load (DuckDB) → Refit model |

## Quick Start

```bash
# 1. Install dependencies
pip install numpy pandas statsmodels matplotlib duckdb

# 2. Generate data
python 1_generate_data.py

# 3. Fit the model
python 2_model.py

# 4. Run the ETL pipeline (simulates a new week arriving + refits model)
python 3_pipeline.py
```

## How the Model Works

```
Raw Spend  →  Adstock (carry-over)  →  Hill Saturation (diminishing returns)
           →  OLS Regression  →  Channel Contributions  →  ROI
```

1. **Adstock** — each week's spend carries over to future weeks (decay rate controls how fast)
2. **Hill Saturation** — diminishing returns on spend (doubling budget ≠ doubling sales)
3. **OLS** — regresses transformed spend + controls against revenue
4. **Decomposition** — splits revenue into base + per-channel contribution
5. **ROI** — revenue attributed / spend for each channel

## Scheduling the Pipeline (cron)

To run every Monday at 9am:

```bash
crontab -e
# Add this line:
0 9 * * MON cd /path/to/your/project && python 3_pipeline.py >> pipeline.log 2>&1
```

## Outputs

After running `2_model.py`:
- `contributions.csv` — weekly revenue decomposition by channel
- `model_params.csv` — fitted transform parameters + coefficients
- `mmm_results.png` — three charts: actual vs fitted, decomposition, ROI

After running `3_pipeline.py`:
- `mmm_warehouse.duckdb` — local data warehouse (all historical data)
- `pipeline_log.csv` — log of every pipeline run with R² and MAPE

## What to Talk About in Interviews

- Why adstock? Marketing effects don't disappear instantly — TV ads have weeks of residual impact
- Why Hill saturation? Spend has diminishing returns — the 10th £1k is worth less than the 1st
- Why OLS? Interpretable coefficients, fast, works well with weekly aggregated data
- Next steps: Bayesian MMM (PyMC-Marketing) for uncertainty quantification, cross-validation