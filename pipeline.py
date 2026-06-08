"""
pipeline.py
-------------
ETL pipeline that:
  1. EXTRACTS new weekly data (simulates an API / CSV drop)
  2. TRANSFORMS it (cleans, validates, applies adstock + saturation)
  3. LOADS it into a local DuckDB database
  4. Re-fits the MMM model on the full updated dataset
  5. Logs each run to pipeline_log.csv

Run manually:    python pipeline.py
Schedule via cron (see README):  0 9 * * MON python 3_pipeline.py
"""

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
import logging
import sys
from datetime import datetime
from pathlib import Path
from transforms import apply_transforms

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH       = "mmm_warehouse.duckdb"
LOG_FILE      = "pipeline_log.csv"
CHANNELS      = ["tv_spend", "search_spend", "social_spend"]
CONTROLS      = ["price", "holiday", "seasonality"]
KPI           = "revenue"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACT
# ═══════════════════════════════════════════════════════════════════════════════
def extract(source_path: str = "data.csv") -> pd.DataFrame:
    """
    In production: replace with API calls to Google Ads, Meta, GA4, etc.
    For portfolio: reads from CSV and appends one synthetic new week.
    """
    log.info(f"[EXTRACT] Reading from {source_path}")
    df = pd.read_csv(source_path, parse_dates=["date"])

    # Simulate a new week arriving
    last = df.iloc[-1]
    new_row = {
        "date":         last["date"] + pd.Timedelta(weeks=1),
        "revenue":      last["revenue"] * np.random.uniform(0.92, 1.08),
        "tv_spend":     max(0, last["tv_spend"]     * np.random.uniform(0.85, 1.15)),
        "search_spend": max(0, last["search_spend"] * np.random.uniform(0.85, 1.15)),
        "social_spend": max(0, last["social_spend"] * np.random.uniform(0.85, 1.15)),
        "price":        round(last["price"] * np.random.uniform(0.97, 1.03), 4),
        "holiday":      0,
        "seasonality":  round(last["seasonality"], 4),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    log.info(f"[EXTRACT] {len(df)} rows (incl. 1 new week: {new_row['date'].date()})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════════
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data quality checks."""
    errors = []
    if df["date"].isnull().any():
        errors.append("Null dates found")
    if (df[CHANNELS] < 0).any().any():
        errors.append("Negative spend values found")
    if (df[KPI] <= 0).any():
        errors.append("Non-positive revenue found")
    if df["date"].duplicated().any():
        errors.append("Duplicate dates found")
    if errors:
        raise ValueError(f"Validation failed: {'; '.join(errors)}")
    log.info("[TRANSFORM] Validation passed ✓")
    return df


def transform(df: pd.DataFrame, params_path: str = "model_params.csv") -> pd.DataFrame:
    """
    Applies adstock + saturation using saved params (or defaults if first run).
    """
    df = validate(df.copy())
    df = df.sort_values("date").reset_index(drop=True)

    # Load transform params if available
    try:
        params = pd.read_csv(params_path).set_index("channel")
        log.info(f"[TRANSFORM] Loaded transform params from {params_path}")
    except FileNotFoundError:
        log.warning("[TRANSFORM] No model_params.csv found — using defaults")
        params = pd.DataFrame({
            "channel": CHANNELS,
            "decay":   [0.5, 0.3, 0.4],
            "alpha":   [2.0, 1.5, 1.8],
            "gamma":   [200, 80,  90],
        }).set_index("channel")

    for ch in CHANNELS:
        p = params.loc[ch]
        df[f"{ch}_transformed"] = apply_transforms(
            df[ch].values, float(p["decay"]), float(p["alpha"]), float(p["gamma"])
        )

    log.info(f"[TRANSFORM] Applied adstock + saturation to {len(CHANNELS)} channels")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════════════════
def load(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    """Upsert rows into DuckDB (insert new, skip existing by date)."""
    con = duckdb.connect(db_path)

    con.execute("""
        CREATE TABLE IF NOT EXISTS mmm_data (
            date          DATE PRIMARY KEY,
            revenue       DOUBLE,
            tv_spend      DOUBLE,
            search_spend  DOUBLE,
            social_spend  DOUBLE,
            price         DOUBLE,
            holiday       INTEGER,
            seasonality   DOUBLE,
            tv_spend_transformed      DOUBLE,
            search_spend_transformed  DOUBLE,
            social_spend_transformed  DOUBLE,
            loaded_at     TIMESTAMP DEFAULT current_timestamp
        )
    """)

    # Insert only rows that don't already exist
    existing = con.execute("SELECT date FROM mmm_data").fetchdf()
    existing_dates = pd.to_datetime(existing["date"]).values if len(existing) else []
    new_rows = df[~df["date"].isin(existing_dates)]

    if len(new_rows) == 0:
        log.info("[LOAD] No new rows to insert — already up to date")
    else:
        con.execute("INSERT INTO mmm_data SELECT *, current_timestamp FROM new_rows")
        log.info(f"[LOAD] Inserted {len(new_rows)} new rows into {db_path}")

    total = con.execute("SELECT COUNT(*) FROM mmm_data").fetchone()[0]
    log.info(f"[LOAD] Total rows in warehouse: {total}")
    con.close()


# ═══════════════════════════════════════════════════════════════════════════════
# REFIT MODEL
# ═══════════════════════════════════════════════════════════════════════════════
def refit_model(db_path: str = DB_PATH) -> dict:
    """Load full dataset from DuckDB and refit OLS MMM."""
    con  = duckdb.connect(db_path)
    df   = con.execute("SELECT * FROM mmm_data ORDER BY date").fetchdf()
    con.close()

    y         = df[KPI].values
    X_ch      = np.column_stack([df[f"{c}_transformed"].values for c in CHANNELS])
    X_ctrl    = df[CONTROLS].values
    X_full    = sm.add_constant(np.hstack([X_ch, X_ctrl]))

    result    = sm.OLS(y, X_full).fit()
    mape      = np.mean(np.abs((y - result.fittedvalues) / y)) * 100

    log.info(f"[MODEL] Refit complete  R²={result.rsquared:.4f}  MAPE={mape:.2f}%")
    return {"r2": round(result.rsquared, 4), "mape": round(mape, 2), "n_obs": len(df)}


# ═══════════════════════════════════════════════════════════════════════════════
# LOG RUN
# ═══════════════════════════════════════════════════════════════════════════════
def log_run(status: str, metrics: dict = None, error: str = "") -> None:
    run = {
        "run_at":  datetime.now().isoformat(timespec="seconds"),
        "status":  status,
        "r2":      metrics.get("r2",    "") if metrics else "",
        "mape":    metrics.get("mape",  "") if metrics else "",
        "n_obs":   metrics.get("n_obs", "") if metrics else "",
        "error":   error,
    }
    log_path = Path(LOG_FILE)
    header   = not log_path.exists()
    pd.DataFrame([run]).to_csv(log_path, mode="a", header=header, index=False)
    log.info(f"[LOG] Run recorded → {LOG_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("MMM PIPELINE  —  starting run")
    log.info("=" * 55)
    try:
        raw_df        = extract()
        clean_df      = transform(raw_df)
        load(clean_df)
        metrics       = refit_model()
        log_run("SUCCESS", metrics)
        log.info("✅  Pipeline completed successfully")
    except Exception as e:
        log.error(f"❌  Pipeline failed: {e}")
        log_run("FAILED", error=str(e))
        sys.exit(1)