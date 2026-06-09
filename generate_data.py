"""
generate_data.py
------------------
Generates a synthetic weekly MMM dataset with known ground truth.
Channels: TV, Paid Search, Paid Social
KPI: Revenue
Controls: Price, Holiday flag, Seasonality
"""

import numpy as np
import pandas as pd

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
N_WEEKS = 156  # 3 years of weekly data


# ── Adstock transform (geometric decay) ───────────────────────────────────────
def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Carry-over effect: each week retains `decay` fraction of prior week."""
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + decay * out[t - 1]
    return out


# ── Hill saturation (diminishing returns) ─────────────────────────────────────
def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Hill function: S(x) = x^alpha / (x^alpha + gamma^alpha)
    Returns values in [0, 1].
    alpha  → steepness of the curve
    gamma  → half-saturation point (spend level at 50% of max effect)
    """
    x = np.array(x, dtype=float)
    return x**alpha / (x**alpha + gamma**alpha)


# ── Spend data (weekly, £000s) ─────────────────────────────────────────────────
dates = pd.date_range("2021-01-04", periods=N_WEEKS, freq="W-MON")

tv_spend     = np.random.gamma(shape=4, scale=25, size=N_WEEKS)   # avg ~£100k/wk
search_spend = np.random.gamma(shape=3, scale=15, size=N_WEEKS)   # avg ~£45k/wk
social_spend = np.random.gamma(shape=2, scale=20, size=N_WEEKS)   # avg ~£40k/wk

# ── Controls ──────────────────────────────────────────────────────────────────
# Seasonality: higher in Q4
week_of_year = np.array([d.isocalendar()[1] for d in dates])
seasonality  = 1 + 0.3 * np.sin(2 * np.pi * week_of_year / 52 - np.pi / 2)

# Price index (normalised around 1.0)
price = 1.0 + np.random.normal(0, 0.05, N_WEEKS)

# Holiday weeks (Christmas, Easter, summer peak)
holiday = np.zeros(N_WEEKS)
for i, d in enumerate(dates):
    if d.month == 12 and d.day >= 18:
        holiday[i] = 1
    if d.month in [3, 4] and d.isocalendar()[1] in [13, 14]:
        holiday[i] = 1

# ── True (ground-truth) parameters ────────────────────────────────────────────
TRUE_PARAMS = {
    "intercept":        50_000,
    "tv":     {"decay": 0.6, "alpha": 2.0, "gamma": 200.0, "beta": 180_000},
    "search": {"decay": 0.3, "alpha": 1.5, "gamma":  80.0, "beta": 120_000},
    "social": {"decay": 0.4, "alpha": 1.8, "gamma":  90.0, "beta":  80_000},
    "price_coef":      -30_000,
    "holiday_coef":     15_000,
    "seasonality_coef": 20_000,
}

# ── Build revenue from ground truth ───────────────────────────────────────────
def channel_contribution(spend, p):
    transformed = adstock(spend, p["decay"])
    saturated   = hill_saturation(transformed, p["alpha"], p["gamma"])
    return p["beta"] * saturated

tv_contrib     = channel_contribution(tv_spend,     TRUE_PARAMS["tv"])
search_contrib = channel_contribution(search_spend, TRUE_PARAMS["search"])
social_contrib = channel_contribution(social_spend, TRUE_PARAMS["social"])

revenue = (
    TRUE_PARAMS["intercept"]
    + tv_contrib
    + search_contrib
    + social_contrib
    + TRUE_PARAMS["price_coef"]      * price
    + TRUE_PARAMS["holiday_coef"]    * holiday
    + TRUE_PARAMS["seasonality_coef"] * seasonality
    + np.random.normal(0, 5_000, N_WEEKS)   # noise
)

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "date":         dates,
    "revenue":      revenue.round(0),
    "tv_spend":     tv_spend.round(2),
    "search_spend": search_spend.round(2),
    "social_spend": social_spend.round(2),
    "price":        price.round(4),
    "holiday":      holiday.astype(int),
    "seasonality":  seasonality.round(4),
})

df.to_csv("data.csv", index=False)
print(f"✅  Saved data.csv  ({len(df)} rows)")
print(df.describe().round(1))
print("\nTrue parameters used to generate data:")
for k, v in TRUE_PARAMS.items():
    print(f"  {k}: {v}")