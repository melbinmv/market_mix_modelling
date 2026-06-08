"""
2_model.py
----------
Builds a Marketing Mix Model from scratch using:
  - Grid search to find best adstock decay + Hill saturation params
  - OLS regression (statsmodels) for coefficients
  - Channel contribution decomposition
  - ROI calculation per channel
  - Model diagnostics + plots
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
from itertools import product
from transforms import apply_transforms

warnings.filterwarnings("ignore")

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data.csv", parse_dates=["date"])
print(f"Loaded {len(df)} rows from data.csv\n")

CHANNELS  = ["tv_spend", "search_spend", "social_spend"]
CONTROLS  = ["price", "holiday", "seasonality"]
KPI       = "revenue"

# ── 2. Grid search for transform parameters ────────────────────────────────────
# For each channel we search over (decay, alpha, gamma).
# We pick the combo that gives the lowest OLS residual sum of squares.

DECAY_GRID = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8]
ALPHA_GRID = [0.5, 1.0, 1.5, 2.0, 2.5]
GAMMA_GRID = [50, 80, 100, 150, 200, 300]   # in same units as spend

print("Running grid search for transform parameters...")
print(f"  Combinations per channel: {len(DECAY_GRID)*len(ALPHA_GRID)*len(GAMMA_GRID)}")

y = df[KPI].values
controls = df[CONTROLS].values


def fit_ols(X_raw: np.ndarray) -> float:
    """Fit OLS with controls and return RSS (lower = better)."""
    X = np.hstack([X_raw, controls, np.ones((len(y), 1))])
    X = sm.add_constant(X, has_constant="add")
    try:
        res = sm.OLS(y, X).fit()
        return res.ssr
    except Exception:
        return np.inf


best_params = {}
for ch in CHANNELS:
    spend = df[ch].values
    best_ssr, best_combo = np.inf, None
    for decay, alpha, gamma in product(DECAY_GRID, ALPHA_GRID, GAMMA_GRID):
        transformed = apply_transforms(spend, decay, alpha, gamma).reshape(-1, 1)
        # Build X with this channel transformed + others untransformed (simple pass)
        other_channels = np.hstack([
            df[c].values.reshape(-1, 1) for c in CHANNELS if c != ch
        ])
        X_raw = np.hstack([transformed, other_channels])
        ssr = fit_ols(X_raw)
        if ssr < best_ssr:
            best_ssr = ssr
            best_combo = (decay, alpha, gamma)
    best_params[ch] = best_combo
    print(f"  {ch:15s} → decay={best_combo[0]}, alpha={best_combo[1]}, gamma={best_combo[2]}")

print()

# ── 3. Build final feature matrix with best params ────────────────────────────
transformed_spends = {}
for ch in CHANNELS:
    decay, alpha, gamma = best_params[ch]
    transformed_spends[ch] = apply_transforms(df[ch].values, decay, alpha, gamma)

X_channels = np.column_stack([transformed_spends[ch] for ch in CHANNELS])
X_controls = df[CONTROLS].values
X_full = np.hstack([X_channels, X_controls])
X_full = sm.add_constant(X_full)

col_names = ["const"] + CHANNELS + CONTROLS

# ── 4. Fit OLS ─────────────────────────────────────────────────────────────────
model = sm.OLS(y, X_full).fit()

print("=" * 60)
print("OLS RESULTS")
print("=" * 60)
summary_df = pd.DataFrame({
    "Variable":  col_names,
    "Coef":      model.params.round(1),
    "Std Err":   model.bse.round(1),
    "t":         model.tvalues.round(2),
    "p-value":   model.pvalues.round(4),
    "Sig":       ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "" for p in model.pvalues],
})
print(summary_df.to_string(index=False))
print(f"\nR²      : {model.rsquared:.4f}")
print(f"Adj. R² : {model.rsquared_adj:.4f}")
print(f"MAPE    : {np.mean(np.abs((y - model.fittedvalues) / y)) * 100:.2f}%")

# ── 5. Channel contribution decomposition ─────────────────────────────────────
contributions = {}
for i, ch in enumerate(CHANNELS):
    coef = model.params[i + 1]   # +1 because index 0 is const
    contributions[ch] = coef * transformed_spends[ch]

# Base (intercept + controls)
base = (
    model.params[0]
    + model.params[len(CHANNELS) + 1] * df["price"].values
    + model.params[len(CHANNELS) + 2] * df["holiday"].values
    + model.params[len(CHANNELS) + 3] * df["seasonality"].values
)

contrib_df = pd.DataFrame(contributions, index=df["date"])
contrib_df["base"] = base
contrib_df["total_predicted"] = model.fittedvalues
contrib_df["actual"] = y

# ── 6. ROI per channel ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CHANNEL ROI")
print("=" * 60)
roi_rows = []
for ch in CHANNELS:
    total_contrib = contributions[ch].sum()
    total_spend   = df[ch].sum()
    roi           = total_contrib / total_spend
    roi_rows.append({"Channel": ch, "Total Spend": f"£{total_spend:,.0f}",
                     "Revenue Attributed": f"£{total_contrib:,.0f}", "ROI": f"{roi:.2f}x"})
print(pd.DataFrame(roi_rows).to_string(index=False))

# ── 7. Save outputs ───────────────────────────────────────────────────────────
contrib_df.to_csv("contributions.csv")
print("\n✅  Saved contributions.csv")

pd.DataFrame([
    {"channel": ch, "decay": best_params[ch][0], "alpha": best_params[ch][1],
     "gamma": best_params[ch][2], "beta": model.params[i + 1]}
    for i, ch in enumerate(CHANNELS)
]).to_csv("model_params.csv", index=False)
print("✅  Saved model_params.csv")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle("Marketing Mix Model — Results", fontsize=15, fontweight="bold")

# Plot 1: Actual vs Fitted
ax = axes[0]
ax.plot(df["date"], y / 1000, label="Actual Revenue", color="#333", linewidth=1.5)
ax.plot(df["date"], model.fittedvalues / 1000, label="Model Fitted", color="#e63946",
        linewidth=1.5, linestyle="--")
ax.set_title("Actual vs Fitted Revenue")
ax.set_ylabel("Revenue (£000s)")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.0f}k"))

# Plot 2: Stacked contribution chart
ax = axes[1]
colors = {"base": "#adb5bd", "tv_spend": "#1d3557", "search_spend": "#457b9d",
          "social_spend": "#a8dadc"}
bottom = np.zeros(len(df))
for col in ["base", "tv_spend", "search_spend", "social_spend"]:
    vals = contrib_df[col].values / 1000
    ax.bar(df["date"], vals, bottom=bottom / 1000, label=col.replace("_spend", "").title(),
           color=colors[col], width=6)
    bottom += contrib_df[col].values
ax.set_title("Revenue Decomposition by Channel")
ax.set_ylabel("Revenue (£000s)")
ax.legend(loc="upper left")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.0f}k"))

# Plot 3: ROI bar chart
ax = axes[2]
rois = {ch: contributions[ch].sum() / df[ch].sum() for ch in CHANNELS}
ch_labels = [c.replace("_spend", "").title() for c in CHANNELS]
bars = ax.bar(ch_labels, list(rois.values()), color=["#1d3557", "#457b9d", "#a8dadc"])
for bar, val in zip(bars, rois.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}x", ha="center", fontsize=11)
ax.set_title("ROI per Channel (Revenue Attributed / Spend)")
ax.set_ylabel("ROI (x)")
ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="Break-even")
ax.legend()

plt.tight_layout()
plt.savefig("mmm_results.png", dpi=150, bbox_inches="tight")
print("✅  Saved mmm_results.png")
plt.show()