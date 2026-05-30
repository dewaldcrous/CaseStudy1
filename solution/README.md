# Smart Traffic Light Optimisation — Senior Quant Case Study (Problem 1)

A machine-learning–driven adaptive traffic signal controller that reduces average
vehicle wait time by **40.1%** at peak hour versus a fixed-timing baseline,
exceeding the required ≥20% target in **all 6 tested scenarios**.

---

## TL;DR — Results

| Metric | Value |
|---|---|
| Peak-hour wait reduction (Mon 8am) | **40.1%** |
| Best scenario (Mon 5pm PM rush) | **+73.8%** |
| Worst scenario (Mon 8am AM rush) | **+38.9%** |
| Scenarios meeting ≥20% target | **6 / 6** |
| Best model | **LightGBM** (Test MAE 0.00545 veh/s, R² 0.9852) |
| CV Val R² (5-fold) | 0.9866 ± 0.0031 |

**The full business case with all charts is in [`report.html`](report.html)** — open it in any browser.

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline — ML training + validation + simulation + 15 PNGs
python main.py

# 3. Run the live animated demos (opens a real-time window)
python main.py demo_am_rush    # Monday 8am — NS/EW asymmetry live
python main.py demo_full_day   # Full 24-hour cycle — adaptation across regimes
python main.py demo_incident   # Accident at t=150s + recovery at t=240s
python main.py demo_all        # All three in sequence

# 4. Save all three demos as GIFs (no display needed — embed in slides)
python save_demos.py

# 5. Scenario benchmark only (no window)
python main.py benchmark

# 5b. Run the interactive Streamlit dashboard (recommended for presentation)
#     Opens http://localhost:8501 in your browser
#     Controls: Start/Pause, speed slider, live charts, incident trigger button
streamlit run dashboard.py

# 6. Open the static business-case report (all charts embedded)
#    report.html — open in any browser

# 7. Load saved model without retraining
python -c "
import joblib
from src.ml_model import predict_rates
model  = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/best_scaler.pkl')
rates  = predict_rates(model, scaler, hour=8, day_of_week=0)
print('NS at Mon 8am:', rates[0]['north'], 'veh/s')
"
```

Requires Python 3.10+. H2O AutoML is optional (needs Java 11+); the pipeline
skips it gracefully if unavailable.

---

## What the pipeline does (10 sections)

| # | Section | What it produces |
|---|---|---|
| 1 | Synthetic data generation | 11,520 rows from 6 documented rules (A–F) |
| 2 | Feature engineering | 12 features incl. circular time encoding |
| 2b | **Correlation analysis** | Feature→target Pearson r + cross-correlation heatmap |
| 3 | ML model comparison | Ridge · RF · GBM · LightGBM · XGBoost ranked by MAE |
| 3b | H2O AutoML | (optional) AutoML leaderboard |
| 3c | Model enhancement | FE / sampling / hyperparameter-tuning experiments |
| 3d | **Model validation** | SHAP · Gini · AUC/ROC · residuals · learning curves · 5-fold CV |
| 4 | Baseline analysis | Why fixed 30/30 timing is suboptimal (with math) |
| 5 | Optimizer walkthrough | Webster's formula, step by step |
| 6 | Simulation & comparison | Baseline vs Webster × each ML model |
| 7 | Scenario benchmark | When is the 20% target met? (6 scenarios) |

---

## Project structure

```
ADHOC/solution/
├── main.py                  # Orchestrator — runs all 10 sections, generates all 15 PNGs
├── generate_diagrams.py     # Architecture + MAE diagrams (also called by main.py)
├── report.html              # ← Business-case report (open this in a browser)
├── README.md                # This file
├── requirements.txt         # numpy · pandas · scikit-learn · lightgbm · xgboost · shap
├── src/
│   ├── simulator.py         # Queue-based physics engine (Poisson arrivals, saturation discharge)
│   ├── ml_model.py          # Rules A–F, feature engineering, 5-model training
│   ├── optimizer.py         # FixedTimingController + WebsterOptimizer (7-step formula)
│   ├── model_enhancement.py # FE / sampling / tuning / feature-reduction experiments
│   ├── model_validation.py  # SHAP, Gini, AUC, residuals, learning curves, cross-validation
│   └── visualization.py     # SUPERSEDED — kept for reference, not called by main.py
├── models/                  # Created on first run
│   ├── best_model.pkl       # Saved LightGBM (joblib) — load without retraining
│   ├── best_scaler.pkl      # StandardScaler or None
│   └── model_info.txt       # Model name, date, feature column order
└── *.png                    # 15 generated figures
```

---

## Key design decisions

### Fair A/B comparison (`demand_model` vs `control_model`)
Every simulation uses the **same best model as the demand source** (what traffic
actually arrives), so all controllers see identical traffic volumes. Only the
**timing strategy** varies. Without this separation, using a weaker model as the
demand source would artificially distort queue lengths and invalidate the comparison.

### Webster's (1958) optimal cycle formula
```
C* = (1.5·L + 5) / (1 − Y)
```
- `L` = total lost time per cycle (4s)
- `Y` = Σ flow ratios = Σ (demand / saturation flow)
- Green split allocated proportionally to flow ratios
- ML prediction blended with live queue feedback (α = 0.7)

### Rush-hour MAE as the enhancement metric
Model enhancements are judged on **rush-hour MAE**, not overall MAE — prediction
errors during peaks cost far more in driver wait time than errors at 3am.

---

## Honest findings (negative results included)

1. **Model enhancement gave only +0.6% rush-hour MAE** — below the 5% deployment
   threshold. The base LightGBM model is already near the irreducible error floor
   set by Rule F's log-normal noise. *Recommendation: keep the simpler base model.*

2. **Gini MDI is biased.** It ranked `intersection_id` as the #1 feature; both
   Permutation importance and SHAP rank it #7. SHAP correctly identifies
   `is_morning_rush` as the true #1 driver — validating the data-generating rules.

3. **Gini = 0.506 (moderate) despite R² = 0.9866 — resolved by the two AUCs.**
   The *ranking AUC* = (Gini+1)/2 = **0.753** carries the same limitation: it is
   dragged down by the near-impossible task of finely ordering thousands of
   near-identical off-peak rows. But the *binary ROC-AUC* for discriminating
   high-demand approaches (top 19%) is **0.9946** — near perfect. The optimizer
   only needs the coarse "busy vs quiet" distinction, which the model nails
   (confirmed by the 40.1% simulation gain).

---

## Generated figures

| File | Content |
|---|---|
| `fig1_time_series.png` | Wait / queue / throughput over 60-min rush hour |
| `fig2_model_comparison.png` | MAE, R², and optimizer gain per model |
| `fig3_scenario_benchmark.png` | Improvement % across 6 scenarios |
| `fig4_feature_importance.png` | Gini MDI feature importances |
| `fig_correlation_analysis.png` | Feature→target r + cross-correlation matrix |
| `fig_shap_summary.png` | SHAP beeswarm |
| `fig_shap_bar.png` | SHAP mean \|φ\| ranking |
| `fig_shap_by_hour.png` | Top features' SHAP by hour |
| `fig_shap_waterfall.png` | Single-prediction explanation |
| `fig_residual_analysis.png` | 5-panel residual diagnostic |
| `fig_learning_curves.png` | Train vs val MAE by dataset size |
| `fig_gini_lorenz.png` | Lorenz curve + Gini coefficient |
| `fig_roc_auc.png` | ROC curve — high-demand discrimination (ROC-AUC = 0.9946) |

---

## References

Webster (1958) · HCM 6th Ed. (TRB, 2016) · Ke et al. LightGBM (2017) ·
Chen & Guestrin XGBoost (2016) · Lundberg & Lee SHAP (2017) · Shapley (1953) ·
Maze et al. weather (2006) · Breiman Random Forests (2001).
Full citations in `report.html` § References.
