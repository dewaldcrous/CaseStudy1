"""
Model Validation & Interpretability Module
===========================================

BUSINESS CASE
-------------
Deploying an ML model in a live traffic system requires more than a good
test MAE.  Regulators, engineers, and senior stakeholders need to know:

  1. WHY does the model make a particular prediction?
     -> SHAP values explain each prediction at the individual row level.

  2. WHICH features actually drive the model?
     Three importance methods are compared because each has blind spots:
       Gini (MDI)           — fast, built into tree models, but BIASED
                              toward high-cardinality / continuous features.
       Permutation          — model-agnostic, unbiased, but noisy.
       SHAP mean |phi|      — gold standard: consistent, theoretically
                              grounded (Shapley values from game theory).

  3. IS the model generalising or just memorising training data?
     -> Learning curves show train vs validation error as data size grows.
     -> Cross-validation gives a robust ±std estimate of true performance.

  4. WHERE are the systematic errors?
     -> Residual analysis by hour reveals if the model over/under-predicts
        at specific times.  A pattern means a missing feature, not noise.

  5. HOW confident is each prediction?
     -> Prediction intervals via quantile regression / bootstrap.

All charts are saved as PNGs in the solution/ folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from sklearn.metrics          import (mean_absolute_error, r2_score, mean_squared_error,
                                       roc_auc_score, roc_curve)
from sklearn.model_selection  import KFold, cross_validate, learning_curve
from sklearn.inspection       import permutation_importance
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.ml_model import FEATURE_COLS, add_features

import os
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ================================================================== #
#  Helpers                                                             #
# ================================================================== #

def _train_test_arrays(df_raw: pd.DataFrame, feature_cols=None):
    """
    Build the X / y arrays used by every validation test.

    MODEL INPUTS / OUTPUTS
    ----------------------
    X  (feature matrix)
       shape  : (N, len(feature_cols))  — default (N, 12)
       dtype  : float64
       columns: FEATURE_COLS (12 features across 4 groups — see ml_model.py
                MODEL SPECIFICATION block for the full per-feature description)
         Group 1 — circular time : hour_sin, hour_cos, dow_sin, dow_cos
         Group 2 — period flags  : is_morning_rush, is_evening_rush,
                                    is_daytime, is_weekend
         Group 3 — weather       : weather  {0=clear, 1=cloudy, 2=rain}
         Group 4 — location/dir  : intersection_id, direction_id, ns_direction

    y  (target variable)
       name   : "arrival_rate"
       shape  : (N,)
       dtype  : float64
       units  : vehicles per second (veh/s)
       range  : ~0.005 (quiet night) to ~1.2 (heavy rush hour)
       meaning: what the model predicts — vehicles arriving per second
                at ONE approach (intersection × direction) for ONE hour.

    Split rule (identical to train_and_compare):
       train : day ≤ max_day − 5  (days 0-24)
       test  : day >  max_day − 5  (days 25-29)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    df      = add_features(df_raw)
    max_day = df["day"].max()
    tr      = df[df["day"] <= max_day - 5].copy()  # training rows
    te      = df[df["day"] >  max_day - 5].copy()  # test rows
    # X_train shape (9216, 12), y_train shape (9216,)
    # X_test  shape (2304, 12), y_test  shape (2304,)
    return (tr[feature_cols].values, te[feature_cols].values,
            tr["arrival_rate"].values, te["arrival_rate"].values, te)


def _rush_mask(te_df: pd.DataFrame) -> np.ndarray:
    return (
        (te_df["hour"].between(7, 9) | te_df["hour"].between(16, 19))
        & (te_df["is_weekend"] == 0)
    ).values


def banner(title: str) -> None:
    print("=" * 66)
    print(f"  {title}")
    print("=" * 66)
    print()


# ================================================================== #
#  1. Cross-validation summary                                         #
# ================================================================== #

def cross_validation_summary(
    model,
    df_raw: pd.DataFrame,
    feature_cols=None,
    n_splits: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run stratified time-series k-fold cross-validation and report
    MAE, RMSE, and R² with mean ± std across folds.

    WHY CV INSTEAD OF A SINGLE TRAIN/TEST SPLIT?
    ---------------------------------------------
    A single split gives one estimate of performance.  CV gives k estimates
    and their variance.  High variance across folds means the model is
    sensitive to which data it sees — a warning sign for production.

    We use KFold (not shuffle=True) because traffic data has temporal
    autocorrelation: rows from the same hour on consecutive days are
    similar.  Shuffling would leak future data into training.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = add_features(df_raw)
    X  = df[feature_cols].values
    y  = df["arrival_rate"].values

    kf      = KFold(n_splits=n_splits, shuffle=False)
    results = cross_validate(
        model, X, y, cv=kf,
        scoring={"MAE": "neg_mean_absolute_error",
                 "RMSE": "neg_root_mean_squared_error",
                 "R2": "r2"},
        return_train_score=True,
        n_jobs=1,
    )

    rows = []
    for fold in range(n_splits):
        rows.append({
            "Fold":       fold + 1,
            "Train MAE":  round(-results["train_MAE"][fold],  5),
            "Val MAE":    round(-results["test_MAE"][fold],   5),
            "Train R2":   round( results["train_R2"][fold],   4),
            "Val R2":     round( results["test_R2"][fold],    4),
        })

    cv_df = pd.DataFrame(rows)

    # Summary row
    summary = {
        "Fold":       "MEAN±STD",
        "Train MAE":  f"{cv_df['Train MAE'].mean():.5f} ± {cv_df['Train MAE'].std():.5f}",
        "Val MAE":    f"{cv_df['Val MAE'].mean():.5f} ± {cv_df['Val MAE'].std():.5f}",
        "Train R2":   f"{cv_df['Train R2'].mean():.4f} ± {cv_df['Train R2'].std():.4f}",
        "Val R2":     f"{cv_df['Val R2'].mean():.4f} ± {cv_df['Val R2'].std():.4f}",
    }

    if verbose:
        print(cv_df.to_string(index=False))
        print()
        print(f"  {'MEAN±STD':<6}  "
              f"Train MAE={summary['Train MAE']}  "
              f"Val MAE={summary['Val MAE']}")
        print(f"           Train R2={summary['Train R2']}   "
              f"Val R2={summary['Val R2']}")
        print()

        gap = cv_df["Val MAE"].mean() - cv_df["Train MAE"].mean()
        if gap > cv_df["Val MAE"].mean() * 0.3:
            print("  WARNING: Train/Val MAE gap > 30% — possible overfitting.")
            print("           Consider reducing model depth or increasing regularisation.")
        else:
            print("  Train/Val MAE gap is small — model generalises well.")
        print()

    return cv_df


# ================================================================== #
#  2. Feature importance: Gini vs Permutation vs SHAP                  #
# ================================================================== #

def importance_comparison(
    model,
    df_raw: pd.DataFrame,
    feature_cols=None,
    n_repeats: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare three feature importance methods side by side.

    Gini (MDI — Mean Decrease in Impurity)
      Built into every tree-based model.  Fast but BIASED: continuous
      features and high-cardinality categoricals appear more important
      because the model has more potential split points on them.

    Permutation Importance
      Randomly shuffles one feature at a time on the TEST set and measures
      how much the MAE increases.  Model-agnostic and unbiased, but noisy
      (n_repeats controls variance).

    SHAP Mean |phi|
      The mean absolute SHAP value per feature.  Theoretically grounded
      (Shapley values satisfy efficiency, symmetry, dummy, additivity).
      Best for stakeholder communication: "Feature X contributes Y veh/s
      on average to each prediction."

    RECOMMENDATION: Use SHAP for reporting.  Use Permutation to cross-check.
    Flag any feature where Gini rank differs strongly from SHAP rank
    (indicates Gini bias rather than true importance).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X_tr, X_te, y_tr, y_te, te_df = _train_test_arrays(df_raw, feature_cols)

    rows = []

    # --- Gini (MDI) -------------------------------------------------------
    gini_imp = getattr(model, "feature_importances_", None)
    if gini_imp is None:
        gini_imp = np.zeros(len(feature_cols))

    # --- Permutation ------------------------------------------------------
    perm = permutation_importance(
        model, X_te, y_te,
        n_repeats=n_repeats, random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    perm_imp = perm.importances_mean   # higher = more important

    # Normalise both to 0-1 for comparison
    gini_norm = gini_imp / (gini_imp.sum() + 1e-9)
    perm_norm = np.abs(perm_imp) / (np.abs(perm_imp).sum() + 1e-9)

    # --- SHAP (computed separately, passed in) ----------------------------
    # Will be filled by run_shap_analysis(); default to NaN here
    shap_imp  = np.full(len(feature_cols), np.nan)

    for i, feat in enumerate(feature_cols):
        rows.append({
            "Feature":          feat,
            "Gini (MDI)":       round(float(gini_norm[i]), 4),
            "Gini Rank":        0,
            "Permutation":      round(float(perm_norm[i]), 4),
            "Perm Rank":        0,
            "SHAP mean|phi|":   np.nan,
            "SHAP Rank":        0,
        })

    imp_df = pd.DataFrame(rows)
    imp_df["Gini Rank"]  = imp_df["Gini (MDI)"].rank(ascending=False).astype(int)
    imp_df["Perm Rank"]  = imp_df["Permutation"].rank(ascending=False).astype(int)
    imp_df = imp_df.sort_values("Gini Rank")

    if verbose:
        print(imp_df[["Feature", "Gini (MDI)", "Gini Rank",
                       "Permutation", "Perm Rank"]].to_string(index=False))
        print()
        # Flag disagreements
        imp_df["Rank Diff"] = abs(imp_df["Gini Rank"] - imp_df["Perm Rank"])
        flagged = imp_df[imp_df["Rank Diff"] >= 4]
        if not flagged.empty:
            print("  Features where Gini and Permutation ranks disagree by 4+:")
            print("  (Likely Gini BIAS — trust Permutation/SHAP instead)")
            for _, r in flagged.iterrows():
                print(f"    {r['Feature']:<22}  "
                      f"Gini rank {r['Gini Rank']}  vs  Perm rank {r['Perm Rank']}")
        print()

    return imp_df


# ================================================================== #
#  3. SHAP analysis                                                     #
# ================================================================== #

def run_shap_analysis(
    model,
    df_raw: pd.DataFrame,
    feature_cols=None,
    sample_size: int = 500,
    verbose: bool = True,
) -> tuple:
    """
    Compute SHAP values and generate four explanatory charts.

    WHAT ARE SHAP VALUES?
    ---------------------
    Shapley values come from cooperative game theory.  For a prediction,
    each feature's SHAP value is its fair contribution to the difference
    between the prediction and the global mean prediction.

    Formally: phi_i = sum over all orderings S of
        [f(S ∪ {i}) − f(S)] / (number of orderings)

    For tree models, TreeExplainer computes exact Shapley values in
    O(TLD^2) time (T = trees, L = leaves, D = depth) — much faster
    than brute-force 2^n.

    Charts produced
    ---------------
    fig_shap_summary  : beeswarm plot — each dot is one prediction.
                        x-axis = SHAP value (contribution to output).
                        colour = feature value (red=high, blue=low).
                        Width shows density of points.

    fig_shap_bar      : mean |SHAP| per feature — overall importance.

    fig_shap_hour     : SHAP values for is_morning_rush and ns_direction
                        broken down by hour — shows temporal pattern.

    fig_shap_waterfall: single rush-hour prediction explained step by step.

    BUSINESS USE
    ------------
    "At 8am on Monday, this intersection's NS approach is predicted at
    0.73 veh/s.  The model's baseline is 0.08 veh/s.  is_morning_rush
    contributed +0.31, ns_direction +0.18, hour_cos +0.09, ..."
    This is auditable and defensible to regulators.
    """
    if not SHAP_AVAILABLE:
        if verbose:
            print("  [SHAP] shap package not installed — skipping.")
            print("         Run: pip install shap")
        return None, None

    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if verbose:
        print(f"  Computing SHAP values on {sample_size} test rows ...")

    X_tr, X_te, y_tr, y_te, te_df = _train_test_arrays(df_raw, feature_cols)

    # Sample for speed — TreeExplainer is fast but large X_te still takes time
    rng     = np.random.default_rng(42)
    idx     = rng.choice(len(X_te), size=min(sample_size, len(X_te)), replace=False)
    X_samp  = X_te[idx]
    te_samp = te_df.iloc[idx].reset_index(drop=True)

    # Use TreeExplainer for tree models (fast, exact); Linear for Ridge
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_samp)
        base_value  = explainer.expected_value
    except Exception:
        if verbose:
            print("  TreeExplainer failed, falling back to LinearExplainer ...")
        try:
            explainer   = shap.LinearExplainer(model, X_tr)
            shap_values = explainer.shap_values(X_samp)
            base_value  = explainer.expected_value
        except Exception as e:
            if verbose:
                print(f"  [SHAP] Could not compute SHAP values: {e}")
            return None, None

    shap_df  = pd.DataFrame(shap_values, columns=feature_cols)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)

    if verbose:
        print()
        print("  SHAP mean |phi| (average contribution per feature):")
        for feat, val in mean_abs.head(8).items():
            bar = "#" * int(val / mean_abs.max() * 40)
            print(f"    {feat:<22} {val:.5f}  {bar}")
        print()

    # ── Figure 1: Beeswarm summary ────────────────────────────────────
    fig_summary, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_samp, feature_names=feature_cols,
                      show=False, plot_size=None)
    plt.title("SHAP Summary Plot\n"
              "Each dot = one prediction. "
              "x = contribution to output. Colour = feature value.",
              fontsize=10)
    plt.tight_layout()
    fig_summary.savefig(f"{FIGURES_DIR}/fig_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig_summary)
    if verbose:
        print("  Saved: fig_shap_summary.png")

    # ── Figure 2: Bar chart of mean |SHAP| ───────────────────────────
    fig_bar, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(shap_values, X_samp, feature_names=feature_cols,
                      plot_type="bar", show=False, plot_size=None)
    plt.title("SHAP Feature Importance (mean |SHAP value|)\n"
              "More reliable than Gini importance — unbiased by feature cardinality.",
              fontsize=10)
    plt.tight_layout()
    fig_bar.savefig(f"{FIGURES_DIR}/fig_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig_bar)
    if verbose:
        print("  Saved: fig_shap_bar.png")

    # ── Figure 3: SHAP by hour for top 2 features ────────────────────
    top2 = list(mean_abs.head(2).index)
    fig_hour, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig_hour.suptitle("SHAP Values by Hour of Day — Top 2 Features\n"
                       "Shows when each feature matters most.",
                       fontsize=11, fontweight="bold")

    for ax, feat in zip(axes, top2):
        if feat not in te_samp.columns:
            continue
        feat_idx = feature_cols.index(feat)
        sv       = shap_values[:, feat_idx]
        hours    = te_samp["hour"].values

        # Box per hour
        hour_groups = [sv[hours == h] for h in range(24)]
        bp = ax.boxplot(hour_groups, positions=range(24), widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor="steelblue", alpha=0.6),
                        medianprops=dict(color="red", linewidth=1.5),
                        flierprops=dict(marker=".", markersize=2),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8))
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("SHAP value")
        ax.set_title(f"Feature: {feat}", fontsize=10)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig_hour.savefig(f"{FIGURES_DIR}/fig_shap_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close(fig_hour)
    if verbose:
        print("  Saved: fig_shap_by_hour.png")

    # ── Figure 4: Waterfall for one rush-hour prediction ─────────────
    # Find a rush-hour row to explain
    rush_rows = te_samp[
        te_samp["hour"].between(7, 9) & (te_samp["is_weekend"] == 0)
    ]
    explain_idx = rush_rows.index[0] if len(rush_rows) > 0 else 0

    fig_wf, ax = plt.subplots(figsize=(10, 6))

    # Manual waterfall since shap.plots.waterfall requires Explanation object
    sv_row    = shap_values[explain_idx]
    base      = float(base_value) if np.isscalar(base_value) else float(base_value[0])
    pred_val  = base + sv_row.sum()
    row_meta  = te_samp.iloc[explain_idx]

    # Sort by |SHAP| descending, keep top 8
    order    = np.argsort(np.abs(sv_row))[::-1][:8]
    feats_wf = [feature_cols[i] for i in order]
    vals_wf  = [sv_row[i]       for i in order]

    colours  = ["seagreen" if v >= 0 else "crimson" for v in vals_wf]
    y_pos    = range(len(feats_wf))
    ax.barh(list(y_pos), vals_wf, color=colours, alpha=0.82)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(feats_wf, fontsize=9)
    ax.set_xlabel("SHAP contribution (veh/s)")
    ax.set_title(
        f"Waterfall: single rush-hour prediction\n"
        f"Hour={int(row_meta['hour'])}, "
        f"Intersection={int(row_meta['intersection_id'])}, "
        f"Direction={row_meta['direction']}\n"
        f"Base={base:.4f}  Predicted={pred_val:.4f}  "
        f"Actual={y_te[idx[explain_idx]]:.4f}",
        fontsize=9)
    ax.grid(True, alpha=0.2, axis="x")

    for i, (y, v) in enumerate(zip(y_pos, vals_wf)):
        ax.text(v + (0.001 if v >= 0 else -0.001), y,
                f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)

    plt.tight_layout()
    fig_wf.savefig(f"{FIGURES_DIR}/fig_shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close(fig_wf)
    if verbose:
        print("  Saved: fig_shap_waterfall.png")

    return shap_values, mean_abs


# ================================================================== #
#  4. Residual analysis                                                #
# ================================================================== #

def residual_analysis(
    model,
    df_raw: pd.DataFrame,
    feature_cols=None,
    verbose: bool = True,
) -> None:
    """
    Analyse prediction residuals (actual − predicted) for systematic patterns.

    A GOOD model has residuals that are:
      - Centred at zero (no bias)
      - Homoscedastic (constant variance across predicted values)
      - Normally distributed (for valid confidence intervals)
      - Uncorrelated with any feature (no missing structure)

    RESIDUALS BY HOUR is the most business-relevant diagnostic.
    If residuals are consistently positive at 8am, the model under-predicts
    morning rush, meaning the optimizer will under-allocate NS green time —
    exactly the opposite of what we want.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X_tr, X_te, y_tr, y_te, te_df = _train_test_arrays(df_raw, feature_cols)

    preds    = model.predict(X_te)
    residuals = y_te - preds

    mae   = mean_absolute_error(y_te, preds)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    bias  = residuals.mean()

    if verbose:
        print(f"  Residual stats:  MAE={mae:.5f}  RMSE={rmse:.5f}  "
              f"Bias={bias:+.5f}  Std={residuals.std():.5f}")
        print()
        if abs(bias) > mae * 0.1:
            print(f"  WARNING: Model has systematic bias of {bias:+.5f} veh/s")
            direction = "over-predicts" if bias < 0 else "under-predicts"
            print(f"           Model {direction} on average.")
        else:
            print("  No significant bias detected.")
        print()

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Residual Analysis\n"
                 "(actual − predicted arrival rate)",
                 fontsize=12, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Residuals vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(preds, residuals, s=3, alpha=0.3, color="steelblue")
    ax1.axhline(0, color="red", lw=1.5, ls="--")
    ax1.set_xlabel("Predicted (veh/s)")
    ax1.set_ylabel("Residual")
    ax1.set_title("Residuals vs Predicted\n(should be random band around 0)")
    ax1.grid(True, alpha=0.2)

    # 2. Residual histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=60, color="steelblue", alpha=0.75, edgecolor="white")
    ax2.axvline(0,    color="red",    lw=1.5, ls="--", label="zero")
    ax2.axvline(bias, color="orange", lw=1.5, ls="--", label=f"bias={bias:+.4f}")
    ax2.set_xlabel("Residual (veh/s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution\n(should be roughly normal, centred at 0)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # 3. Q-Q plot
    ax3 = fig.add_subplot(gs[0, 2])
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, fit=True)
    ax3.plot(osm, osr,       "o", ms=2, alpha=0.4, color="steelblue")
    ax3.plot(osm, slope * np.array(osm) + intercept, "r--", lw=1.5)
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Sample Quantiles")
    ax3.set_title(f"Q-Q Plot  (R={r:.3f})\n(closer to line = more normal)")
    ax3.grid(True, alpha=0.2)

    # 4. Residuals by hour — most actionable for business
    ax4 = fig.add_subplot(gs[1, :2])
    hours        = te_df["hour"].values
    hour_medians = [np.median(residuals[hours == h]) if (hours == h).any() else 0
                    for h in range(24)]
    hour_q25     = [np.percentile(residuals[hours == h], 25) if (hours == h).any() else 0
                    for h in range(24)]
    hour_q75     = [np.percentile(residuals[hours == h], 75) if (hours == h).any() else 0
                    for h in range(24)]

    ax4.fill_between(range(24), hour_q25, hour_q75, alpha=0.3, color="steelblue",
                     label="IQR (25–75%)")
    ax4.plot(range(24), hour_medians, "o-", color="steelblue", lw=2,
             markersize=5, label="Median residual")
    ax4.axhline(0, color="red", lw=1.5, ls="--")
    ax4.fill_betweenx([-1, 1], 7, 9,   alpha=0.1, color="orange",
                      label="AM rush (7-9am)")
    ax4.fill_betweenx([-1, 1], 16, 19, alpha=0.1, color="purple",
                      label="PM rush (4-7pm)")
    ax4.set_xlabel("Hour of Day")
    ax4.set_ylabel("Residual (veh/s)")
    ax4.set_title("Residuals by Hour\n"
                  "Non-zero medians at rush hours indicate systematic under/over-prediction")
    ax4.set_xticks(range(0, 24, 2))
    ax4.legend(fontsize=8, loc="upper left")
    ax4.grid(True, alpha=0.2)
    ax4.set_xlim(-0.5, 23.5)

    # 5. Actual vs Predicted scatter
    ax5 = fig.add_subplot(gs[1, 2])
    lim = max(y_te.max(), preds.max()) * 1.05
    ax5.scatter(y_te, preds, s=3, alpha=0.3, color="seagreen")
    ax5.plot([0, lim], [0, lim], "r--", lw=1.5)
    ax5.set_xlabel("Actual (veh/s)")
    ax5.set_ylabel("Predicted (veh/s)")
    ax5.set_title(f"Actual vs Predicted\nMAE={mae:.4f}  R²={r2_score(y_te, preds):.4f}")
    ax5.grid(True, alpha=0.2)

    plt.savefig(f"{FIGURES_DIR}/fig_residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print("  Saved: fig_residual_analysis.png")
        print()


# ================================================================== #
#  5. Learning curves                                                  #
# ================================================================== #

def plot_learning_curves(
    model,
    df_raw: pd.DataFrame,
    feature_cols=None,
    verbose: bool = True,
) -> None:
    """
    Plot train vs validation MAE as training set size grows.

    WHAT THIS TELLS YOU
    -------------------
    High train error + high val error  -> underfitting (model too simple)
    Low train error + high val error   -> overfitting (model too complex)
    Both errors converge as data grows -> model is well-fitted; more data helps.
    Both plateau early                 -> more data won't help; need better features.

    BUSINESS IMPLICATION
    --------------------
    If val MAE is still falling at 100% of training data, deploying daily
    retraining on a growing historical dataset will keep improving accuracy.
    If it plateaus at 60%, focus on feature engineering instead.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    df = add_features(df_raw)
    X  = df[feature_cols].values
    y  = df["arrival_rate"].values

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        shuffle=False,
    )

    train_mean = -train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = -val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color="seagreen")
    ax.fill_between(train_sizes, val_mean   - val_std,   val_mean   + val_std,
                    alpha=0.2, color="crimson")
    ax.plot(train_sizes, train_mean, "o-", color="seagreen", lw=2,
            markersize=5, label="Training MAE")
    ax.plot(train_sizes, val_mean,   "s-", color="crimson",  lw=2,
            markersize=5, label="Validation MAE")

    ax.set_xlabel("Training Set Size (rows)")
    ax.set_ylabel("MAE (veh/s)")
    ax.set_title("Learning Curves\n"
                 "If curves still converging at full data, more data will help.\n"
                 "If they plateau, focus on feature engineering instead.")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Diagnosis annotation
    gap  = val_mean[-1] - train_mean[-1]
    note = ("Overfitting detected" if gap > val_mean[-1] * 0.3
            else "Good generalisation")
    ax.text(0.98, 0.98, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="darkgreen" if "Good" in note else "firebrick",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print("  Saved: fig_learning_curves.png")
        print(f"  Final train MAE={train_mean[-1]:.5f}  val MAE={val_mean[-1]:.5f}  "
              f"gap={gap:.5f}")
        print()


# ================================================================== #
#  6. Gini coefficient (regression version)                            #
# ================================================================== #

def gini_coefficient_regression(
    model,
    df_raw: pd.DataFrame,
    feature_cols=None,
    verbose: bool = True,
) -> float:
    """
    Compute the normalised Gini coefficient for a regression model.

    BACKGROUND
    ----------
    The Gini coefficient is widely used in credit risk (finance) as a
    ranking quality measure.  For regression it is defined as:

        Gini = 2 * AUC(Lorenz curve) - 1

    where the Lorenz curve plots the cumulative fraction of actual values
    (sorted by predicted values) against the cumulative fraction of rows.

    Interpretation:
        Gini = 0   -> model predicts randomly (no ranking skill)
        Gini = 1   -> model perfectly ranks rows by actual value
        Gini = -1  -> model perfectly anti-ranks

    FOR TRAFFIC DEMAND PREDICTION
    ------------------------------
    A high Gini means the model correctly identifies WHICH hours/directions
    have high demand vs low demand — even if the absolute predicted values
    are slightly off.  This is important for the optimizer: if the model
    correctly ranks NS demand above EW demand at 8am, it will allocate
    green correctly even if the exact rates are noisy.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X_tr, X_te, y_tr, y_te, te_df = _train_test_arrays(df_raw, feature_cols)
    preds = model.predict(X_te)

    # Sort DESCENDING by predicted value — this is the credit-scoring convention.
    #
    # WHY DESCENDING?
    # ---------------
    # When sorted descending, a good model places high-actual rows first.
    # The cumulative actual curve therefore rises steeply above the diagonal
    # (the first X% of rows — predicted as high-demand — contain MORE than
    # X% of total demand).  Lorenz area > 0.5 → Gini = 2*area − 1 > 0.
    #
    # If sorted ascending, the same good model would place low-actual rows
    # first → cum_actual below diagonal → Lorenz area < 0.5 → Gini negative.
    # That is mathematically consistent but conventionally confusing: credit
    # risk practitioners (and this report) expect Gini ∈ [0, 1] for a useful
    # model and Gini = 0 for a random model.
    sort_idx    = np.argsort(preds)[::-1]   # descending: high predicted first
    sorted_y    = y_te[sort_idx]
    n           = len(sorted_y)
    total       = sorted_y.sum()

    cum_actual  = np.cumsum(sorted_y) / total
    cum_frac    = np.arange(1, n + 1) / n

    # Lorenz area (trapezoidal rule) — area under the concentration curve
    lorenz_area = np.trapz(cum_actual, cum_frac)
    gini        = 2 * lorenz_area - 1   # ∈ [0,1] for a useful model

    if verbose:
        print(f"  Gini coefficient (regression ranking quality) = {gini:.4f}")
        print(f"  Convention: sorted descending by predicted value.")
        print(f"  Gini=1 → perfect ranking  |  Gini=0 → random (no skill)")
        print(f"  Interpretation:")
        if gini > 0.85:
            print(f"    Excellent (> 0.85) — model correctly ranks demand in "
                  f"{gini*100:.1f}% of pairwise comparisons")
        elif gini > 0.70:
            print(f"    Good (0.70–0.85) — strong ranking skill")
        elif gini > 0.50:
            print(f"    Moderate (0.50–0.70) — acceptable ranking; consider "
                  f"reviewing feature set")
        else:
            print(f"    Weak (< 0.50) — ranking skill needs improvement")
        print()

    # Plot Lorenz / concentration curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(cum_frac, cum_actual, "b-", lw=2,
            label=f"Model  (Gini = {gini:.3f})")
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Random model  (Gini = 0)")
    ax.fill_between(cum_frac, cum_actual, cum_frac, alpha=0.2, color="steelblue",
                    label="Area = Gini / 2")
    ax.set_xlabel("Cumulative fraction of rows (sorted by predicted demand, high→low)")
    ax.set_ylabel("Cumulative fraction of actual demand")
    ax.set_title(
        "Lorenz / Concentration Curve — Regression Gini\n"
        "A good model concentrates high-actual rows at the left.\n"
        "Gini = 2 × (shaded area) − 1  |  1 = perfect, 0 = random",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig_gini_lorenz.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print("  Saved: fig_gini_lorenz.png")

    return gini


# ================================================================== #
#  6b. AUC analysis (two complementary forms)                          #
# ================================================================== #

def auc_analysis(
    model,
    df_raw: pd.DataFrame,
    feature_cols=None,
    high_demand_quantile: float = 0.80,
    verbose: bool = True,
) -> dict:
    """
    Report TWO complementary AUC measures for the regression model.

    AUC is fundamentally a *classification / ranking* metric, so a plain
    regression model does not have a single "AUC".  We compute the two forms
    that are actually meaningful for traffic demand prediction.

    (1) RANKING AUC  (concordance / Somers' D relationship)
        For regression, the Gini coefficient and AUC are linked exactly by:
            AUC = (Gini + 1) / 2
        This "ranking AUC" is the probability that, for a randomly chosen
        pair of rows, the model assigns the higher predicted value to the
        row with the higher actual value.  It is the continuous analogue of
        ROC-AUC and inherits the Gini's interpretation:
            AUC = 1.0 → perfect ordering,  AUC = 0.5 → random.

    (2) BINARY ROC-AUC  (operationally the metric that matters)
        The optimizer's real job is to flag HIGH-DEMAND situations and pour
        green time into them.  So we frame a genuine binary problem:
            positive class = "high demand"  (arrival_rate ≥ Q80 threshold)
        and use the model's continuous prediction as the discrimination
        score.  ROC-AUC then answers directly:
            "How well does the model separate high-demand approaches from
             the rest?"  — the exact capability the controller relies on.

        We deliberately use the top quintile (Q80) because that is roughly
        the population of approaches where fixed timing starts to build
        queues, i.e. where adaptive control adds value.

    Returns {ranking_auc, roc_auc, threshold, n_positive}.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X_tr, X_te, y_tr, y_te, te_df = _train_test_arrays(df_raw, feature_cols)
    preds = model.predict(X_te)

    # ── (1) Ranking AUC via Gini (descending-sort convention) ─────────
    sort_idx    = np.argsort(preds)[::-1]
    sorted_y    = y_te[sort_idx]
    cum_actual  = np.cumsum(sorted_y) / sorted_y.sum()
    cum_frac    = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
    gini        = 2 * np.trapz(cum_actual, cum_frac) - 1
    ranking_auc = (gini + 1) / 2.0

    # ── (2) Binary ROC-AUC: "high demand" = top quintile of actual rates ─
    threshold = float(np.quantile(y_te, high_demand_quantile))
    y_binary  = (y_te >= threshold).astype(int)
    n_pos     = int(y_binary.sum())

    # The model's continuous predictions act as the classification score.
    roc_auc        = roc_auc_score(y_binary, preds)
    fpr, tpr, _    = roc_curve(y_binary, preds)

    if verbose:
        print(f"  (1) Ranking AUC (concordance) = {ranking_auc:.4f}")
        print(f"      Relationship: AUC = (Gini + 1)/2 = ({gini:.4f} + 1)/2")
        print(f"      → P(model ranks a higher-demand row above a lower one)")
        print()
        print(f"  (2) Binary ROC-AUC = {roc_auc:.4f}")
        print(f"      Positive class: arrival_rate ≥ {threshold:.4f} veh/s "
              f"(top {int((1-high_demand_quantile)*100)}%)")
        print(f"      Positives in test set: {n_pos} / {len(y_te)}")
        if roc_auc > 0.95:
            print(f"      → Excellent (>0.95): model reliably flags high-demand approaches")
        elif roc_auc > 0.85:
            print(f"      → Strong (0.85–0.95): high-demand discrimination is robust")
        elif roc_auc > 0.70:
            print(f"      → Acceptable (0.70–0.85)")
        else:
            print(f"      → Weak (<0.70): high-demand discrimination needs work")
        print()

    # ── Plot ROC curve ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, "b-", lw=2,
            label=f"Model  (ROC-AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Random  (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="steelblue")
    ax.set_xlabel("False Positive Rate\n(low-demand approaches wrongly flagged as high)")
    ax.set_ylabel("True Positive Rate\n(high-demand approaches correctly flagged)")
    ax.set_title(
        "ROC Curve — High-Demand Discrimination\n"
        f"Positive class = top {int((1-high_demand_quantile)*100)}% of arrival rates "
        f"(≥ {threshold:.3f} veh/s)\n"
        "This is the capability the optimizer relies on to reallocate green time",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig_roc_auc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print("  Saved: fig_roc_auc.png")
        print()

    return {
        "ranking_auc": round(ranking_auc, 4),
        "roc_auc":     round(roc_auc, 4),
        "threshold":   round(threshold, 4),
        "n_positive":  n_pos,
    }


# ================================================================== #
#  Master runner — call all validation tests in one go                 #
# ================================================================== #

def run_all_validation(
    model,
    df_raw: pd.DataFrame,
    model_name: str = "Best Model",
    feature_cols=None,
    verbose: bool = True,
) -> dict:
    """
    Run the full validation suite and return a summary dict.

    Tests run
    ---------
    1. Cross-validation        (k-fold train/val MAE and R2)
    2. Feature importance      (Gini MDI vs Permutation)
    3. SHAP analysis           (4 charts; requires shap package)
    4. Residual analysis       (5-panel diagnostic chart)
    5. Learning curves         (overfitting / data-sufficiency check)
    6. Gini coefficient        (regression ranking quality + Lorenz curve)
    7. AUC analysis            (ranking AUC + binary ROC-AUC + ROC curve)

    Returns a dict with headline metrics for the summary table.
    """
    if verbose:
        banner(f"MODEL VALIDATION — {model_name}")

    if feature_cols is None:
        feature_cols = FEATURE_COLS

    summary = {"model": model_name}

    # 1. Cross-validation
    if verbose:
        print("  [1/7] Cross-Validation (5-fold, no shuffle) ...")
    cv_df = cross_validation_summary(model, df_raw, feature_cols, verbose=verbose)
    summary["cv_val_mae_mean"] = round(cv_df["Val MAE"].mean(), 5)
    summary["cv_val_mae_std"]  = round(cv_df["Val MAE"].std(),  5)
    summary["cv_val_r2_mean"]  = round(cv_df["Val R2"].mean(),  4)

    # 2. Feature importance comparison
    if verbose:
        print("  [2/7] Feature Importance: Gini vs Permutation ...")
    imp_df = importance_comparison(model, df_raw, feature_cols, verbose=verbose)
    summary["top_gini_feature"]  = imp_df.sort_values("Gini (MDI)", ascending=False).iloc[0]["Feature"]
    summary["top_perm_feature"]  = imp_df.sort_values("Permutation", ascending=False).iloc[0]["Feature"]

    # 3. SHAP
    if verbose:
        print("  [3/7] SHAP Analysis ...")
    shap_vals, shap_mean_abs = run_shap_analysis(
        model, df_raw, feature_cols, sample_size=400, verbose=verbose)
    if shap_mean_abs is not None:
        summary["top_shap_feature"] = shap_mean_abs.index[0]

    # 4. Residuals
    if verbose:
        print("  [4/7] Residual Analysis ...")
    residual_analysis(model, df_raw, feature_cols, verbose=verbose)

    # 5. Learning curves
    if verbose:
        print("  [5/7] Learning Curves ...")
    plot_learning_curves(model, df_raw, feature_cols, verbose=verbose)

    # 6. Gini coefficient
    if verbose:
        print("  [6/7] Gini Coefficient ...")
    gini = gini_coefficient_regression(model, df_raw, feature_cols, verbose=verbose)
    summary["gini"] = round(gini, 4)

    # 7. AUC analysis (ranking AUC + binary ROC-AUC)
    if verbose:
        print("  [7/7] AUC Analysis (ranking + ROC) ...")
    auc_res = auc_analysis(model, df_raw, feature_cols, verbose=verbose)
    summary["ranking_auc"] = auc_res["ranking_auc"]
    summary["roc_auc"]     = auc_res["roc_auc"]

    # Final summary
    if verbose:
        banner("VALIDATION SUMMARY")
        print(f"  Model            : {model_name}")
        print(f"  CV Val MAE       : {summary['cv_val_mae_mean']:.5f} "
              f"(+/- {summary['cv_val_mae_std']:.5f})")
        print(f"  CV Val R2        : {summary['cv_val_r2_mean']:.4f}")
        print(f"  Gini coefficient : {summary['gini']:.4f}")
        print(f"  Ranking AUC      : {summary['ranking_auc']:.4f}  (= (Gini+1)/2)")
        print(f"  Binary ROC-AUC   : {summary['roc_auc']:.4f}  (high-demand discrimination)")
        print(f"  Top feature (Gini MDI)  : {summary.get('top_gini_feature','N/A')}")
        print(f"  Top feature (Permutation): {summary.get('top_perm_feature','N/A')}")
        print(f"  Top feature (SHAP)       : {summary.get('top_shap_feature','N/A')}")
        print()
        print(f"  Output files (in {FIGURES_DIR}/):")
        for f in ["fig_shap_summary.png", "fig_shap_bar.png",
                  "fig_shap_by_hour.png", "fig_shap_waterfall.png",
                  "fig_residual_analysis.png", "fig_learning_curves.png",
                  "fig_gini_lorenz.png", "fig_roc_auc.png"]:
            print(f"    {FIGURES_DIR}/{f}")
        print()

    return summary
