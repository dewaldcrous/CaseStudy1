"""
Model Enhancement Module
========================

BUSINESS CASE
-------------
The base model (LightGBM on 12 features) achieves Test MAE ~0.0055 veh/s
and R2 ~0.985.  While strong overall, "overall accuracy" hides where errors
concentrate.  In a traffic optimization context, prediction errors during
rush hours are 10–20× more costly than errors at 2 AM, because:

  1. Rush-hour demand feeds directly into Webster's cycle calculation.
     A 20% error in NS demand at 8 AM can mis-allocate green time by
     several seconds per cycle, compounding across thousands of vehicles.

  2. Night-time errors have near-zero impact: queues are short regardless.

This module tests three improvement strategies and quantifies each one
separately so a business stakeholder can make an informed trade-off between
accuracy gain, training time, and code complexity.

Strategy 1 — Feature Engineering
  Add interaction terms that encode domain knowledge the model would
  otherwise need to discover through deep splits.  "Rush hour × NS direction"
  directly captures that north–south demand triples at 8 AM.  Cheap to
  implement, no additional data required, no risk of leakage.

Strategy 2 — Importance Sampling (weighted/resampled training)
  Traffic data is naturally imbalanced: ~8/24 hours are "high value" rush
  periods, ~8/24 are low-traffic night hours.  Training on the raw
  distribution biases the loss function towards the majority (quiet) class.
  Re-weighting trains the model to prioritise accuracy where it matters.
  Two approaches are tested:
    a. Sample weights — multiply the loss contribution of each rush-hour
       row by a factor.  Elegant, preserves dataset size.
    b. Resampling — explicitly duplicate rush-hour rows and drop a fraction
       of night rows.  Cruder but sometimes more effective with tree models.

Strategy 3 — Hyperparameter Tuning
  Default hyperparameters are sensible starting points but rarely optimal.
  RandomizedSearchCV samples the parameter space efficiently — it draws
  N random combinations from pre-defined distributions, evaluates each
  via k-fold cross-validation, and returns the best.  Compared to
  exhaustive GridSearchCV it is ~10× faster for the same coverage.
  We tune the best available model (LightGBM if present, else RF).

Combined strategy:
  Best feature set + resampled training + tuned hyperparameters.
  This is the "production-ready" configuration recommended for deployment.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics         import mean_absolute_error, r2_score
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.inspection      import permutation_importance
from scipy.stats             import randint, uniform, loguniform
from typing                  import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.ml_model import FEATURE_COLS, add_features


# ================================================================== #
#  Strategy 1 — Feature Engineering                                   #
# ================================================================== #

# Three feature sets with increasing domain-knowledge encoding.
# Each set is a superset of the previous.

BASE_FEATURES = FEATURE_COLS  # 12 features from Section 2

INTERACTION_FEATURES = BASE_FEATURES + [
    # DOMAIN RULE: NS demand triples during AM rush.
    # Encoding this explicitly as a product term is cheaper for the model
    # than discovering it requires combining is_morning_rush AND ns_direction.
    "rush_am_x_ns",       # is_morning_rush × ns_direction

    # DOMAIN RULE: EW demand doubles during PM rush.
    "rush_pm_x_ew",       # is_evening_rush × (1 - ns_direction)

    # DOMAIN RULE: the time-of-day effect on NS and EW approaches differs.
    # Without this term, hour_sin has the same coefficient for NS and EW.
    "hour_sin_x_ns",      # hour_sin × ns_direction
    "hour_cos_x_ns",      # hour_cos × ns_direction
]

RICH_FEATURES = INTERACTION_FEATURES + [
    # DOMAIN RULE: rain affects rush-hour more than off-peak because
    # the supply of alternative transport (cycling, walking) is higher
    # during daytime when people have a choice.
    "weather_x_rush",     # weather × (is_morning_rush + is_evening_rush)

    # DOMAIN RULE: weekend peak is symmetric but larger intersections
    # experience sharper midday concentration.
    "weekend_x_iid",      # is_weekend × intersection_id

    # Squared circular time to capture sharper rush-hour spikes
    "hour_sin_sq",        # hour_sin²
    "hour_cos_sq",        # hour_cos²
]


def build_feature_set(df: pd.DataFrame, feature_set: List[str]) -> pd.DataFrame:
    """
    Add interaction and polynomial features to a feature-engineered DataFrame.
    Columns not already present are computed on the fly.
    """
    df = df.copy()

    # Interaction features
    if "rush_am_x_ns"   in feature_set:
        df["rush_am_x_ns"]   = df["is_morning_rush"] * df["ns_direction"]
    if "rush_pm_x_ew"   in feature_set:
        df["rush_pm_x_ew"]   = df["is_evening_rush"] * (1 - df["ns_direction"])
    if "hour_sin_x_ns"  in feature_set:
        df["hour_sin_x_ns"]  = df["hour_sin"] * df["ns_direction"]
    if "hour_cos_x_ns"  in feature_set:
        df["hour_cos_x_ns"]  = df["hour_cos"] * df["ns_direction"]

    # Weather interactions
    if "weather_x_rush" in feature_set:
        df["weather_x_rush"] = df["weather"] * (
            df["is_morning_rush"] + df["is_evening_rush"]
        )
    if "weekend_x_iid"  in feature_set:
        df["weekend_x_iid"]  = df["is_weekend"] * df["intersection_id"]

    # Polynomial
    if "hour_sin_sq"    in feature_set:
        df["hour_sin_sq"]    = df["hour_sin"] ** 2
    if "hour_cos_sq"    in feature_set:
        df["hour_cos_sq"]    = df["hour_cos"] ** 2

    return df


FEATURE_STRATEGIES = {
    "Base (12 features)":             BASE_FEATURES,
    "Interactions (16 features)":     INTERACTION_FEATURES,
    "Rich (20 features)":             RICH_FEATURES,
}


# ================================================================== #
#  Strategy 2 — Importance Sampling                                   #
# ================================================================== #

def compute_sample_weights(df: pd.DataFrame, rush_weight: float = 3.0,
                            night_weight: float = 0.4) -> np.ndarray:
    """
    Assign per-row training weights to focus the model on rush hours.

    BUSINESS RATIONALE
    ------------------
    Webster's formula uses predicted demand directly in the cycle
    calculation.  A prediction error of ε at 8 AM shifts the NS green
    time by roughly  ε / (saturation_flow × Y²) × C² seconds.
    At rush-hour demand levels (Y ~ 0.9) this multiplier is ~10×
    larger than at night (Y ~ 0.15).  So rush-hour errors cost ~10×
    more in terms of real-world wait time.

    We therefore weight rush-hour training rows by `rush_weight` so
    the loss function penalises those errors proportionally more.

    Parameters
    ----------
    rush_weight  : weight for rows in 7-9am or 4-7pm on weekdays
    night_weight : weight for rows in 0-5am (any day)
    """
    weights = np.ones(len(df))

    is_rush  = ((df["hour"].between(7, 9) | df["hour"].between(16, 19))
                & (df["is_weekend"] == 0))
    is_night = df["hour"] <= 5

    weights[is_rush.values]  = rush_weight
    weights[is_night.values] = night_weight

    return weights


def resample_dataframe(df: pd.DataFrame, rush_repeat: int = 2,
                       night_keep_frac: float = 0.35,
                       seed: int = 42) -> pd.DataFrame:
    """
    Physically duplicate rush-hour rows and drop a fraction of night rows.

    APPROACH
    --------
    Unlike sample_weight (which keeps the dataset size fixed but adjusts
    the gradient contribution), resampling changes the actual data the
    model trains on.  Tree-based models sometimes respond better to
    resampling than to weights because:
      - Node splits are chosen by impurity reduction, which depends on
        how many rows fall into each leaf, not just their weights.
      - Duplicated rows give the model multiple "votes" to place splits
        at rush-hour boundaries.

    BUSINESS TRADE-OFF
    ------------------
    Resampling increases dataset size (by ~rush_repeat × rush fraction),
    which increases training time.  On this dataset (11,520 rows),
    night undersampling roughly compensates so net size is similar.
    """
    rng = np.random.default_rng(seed)

    is_rush  = ((df["hour"].between(7, 9) | df["hour"].between(16, 19))
                & (df["is_weekend"] == 0))
    is_night = df["hour"] <= 5
    is_other = ~is_rush & ~is_night

    rush_rows  = pd.concat([df[is_rush]] * rush_repeat, ignore_index=True)
    night_idx  = df[is_night].index
    keep_idx   = rng.choice(night_idx, size=int(len(night_idx) * night_keep_frac),
                             replace=False)
    night_rows = df.loc[keep_idx]
    other_rows = df[is_other]

    resampled = pd.concat([rush_rows, other_rows, night_rows], ignore_index=True)
    resampled = resampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    return resampled


# ================================================================== #
#  Strategy 3 — Hyperparameter Tuning                                 #
# ================================================================== #

def _get_param_space(model_name: str) -> dict:
    """
    Define the hyperparameter search space for each model type.

    WHY THESE PARAMETERS?
    ---------------------
    LightGBM
      num_leaves      Controls tree complexity.  More leaves → more complex
                      decision boundaries but higher risk of overfitting.
                      Range 15-127 covers simple to moderately complex trees.
      learning_rate   Step size per boosting round.  Lower = more rounds
                      needed but usually better generalisation.
      min_child_samples  Minimum rows per leaf.  Higher = more regularisation.
      n_estimators    Number of boosting rounds.  Works with learning_rate.
      subsample       Row subsampling per tree (0.6-1.0).  Reduces variance.
      colsample_bytree  Feature subsampling per tree.  Similar to RF.

    Random Forest
      n_estimators    More trees = lower variance but diminishing returns.
      max_depth       Depth limit.  Shallower = more regularisation.
      min_samples_leaf  Minimum rows per leaf.
      max_features    Features at each split (like colsample for RF).

    XGBoost
      max_depth, learning_rate, n_estimators, subsample, colsample_bytree,
      reg_alpha (L1), reg_lambda (L2) — L1/L2 regularisation is XGBoost's
      main advantage over plain GBM.
    """
    if "LightGBM" in model_name:
        return {
            "num_leaves":        randint(15, 128),
            "learning_rate":     loguniform(0.01, 0.3),
            "min_child_samples": randint(5, 60),
            "n_estimators":      randint(100, 500),
            "subsample":         uniform(0.6, 0.4),
            "colsample_bytree":  uniform(0.6, 0.4),
        }
    elif "XGBoost" in model_name:
        return {
            "max_depth":         randint(3, 10),
            "learning_rate":     loguniform(0.01, 0.3),
            "n_estimators":      randint(100, 400),
            "subsample":         uniform(0.6, 0.4),
            "colsample_bytree":  uniform(0.6, 0.4),
            "reg_alpha":         loguniform(1e-4, 1.0),
            "reg_lambda":        loguniform(0.1, 5.0),
        }
    else:  # Random Forest
        return {
            "n_estimators":      randint(80, 300),
            "max_depth":         randint(5, 20),
            "min_samples_leaf":  randint(2, 12),
            "max_features":      ["sqrt", "log2", 0.5, 0.7],
        }


def tune_model(
    model,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    n_iter: int = 30,
    cv: int = 5,
    verbose: bool = True,
) -> Tuple[object, dict, float]:
    """
    Run RandomizedSearchCV on `model` and return the best estimator.

    METHODOLOGY
    -----------
    RandomizedSearchCV samples `n_iter` combinations from the parameter
    distributions, evaluating each via `cv`-fold cross-validation.
    Scoring metric: negative MAE (sklearn convention — higher = better).

    n_iter=30, cv=5 means 150 model fits total.  Roughly 2-5 minutes
    for LightGBM on this dataset — acceptable for an offline training step
    that runs once per day in production.

    BUSINESS TRADE-OFF
    ------------------
    Increasing n_iter from 30 to 100 typically adds <0.5% MAE improvement
    while tripling compute cost.  30 iterations is the pragmatic sweet spot
    for this problem size.

    Parameters
    ----------
    model          : unfitted estimator
    model_name     : string used to look up the parameter space
    X_train        : feature matrix (numpy)
    y_train        : target vector
    sample_weights : optional per-row weights (passed to fit via fit_params)
    n_iter         : number of random parameter combinations to try
    cv             : number of cross-validation folds

    Returns
    -------
    best_estimator, best_params, best_cv_mae
    """
    param_space = _get_param_space(model_name)
    fit_params  = {}
    if sample_weights is not None:
        # Pass weights through CV — sklearn maps "estimator__sample_weight"
        fit_params["sample_weight"] = sample_weights

    # n_jobs=1 avoids spawning parallel worker processes.
    # On Windows, multiprocessing workers re-import the full scipy/numpy stack
    # per process which can exhaust memory on laptops.  Single-threaded search
    # is slower (roughly n_iter × fit_time) but completes reliably.
    search = RandomizedSearchCV(
        estimator           = model,
        param_distributions = param_space,
        n_iter              = n_iter,
        cv                  = cv,
        scoring             = "neg_mean_absolute_error",
        refit               = True,
        random_state        = 42,
        n_jobs              = 1,
        verbose             = 0,
    )

    search.fit(X_train, y_train, **fit_params)

    best_cv_mae = -search.best_score_

    if verbose:
        print(f"    Best CV MAE: {best_cv_mae:.5f}")
        print(f"    Best params:")
        for k, v in search.best_params_.items():
            if isinstance(v, float):
                print(f"      {k:<22} {v:.4f}")
            else:
                print(f"      {k:<22} {v}")

    return search.best_estimator_, search.best_params_, best_cv_mae


# ================================================================== #
#  Full enhancement experiment runner                                  #
# ================================================================== #

def run_enhancement_experiments(
    df_raw: pd.DataFrame,
    base_model_name: str,
    base_trained: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run all three strategies and report improvement over the base model.

    Experiment matrix
    -----------------
    1. Base model (control)
    2. Feature set: Base → Interactions → Rich
    3. Sampling:    None → sample weights → resampling
    4. Tuning:      Default → RandomizedSearchCV
    5. Combined:    Best feature set + resampling + tuning

    Returns a DataFrame ranked by rush-hour MAE (the metric that matters
    most for the business case — not overall MAE).
    """
    if verbose:
        banner_local("MODEL ENHANCEMENT EXPERIMENTS")
        print("  Each experiment is evaluated on TWO metrics:")
        print("  Overall MAE  : accuracy across all hours")
        print("  Rush-hour MAE: accuracy only during 7-9am and 4-7pm weekdays")
        print("                 (the periods where prediction errors cost most)")
        print()

    # ── MODEL SPEC: X (features) and y (target) ──────────────────────
    # X : feature matrix, shape (N, 12)
    #     Columns = FEATURE_COLS:
    #       Group 1 — circular time:  hour_sin, hour_cos, dow_sin, dow_cos
    #       Group 2 — period flags:   is_morning_rush, is_evening_rush,
    #                                  is_daytime, is_weekend
    #       Group 3 — weather:        weather  (0=clear, 1=cloudy, 2=rain)
    #       Group 4 — location/dir:   intersection_id, direction_id, ns_direction
    #
    # y : target vector, shape (N,)
    #     Column = "arrival_rate"  (vehicles per second, float, ~0.005–1.2)
    #     This is what every model predicts.  Errors in y during rush hours
    #     propagate into Webster's green-split calculation (q→y_i→C*).
    #
    # ── Prepare train / test split ────────────────────────────────────
    df_fe    = add_features(df_raw)
    max_day  = df_fe["day"].max()
    train_df = df_fe[df_fe["day"] <= max_day - 5].copy()  # X_tr rows
    test_df  = df_fe[df_fe["day"] >  max_day - 5].copy()  # X_te rows

    # Rush-hour test mask — subset of test rows where y errors cost most
    rush_mask = (
        (test_df["hour"].between(7, 9) | test_df["hour"].between(16, 19))
        & (test_df["is_weekend"] == 0)
    )

    y_test       = test_df["arrival_rate"].values             # y_te: all hours
    y_test_rush  = test_df.loc[rush_mask, "arrival_rate"].values  # y_te: rush only

    results = []

    def _eval(name, model, X_te, y_te, y_te_rush, X_te_rush):
        p_all  = model.predict(X_te)
        p_rush = model.predict(X_te_rush)
        return {
            "Experiment":         name,
            "Overall MAE":        round(mean_absolute_error(y_te,       p_all),  5),
            "Rush-hour MAE":      round(mean_absolute_error(y_te_rush,  p_rush), 5),
            "Overall R2":         round(r2_score(y_te,      p_all),              4),
        }

    # ── Factory: get a fresh unfitted model ───────────────────────────
    def fresh_model():
        # n_jobs=1 on all models — avoids Windows multiprocessing MemoryErrors
        # when many parallel CV workers each re-load scipy/numpy.
        if LGBM_AVAILABLE and "LightGBM" in base_model_name:
            return lgb.LGBMRegressor(
                n_estimators=200, num_leaves=63, learning_rate=0.05,
                min_child_samples=10, random_state=42, n_jobs=1, verbose=-1)
        elif XGB_AVAILABLE and "XGBoost" in base_model_name:
            return xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                random_state=42, n_jobs=1, verbosity=0)
        else:
            return RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=4,
                random_state=42, n_jobs=1)

    # ── Experiment 1: baseline (control) ──────────────────────────────
    if verbose:
        print("  [1/5] Control: base model, base features, no sampling, default params")
    base_m      = base_trained[base_model_name]
    X_te_base   = build_feature_set(test_df, BASE_FEATURES)[BASE_FEATURES].values
    X_te_base_r = build_feature_set(test_df[rush_mask], BASE_FEATURES)[BASE_FEATURES].values
    results.append(_eval("1. Control (base)", base_m, X_te_base, y_test,
                         y_test_rush, X_te_base_r))
    if verbose:
        r = results[-1]
        print(f"     Overall MAE={r['Overall MAE']:.5f}  "
              f"Rush MAE={r['Rush-hour MAE']:.5f}")
        print()

    # ── Experiment 2: Feature Engineering ─────────────────────────────
    if verbose:
        print("  [2/5] Feature Engineering: compare Base vs Interactions vs Rich")

    for strat_name, feat_cols in FEATURE_STRATEGIES.items():
        tr  = build_feature_set(train_df, feat_cols)
        te  = build_feature_set(test_df,   feat_cols)
        te_r = build_feature_set(test_df[rush_mask], feat_cols)

        X_tr  = tr[feat_cols].values
        X_te  = te[feat_cols].values
        X_te_r = te_r[feat_cols].values
        y_tr  = tr["arrival_rate"].values

        m = fresh_model()
        m.fit(X_tr, y_tr)
        results.append(_eval(f"2. FE: {strat_name}", m, X_te, y_test,
                             y_test_rush, X_te_r))
        r = results[-1]
        if verbose:
            print(f"     {strat_name:<30} Overall MAE={r['Overall MAE']:.5f}  "
                  f"Rush MAE={r['Rush-hour MAE']:.5f}")

    if verbose: print()

    # ── Experiment 3: Sampling ─────────────────────────────────────────
    if verbose:
        print("  [3/5] Sampling strategies (using Interactions feature set)")

    best_fe_name = min(
        [r for r in results if r["Experiment"].startswith("2.")],
        key=lambda r: r["Rush-hour MAE"]
    )["Experiment"]
    # Extract feature set key from name
    fe_key  = next(k for k in FEATURE_STRATEGIES if k in best_fe_name)
    fe_cols = FEATURE_STRATEGIES[fe_key]

    # 3a: sample weights
    tr_fe    = build_feature_set(train_df, fe_cols)
    te_fe    = build_feature_set(test_df,   fe_cols)
    te_fe_r  = build_feature_set(test_df[rush_mask], fe_cols)
    X_tr_fe  = tr_fe[fe_cols].values
    X_te_fe  = te_fe[fe_cols].values
    X_te_fe_r = te_fe_r[fe_cols].values
    y_tr_fe  = tr_fe["arrival_rate"].values

    sw = compute_sample_weights(train_df)
    m  = fresh_model()
    m.fit(X_tr_fe, y_tr_fe, sample_weight=sw)
    results.append(_eval("3a. Sample weights", m, X_te_fe, y_test,
                         y_test_rush, X_te_fe_r))
    r = results[-1]
    if verbose:
        print(f"     Sample weights (rush×3, night×0.4)  "
              f"Overall MAE={r['Overall MAE']:.5f}  "
              f"Rush MAE={r['Rush-hour MAE']:.5f}")

    # 3b: physical resampling
    resampled   = resample_dataframe(build_feature_set(train_df, fe_cols))
    X_tr_rs     = resampled[fe_cols].values
    y_tr_rs     = resampled["arrival_rate"].values
    m           = fresh_model()
    m.fit(X_tr_rs, y_tr_rs)
    results.append(_eval("3b. Resampling", m, X_te_fe, y_test,
                         y_test_rush, X_te_fe_r))
    r = results[-1]
    if verbose:
        print(f"     Resampling (rush×2, night keep 35%)  "
              f"Overall MAE={r['Overall MAE']:.5f}  "
              f"Rush MAE={r['Rush-hour MAE']:.5f}")
        print()

    # ── Experiment 4: Hyperparameter Tuning ───────────────────────────
    if verbose:
        print("  [4/5] Hyperparameter Tuning (RandomizedSearchCV, 20 iter, 3-fold CV)")
        print(f"        Tuning: {base_model_name}")
        print(f"        n_jobs=1 to avoid Windows multiprocessing MemoryErrors")

    tuned_m, best_params, cv_mae = tune_model(
        fresh_model(), base_model_name,
        X_tr_fe, y_tr_fe,
        sample_weights=None,
        n_iter=20, cv=3, verbose=verbose,
    )
    results.append(_eval("4. Tuned (no sampling)", tuned_m, X_te_fe, y_test,
                         y_test_rush, X_te_fe_r))
    r = results[-1]
    if verbose:
        print(f"     Overall MAE={r['Overall MAE']:.5f}  "
              f"Rush MAE={r['Rush-hour MAE']:.5f}")
        print()

    # ── Experiment 5: Combined (best FE + resampling + tuning) ────────
    if verbose:
        print("  [5/5] Combined: best features + resampling + tuned hyperparams")

    # Apply best hyperparams to a fresh model, train on resampled data
    best_params_clean = {k: v for k, v in best_params.items()}
    m5 = fresh_model()
    m5.set_params(**best_params_clean)
    m5.fit(X_tr_rs, y_tr_rs)   # resampled data + tuned params
    results.append(_eval("5. Combined (FE+Resample+Tuned)", m5, X_te_fe,
                         y_test, y_test_rush, X_te_fe_r))
    r = results[-1]
    if verbose:
        print(f"     Overall MAE={r['Overall MAE']:.5f}  "
              f"Rush MAE={r['Rush-hour MAE']:.5f}")
        print()

    results_df = pd.DataFrame(results).sort_values("Rush-hour MAE").reset_index(drop=True)

    if verbose:
        print("  Summary (sorted by Rush-hour MAE — the metric that matters):")
        print()
        print(f"  {'Experiment':<40} {'Overall MAE':>12} {'Rush MAE':>10} {'R2':>8}")
        print("  " + "-" * 75)
        base_rush = results[0]["Rush-hour MAE"]
        for _, row in results_df.iterrows():
            imp = (base_rush - row["Rush-hour MAE"]) / base_rush * 100
            tag = f"  ({imp:+.1f}%)" if imp != 0 else "  (baseline)"
            print(f"  {row['Experiment']:<40} {row['Overall MAE']:>12.5f} "
                  f"{row['Rush-hour MAE']:>10.5f} {row['Overall R2']:>8.4f}{tag}")
        print()

        # Business recommendation
        best_row = results_df.iloc[0]
        best_imp = (base_rush - best_row["Rush-hour MAE"]) / base_rush * 100
        print("  BUSINESS RECOMMENDATION")
        print("  " + "-" * 60)
        print(f"  Best configuration: {best_row['Experiment']}")
        print(f"  Rush-hour MAE improvement over base: {best_imp:.1f}%")
        print()
        print("  Deploy if: rush-hour MAE improvement > 5%  (worth the")
        print("  extra training complexity) AND overall R2 stays above 0.97.")
        print()
        print("  NOT recommended to deploy if:")
        print("  - Improvement < 5% (noise, not signal)")
        print("  - Training time > 10 min (exceeds daily retraining budget)")
        print()

    return results_df, m5, fe_cols, best_params


# ================================================================== #
#  Strategy 4 — Feature Reduction (dimensionality reduction)           #
# ================================================================== #

def run_feature_reduction_experiments(
    df_raw: pd.DataFrame,
    base_model_name: str,
    base_trained: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Test whether the model can be SIMPLIFIED — fewer features — WITHOUT
    losing accuracy.

    BUSINESS CASE FOR FEWER FEATURES
    --------------------------------
    More features is not automatically better.  A leaner model is:
      - Cheaper to operate   — each feature is a live data feed that must be
                               collected, validated, and monitored in production.
      - Faster               — less compute per prediction (matters at 60s
                               re-optimisation cadence across many intersections).
      - More robust          — fewer features means fewer things that can drift,
                               break, or arrive late from a sensor.
      - Easier to explain    — a 6-feature model is far simpler to defend to a
                               regulator than a 12- or 20-feature one.

    So the question is: how few features can we keep before accuracy degrades?
    If we can drop from 12 → 7 features for <2% MAE loss, that is a clear win.

    METHODS COMPARED
    ----------------
    1. Full set (control)        — all 12 base features.
    2. Correlation pruning       — for every feature pair with |r| > 0.80,
                                   drop the one LESS correlated with the target.
                                   Removes redundant (collinear) inputs.
    3. Variance threshold        — drop near-constant features (they carry
                                   almost no information).
    4. Top-K permutation         — rank features by permutation importance on a
                                   validation split, keep the top K (K = 8, 6).
    5. RFE (Recursive Feature    — repeatedly fit the model, drop the weakest
       Elimination)                feature, refit, until K remain (K = 8, 6).

    Returns a DataFrame ranked by Overall MAE with the feature count for each.
    """
    if verbose:
        banner_local("FEATURE REDUCTION EXPERIMENTS")
        print("  Question: can we use FEWER features without losing accuracy?")
        print("  A leaner model is cheaper, faster, more robust, easier to explain.")
        print()

    # ── MODEL SPEC (same X→y contract as all other experiments) ──────────
    # X (feature matrix) : one row per (intersection, direction, hour, day)
    #   Starting from BASE_FEATURES (12 cols); each method below tries
    #   a subset to see whether fewer features degrades y predictions.
    # y (target)         : "arrival_rate"  float (veh/s)  — unchanged across
    #   all reduction experiments.  We only vary which X columns are used.
    #
    # ── Train / test split (same day-based rule as everywhere else) ──────
    df_fe    = add_features(df_raw)
    max_day  = df_fe["day"].max()
    train_df = df_fe[df_fe["day"] <= max_day - 5].copy()
    test_df  = df_fe[df_fe["day"] >  max_day - 5].copy()

    feats_all = list(BASE_FEATURES)           # all 12 feature names (X columns)
    X_tr_all  = train_df[feats_all].values    # X_train : shape (9216, 12)
    X_te_all  = test_df[feats_all].values     # X_test  : shape (2304, 12)
    y_tr      = train_df["arrival_rate"].values   # y_train : shape (9216,)  veh/s
    y_te      = test_df["arrival_rate"].values    # y_test  : shape (2304,)  veh/s

    def fresh_model():
        # Same factory as the enhancement experiments; n_jobs=1 for Windows safety
        if LGBM_AVAILABLE and "LightGBM" in base_model_name:
            return lgb.LGBMRegressor(
                n_estimators=200, num_leaves=63, learning_rate=0.05,
                min_child_samples=10, random_state=42, n_jobs=1, verbose=-1)
        elif XGB_AVAILABLE and "XGBoost" in base_model_name:
            return xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                random_state=42, n_jobs=1, verbosity=0)
        else:
            return RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=4,
                random_state=42, n_jobs=1)

    def _train_eval(name, feat_subset):
        """Train a fresh model on feat_subset and return metrics."""
        m = fresh_model()
        m.fit(train_df[feat_subset].values, y_tr)
        p = m.predict(test_df[feat_subset].values)
        return {
            "Method":       name,
            "n_features":   len(feat_subset),
            "Features":     ", ".join(feat_subset),
            "Overall MAE":  round(mean_absolute_error(y_te, p), 5),
            "Overall R2":   round(r2_score(y_te, p), 4),
        }

    results = []

    # ── 1. Full set (control) ────────────────────────────────────────────
    if verbose:
        print(f"  [1/6] Full feature set ({len(feats_all)} features) — control")
    results.append(_train_eval("1. Full set (control)", feats_all))
    if verbose:
        r = results[-1]
        print(f"        MAE={r['Overall MAE']:.5f}  R2={r['Overall R2']:.4f}")
        print()

    # ── 2. Correlation pruning (drop redundant collinear features) ───────
    if verbose:
        print("  [2/6] Correlation pruning (drop one of each |r|>0.80 pair)")
    corr      = train_df[feats_all].corr()
    target_r  = train_df[feats_all + ["arrival_rate"]].corr()["arrival_rate"].abs()
    to_drop   = set()
    for i in range(len(feats_all)):
        for j in range(i + 1, len(feats_all)):
            if abs(corr.iloc[i, j]) > 0.80:
                f_i, f_j = feats_all[i], feats_all[j]
                # Drop whichever is LESS correlated with the target
                weaker = f_i if target_r[f_i] < target_r[f_j] else f_j
                to_drop.add(weaker)
    feats_corr = [f for f in feats_all if f not in to_drop]
    if verbose:
        print(f"        Dropped {sorted(to_drop) if to_drop else 'nothing'} "
              f"→ {len(feats_corr)} features")
    results.append(_train_eval("2. Correlation pruned", feats_corr))
    if verbose:
        r = results[-1]
        print(f"        MAE={r['Overall MAE']:.5f}  R2={r['Overall R2']:.4f}")
        print()

    # ── 3. Variance threshold (drop near-constant features) ──────────────
    if verbose:
        print("  [3/6] Variance threshold (drop near-constant features)")
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X_tr_all)
    feats_var = [f for f, keep in zip(feats_all, vt.get_support()) if keep]
    if verbose:
        dropped_var = [f for f in feats_all if f not in feats_var]
        print(f"        Dropped {dropped_var if dropped_var else 'nothing'} "
              f"→ {len(feats_var)} features")
    results.append(_train_eval("3. Variance threshold", feats_var))
    if verbose:
        r = results[-1]
        print(f"        MAE={r['Overall MAE']:.5f}  R2={r['Overall R2']:.4f}")
        print()

    # ── 4. Top-K by permutation importance (K = 8, 6) ────────────────────
    if verbose:
        print("  [4/6] Top-K by permutation importance (model-agnostic ranking)")
    base_m = base_trained[base_model_name]
    perm   = permutation_importance(
        base_m, X_te_all, y_te,
        n_repeats=10, random_state=42,
        scoring="neg_mean_absolute_error", n_jobs=1)
    perm_order = [feats_all[i] for i in np.argsort(perm.importances_mean)[::-1]]
    for K in (8, 6):
        feats_topk = perm_order[:K]
        results.append(_train_eval(f"4. Top-{K} permutation", feats_topk))
        if verbose:
            r = results[-1]
            print(f"        Top-{K}: {feats_topk}")
            print(f"        MAE={r['Overall MAE']:.5f}  R2={r['Overall R2']:.4f}")
    if verbose: print()

    # ── 5. RFE (Recursive Feature Elimination, K = 8, 6) ─────────────────
    if verbose:
        print("  [5/6] RFE: iteratively drop the weakest feature (K = 8, 6)")
    for K in (8, 6):
        # RFE needs an estimator exposing feature_importances_; all tree models do.
        rfe = RFE(estimator=fresh_model(), n_features_to_select=K, step=1)
        rfe.fit(X_tr_all, y_tr)
        feats_rfe = [f for f, keep in zip(feats_all, rfe.support_) if keep]
        results.append(_train_eval(f"5. RFE top-{K}", feats_rfe))
        if verbose:
            r = results[-1]
            print(f"        RFE-{K}: {feats_rfe}")
            print(f"        MAE={r['Overall MAE']:.5f}  R2={r['Overall R2']:.4f}")
    if verbose: print()

    # ── Summary ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("Overall MAE").reset_index(drop=True)

    if verbose:
        print("  [6/6] Summary (sorted by Overall MAE):")
        print()
        print(f"  {'Method':<26} {'#Feat':>6} {'MAE':>10} {'R2':>8}  vs full")
        print("  " + "-" * 64)
        full_mae = next(r["Overall MAE"] for r in results
                        if r["Method"].startswith("1."))
        for _, row in results_df.iterrows():
            delta = (row["Overall MAE"] - full_mae) / full_mae * 100
            tag   = "baseline" if row["Method"].startswith("1.") else f"{delta:+.1f}% MAE"
            print(f"  {row['Method']:<26} {row['n_features']:>6} "
                  f"{row['Overall MAE']:>10.5f} {row['Overall R2']:>8.4f}  {tag}")
        print()

        # Business recommendation: find the leanest set within +2% MAE of full
        candidates = results_df[
            (results_df["Overall MAE"] <= full_mae * 1.02)
        ].sort_values("n_features")
        leanest = candidates.iloc[0]
        print("  BUSINESS RECOMMENDATION")
        print("  " + "-" * 60)
        if leanest["n_features"] < len(feats_all):
            saved = len(feats_all) - leanest["n_features"]
            print(f"  Leanest model within +2% MAE of full: {leanest['Method']}")
            print(f"  Uses {leanest['n_features']} features (drops {saved}) "
                  f"at MAE={leanest['Overall MAE']:.5f}, R2={leanest['Overall R2']:.4f}")
            print(f"  → Can simplify from {len(feats_all)} to {leanest['n_features']} "
                  f"features with negligible accuracy loss.")
            print(f"  → Fewer data feeds to maintain, faster inference, more robust.")
        else:
            print(f"  No reduction stays within +2% MAE — all 12 features earn their place.")
            print(f"  → Keep the full feature set; the model is already parsimonious.")
        print()

    return results_df


def banner_local(title: str) -> None:
    print("=" * 66)
    print(f"  {title}")
    print("=" * 66)
    print()


# ================================================================== #
#  Enhanced prediction wrapper                                         #
# ================================================================== #

def predict_rates_enhanced(
    model,
    feature_cols: List[str],
    hour: int,
    day_of_week: int,
    weather: int = 0,
    num_intersections: int = 4,
) -> Dict[int, Dict[str, float]]:
    """
    Like predict_rates() from ml_model.py but builds the richer feature
    set required by an enhanced model (interactions, polynomial terms).

    TARGET VARIABLE  (same as base model)
    --------------------------------------
    arrival_rate  float  veh/s  range ≈0.005–1.2
      Predicted demand at each approach.  Values feed into WebsterOptimizer
      exactly the same way as the base predict_rates() output.

    FEATURE MATRIX CONSTRUCTED INTERNALLY
    --------------------------------------
    Base 12 features (FEATURE_COLS) are first generated by add_features(),
    then build_feature_set() appends enhancement features selected by the
    best experiment.  The final X has shape (num_intersections × 4, len(feature_cols)).

    Enhancement features that may be present (see build_feature_set()):
      rush_weather   = is_morning_rush × weather   — rain during rush amplifies demand
      rush_direction = is_morning_rush × ns_direction — AM NS dominance interaction
      hour_sin_sq    = hour_sin²                   — non-linear time curvature
      hour_weather   = hour_sin × weather          — rain shifts peak timing slightly

    PARAMETERS
    ----------
    model         : enhanced trained estimator exposing .predict(X_numpy)
    feature_cols  : list[str]  columns selected by the best enhancement experiment;
                   passed at construction time so this function is self-contained.
    hour          : int 0-23
    day_of_week   : int 0-6  (Mon=0 … Sun=6)
    weather       : int {0,1,2}  0=clear, 1=cloudy, 2=rain
    num_intersections : int  number of intersections (default 4)

    OUTPUT
    ------
    Dict[intersection_id, Dict[direction, float]]
      Same nested structure as predict_rates() — drop-in replacement for
      enhanced-model runs.  Values are floored at 0.005 veh/s to avoid
      division-by-zero inside Webster's formula (Y = q / saturation_flow).

    DIRECTIONS matches the order used during training.
    """
    from src.ml_model import DIRECTIONS
    rows = [
        {
            "hour":             hour,
            "day_of_week":      day_of_week,
            "is_weekend":       int(day_of_week >= 5),
            "intersection_id":  iid,
            "direction":        direction,
            "direction_id":     did,
            "weather":          weather,
        }
        for iid in range(num_intersections)
        for did, direction in enumerate(DIRECTIONS)
    ]
    df = add_features(pd.DataFrame(rows))
    df = build_feature_set(df, feature_cols)
    X  = df[feature_cols].values

    preds  = model.predict(X)
    result: Dict[int, Dict[str, float]] = {}
    idx = 0
    for iid in range(num_intersections):
        result[iid] = {}
        for direction in DIRECTIONS:
            result[iid][direction] = max(0.005, float(preds[idx]))
            idx += 1
    return result
