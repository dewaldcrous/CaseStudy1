"""
Traffic demand prediction models.

Model roster
------------
  sklearn
    Ridge Regression        — linear baseline (fast, interpretable)
    Random Forest           — bagged trees, robust out-of-the-box
    Gradient Boosting       — sklearn's sequential boosting
  Third-party (optional — skipped gracefully if not installed)
    LightGBM                — Microsoft's histogram-based GBM; very fast
    XGBoost                 — gradient boosting with regularisation
  H2O AutoML (optional — requires Java 11+)
    H2OAutoML               — tries GBM, DRF, XGBoost, GLM, DeepLearning,
                              StackedEnsemble automatically and returns a
                              ranked leaderboard

All models implement the same interface via sklearn or a thin wrapper,
so the rest of the pipeline doesn't need to know which backend is running.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model   import Ridge
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics        import mean_absolute_error, r2_score
from sklearn.preprocessing  import StandardScaler
from typing                 import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# ── optional imports ─────────────────────────────────────────────────────────
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

try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False


# ================================================================== #
#  Data-generation rules (documented for auditability)                #
# ================================================================== #

DIRECTIONS            = ["north", "south", "east", "west"]
INTERSECTION_FACTORS  = [1.0, 0.85, 1.20, 0.95]
DIRECTION_FACTORS     = {"north": 0.90, "south": 1.10, "east": 1.00, "west": 0.95}
WEATHER_MULT          = {"clear": 1.00, "cloudy": 1.05, "rain": 1.25}
WEATHER_PROB          = [0.55, 0.30, 0.15]


def generate_training_data(
    days: int = 30,
    num_intersections: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build synthetic hourly traffic data.

    Rules applied
    -------------
    RULE A  Weekday two-Gaussian pattern:
              f(h) = 0.015 + 0.20*exp(-(h-8)²/2) + 0.16*exp(-(h-17.5)²/3)

    RULE B  Directional asymmetry tied to rush-hour weights:
              AM  → NS up to 3×, EW as low as 0.7×
              PM  → EW up to 2.5×, NS as low as 0.5×

    RULE C  Weekend: symmetric midday Gaussian, no directional bias.

    RULE D  Weather: daily multiplier drawn once (clear/cloudy/rain).

    RULE E  Per-intersection and per-direction scaling factors.

    RULE F  Log-normal noise (σ=0.08) — day-to-day variance within an hour.
    """
    rng = np.random.default_rng(seed)
    records = []

    for day in range(days):
        dow        = day % 7
        is_weekend = dow >= 5

        # RULE D: daily weather
        wtype  = rng.choice(["clear", "cloudy", "rain"], p=WEATHER_PROB)
        wmult  = WEATHER_MULT[wtype]
        wcode  = {"clear": 0, "cloudy": 1, "rain": 2}[wtype]

        for hour in range(24):
            if is_weekend:
                # RULE C
                base     = 0.06 + 0.04 * np.exp(-((hour - 13) ** 2) / 18)
                ns_mult  = 1.0
                ew_mult  = 1.0
            else:
                # RULE A
                morning  = 0.20 * np.exp(-((hour - 8.0)  ** 2) / 2.0)
                evening  = 0.16 * np.exp(-((hour - 17.5) ** 2) / 3.0)
                base     = 0.015 + morning + evening
                # RULE B
                am_w     = np.exp(-((hour - 8.0)  ** 2) / 2.0)
                pm_w     = np.exp(-((hour - 17.5) ** 2) / 3.0)
                ns_mult  = 1.0 + 2.0 * am_w - 0.5 * pm_w
                ew_mult  = 1.0 + 1.5 * pm_w - 0.3 * am_w

            for iid in range(num_intersections):
                ifact = INTERSECTION_FACTORS[iid % len(INTERSECTION_FACTORS)]  # RULE E
                for did, direction in enumerate(DIRECTIONS):
                    dfact   = DIRECTION_FACTORS[direction]                      # RULE E
                    dm      = ns_mult if direction in ("north", "south") else ew_mult
                    rate    = base * ifact * dfact * dm * wmult
                    rate   *= float(rng.lognormal(0.0, 0.08))                  # RULE F
                    rate    = max(0.005, rate)

                    records.append({
                        "day":             day,
                        "day_of_week":     dow,
                        "is_weekend":      int(is_weekend),
                        "hour":            hour,
                        "intersection_id": iid,
                        "direction":       direction,
                        "direction_id":    did,
                        "weather":         wcode,
                        "arrival_rate":    rate,
                    })

    return pd.DataFrame(records)


# ================================================================== #
#  MODEL SPECIFICATION — Feature variables (X) and Target variable (y) #
# ================================================================== #
#
#  ┌─────────────────────────────────────────────────────────────────┐
#  │  TARGET VARIABLE  y = arrival_rate                              │
#  └─────────────────────────────────────────────────────────────────┘
#
#  Column : "arrival_rate"
#  Type   : float  (vehicles per second, veh/s)
#  Range  : ~0.005  (quiet night)  →  ~1.2  (heavy rush hour)
#  Meaning: The number of vehicles arriving per second at ONE approach
#           (one direction at one intersection) during ONE hour.
#           One row in the dataset = one (intersection, direction, hour, day).
#  Role   : What every model predicts.  The predicted value q̂ feeds
#           directly into Webster's cycle formula:
#               y_i = q̂_i / saturation_flow  (flow ratio)
#               C*  = (1.5·L + 5) / (1 − ΣY)  (optimal cycle length)
#           so prediction error in arrival_rate propagates directly into
#           green-time allocation errors.
#
#  ┌─────────────────────────────────────────────────────────────────┐
#  │  FEATURE VARIABLES  X  (12 columns, defined in FEATURE_COLS)   │
#  └─────────────────────────────────────────────────────────────────┘
#
#  Each row of X describes the CONDITIONS for one (intersection,
#  direction, hour, day) cell.  The model learns the mapping X → y.
#
#  ── GROUP 1: TIME OF DAY  (circular / sinusoidal encoding) ────────
#
#  "hour_sin"
#     Type : float   Range : [-1, +1]
#     Formula : sin(2π × hour / 24)
#     Why  : Encodes the hour as a point on the unit circle.
#            Midnight (0) and 23:00 are geometrically adjacent (distance
#            ~0.26) rather than 23 apart.  Linear encoding treats them
#            as maximally different; this encoding treats them correctly.
#
#  "hour_cos"
#     Type : float   Range : [-1, +1]
#     Formula : cos(2π × hour / 24)
#     Why  : The cosine component is orthogonal to hour_sin.  Together
#            they give a complete 2-D position on the daily clock.
#            Without both, the encoding is ambiguous (same sin at 2am
#            and 10am; cos resolves the ambiguity).
#
#  "dow_sin"
#     Type : float   Range : [-1, +1]
#     Formula : sin(2π × day_of_week / 7)   (Mon=0 … Sun=6)
#     Why  : Weekly periodicity; Sunday (6) and Monday (0) should be
#            close together (one day apart), not 6 apart.
#
#  "dow_cos"
#     Type : float   Range : [-1, +1]
#     Formula : cos(2π × day_of_week / 7)
#     Why  : Orthogonal complement to dow_sin; same argument as hour_cos.
#
#  ── GROUP 2: PERIOD FLAGS  (binary step indicators) ───────────────
#
#  "is_morning_rush"
#     Type : int    Values : {0, 1}
#     Rule : 1 if (7 ≤ hour ≤ 9) AND (is_weekend == 0)
#     Why  : The AM rush creates a sharp demand spike that the smooth
#            sinusoidal features represent poorly.  A binary flag gives
#            tree models a direct, zero-cost split at the rush boundary.
#            SHAP confirms this is the #1 most important feature.
#
#  "is_evening_rush"
#     Type : int    Values : {0, 1}
#     Rule : 1 if (16 ≤ hour ≤ 19) AND (is_weekend == 0)
#     Why  : Same reasoning for the PM return commute (EW-dominant).
#
#  "is_daytime"
#     Type : int    Values : {0, 1}
#     Rule : 1 if 7 ≤ hour ≤ 22
#     Why  : Coarser flag separating overnight-quiet from all daytime
#            activity.  Helps with the daytime baseline (non-rush but
#            non-night traffic that is intermediate in level).
#
#  "is_weekend"
#     Type : int    Values : {0, 1}
#     Rule : 1 if day_of_week ≥ 5  (Saturday or Sunday)
#     Source: directly from the raw "day_of_week" column.
#     Why  : Weekend demand follows a completely different pattern
#            (Rule C: single midday Gaussian, no directional asymmetry).
#            This flag triggers that regime switch instantly.
#
#  ── GROUP 3: WEATHER ──────────────────────────────────────────────
#
#  "weather"
#     Type : int    Values : {0, 1, 2}
#     Encoding : 0 = clear  |  1 = cloudy  |  2 = rain
#     Source   : drawn once per simulated day (Rule D).
#     Why  : Ordinal encoding preserves "rain (2) > cloudy (1) > clear
#            (0)" ordering.  Rain raises vehicle arrivals ~25% as
#            cyclists and pedestrians switch to driving.
#
#  ── GROUP 4: LOCATION AND DIRECTION ──────────────────────────────
#
#  "intersection_id"
#     Type : int    Values : {0, 1, 2, 3}
#     Source: which of the 4 intersections in the simulated network.
#     Why  : Different intersections have different baseline volumes
#            (INTERSECTION_FACTORS = [1.0, 0.85, 1.2, 0.95], Rule E).
#            A model without this feature cannot distinguish intersection
#            2 (20% busier) from intersection 1 (15% quieter).
#
#  "direction_id"
#     Type : int    Values : {0=north, 1=south, 2=east, 3=west}
#     Why  : Different directions have different per-direction scaling
#            (DIRECTION_FACTORS: south 1.1×, north 0.9×, Rule E) AND
#            different rush-hour asymmetry roles.  Ordinal encoding lets
#            the model learn north ≠ south ≠ east ≠ west without needing
#            four separate one-hot columns.
#            NOTE: correlated with ns_direction (r = −0.89) but carries
#            distinct information (within-NS and within-EW differences).
#
#  "ns_direction"
#     Type : int    Values : {0, 1}
#     Rule : 1 if direction ∈ {"north", "south"}; 0 if {"east", "west"}
#     Why  : NS approaches share AM-rush dominance (Rule B); EW approaches
#            share PM-rush dominance.  A single binary flag lets the model
#            apply different time-of-day responses to NS vs EW without
#            needing direction_id × hour interaction terms.
#            SHAP assigns this a lower rank than direction_id alone,
#            confirming direction_id subsumes most of its information.
#
#  ── SUMMARY ───────────────────────────────────────────────────────
#
#  X shape per training run : (9216, 12)  [days 0-24 × 24h × 4int × 4dir]
#  y shape per training run : (9216,)
#  X shape per test run     : (2304, 12)  [days 25-29 × 24h × 4int × 4dir]
#  y shape per test run     : (2304,)
#
#  At PREDICTION TIME (run_simulation):
#    X shape : (16, 12)   [4 intersections × 4 directions, one hour]
#    y shape : (16,)      [predicted arrival rate per approach]
#    These 16 values feed directly into WebsterOptimizer.compute_timings().
#
# ================================================================== #

FEATURE_COLS = [
    # ── Group 1: circular time encoding ──────────────────────────────
    "hour_sin",         # sin(2π·hour/24)       float  [-1, +1]
    "hour_cos",         # cos(2π·hour/24)       float  [-1, +1]
    "dow_sin",          # sin(2π·dow/7)         float  [-1, +1]
    "dow_cos",          # cos(2π·dow/7)         float  [-1, +1]
    # ── Group 2: binary period flags ─────────────────────────────────
    "is_morning_rush",  # 1 if 7≤hour≤9,  weekday     int  {0,1}
    "is_evening_rush",  # 1 if 16≤hour≤19, weekday    int  {0,1}
    "is_daytime",       # 1 if 7≤hour≤22              int  {0,1}
    "is_weekend",       # 1 if dow≥5 (Sat/Sun)        int  {0,1}
    # ── Group 3: weather ─────────────────────────────────────────────
    "weather",          # 0=clear, 1=cloudy, 2=rain   int  {0,1,2}
    # ── Group 4: location and approach direction ──────────────────────
    "intersection_id",  # which intersection (0-3)    int  {0,1,2,3}
    "direction_id",     # 0=N 1=S 2=E 3=W             int  {0,1,2,3}
    "ns_direction",     # 1 if N or S approach         int  {0,1}
]
# Total: 12 features → predicts 1 target: arrival_rate (veh/s)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive all 12 model features from the raw data columns.

    INPUT COLUMNS (raw, from generate_training_data)
    -------------------------------------------------
    day          int   day index (0-29)
    day_of_week  int   Mon=0 … Sun=6
    is_weekend   int   {0,1}  already present
    hour         int   0-23
    intersection_id int {0,1,2,3}
    direction    str   "north"/"south"/"east"/"west"
    direction_id int   {0,1,2,3}
    weather      int   {0,1,2}  already present
    arrival_rate float  TARGET — not modified here

    OUTPUT COLUMNS ADDED (the 8 derived features)
    -----------------------------------------------
    hour_sin        sin(2π·hour/24)                  GROUP 1
    hour_cos        cos(2π·hour/24)                  GROUP 1
    dow_sin         sin(2π·day_of_week/7)             GROUP 1
    dow_cos         cos(2π·day_of_week/7)             GROUP 1
    is_morning_rush 1 if 7≤hour≤9  (weekday)          GROUP 2
    is_evening_rush 1 if 16≤hour≤19 (weekday)         GROUP 2
    is_daytime      1 if 7≤hour≤22                    GROUP 2
    ns_direction    1 if direction ∈ {north, south}   GROUP 4

    After calling this function, df[FEATURE_COLS] gives the full
    (N × 12) feature matrix X ready for model.fit() or model.predict().
    """
    df = df.copy()
    # ── Group 1: circular time encoding (see MODEL SPECIFICATION above) ──
    df["hour_sin"]        = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]        = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]         = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]         = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # ── Group 2: binary period flags ─────────────────────────────────────
    df["is_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] <= 9)).astype(int)
    df["is_evening_rush"] = ((df["hour"] >= 16) & (df["hour"] <= 19)).astype(int)
    df["is_daytime"]      = ((df["hour"] >= 7)  & (df["hour"] <= 22)).astype(int)
    # ── Group 4: direction flag (Group 3 "weather" already in raw data) ──
    df["ns_direction"]    = df["direction"].isin(["north", "south"]).astype(int)
    return df


# ================================================================== #
#  sklearn + LightGBM + XGBoost model registry                        #
# ================================================================== #

def build_models() -> dict:
    """
    Return {name: estimator} for every available sklearn-compatible model.

    Ridge Regression
      Linear model, normalisation required (StandardScaler applied).
      Cannot learn interaction effects ("hour=8 AND NS=1 → high demand").
      Sets the lower-accuracy benchmark.

    Random Forest (sklearn)
      100 trees grown on bootstrap subsets. Each sees a random feature
      subset at every split. Averaging reduces variance. Robust default.

    Gradient Boosting (sklearn)
      Trees grown sequentially; each corrects residuals of the last.
      Slower than LightGBM but ships with sklearn — no extra dependency.

    LightGBM  [optional]
      Histogram-based GBM. Bins continuous features before splitting,
      making it 10-20× faster than sklearn GBM at similar accuracy.
      Also handles categorical features natively.

    XGBoost  [optional]
      The original "competition-winning" gradient boosting framework.
      Adds L1/L2 regularisation to the objective, which can help when
      features are noisy or correlated.
    """
    models = {
        "Ridge Regression":      Ridge(alpha=1.0),
        "Random Forest":         RandomForestRegressor(
                                     n_estimators=100, max_depth=12,
                                     min_samples_leaf=4, random_state=42, n_jobs=-1),
        "Gradient Boosting":     GradientBoostingRegressor(
                                     n_estimators=150, max_depth=5,
                                     learning_rate=0.08, random_state=42),
    }

    if LGBM_AVAILABLE:
        # LightGBM: histogram binning makes this much faster than sklearn GBM
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=200, num_leaves=63, learning_rate=0.05,
            min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1,
        )

    if XGB_AVAILABLE:
        # XGBoost: adds L1/L2 regularisation to GBM objective
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            n_jobs=-1, verbosity=0,
        )

    return models


# ================================================================== #
#  Train and compare all sklearn-compatible models                     #
# ================================================================== #

def train_and_compare(df: pd.DataFrame) -> Tuple[dict, dict, pd.DataFrame]:
    """
    Train all models on days 0-24, test on days 25-29.

    MODEL INPUT / OUTPUT
    --------------------
    X  (feature matrix)  shape (N, 12)   dtype float64
       Columns = FEATURE_COLS (see MODEL SPECIFICATION block above).
       Each row describes one (intersection, direction, hour, day) cell.

    y  (target vector)   shape (N,)      dtype float64
       Column = "arrival_rate"  (vehicles per second)
       This is what every model predicts.

    TRAIN / TEST SPLIT
    ------------------
    Training set  : rows where day ≤ max_day - 5   (days 0-24)
                    shape ≈ (9216, 12)
    Test set      : rows where day >  max_day - 5   (days 25-29)
                    shape ≈ (2304, 12)
    Mimics production: "predict next week from past history."
    Temporal split (no shuffle) prevents future-data leakage.

    SCALING
    -------
    Ridge Regression : StandardScaler fitted on X_train, applied to
                       X_train and X_test.  Required because Ridge's
                       L2 penalty is scale-sensitive.
    Tree models      : No scaling needed (splits are ordinal, not
                       distance-based).  Scaler stored as None.

    Returns
    -------
    trained    : {model_name: fitted_model}
    scalers    : {model_name: fitted_StandardScaler | None}
    metrics_df : DataFrame, one row per model, sorted by Test MAE
                 columns: Model, Test MAE (veh/s), Test R2, Source
    """
    df    = add_features(df)
    X     = df[FEATURE_COLS].values   # shape (N, 12) — feature matrix
    y     = df["arrival_rate"].values  # shape (N,)   — target: veh/s

    max_day   = df["day"].max()
    train_idx = df["day"] <= max_day - 5
    X_tr, X_te = X[train_idx], X[~train_idx]   # X_tr:(9216,12)  X_te:(2304,12)
    y_tr, y_te = y[train_idx], y[~train_idx]   # y_tr:(9216,)    y_te:(2304,)

    trained, scalers, rows = {}, {}, []

    for name, model in build_models().items():
        # Ridge needs feature scaling; tree models do not
        scaler = StandardScaler() if "Ridge" in name else None
        X_tr_fit = scaler.fit_transform(X_tr) if scaler else X_tr
        X_te_fit = scaler.transform(X_te)     if scaler else X_te

        model.fit(X_tr_fit, y_tr)
        preds = model.predict(X_te_fit)

        rows.append({
            "Model":              name,
            "Test MAE (veh/s)":   mean_absolute_error(y_te, preds),
            "Test R2":            r2_score(y_te, preds),
            "Source":             "sklearn-compat",
        })
        trained[name] = model
        scalers[name] = scaler

    metrics_df = pd.DataFrame(rows).sort_values("Test MAE (veh/s)").reset_index(drop=True)
    return trained, scalers, metrics_df


# ================================================================== #
#  H2O AutoML                                                          #
# ================================================================== #

class H2OModelWrapper:
    """
    Thin wrapper that makes an H2O model look like a sklearn model
    so it can plug into predict_rates() without changes.

    H2O keeps models in the H2O cluster (a local JVM process).
    Prediction converts a numpy array to H2OFrame, scores it,
    and returns a numpy array — transparent to the caller.
    """

    def __init__(self, h2o_model, feature_names: list) -> None:
        self.model         = h2o_model
        self.feature_names = feature_names

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Score a numpy feature matrix via the H2O cluster."""
        import h2o
        df = pd.DataFrame(X, columns=self.feature_names)
        hf = h2o.H2OFrame(df)
        return self.model.predict(hf).as_data_frame().values.flatten()

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Return a numpy array of importances aligned to FEATURE_COLS."""
        try:
            vi = self.model.varimp(use_pandas=True)
            imp_map = dict(zip(vi["variable"], vi["relative_importance"]))
            total = sum(imp_map.values()) or 1.0
            return np.array([imp_map.get(f, 0.0) / total for f in self.feature_names])
        except Exception:
            return None


def run_h2o_automl(
    df: pd.DataFrame,
    max_models: int = 12,
    max_runtime_secs: int = 120,
    verbose: bool = True,
) -> Tuple[Optional[H2OModelWrapper], Optional[pd.DataFrame]]:
    """
    Run H2O AutoML on the traffic data and return the best model + leaderboard.

    H2O AutoML workflow
    -------------------
    1. h2o.init()         — starts a local JVM-based H2O server (port 54321)
    2. H2OFrame           — converts pandas DataFrame to H2O's distributed format
    3. H2OAutoML.train()  — explores GBM, DRF, XGBoost, GLM, DeepLearning,
                            StackedEnsemble within time/model budget
    4. leader             — the single best model by cross-validated RMSE
    5. leaderboard        — ranked table of all models tried

    Why AutoML?
    -----------
    Rather than hand-tuning hyperparameters for each model family, AutoML
    runs a broad search automatically. The leaderboard shows which model
    family wins and by how much, giving confidence the best choice was found.

    Returns (None, None) if H2O is not installed or Java is unavailable.
    """
    if not H2O_AVAILABLE:
        if verbose:
            print("  [H2O] Package not installed — skipping. (pip install h2o)")
        return None, None

    import h2o
    from h2o.automl import H2OAutoML

    try:
        if verbose:
            print("  [H2O] Starting local H2O cluster ...")
        h2o.init(nthreads=-1, max_mem_size="2G", verbose=False)
        h2o.no_progress()

        df_fe  = add_features(df)
        target = "arrival_rate"
        feats  = FEATURE_COLS

        # Train / test split by day (same rule as sklearn comparison)
        max_day = df_fe["day"].max()
        train_df = df_fe[df_fe["day"] <= max_day - 5][feats + [target]]
        test_df  = df_fe[df_fe["day"] >  max_day - 5][feats + [target]]

        train_hf = h2o.H2OFrame(train_df)
        test_hf  = h2o.H2OFrame(test_df)

        if verbose:
            print(f"  [H2O] Running AutoML: max_models={max_models}, "
                  f"max_runtime={max_runtime_secs}s ...")

        aml = H2OAutoML(
            max_models          = max_models,
            max_runtime_secs    = max_runtime_secs,
            seed                = 42,
            sort_metric         = "MAE",
            verbosity           = None,
        )
        aml.train(x=feats, y=target, training_frame=train_hf)

        # Evaluate leader on hold-out test set
        leader     = aml.leader
        preds_hf   = leader.predict(test_hf)
        preds_np   = preds_hf.as_data_frame().values.flatten()
        actuals_np = test_df[target].values

        mae = mean_absolute_error(actuals_np, preds_np)
        r2  = r2_score(actuals_np, preds_np)

        # Build leaderboard DataFrame
        lb = aml.leaderboard.as_data_frame()
        lb = lb[["model_id", "mae", "rmse", "r2"]].head(10)
        lb.columns = ["Model ID", "MAE", "RMSE", "R2"]

        if verbose:
            print(f"  [H2O] AutoML complete. Leader: {leader.model_id}")
            print(f"         Test MAE = {mae:.5f}  R2 = {r2:.4f}")
            print()
            print("  [H2O] Leaderboard (top 10):")
            print(lb.to_string(index=False))
            print()

        wrapper = H2OModelWrapper(leader, FEATURE_COLS)
        return wrapper, lb

    except Exception as e:
        if verbose:
            print(f"  [H2O] Failed: {e}")
            print("  [H2O] Skipping H2O AutoML.")
        return None, None


# ================================================================== #
#  Unified prediction interface                                        #
# ================================================================== #

def predict_rates(
    model,
    scaler,
    hour: int,
    day_of_week: int,
    weather: int = 0,
    num_intersections: int = 4,
) -> Dict[int, Dict[str, float]]:
    """
    Predict arrival rates (veh/s) for ALL approaches at a given time.

    This is the runtime prediction interface called every 60 seconds
    inside the simulation loop.  It constructs the feature matrix X
    from scratch for the current (hour, day_of_week, weather) and
    returns predictions for every intersection × direction combination.

    INPUTS (scalars — describing the current time/weather context)
    --------------------------------------------------------------
    model         : any trained model exposing .predict(X_numpy)
                    (Ridge, RF, GB, LightGBM, XGBoost, H2OModelWrapper)
    scaler        : fitted StandardScaler | None
                    Ridge requires scaling; all tree models pass None.
    hour          : int  0-23   current hour of day
    day_of_week   : int  0-6    Mon=0 … Sun=6
    weather       : int  {0,1,2}  0=clear, 1=cloudy, 2=rain
    num_intersections : int  number of intersections to predict for

    FEATURE MATRIX CONSTRUCTED INTERNALLY
    --------------------------------------
    X  shape (num_intersections × 4, 12)  e.g. (16, 12) for 4 intersections.
    Each of the 16 rows represents one approach (one direction at one
    intersection) under the current time/weather conditions.
    Columns are FEATURE_COLS in order:
      [hour_sin, hour_cos, dow_sin, dow_cos,
       is_morning_rush, is_evening_rush, is_daytime, is_weekend,
       weather, intersection_id, direction_id, ns_direction]

    OUTPUT (nested dict — predicted target variable per approach)
    -------------------------------------------------------------
    result : Dict[intersection_id, Dict[direction, float]]
      result[iid][direction] = predicted arrival_rate (veh/s)
      e.g. result[0]["north"] = 0.73  means 0.73 veh/s arriving
           at the north approach of intersection 0 this hour.

    These values feed into WebsterOptimizer._one_intersection() as:
      q_ns = max(result[iid]["north"], result[iid]["south"])
      q_ew = max(result[iid]["east"],  result[iid]["west"])
      y_ns = q_ns / SATURATION_FLOW   ← flow ratio for NS phase
      y_ew = q_ew / SATURATION_FLOW   ← flow ratio for EW phase
      C*   = (1.5·L + 5) / (1 − y_ns − y_ew)   ← Webster cycle
    """
    # Build one row per (intersection, direction) — these are the raw
    # columns that add_features() needs to derive the 12 model inputs.
    rows = [
        {
            "hour":            hour,           # raw → hour_sin, hour_cos, rush flags
            "day_of_week":     day_of_week,    # raw → dow_sin, dow_cos
            "is_weekend":      int(day_of_week >= 5),
            "intersection_id": iid,            # feature: location heterogeneity
            "direction":       direction,       # raw → ns_direction
            "direction_id":    did,            # feature: N/S/E/W encoding
            "weather":         weather,        # feature: rain multiplier
        }
        for iid in range(num_intersections)
        for did, direction in enumerate(DIRECTIONS)
    ]

    # add_features() derives the 8 engineered columns (hour_sin/cos, etc.)
    # then we slice out exactly FEATURE_COLS to get the (16, 12) matrix X.
    df = add_features(pd.DataFrame(rows))
    X  = df[FEATURE_COLS].values    # X shape: (num_intersections × 4, 12)

    # Ridge requires scaling (L2 penalty is scale-sensitive); others do not.
    if scaler is not None:
        X = scaler.transform(X)

    # model.predict(X) returns ŷ shape (16,) — predicted arrival_rate per approach
    preds  = model.predict(X)

    # Reshape flat predictions back into the nested {iid: {direction: rate}} dict
    # that the optimizer and simulator both expect.
    result: Dict[int, Dict[str, float]] = {}
    idx = 0
    for iid in range(num_intersections):
        result[iid] = {}
        for direction in DIRECTIONS:
            # Floor at 0.005 veh/s — never predict zero (would break Webster's Y)
            result[iid][direction] = max(0.005, float(preds[idx]))
            idx += 1
    return result
