"""
Standalone diagram generator.
Run: python generate_diagrams.py
Produces: figures/fig_architecture.png  (system architecture)
          figures/fig_mae_explained.png  (MAE visual explanation)
"""

import os
import numpy as np

# ── Output directory ────────────────────────────────────────────────
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── colour palette ─────────────────────────────────────────────────────
C_DATA    = "#1565C0"   # dark blue  — data / rules layer
C_FEAT    = "#00838F"   # teal       — feature engineering
C_MODEL   = "#2E7D32"   # green      — ML models
C_ENH     = "#EF6C00"   # orange     — enhancement / reduction
C_VAL     = "#6A1B9A"   # purple     — validation
C_OPT     = "#AD1457"   # pink/red   — optimizer
C_SIM     = "#4E342E"   # brown      — simulator
C_OUT     = "#37474F"   # slate      — outputs / results
C_ARROW   = "#546E7A"   # grey-blue  — arrows
C_BG      = "#FAFAFA"


def _box(ax, x, y, w, h, label, sublabel="", color="#333", fontsize=9,
         radius=0.3, alpha=0.92):
    """Draw a rounded rectangle with a label."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=1.4, edgecolor="white",
        facecolor=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4,
            wrap=True)
    if sublabel:
        ax.text(x, y - 0.28, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color="white", alpha=0.88, zorder=4, style="italic")


def _arrow(ax, x1, y1, x2, y2, label="", color=C_ARROW):
    """Draw a labelled arrow."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.5,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.18, label, ha="center", va="bottom",
                fontsize=7, color=color, style="italic", zorder=5)


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Full System Architecture Diagram
# ══════════════════════════════════════════════════════════════════════

def architecture_diagram():
    fig = plt.figure(figsize=(24, 13), facecolor=C_BG)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 13)
    ax.axis("off")

    # ── title ─────────────────────────────────────────────────────────
    ax.text(12, 12.4, "Smart Traffic Light Optimisation — System Architecture",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#1a1a2e")
    ax.text(12, 11.95,
            "Read left → right.  Each box = one module/function.  "
            "Arrows show data flow.  Colours show layer type.",
            ha="center", va="center", fontsize=9, color="#555")

    # ── LAYER LABELS (background bands) ───────────────────────────────
    bands = [
        (0.2,  3.1,  "① DATA",        C_DATA),
        (3.2,  5.9,  "② FEATURES",    C_FEAT),
        (6.0,  9.3,  "③ ML MODELS",   C_MODEL),
        (9.4,  13.2, "④ ENHANCE/REDUCE", C_ENH),
        (13.3, 16.6, "⑤ VALIDATION",  C_VAL),
        (16.7, 19.6, "⑥ OPTIMIZER",   C_OPT),
        (19.7, 22.3, "⑦ SIMULATION",  C_SIM),
        (22.4, 23.8, "⑧ RESULTS",     C_OUT),
    ]
    for x0, x1, lbl, col in bands:
        rect = plt.Rectangle((x0, 0.5), x1 - x0, 11.2,
                              facecolor=col, alpha=0.06,
                              edgecolor=col, linewidth=1.2, zorder=0)
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, 11.45, lbl,
                ha="center", va="bottom", fontsize=8,
                color=col, fontweight="bold")

    # ─────────────────────────────────────────────────────────────────
    #  LAYER 1 — DATA  (x ≈ 0.2 – 3.1)
    # ─────────────────────────────────────────────────────────────────
    _box(ax, 1.65, 9.7, 2.6, 1.0,
         "generate_training_data()", "ml_model.py", C_DATA)
    _box(ax, 1.65, 7.8, 2.6, 1.5,
         "Rules A–F", "A: 2-peak weekday\nB: NS/EW asymmetry\nC: weekend\nD: weather\nE: intersection\nF: noise",
         C_DATA, fontsize=7.5)
    _box(ax, 1.65, 5.4, 2.6, 1.1,
         "TrafficSimulator", "simulator.py\nPoisson arrivals\n0.5 veh/s discharge",
         C_SIM, fontsize=7.5)
    _box(ax, 1.65, 3.2, 2.6, 0.9,
         "Raw DataFrame", "11,520 rows\n(day×hour×int×dir)", C_DATA, fontsize=8)

    _arrow(ax, 1.65, 9.2, 1.65, 8.6)       # gen → rules
    _arrow(ax, 1.65, 7.05, 1.65, 3.65)     # rules → raw df

    # ─────────────────────────────────────────────────────────────────
    #  LAYER 2 — FEATURES  (x ≈ 3.2 – 5.9)
    # ─────────────────────────────────────────────────────────────────
    _box(ax, 4.55, 9.5, 2.5, 1.0,
         "add_features()", "ml_model.py", C_FEAT)
    _box(ax, 4.55, 7.5, 2.5, 2.5,
         "X — 12 Features",
         "G1: hour_sin/cos, dow_sin/cos\n"
         "G2: morning_rush, evening_rush\n"
         "    daytime, weekend\n"
         "G3: weather\n"
         "G4: intersection_id\n"
         "    direction_id, ns_direction",
         C_FEAT, fontsize=7.5)
    _box(ax, 4.55, 5.2, 2.5, 0.9,
         "y — Target variable",
         "arrival_rate  (veh/s)\nrange 0.005 – 1.2",
         C_FEAT, fontsize=8)
    _box(ax, 4.55, 3.5, 2.5, 1.1,
         "Correlation Analysis",
         "section2b\nPearson r (feat→target)\nCross-corr heatmap",
         C_FEAT, fontsize=7.5)

    _arrow(ax, 2.95, 3.2, 3.3, 3.2, "raw df")
    _arrow(ax, 3.3, 9.5, 4.3, 9.5)
    _arrow(ax, 4.55, 9.0, 4.55, 8.75)
    _arrow(ax, 4.55, 6.25, 4.55, 5.65)     # X → y
    _arrow(ax, 4.55, 4.75, 4.55, 4.05)     # features → corr

    # ─────────────────────────────────────────────────────────────────
    #  LAYER 3 — ML MODELS  (x ≈ 6.0 – 9.3)
    # ─────────────────────────────────────────────────────────────────
    _box(ax, 7.65, 10.4, 2.8, 0.85,
         "train_and_compare()", "ml_model.py · train days 0-24, test 25-29",
         C_MODEL, fontsize=7.5)

    models = [
        (7.65, 9.1, "LightGBM ★ BEST", "MAE 0.00545  R²0.985"),
        (7.65, 7.9, "Random Forest",    "MAE 0.00547  R²0.985"),
        (7.65, 6.7, "XGBoost",          "MAE 0.00574  R²0.986"),
        (7.65, 5.5, "Gradient Boosting","MAE 0.00625  R²0.984"),
        (7.65, 4.3, "Ridge Regression", "MAE ~0.018   R²~0.95"),
    ]
    for mx, my, ml, ms in models:
        col = "#1B5E20" if "BEST" in ml else C_MODEL
        _box(ax, mx, my, 2.8, 0.75, ml, ms, col, fontsize=7.5)

    _box(ax, 7.65, 3.0, 2.8, 0.85,
         "H2O AutoML (optional)",
         "GBM, DRF, XGB, GLM, DL, Ensemble",
         "#004D40", fontsize=7.5)

    _arrow(ax, 5.8, 7.5, 6.25, 9.1,  "X (12 feat)")
    _arrow(ax, 5.8, 5.2, 6.25, 5.0,  "y (arr_rate)")
    _arrow(ax, 7.65, 9.98, 7.65, 9.5)
    for _, my, _, _ in models[:-1]:
        _arrow(ax, 7.65, my - 0.38, 7.65, my - 0.62)
    _arrow(ax, 7.65, 3.92, 7.65, 3.45)

    # best-model callout
    ax.annotate("",
        xy=(9.1, 9.1), xytext=(9.7, 9.4),
        arrowprops=dict(arrowstyle="-|>", color=C_ENH, lw=1.3), zorder=2)

    # ─────────────────────────────────────────────────────────────────
    #  LAYER 4 — ENHANCEMENT / REDUCTION  (x ≈ 9.4 – 13.2)
    # ─────────────────────────────────────────────────────────────────
    _box(ax, 11.3, 10.4, 3.5, 0.85,
         "model_enhancement.py", "section 3c + 3c-2", C_ENH, fontsize=8)

    enh_items = [
        (11.3, 9.1,  "Feature Engineering",
                     "Base(12) / Interactions(16) / Rich(20)"),
        (11.3, 7.85, "Over-sampling",
                     "Rush rows ×2 (result: neutral −0.1%)"),
        (11.3, 6.6,  "Under-weighting",
                     "Night weight ×0.4 (result: hurt −1.1%)"),
        (11.3, 5.35, "Hyperparameter Tuning",
                     "RandomizedSearchCV  20 iter  3-fold"),
        (11.3, 4.1,  "Feature Reduction",
                     "Corr prune / Variance / Top-K / RFE"),
    ]
    for ex, ey, el, es in enh_items:
        _box(ax, ex, ey, 3.5, 0.75, el, es, C_ENH, fontsize=7.5)

    _box(ax, 11.3, 2.7, 3.5, 0.85,
         "RFE top-6 ← best result",
         "6 features,  MAE 0.00539 (−1.1% vs full)\nhour_sin/cos, dow_sin, weather, int_id, dir_id",
         "#BF360C", fontsize=7.5)

    _arrow(ax, 9.06, 9.1, 9.55, 9.8)
    for _, ey, _, _ in enh_items[:-1]:
        _arrow(ax, 11.3, ey - 0.38, 11.3, ey - 0.62)
    _arrow(ax, 11.3, 3.72, 11.3, 3.13)

    # ─────────────────────────────────────────────────────────────────
    #  LAYER 5 — VALIDATION  (x ≈ 13.3 – 16.6)
    # ─────────────────────────────────────────────────────────────────
    _box(ax, 14.95, 10.4, 3.0, 0.85,
         "model_validation.py", "section 3d  (7 tests)", C_VAL, fontsize=8)

    val_items = [
        (14.95, 9.2,  "1. Cross-Validation",   "5-fold  Val MAE 0.00563±0.00047"),
        (14.95, 8.05, "2. Gini MDI vs Perm",    "Bias caught: int_id#1→#7"),
        (14.95, 6.9,  "3. SHAP Analysis",       "is_morning_rush φ=0.034  #1"),
        (14.95, 5.75, "4. Residual Analysis",   "Bias=−0.00013  no pattern"),
        (14.95, 4.6,  "5. Learning Curves",     "No overfitting, gap=0.00162"),
        (14.95, 3.45, "6. Gini Coefficient",    "Gini=0.506  Ranking AUC=0.753"),
        (14.95, 2.3,  "7. Binary ROC-AUC",      "AUC=0.9946  Excellent"),
    ]
    for vx, vy, vl, vs in val_items:
        _box(ax, vx, vy, 3.0, 0.75, vl, vs, C_VAL, fontsize=7.5)

    _arrow(ax, 9.06, 9.1, 13.45, 9.2,  "best model")
    _arrow(ax, 14.95, 9.98, 14.95, 9.58)
    for _, vy, _, _ in val_items[:-1]:
        _arrow(ax, 14.95, vy - 0.38, 14.95, vy - 0.62)

    # ─────────────────────────────────────────────────────────────────
    #  LAYER 6 — OPTIMIZER  (x ≈ 16.7 – 19.6)
    # ─────────────────────────────────────────────────────────────────
    _box(ax, 18.15, 9.8, 2.7, 0.85,
         "optimizer.py", "section 4 + 5", C_OPT, fontsize=8)
    _box(ax, 18.15, 8.5, 2.7, 1.1,
         "WebsterOptimizer",
         "C* = (1.5L+5)/(1−Y)\nGreen ∝ flow ratios\nML + queue blend α=0.7",
         C_OPT, fontsize=7.5)
    _box(ax, 18.15, 6.95, 2.7, 0.85,
         "FixedTimingController",
         "Baseline: 30s NS / 30s EW\n(always, ignores demand)",
         "#880E4F", fontsize=7.5)

    _arrow(ax, 16.45, 9.2, 16.8, 9.2, "predicted\nrates")
    _arrow(ax, 18.15, 9.38, 18.15, 9.05)
    _arrow(ax, 18.15, 7.95, 18.15, 7.38)

    # ─────────────────────────────────────────────────────────────────
    #  LAYER 7 — SIMULATION  (x ≈ 19.7 – 22.3)
    # ─────────────────────────────────────────────────────────────────
    _box(ax, 21.0, 9.5, 2.4, 1.0,
         "run_simulation()", "main.py · 3600s loop\n60s re-optimise cadence",
         C_SIM, fontsize=7.5)
    _box(ax, 21.0, 7.8, 2.4, 1.3,
         "TrafficSimulator.step()",
         "Poisson arrivals\nSaturation discharge\nWait accumulation\nPhase transitions",
         C_SIM, fontsize=7.5)
    _box(ax, 21.0, 5.9, 2.4, 1.1,
         "demand_model  ≠\ncontrol_model",
         "SAME traffic for all runs\nOnly timing strategy varies\n(fair A/B comparison)",
         "#3E2723", fontsize=7.0)

    _arrow(ax, 19.5, 8.5, 19.8, 9.0,  "timings")
    _arrow(ax, 21.0, 9.0, 21.0, 8.45)

    # ─────────────────────────────────────────────────────────────────
    #  LAYER 8 — RESULTS  (x ≈ 22.4 – 23.8)
    # ─────────────────────────────────────────────────────────────────
    _box(ax, 23.1, 9.5, 1.9, 1.2,
         "+40.1%",
         "wait reduction\nAM rush peak",
         C_OUT, fontsize=9)
    _box(ax, 23.1, 7.8, 1.9, 1.3,
         "6 / 6",
         "scenarios\nmeet ≥20%\ntarget",
         C_OUT, fontsize=9)
    _box(ax, 23.1, 5.9, 1.9, 1.0,
         "12 figures",
         "PNG outputs\nreport.html",
         C_OUT, fontsize=8)

    _arrow(ax, 22.2, 9.5, 22.15, 9.5)
    _arrow(ax, 22.2, 7.8, 22.15, 7.8)

    # ─────────────────────────────────────────────────────────────────
    #  LEGEND
    # ─────────────────────────────────────────────────────────────────
    legend_items = [
        (C_DATA, "Data / Rules"),
        (C_FEAT, "Feature Engineering"),
        (C_MODEL,"ML Models"),
        (C_ENH,  "Enhancement / Reduction"),
        (C_VAL,  "Validation"),
        (C_OPT,  "Optimizer"),
        (C_SIM,  "Simulator"),
        (C_OUT,  "Outputs"),
    ]
    for i, (col, lbl) in enumerate(legend_items):
        px = 0.5 + i * 3.0
        ax.add_patch(plt.Rectangle((px, 0.55), 0.35, 0.35,
                                   facecolor=col, edgecolor="none", zorder=5))
        ax.text(px + 0.45, 0.72, lbl, va="center", fontsize=7.5,
                color="#333", zorder=5)

    plt.savefig(f"{FIGURES_DIR}/fig_architecture.png", dpi=150, bbox_inches="tight",
                facecolor=C_BG)
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR}/fig_architecture.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 2 — MAE Explained Visually
# ══════════════════════════════════════════════════════════════════════

def mae_explanation_diagram():
    rng = np.random.default_rng(42)

    # Simulate 20 realistic test points
    hours   = np.array([0, 2, 5, 7, 8, 8, 9, 10, 12, 14,
                         16, 17, 17, 18, 19, 20, 21, 22, 23, 23])
    actuals = np.array([0.012, 0.010, 0.015, 0.18, 0.52, 0.65, 0.48, 0.09, 0.07, 0.06,
                         0.14, 0.38, 0.55, 0.48, 0.22, 0.10, 0.06, 0.04, 0.02, 0.013])
    # LightGBM is very accurate — small errors
    noise   = rng.normal(0, 0.008, len(actuals))
    preds   = np.clip(actuals + noise, 0.005, None)
    errors  = np.abs(preds - actuals)
    mae_val = errors.mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "MAE — Mean Absolute Error  :  What It Means for This Project",
        fontsize=14, fontweight="bold", y=1.01
    )

    # ── Panel 1: Actual vs predicted scatter ─────────────────────────
    ax = axes[0]
    ax.scatter(actuals, preds, s=60, color="#1565C0", alpha=0.8, zorder=3,
               label="Each dot = one test row")
    # Perfect-prediction line
    lim = max(actuals.max(), preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction (error=0)")
    # Show a few error lines
    for i in [4, 10, 12]:
        ax.plot([actuals[i], actuals[i]], [actuals[i], preds[i]],
                color="orange", lw=2.0, zorder=4)
    ax.plot([], [], color="orange", lw=2.0, label="Prediction error (|ŷ − y|)")
    ax.set_xlabel("Actual arrival_rate  (veh/s)", fontsize=10)
    ax.set_ylabel("Predicted arrival_rate  (veh/s)", fontsize=10)
    ax.set_title("Actual vs Predicted\n(closer to red line = better)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.text(0.04, 0.92,
            f"MAE = {mae_val:.4f} veh/s\n≈ {mae_val*3600:.0f} veh/hr avg error",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="#e3f2fd", alpha=0.9))

    # ── Panel 2: Error bars by hour ───────────────────────────────────
    ax2 = axes[1]
    colours = ["#E53935" if (7 <= h <= 9 or 16 <= h <= 19) else "#1565C0"
               for h in hours]
    ax2.bar(range(len(hours)), errors, color=colours, alpha=0.82, width=0.7)
    ax2.axhline(mae_val, color="black", lw=2, ls="--",
                label=f"MAE = {mae_val:.4f} veh/s\n(average of all bars)")
    ax2.set_xticks(range(len(hours)))
    ax2.set_xticklabels([f"{h:02d}h" for h in hours], rotation=45, fontsize=7)
    ax2.set_ylabel("|Predicted − Actual|  (veh/s)", fontsize=9)
    ax2.set_title("Absolute error per test row\n"
                  "(red = rush hour rows — errors matter most here)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis="y")
    red_patch  = mpatches.Patch(color="#E53935", alpha=0.82, label="Rush hour")
    blue_patch = mpatches.Patch(color="#1565C0", alpha=0.82, label="Off-peak")
    ax2.legend(handles=[red_patch, blue_patch,
                         plt.Line2D([0],[0], color="black", lw=2, ls="--",
                                    label=f"MAE = {mae_val:.4f}")],
               fontsize=8, loc="upper left")

    # ── Panel 3: What MAE = 0.00545 means in plain English ───────────
    ax3 = axes[2]
    ax3.axis("off")
    lines = [
        ("WHAT IS MAE?", 0.94, 12, "bold", "#1565C0"),
        ("MAE = Mean Absolute Error", 0.87, 10, "bold", "#1a1a2e"),
        ("= average of  | predicted − actual |", 0.81, 10, "normal", "#333"),
        ("= average SIZE of the model's mistake,", 0.75, 9.5, "normal", "#333"),
        ("  regardless of direction.", 0.70, 9.5, "normal", "#333"),
        ("", 0.64, 9, "normal", "#333"),
        ("IN THIS PROJECT:", 0.63, 10, "bold", "#1565C0"),
        ("LightGBM MAE = 0.00545 veh/s", 0.57, 10, "bold", "#1B5E20"),
        ("= 0.00545 × 3600 = 19.6 veh/hr", 0.51, 9.5, "normal", "#333"),
        ("  average error in arrival rate", 0.46, 9.5, "normal", "#333"),
        ("", 0.40, 9, "normal", "#333"),
        ("PUTTING IT IN CONTEXT:", 0.39, 10, "bold", "#1565C0"),
        ("Rush hour rate:  0.2–0.7 veh/s", 0.33, 9.5, "normal", "#333"),
        ("  Error ÷ true rate  ≈  1–3%  ✓", 0.28, 9.5, "bold", "#2E7D32"),
        ("Night rate:  0.01–0.03 veh/s", 0.22, 9.5, "normal", "#333"),
        ("  Error ÷ true rate  ≈  18–50%", 0.17, 9.5, "normal", "#888"),
        ("  (but wait times near zero anyway)", 0.12, 9, "normal", "#888"),
        ("", 0.07, 9, "normal", "#333"),
        ("R²  = 0.9852  →  model explains", 0.04, 9.5, "bold", "#1565C0"),
        ("  98.5% of all variance in demand", -0.01, 9.5, "bold", "#1565C0"),
    ]
    for txt, ypos, fsize, fweight, fcolor in lines:
        ax3.text(0.05, ypos, txt, transform=ax3.transAxes,
                 fontsize=fsize, fontweight=fweight, color=fcolor, va="top")

    # Traffic light icon strip (decorative)
    for yi, col in zip([0.90, 0.78, 0.66], ["#E53935", "#FDD835", "#43A047"]):
        ax3.add_patch(plt.Circle((0.90, yi), 0.035, color=col,
                                  transform=ax3.transAxes, zorder=5))

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_mae_explained.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR}/fig_mae_explained.png")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    architecture_diagram()
    mae_explanation_diagram()
    print("Done.")
