"""
Visualization helpers — SUPERSEDED / LEGACY FILE
=================================================

STATUS: This module is no longer called by main.py.
        It is kept for historical reference only.

All visualisation is now handled in two places:
  main.py             — plot_all_results()  produces fig1–fig4
  generate_diagrams.py — produces fig_architecture.png + fig_mae_explained.png
  model_validation.py  — produces all fig_shap_*, fig_residual_analysis,
                          fig_learning_curves, fig_gini_lorenz, fig_roc_auc

DO NOT import this file in new code.  If you need to modify a chart,
find the relevant function in the files listed above instead.

Original functions preserved below for audit / reference:
  plot_comparison()        — time-series + bar summary of baseline vs optimised
  plot_ml_performance()    — predicted vs actual scatter, hourly pattern, feature importances
  plot_timing_comparison() — green time allocation per intersection
  plot_network_snapshot()  — grid diagram of intersection states
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List


# ------------------------------------------------------------------ #
#  Figure 1: Performance comparison over time                          #
# ------------------------------------------------------------------ #

def plot_comparison(
    baseline: Dict,
    optimized: Dict,
    scenario_label: str = "Rush Hour (8 AM, Monday)",
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Traffic Light Optimization — {scenario_label}\n"
        "Fixed Timing (Baseline) vs. Webster Optimized (ML + Queue Feedback)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    b_time = baseline["time"]
    o_time = optimized["time"]

    _line(axes[0, 0], b_time, baseline["avg_wait_time"],
          o_time, optimized["avg_wait_time"],
          "Average Wait Time per Vehicle (s)", "Time (s)", "Wait Time (s)")

    _line(axes[0, 1], b_time, baseline["total_queue"],
          o_time, optimized["total_queue"],
          "Total Queue Length (all intersections)", "Time (s)", "Vehicles in Queue")

    _line(axes[1, 0], b_time, baseline["throughput_rate"],
          o_time, optimized["throughput_rate"],
          "Throughput Rate (vehicles/s)", "Time (s)", "Vehicles / s")

    _bar_summary(axes[1, 1], baseline, optimized)

    plt.tight_layout()
    return fig


def _line(ax, t_b, y_b, t_o, y_o, title, xlabel, ylabel):
    ax.plot(t_b, y_b, color="crimson", linewidth=1.8, label="Fixed Timing", alpha=0.85)
    ax.plot(t_o, y_o, color="seagreen", linewidth=1.8, label="Optimized", alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def _bar_summary(ax, baseline, optimized):
    b_wait = baseline["avg_wait_time"][-1]
    o_wait = optimized["avg_wait_time"][-1]
    b_thru = baseline["throughput_rate"][-1]
    o_thru = optimized["throughput_rate"][-1]
    b_dep = baseline["total_departed"][-1]
    o_dep = optimized["total_departed"][-1]

    # Normalise to baseline = 1 for relative comparison
    categories = ["Avg Wait\nTime", "Throughput\nRate", "Total\nDeparted"]
    b_vals = [b_wait, b_thru * 100, b_dep / 50]
    o_vals = [o_wait, o_thru * 100, o_dep / 50]

    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w / 2, b_vals, w, label="Fixed Timing", color="crimson", alpha=0.75)
    ax.bar(x + w / 2, o_vals, w, label="Optimized", color="seagreen", alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title("Final Performance (scaled)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")

    if b_wait > 0:
        pct = (b_wait - o_wait) / b_wait * 100
        colour = "darkgreen" if pct > 0 else "firebrick"
        ax.text(
            0.5, 0.97,
            f"Wait time reduction: {pct:+.1f}%",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, fontweight="bold", color=colour,
        )


# ------------------------------------------------------------------ #
#  Figure 2: ML model diagnostics                                      #
# ------------------------------------------------------------------ #

def plot_ml_performance(
    actuals: np.ndarray,
    predictions: np.ndarray,
    hourly_actuals: np.ndarray,   # shape (24,)
    hourly_preds: np.ndarray,     # shape (24,)
    importances: Dict[str, float],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Traffic Demand Prediction — RandomForest Model", fontsize=13, fontweight="bold")

    # Scatter: actual vs predicted
    ax = axes[0]
    ax.scatter(actuals, predictions, s=4, alpha=0.35, color="steelblue")
    lim = max(actuals.max(), predictions.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5)
    ax.set_xlabel("Actual (veh/s)")
    ax.set_ylabel("Predicted (veh/s)")
    ax.set_title("Actual vs. Predicted")
    ax.grid(True, alpha=0.25)

    # Hourly pattern
    ax = axes[1]
    ax.plot(range(24), hourly_actuals, "b-o", markersize=4, linewidth=1.8, label="Actual")
    ax.plot(range(24), hourly_preds, "r--s", markersize=4, linewidth=1.8, label="Predicted")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Arrival Rate (veh/s)")
    ax.set_title("Hourly Traffic Pattern")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Feature importances (horizontal bar)
    ax = axes[2]
    feats = list(importances.keys())
    vals = list(importances.values())
    order = np.argsort(vals)
    ax.barh([feats[i] for i in order], [vals[i] for i in order], color="steelblue", alpha=0.8)
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.25, axis="x")

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ #
#  Figure 3: Green time allocation comparison                          #
# ------------------------------------------------------------------ #

def plot_timing_comparison(
    baseline_timings: Dict[int, Dict[str, float]],
    optimized_timings: Dict[int, Dict[str, float]],
    intersection_ids: List[int],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Green Time Allocation — Baseline vs. Optimized (Rush Hour)",
        fontsize=13, fontweight="bold",
    )

    labels = [f"Int {iid}" for iid in intersection_ids]
    x = np.arange(len(intersection_ids))
    w = 0.35

    for ax, timings, title, cns, cew in [
        (axes[0], baseline_timings, "Fixed Timing (Baseline)", "steelblue", "coral"),
        (axes[1], optimized_timings, "Webster Optimized", "seagreen", "darkorange"),
    ]:
        ns_g = [timings[iid]["ns_green"] for iid in intersection_ids]
        ew_g = [timings[iid]["ew_green"] for iid in intersection_ids]
        total = [ns + ew for ns, ew in zip(ns_g, ew_g)]

        ax.bar(x - w / 2, ns_g, w, label="NS Green (s)", color=cns, alpha=0.82)
        ax.bar(x + w / 2, ew_g, w, label="EW Green (s)", color=cew, alpha=0.82)

        for xi, (ng, eg, tot) in enumerate(zip(ns_g, ew_g, total)):
            ax.text(xi - w / 2, ng + 0.5, f"{ng:.0f}s", ha="center", va="bottom", fontsize=8)
            ax.text(xi + w / 2, eg + 0.5, f"{eg:.0f}s", ha="center", va="bottom", fontsize=8)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Intersection")
        ax.set_ylabel("Green Duration (s)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 75)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ #
#  Figure 4: Network snapshot (queue lengths at a point in time)       #
# ------------------------------------------------------------------ #

def plot_network_snapshot(
    queue_data: Dict[int, Dict[str, int]],
    phase_data: Dict[int, str],   # iid → 'NS_GREEN' or 'EW_GREEN'
    sim_time: float,
    grid_side: int = 2,
) -> plt.Figure:
    n = len(queue_data)
    fig, axes = plt.subplots(grid_side, grid_side, figsize=(4 * grid_side, 4 * grid_side))
    fig.suptitle(f"Network Snapshot — t = {sim_time:.0f}s", fontsize=12, fontweight="bold")

    axes_flat = axes.flatten() if n > 1 else [axes]

    for iid in range(n):
        ax = axes_flat[iid]
        queues = queue_data.get(iid, {d: 0 for d in ["north", "south", "east", "west"]})
        phase = phase_data.get(iid, "NS_GREEN")
        _draw_intersection(ax, iid, queues, phase)

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    return fig


def _draw_intersection(ax, iid, queues, phase):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Intersection {iid}", fontsize=9, pad=3)

    # Roads
    for x, y, w, h in [(-0.4, -3, 0.8, 6), (-3, -0.4, 6, 0.8)]:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                                              fc="#cccccc", ec="none", zorder=1))
    # Centre box
    ax.add_patch(mpatches.FancyBboxPatch((-0.4, -0.4), 0.8, 0.8,
                                          boxstyle="square,pad=0",
                                          fc="#999999", ec="none", zorder=2))

    ns_green = phase == "NS_GREEN"

    # direction → (light_x, light_y, queue_start, queue_end, is_vertical)
    config = {
        "north": ( 0.55,  1.8,  0.5,  2.5, True),
        "south": (-0.55, -1.8, -0.5, -2.5, True),
        "east":  ( 1.8,  0.55,  0.5,  2.5, False),
        "west":  (-1.8, -0.55, -0.5, -2.5, False),
    }

    for direction, (lx, ly, qs, qe, vertical) in config.items():
        is_ns = direction in ("north", "south")
        green = (ns_green and is_ns) or (not ns_green and not is_ns)
        colour = "#22cc44" if green else "#dd2222"

        ax.add_patch(plt.Circle((lx, ly), 0.18, color=colour, zorder=3))

        q = queues.get(direction, 0)
        if q > 0:
            bar_len = min(q / 15.0, 1.0) * 1.5
            bar_colour = "#ff9900" if q < 8 else "#dd2222"
            if vertical:
                sign = 1 if direction == "north" else -1
                ax.barh(
                    [sign * (0.55 + bar_len / 2)],
                    [0.3],
                    height=bar_len,
                    left=[-0.15],
                    color=bar_colour, alpha=0.7, zorder=2,
                )
            else:
                sign = 1 if direction == "east" else -1
                ax.bar(
                    [sign * (0.55 + bar_len / 2)],
                    [0.3],
                    width=bar_len,
                    bottom=[-0.15],
                    color=bar_colour, alpha=0.7, zorder=2,
                )

        # Queue count label
        label_x = lx * 1.35
        label_y = ly * 1.35
        ax.text(label_x, label_y, str(q), ha="center", va="center",
                fontsize=8, fontweight="bold", color="black", zorder=4)
