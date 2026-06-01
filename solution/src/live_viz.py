"""
Real-time animated traffic visualization
=========================================

Produces a side-by-side animated window:
  LEFT  — 2×2 intersection grid showing light states (RED/GREEN),
           per-approach queue bars, and moving vehicle dots.
  RIGHT — three live-updating metric charts:
           top    : average wait time (baseline vs optimized)
           middle : total queue length over time
           bottom : cumulative throughput

Usage (called from main.py demo scenarios):
    from src.live_viz import run_live_demo
    run_live_demo(scenario='am_rush')   # or 'full_day' or 'incident'
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import warnings
import os
warnings.filterwarnings("ignore")

# ── Output directory ────────────────────────────────────────────────
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── colours ─────────────────────────────────────────────────────────
COL_GREEN  = "#2ecc71"
COL_RED    = "#e74c3c"
COL_YELLOW = "#f39c12"
COL_ROAD   = "#555555"
COL_BG     = "#2c3e50"
COL_TEXT   = "#ecf0f1"
COL_BASELINE = "#e74c3c"
COL_OPT      = "#2ecc71"

# Color by origin intersection - makes network flow visible
ORIGIN_COLORS = {
    0: "#9b59b6",  # Purple - vehicles from Int 0 (top-left)
    1: "#ffffff",  # White - vehicles from Int 1 (top-right)
    2: "#2c3e50",  # Black/Dark Slate - vehicles from Int 2 (bottom-left)
    3: "#f39c12",  # Orange - vehicles from Int 3 (bottom-right)
}


def run_live_demo(
    scenario: str = "am_rush",
    duration_s: int = 300,
    speed_factor: int = 5,
    save_gif: bool = False,
) -> None:
    """
    Run a real-time animated simulation and display both the
    grid visualization and live metrics side by side.

    Parameters
    ----------
    scenario     : 'am_rush' | 'full_day' | 'incident'
    duration_s   : real-world seconds to simulate
    speed_factor : animation speed (how many sim-seconds per frame)
    save_gif     : if True, save to fig_live_demo_{scenario}.gif
    """
    # ── lazy imports so the module loads without circular deps ────────
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.simulator  import TrafficSimulator, LightState, TrafficDataGenerator
    from src.optimizer  import FixedTimingController, WebsterOptimizer
    from src.ml_model   import (generate_training_data, train_and_compare,
                                 add_features, predict_rates)

    # ── train the ML model once up front ─────────────────────────────
    print(f"  [live_viz] Training model for '{scenario}' demo...")
    df        = generate_training_data(days=30, num_intersections=4)
    df_fe     = add_features(df)
    trained, scalers, metrics_df = train_and_compare(df_fe)
    best_name = metrics_df.iloc[0]["Model"]
    model     = trained[best_name]
    scaler    = scalers[best_name]
    print(f"  [live_viz] Best model: {best_name} — starting animation")

    # ── scenario settings ─────────────────────────────────────────────
    scenario_cfg = {
        "am_rush":  {"hour": 8,  "dow": 0, "label": "Monday 8 AM — AM Rush"},
        "full_day": {"hour": 6,  "dow": 0, "label": "Monday Full Day Cycle (6am → evening)"},
        "incident": {"hour": 8,  "dow": 0, "label": "Monday 8 AM + Incident at t=150s"},
    }
    cfg   = scenario_cfg.get(scenario, scenario_cfg["am_rush"])
    hour0 = cfg["hour"]
    dow   = cfg["dow"]
    label = cfg["label"]

    # ── two simulators: baseline (fixed) and optimized (Webster) ─────
    sim_base = TrafficSimulator(num_intersections=4, seed=42)
    sim_opt  = TrafficSimulator(num_intersections=4, seed=42)
    baseline_ctrl = FixedTimingController(ns_green=30.0, ew_green=30.0)
    optimizer     = WebsterOptimizer()

    iids = [i.id for i in sim_base.intersections]

    # Set initial timings
    base_timings = baseline_ctrl.compute_timings(iids)
    sim_base.update_lights(base_timings)
    rates0  = predict_rates(model, scaler, hour0, dow, num_intersections=4)
    opt_timings = optimizer.compute_timings(iids, predicted_rates=rates0)
    sim_opt.update_lights(opt_timings)

    # ── history buffers for charts ────────────────────────────────────
    times_hist: List[float]  = []
    wait_base:  List[float]  = []
    wait_opt:   List[float]  = []
    queue_base: List[float]  = []
    queue_opt:  List[float]  = []
    thru_base:  List[float]  = []
    thru_opt:   List[float]  = []

    # ── figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9), facecolor=COL_BG)
    fig.suptitle(
        f"Smart Traffic Light Optimisation — Live Demo\n{label}",
        fontsize=13, fontweight="bold", color=COL_TEXT, y=0.99,
    )
    gs = GridSpec(3, 3, figure=fig,
                  left=0.04, right=0.98, top=0.92, bottom=0.06,
                  wspace=0.35, hspace=0.5)

    # Left two columns: grid panels (baseline top, optimized bottom)
    ax_base_grid = fig.add_subplot(gs[:, :2])
    ax_base_grid.set_facecolor(COL_BG)

    # Right column: three metric charts
    ax_wait  = fig.add_subplot(gs[0, 2])
    ax_queue = fig.add_subplot(gs[1, 2])
    ax_thru  = fig.add_subplot(gs[2, 2])

    for ax in (ax_wait, ax_queue, ax_thru):
        ax.set_facecolor("#1a252f")
        ax.tick_params(colors=COL_TEXT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#4a5568")

    # ── helper: draw one intersection grid ───────────────────────────
    def draw_grid(ax, sim: TrafficSimulator,
                  title: str, elapsed: float) -> None:
        ax.clear()
        ax.set_facecolor(COL_BG)
        ax.set_xlim(-0.8, 2.8)
        ax.set_ylim(-0.8, 2.8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color=COL_TEXT, fontsize=10, pad=6)

        # Draw road grid lines
        for x in [0, 1]:
            ax.axvline(x, color=COL_ROAD, lw=6, alpha=0.5, zorder=1)
        for y in [0, 1]:
            ax.axhline(y, color=COL_ROAD, lw=6, alpha=0.5, zorder=1)

        for inter in sim.intersections:
            cx, cy = inter.position
            ql     = inter.queue_lengths()

            # Intersection circle
            circle = plt.Circle((cx, cy), 0.12, color="#333", zorder=3)
            ax.add_patch(circle)
            ax.text(cx, cy, str(inter.id), ha="center", va="center",
                    color=COL_TEXT, fontsize=8, fontweight="bold", zorder=4)

            # Light state indicators per direction
            offsets = {
                "north": (cx,       cy + 0.22),
                "south": (cx,       cy - 0.22),
                "east":  (cx + 0.22, cy),
                "west":  (cx - 0.22, cy),
            }
            lstate_col = {
                LightState.GREEN:  COL_GREEN,
                LightState.RED:    COL_RED,
                LightState.YELLOW: COL_YELLOW,
            }
            for direction, (lx, ly) in offsets.items():
                col  = lstate_col[inter.light_states[direction]]
                dot  = plt.Circle((lx, ly), 0.055, color=col, zorder=5)
                ax.add_patch(dot)

            # Queue bars (length proportional to queue size, capped for display)
            bar_offsets = {
                "north": (cx - 0.18, cy + 0.35, 0.10, None),
                "south": (cx - 0.18, cy - 0.60, 0.10, None),
                "east":  (cx + 0.35, cy - 0.10, None, 0.08),
                "west":  (cx - 0.65, cy - 0.10, None, 0.08),
            }
            for direction, (bx, by, bw, bh) in bar_offsets.items():
                n = min(ql[direction], 20)  # cap for display
                col = COL_RED if inter.light_states[direction] == LightState.RED else COL_GREEN
                if bw is not None:
                    bar_height = n * 0.015 + 0.005
                    rect = plt.Rectangle((bx, by), bw, bar_height,
                                         color=col, alpha=0.7, zorder=2)
                else:
                    bar_width = n * 0.015 + 0.005
                    rect = plt.Rectangle((bx, by), bar_width, bh,
                                         color=col, alpha=0.7, zorder=2)
                ax.add_patch(rect)

                # Queue count label
                ax.text(bx + (bw or 0.04) / 2,
                        by + (bh or 0.04) / 2,
                        str(ql[direction]),
                        ha="center", va="center",
                        fontsize=6, color=COL_TEXT, zorder=6)

        # Incident marker
        for iid, dirs in sim._incidents.items():
            for d in dirs:
                inter = sim.intersections[iid]
                cx, cy = inter.position
                ax.text(cx, cy + 0.45, "⚠ INCIDENT", ha="center",
                        color=COL_YELLOW, fontsize=7, fontweight="bold",
                        zorder=7)

        # Draw traveling vehicles as colored dots on roads between intersections
        for v in getattr(sim, 'traveling_vehicles', []):
            route = getattr(v, 'route', None)
            current_route_idx = getattr(v, 'current_route_idx', 0)
            if route is None or current_route_idx < 1:
                continue
            # Calculate progress along the road
            travel_time = getattr(sim, 'travel_time', 2.0)
            travel_start_time = getattr(v, 'travel_start_time', 0.0)
            progress = (sim.time - travel_start_time) / travel_time
            progress = min(max(progress, 0.0), 1.0)

            # Get start and end intersection positions
            prev_int = sim.intersections[route[current_route_idx - 1]]
            curr_int = sim.intersections[route[current_route_idx]]
            start_pos = prev_int.position
            end_pos = curr_int.position

            # Interpolate position
            vx = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
            vy = start_pos[1] + (end_pos[1] - start_pos[1]) * progress

            # Color by origin intersection
            origin_int = getattr(v, 'origin_intersection', -1)
            origin_int = origin_int if origin_int >= 0 else 0
            color = ORIGIN_COLORS.get(origin_int, "#888888")

            dot = plt.Circle((vx, vy), 0.04, color=color, zorder=8, alpha=0.9,
                             edgecolor='white', linewidth=1.5)
            ax.add_patch(dot)

        # Draw queued vehicles with origin colors (show a few dots per queue)
        for inter in sim.intersections:
            cx, cy = inter.position
            queue_dot_offsets = {
                "north": [(cx + 0.08, cy - 0.35 - i * 0.06) for i in range(5)],
                "south": [(cx + 0.08, cy + 0.35 + i * 0.06) for i in range(5)],
                "east":  [(cx + 0.35 + i * 0.06, cy + 0.08) for i in range(5)],
                "west":  [(cx - 0.35 - i * 0.06, cy + 0.08) for i in range(5)],
            }
            for direction, queue in inter.queues.items():
                offsets = queue_dot_offsets[direction]
                for i, v in enumerate(queue[:5]):  # Show up to 5 vehicles per queue
                    if i >= len(offsets):
                        break
                    origin_int = getattr(v, 'origin_intersection', -1)
                    origin_int = origin_int if origin_int >= 0 else inter.id
                    color = ORIGIN_COLORS.get(origin_int, "#888888")
                    dx, dy = offsets[i]
                    dot = plt.Circle((dx, dy), 0.025, color=color, zorder=8, alpha=0.85)
                    ax.add_patch(dot)

        # Stats box
        m = sim.get_metrics()
        in_transit = len(getattr(sim, 'traveling_vehicles', []))
        ax.text(0.02, 0.02,
                f"t={elapsed:.0f}s  wait={m['avg_wait_time']:.1f}s  "
                f"depart={m['total_departed']}  in-transit={in_transit}",
                transform=ax.transAxes, fontsize=7.5,
                color=COL_TEXT, va="bottom")

        # Per-intersection origin breakdown (shows vehicles from other intersections)
        for inter in sim.intersections:
            cx, cy = inter.position
            # Count vehicles at this intersection by origin
            origin_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for queue in inter.queues.values():
                for v in queue:
                    origin = getattr(v, 'origin_intersection', -1)
                    origin = origin if origin >= 0 else inter.id
                    if origin in origin_counts:
                        origin_counts[origin] += 1

            # Display counts as colored numbers below intersection
            count_str_parts = []
            for orig_id, count in origin_counts.items():
                if count > 0:
                    count_str_parts.append(f"{count}")
                else:
                    count_str_parts.append("-")

            # Show as a small table below the intersection
            y_offset = -0.55
            ax.text(cx, cy + y_offset, "From:", ha="center", fontsize=5,
                    color=COL_TEXT, zorder=10)
            for i, (orig_id, count) in enumerate(origin_counts.items()):
                dot_x = cx - 0.15 + i * 0.10
                dot = plt.Circle((dot_x, cy + y_offset - 0.08), 0.025,
                                 color=ORIGIN_COLORS[orig_id], zorder=10)
                ax.add_patch(dot)
                ax.text(dot_x, cy + y_offset - 0.16, str(count), ha="center",
                        fontsize=5, color=COL_TEXT, zorder=10)

        # Origin color legend
        legend_y = 0.95
        ax.text(0.98, legend_y, "Origin:", transform=ax.transAxes, fontsize=7,
                color=COL_TEXT, ha="right", va="top")
        for i, (int_id, color) in enumerate(ORIGIN_COLORS.items()):
            dot = plt.Circle((2.55, 1.6 - i * 0.18), 0.05, color=color, zorder=10)
            ax.add_patch(dot)
            ax.text(2.65, 1.6 - i * 0.18, f"Int {int_id}", fontsize=6,
                    color=COL_TEXT, va="center")

    # ── helper: update metric charts ─────────────────────────────────
    def update_charts() -> None:
        if not times_hist:
            return
        t = times_hist

        ax_wait.clear()
        ax_wait.set_facecolor("#1a252f")
        ax_wait.plot(t, wait_base,  COL_BASELINE, lw=1.5,
                     label="Fixed (baseline)")
        ax_wait.plot(t, wait_opt,   COL_OPT,      lw=1.5,
                     label="Webster (optimized)")
        ax_wait.set_ylabel("Avg wait (s)", color=COL_TEXT, fontsize=8)
        ax_wait.set_title("Average Wait Time", color=COL_TEXT, fontsize=9)
        ax_wait.tick_params(colors=COL_TEXT, labelsize=7)
        ax_wait.legend(fontsize=7, facecolor="#1a252f",
                       labelcolor=COL_TEXT, loc="upper left")

        ax_queue.clear()
        ax_queue.set_facecolor("#1a252f")
        ax_queue.plot(t, queue_base, COL_BASELINE, lw=1.5)
        ax_queue.plot(t, queue_opt,  COL_OPT,      lw=1.5)
        ax_queue.set_ylabel("Total queue", color=COL_TEXT, fontsize=8)
        ax_queue.set_title("Total Queue Length", color=COL_TEXT, fontsize=9)
        ax_queue.tick_params(colors=COL_TEXT, labelsize=7)

        ax_thru.clear()
        ax_thru.set_facecolor("#1a252f")
        ax_thru.plot(t, thru_base, COL_BASELINE, lw=1.5)
        ax_thru.plot(t, thru_opt,  COL_OPT,      lw=1.5)
        ax_thru.set_ylabel("Throughput (veh/s)", color=COL_TEXT, fontsize=8)
        ax_thru.set_title("Throughput Rate", color=COL_TEXT, fontsize=9)
        ax_thru.set_xlabel("Simulation time (s)", color=COL_TEXT, fontsize=8)
        ax_thru.tick_params(colors=COL_TEXT, labelsize=7)

    # ── animation update function ─────────────────────────────────────
    sim_step = [0]   # mutable counter

    def update(frame):
        step = sim_step[0]
        elapsed = step * DT * speed_factor

        if elapsed > duration_s:
            return

        # Run speed_factor sim-seconds per animation frame
        for _ in range(speed_factor):
            current_hour = int((hour0 * 3600 + step) % 86400 / 3600)

            rates = predict_rates(model, scaler, current_hour, dow,
                                  num_intersections=4)

            # Re-optimise every 60 sim-seconds
            if step % 60 == 0:
                ql_base = sim_base.get_queue_lengths()
                ql_opt  = sim_opt.get_queue_lengths()
                sim_base.update_lights(baseline_ctrl.compute_timings(iids))
                opt_t = optimizer.compute_timings(iids,
                            predicted_rates=rates, queue_lengths=ql_opt)
                sim_opt.update_lights(opt_t)

            # Trigger incident at t=150s for incident scenario
            if scenario == "incident" and step == 150:
                sim_base.trigger_incident(0, "north", capacity_factor=0.15)
                sim_opt.trigger_incident(0, "north", capacity_factor=0.15)
                print("  [live_viz] ⚠ Incident triggered: north approach "
                      "at intersection 0 blocked 85%")

            # Clear incident at t=240s
            if scenario == "incident" and step == 240:
                sim_base.clear_incident(0, "north")
                sim_opt.clear_incident(0, "north")
                print("  [live_viz] ✓ Incident cleared: road reopened")

            sim_base.step(rates)
            sim_opt.step(rates)
            step += 1

        sim_step[0] = step

        # Record metrics
        m_b = sim_base.get_metrics()
        m_o = sim_opt.get_metrics()
        times_hist.append(elapsed)
        wait_base.append(m_b["avg_wait_time"])
        wait_opt.append(m_o["avg_wait_time"])
        queue_base.append(m_b["total_queue"])
        queue_opt.append(m_o["total_queue"])
        thru_base.append(m_b["throughput"])
        thru_opt.append(m_o["throughput"])

        # Draw
        draw_grid(ax_base_grid,
                  sim_opt,   # show the optimized sim on the grid
                  f"Intersection Grid  |  Webster Optimized\n"
                  f"Baseline wait: {m_b['avg_wait_time']:.1f}s  →  "
                  f"Optimized: {m_o['avg_wait_time']:.1f}s  "
                  f"(+{max(0,(m_b['avg_wait_time']-m_o['avg_wait_time'])/max(m_b['avg_wait_time'],0.01)*100):.1f}%)",
                  elapsed)
        update_charts()

    # ── import DT from simulator ──────────────────────────────────────
    from src.simulator import DT

    total_frames = duration_s // speed_factor + 1
    interval_ms  = 80   # ~12 fps

    anim = animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=interval_ms,
        repeat=False,
        blit=False,
    )

    if save_gif:
        fname = f"{FIGURES_DIR}/fig_live_demo_{scenario}.gif"
        print(f"  [live_viz] Saving animation to {fname} ...")
        writer = animation.PillowWriter(fps=12)
        anim.save(fname, writer=writer)
        print(f"  [live_viz] Saved {fname}")
    else:
        plt.show()

    plt.close(fig)

    # Final improvement summary
    final_base = wait_base[-1] if wait_base else 0
    final_opt  = wait_opt[-1]  if wait_opt  else 0
    if final_base > 0:
        improvement = (final_base - final_opt) / final_base * 100
        print(f"  [live_viz] {scenario}: baseline={final_base:.1f}s  "
              f"optimized={final_opt:.1f}s  improvement={improvement:.1f}%")
