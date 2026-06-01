"""
Smart Traffic Light Optimisation — Standalone Demo
===================================================

A smooth, standalone visualization that matches the demo.html style.
Can be packaged as an executable with PyInstaller.

Run:
    python run_demo.py              # Default: AM Rush
    python run_demo.py pm_rush      # PM Rush scenario
    python run_demo.py full_day     # Full day cycle

Package as executable:
    pip install pyinstaller
    pyinstaller --onefile --windowed run_demo.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for smoother rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# ── Colours (matching demo.html) ─────────────────────────────────
C_BG = "#0e1117"
C_ROAD = "#2a3441"
C_GREEN = "#00ff00"
C_RED = "#ff0000"
C_TEXT = "#ffffff"
C_BASELINE_LINE = "#e74c3c"
C_OPT_LINE = "#2ecc71"

# Origin colors - vehicles colored by entry intersection
ORIGIN_COLORS = {
    0: "#9b59b6",  # Purple - Int 0 (top-left)
    1: "#ffffff",  # White - Int 1 (top-right)
    2: "#2c3e50",  # Black/Dark Slate - Int 2 (bottom-left)
    3: "#f39c12",  # Orange - Int 3 (bottom-right)
}

ORIGIN_NAMES = {0: "Int 0", 1: "Int 1", 2: "Int 2", 3: "Int 3"}


def main():
    # ── Parse scenario from command line ────────────────────────────
    scenario = sys.argv[1] if len(sys.argv) > 1 else "am_rush"

    scenarios = {
        "am_rush":   {"hour": 8,  "dow": 0, "label": "Monday 8 AM — AM Rush"},
        "pm_rush":   {"hour": 17, "dow": 0, "label": "Monday 5 PM — PM Rush"},
        "full_day":  {"hour": 6,  "dow": 0, "label": "Full Day Cycle"},
        "midday":    {"hour": 12, "dow": 0, "label": "Monday 12 PM — Midday"},
    }

    cfg = scenarios.get(scenario, scenarios["am_rush"])
    print(f"\n{'='*60}")
    print(f"  Smart Traffic Light Optimisation — Standalone Demo")
    print(f"  Scenario: {cfg['label']}")
    print(f"{'='*60}\n")

    # ── Load ML model ───────────────────────────────────────────────
    print("  Loading ML model...")
    from src.ml_model import generate_training_data, add_features, train_and_compare, predict_rates
    from src.simulator import TrafficSimulator, LightState
    from src.optimizer import FixedTimingController, WebsterOptimizer

    df = generate_training_data(days=30, num_intersections=4)
    df_fe = add_features(df)
    trained, scalers, metrics_df = train_and_compare(df_fe)
    best_name = metrics_df.iloc[0]["Model"]
    model = trained[best_name]
    scaler = scalers[best_name]
    print(f"  Best model: {best_name} (R² = {metrics_df.iloc[0]['Test R2']:.4f})")

    # ── Initialize simulators ───────────────────────────────────────
    sim_base = TrafficSimulator(num_intersections=4, seed=42)
    sim_opt = TrafficSimulator(num_intersections=4, seed=42)
    baseline_ctrl = FixedTimingController(ns_green=30.0, ew_green=30.0)
    optimizer = WebsterOptimizer()

    iids = [i.id for i in sim_base.intersections]
    hour = cfg["hour"]
    dow = cfg["dow"]

    # Initial timings
    base_timings = baseline_ctrl.compute_timings(iids)
    sim_base.update_lights(base_timings)
    rates = predict_rates(model, scaler, hour, dow, num_intersections=4)
    opt_timings = optimizer.compute_timings(iids, predicted_rates=rates)
    sim_opt.update_lights(opt_timings)

    # ── History for charts ──────────────────────────────────────────
    history = {
        "time": [], "wait_base": [], "wait_opt": [],
        "queue_base": [], "queue_opt": [], "thru_base": [], "thru_opt": []
    }

    # ── Create figure ───────────────────────────────────────────────
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    fig.canvas.manager.set_window_title("Smart Traffic Light Optimisation")

    gs = GridSpec(3, 4, figure=fig, left=0.03, right=0.97, top=0.92, bottom=0.05,
                  wspace=0.3, hspace=0.4)

    # Grid panels (2 columns each)
    ax_base = fig.add_subplot(gs[:, 0:2])
    ax_opt = fig.add_subplot(gs[:, 2:4])

    # Set up axes
    for ax in [ax_base, ax_opt]:
        ax.set_facecolor(C_BG)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')

    # Title
    fig.suptitle(f"Smart Traffic Light Optimisation — {cfg['label']}",
                 fontsize=14, fontweight='bold', color=C_TEXT, y=0.98)

    # ── State ───────────────────────────────────────────────────────
    state = {"step": 0, "hour": hour}

    # ── Drawing functions ───────────────────────────────────────────
    def draw_intersection_grid(ax, sim, title, show_timings=False):
        ax.clear()
        ax.set_facecolor(C_BG)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title with metrics
        mb = sim.get_metrics()
        title_text = f"{title}\nQueue: {mb['total_queue']} | Wait: {mb['avg_wait_time']:.1f}s"
        ax.set_title(title_text, fontsize=11, color=C_TEXT, pad=10)

        # Intersection positions (2x2 grid)
        positions = {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}

        # Draw roads
        road_width = 0.12
        for x in [0, 1]:
            ax.add_patch(patches.Rectangle((x - road_width/2, -0.25), road_width, 1.5,
                                           facecolor=C_ROAD, edgecolor='none'))
        for y in [0, 1]:
            ax.add_patch(patches.Rectangle((-0.25, y - road_width/2), 1.5, road_width,
                                           facecolor=C_ROAD, edgecolor='none'))

        # Draw each intersection
        for inter in sim.intersections:
            x, y = positions[inter.id]

            # Intersection box
            box_size = 0.08
            ax.add_patch(patches.Rectangle((x - box_size, y - box_size),
                                           box_size*2, box_size*2,
                                           facecolor='#1a1f2e', edgecolor='#4fc3f7', linewidth=1.5))

            # Traffic lights (small circles at each direction)
            light_offset = 0.06
            light_positions = {
                'north': (x, y + light_offset),
                'south': (x, y - light_offset),
                'east': (x + light_offset, y),
                'west': (x - light_offset, y),
            }

            for direction, (lx, ly) in light_positions.items():
                is_green = inter.light_states[direction] == LightState.GREEN
                color = C_GREEN if is_green else C_RED
                ax.add_patch(patches.Circle((lx, ly), 0.015, facecolor=color, edgecolor='white', linewidth=0.5))

            # Draw queued vehicles as car shapes
            queue_positions = {
                'north': (0, 1),   # Cars stack upward
                'south': (0, -1),  # Cars stack downward
                'east': (1, 0),    # Cars stack right
                'west': (-1, 0),   # Cars stack left
            }

            for direction, queue in inter.queues.items():
                dx, dy = queue_positions[direction]
                for i, v in enumerate(queue[:8]):  # Show up to 8 cars
                    offset = (i + 1) * 0.04
                    vx = x + dx * (0.1 + offset)
                    vy = y + dy * (0.1 + offset)

                    # Get origin color
                    origin = v.route[0] if hasattr(v, 'route') and v.route else inter.id
                    color = ORIGIN_COLORS.get(origin, '#888888')

                    # Draw car shape (small rectangle)
                    car_w, car_h = 0.025, 0.015
                    if direction in ['north', 'south']:
                        car_w, car_h = car_h, car_w
                    ax.add_patch(patches.FancyBboxPatch(
                        (vx - car_w/2, vy - car_h/2), car_w, car_h,
                        boxstyle="round,pad=0.003", facecolor=color, edgecolor='white', linewidth=0.3))

                # Show overflow count
                overflow = len(queue) - 8
                if overflow > 0:
                    ox = x + dx * 0.45
                    oy = y + dy * 0.45
                    ax.text(ox, oy, f"+{overflow}", fontsize=7, color='#ffff00',
                            ha='center', va='center', fontweight='bold')

            # Show intersection ID
            ax.text(x, y, str(inter.id), fontsize=8, color='#4fc3f7',
                    ha='center', va='center', fontweight='bold')

        # Draw traveling vehicles
        for v in getattr(sim, 'traveling_vehicles', []):
            if not hasattr(v, 'route') or v.route_idx < 1:
                continue

            progress = min((sim.time - v.travel_start_time) / 2.0, 1.0)
            prev_id = v.route[v.route_idx - 1]
            curr_id = v.route[v.route_idx]

            px, py = positions[prev_id]
            cx, cy = positions[curr_id]

            vx = px + (cx - px) * progress
            vy = py + (cy - py) * progress

            origin = v.route[0] if v.route else 0
            color = ORIGIN_COLORS.get(origin, '#888888')

            ax.add_patch(patches.Circle((vx, vy), 0.02, facecolor=color, edgecolor='white', linewidth=0.5))

        # Legend
        legend_y = -0.22
        for i, (oid, color) in enumerate(ORIGIN_COLORS.items()):
            lx = 0.1 + i * 0.28
            ax.add_patch(patches.FancyBboxPatch((lx, legend_y), 0.04, 0.02,
                                                boxstyle="round,pad=0.002",
                                                facecolor=color, edgecolor='white', linewidth=0.5))
            ax.text(lx + 0.055, legend_y + 0.01, ORIGIN_NAMES[oid], fontsize=7, color='#aaa', va='center')

    # ── Animation update function ───────────────────────────────────
    def update(frame):
        nonlocal hour

        # Advance simulation
        for _ in range(5):  # 5 steps per frame for smooth animation
            state["step"] += 1

            # Update hour every 60 steps (1 minute sim time)
            if state["step"] % 60 == 0:
                hour = (cfg["hour"] + state["step"] // 3600) % 24
                state["hour"] = hour

            # Get arrival rates from ML
            rates = predict_rates(model, scaler, hour, dow, num_intersections=4)

            # Step both simulations
            sim_base.step(arrival_rates=rates)
            sim_opt.step(arrival_rates=rates)

            # Update timings every 30 steps
            if state["step"] % 30 == 0:
                base_timings = baseline_ctrl.compute_timings(iids)
                sim_base.update_lights(base_timings)

                queue_lengths = {i.id: {d: len(q) for d, q in i.queues.items()}
                                 for i in sim_opt.intersections}
                opt_timings = optimizer.compute_timings(iids, predicted_rates=rates,
                                                        queue_lengths=queue_lengths)
                sim_opt.update_lights(opt_timings)

        # Record history
        mb = sim_base.get_metrics()
        mo = sim_opt.get_metrics()
        history["time"].append(state["step"])
        history["wait_base"].append(mb["avg_wait_time"])
        history["wait_opt"].append(mo["avg_wait_time"])
        history["queue_base"].append(mb["total_queue"])
        history["queue_opt"].append(mo["total_queue"])

        # Draw grids
        draw_intersection_grid(ax_base, sim_base, "BASELINE (Fixed 30/30)")
        draw_intersection_grid(ax_opt, sim_opt, "OPTIMIZED (Webster + ML)")

        # Calculate improvement
        if mb["avg_wait_time"] > 0:
            imp = (mb["avg_wait_time"] - mo["avg_wait_time"]) / mb["avg_wait_time"] * 100
        else:
            imp = 0

        # Add improvement badge to optimized panel
        imp_color = '#2ecc71' if imp >= 20 else '#f39c12' if imp >= 0 else '#e74c3c'
        ax_opt.text(0.5, 1.22, f"+{imp:.0f}% improvement", fontsize=10, color=imp_color,
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1f2e', edgecolor=imp_color))

        # Time display
        sim_hour = (cfg["hour"] + state["step"] // 3600) % 24
        sim_min = (state["step"] % 3600) // 60
        time_str = f"{sim_hour:02d}:{sim_min:02d}"
        ax_base.text(0.5, 1.22, f"Time: {time_str} | Step: {state['step']}", fontsize=9, color='#aaa',
                     ha='center', va='center')

        return []

    # ── Run animation ───────────────────────────────────────────────
    print("  Starting animation... (close window to exit)")
    ani = FuncAnimation(fig, update, frames=None, interval=50, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
