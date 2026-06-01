"""
Smart Traffic Light Optimisation — Standalone Demo
===================================================

Controls:
  1/2/3    - Switch scenario (AM Rush / PM Rush / Midday)
  I        - Toggle incident on Int 0 North
  +/=      - Speed up (max 10x)
  -        - Slow down (min 0.5x)
  R        - Reset simulation
  Space    - Pause/Resume
  Q/Esc    - Quit
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from collections import deque

# Colors
C_BG = "#0e1117"
C_ROAD = "#2a3441"
C_GREEN = "#00ff00"
C_RED = "#ff0000"
C_BASELINE = "#e74c3c"
C_OPT = "#2ecc71"

ORIGIN_COLORS = {
    0: "#9b59b6",  # Purple
    1: "#ffffff",  # White
    2: "#2c3e50",  # Dark
    3: "#f39c12",  # Orange
}

SCENARIOS = {
    "am_rush": {"hour": 8, "dow": 0, "label": "AM Rush (8 AM)", "key": "1"},
    "pm_rush": {"hour": 17, "dow": 0, "label": "PM Rush (5 PM)", "key": "2"},
    "midday": {"hour": 12, "dow": 0, "label": "Midday (12 PM)", "key": "3"},
}

SPEED_LEVELS = [0.5, 1, 2, 3, 5, 10]


def draw_grid(ax, sim, title, title_color, LightState, incidents=None):
    """Draw a 2x2 intersection grid."""
    ax.clear()
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    m = sim.get_metrics()
    ax.set_title(f"{title}\nQueue: {m['total_queue']} | Wait: {m['avg_wait_time']:.1f}s",
                 color=title_color, fontsize=10, fontweight='bold', pad=5)

    positions = {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}

    # Roads
    rw = 0.15
    for x in [0, 1]:
        ax.add_patch(patches.Rectangle((x - rw/2, -0.4), rw, 1.8, fc=C_ROAD, ec='none'))
    for y in [0, 1]:
        ax.add_patch(patches.Rectangle((-0.4, y - rw/2), 1.8, rw, fc=C_ROAD, ec='none'))

    # Intersections
    for inter in sim.intersections:
        x, y = positions[inter.id]

        # Check for incident
        has_incident = incidents and inter.id in incidents and any(incidents[inter.id].values())
        box_color = '#ff6b6b' if has_incident else '#4fc3f7'

        ax.add_patch(patches.Rectangle((x-0.08, y-0.08), 0.16, 0.16,
                                        fc='#1a1f2e', ec=box_color, lw=2))

        # Incident marker
        if has_incident:
            ax.text(x, y+0.12, "⚠", color='#ff6b6b', fontsize=10, ha='center', va='center')

        # Lights
        for d, (dx, dy) in [('north', (0, 0.06)), ('south', (0, -0.06)),
                            ('east', (0.06, 0)), ('west', (-0.06, 0))]:
            c = C_GREEN if inter.light_states[d] == LightState.GREEN else C_RED
            ax.add_patch(patches.Circle((x+dx, y+dy), 0.02, fc=c, ec='white', lw=0.5))

        # ID
        ax.text(x, y, str(inter.id), color='#4fc3f7', ha='center', va='center',
                fontsize=9, fontweight='bold')

        # Queued vehicles
        offsets = {'north': (0, 1), 'south': (0, -1), 'east': (1, 0), 'west': (-1, 0)}
        for d, q in inter.queues.items():
            dx, dy = offsets[d]
            for i, v in enumerate(q[:6]):
                off = (i + 1) * 0.05
                vx, vy = x + dx * (0.12 + off), y + dy * (0.12 + off)
                origin = v.route[0] if hasattr(v, 'route') and v.route else inter.id
                col = ORIGIN_COLORS.get(origin, '#888')

                w, h = (0.015, 0.03) if d in ['north', 'south'] else (0.03, 0.015)
                ax.add_patch(patches.FancyBboxPatch((vx-w/2, vy-h/2), w, h,
                             boxstyle="round,pad=0.003", fc=col, ec='white', lw=0.3))

            if len(q) > 6:
                ox, oy = x + dx * 0.5, y + dy * 0.5
                ax.text(ox, oy, f"+{len(q)-6}", color='#ff0', fontsize=7,
                        ha='center', va='center', fontweight='bold')

    # Traveling vehicles
    for v in getattr(sim, 'traveling_vehicles', []):
        if not hasattr(v, 'route') or v.current_route_idx < 1:
            continue
        progress = min((sim.time - v.travel_start_time) / 2.0, 1.0)
        p1, p2 = positions[v.route[v.current_route_idx - 1]], positions[v.route[v.current_route_idx]]
        vx = p1[0] + (p2[0] - p1[0]) * progress
        vy = p1[1] + (p2[1] - p1[1]) * progress
        col = ORIGIN_COLORS.get(v.route[0], '#888')
        ax.add_patch(patches.Circle((vx, vy), 0.025, fc=col, ec='white', lw=0.5))


def main():
    print("\n  Loading ML model...")

    from src.ml_model import generate_training_data, add_features, train_and_compare, predict_rates
    from src.simulator import TrafficSimulator, LightState
    from src.optimizer import FixedTimingController, WebsterOptimizer

    df = generate_training_data(days=30, num_intersections=4)
    df_fe = add_features(df)
    trained, scalers, metrics_df = train_and_compare(df_fe)
    model = trained[metrics_df.iloc[0]["Model"]]
    scaler = scalers[metrics_df.iloc[0]["Model"]]
    print(f"  Model: {metrics_df.iloc[0]['Model']}")

    # Simulators
    sim_base = TrafficSimulator(num_intersections=4, seed=42)
    sim_opt = TrafficSimulator(num_intersections=4, seed=42)
    baseline = FixedTimingController(ns_green=30.0, ew_green=30.0)
    optimizer = WebsterOptimizer()

    iids = [i.id for i in sim_base.intersections]

    # State
    state = {
        'step': 0,
        'scenario': 'am_rush',
        'paused': False,
        'incident': False,
        'speed_idx': 1,  # Index into SPEED_LEVELS (1 = 1x speed)
        'total_departed_base': 0,
        'total_departed_opt': 0,
    }

    # History
    history = {k: deque(maxlen=200) for k in ['t', 'wb', 'wo', 'qb', 'qo', 'tb', 'to']}

    # Figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8), facecolor=C_BG)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1],
                  left=0.05, right=0.95, top=0.88, bottom=0.08, wspace=0.15, hspace=0.25)

    ax_base = fig.add_subplot(gs[0, 0])
    ax_opt = fig.add_subplot(gs[0, 1])
    ax_wait = fig.add_subplot(gs[1, 0])
    ax_queue = fig.add_subplot(gs[1, 1])

    # Chart setup
    for ax, ylabel in [(ax_wait, 'Wait Time (s)'), (ax_queue, 'Queue Length')]:
        ax.set_facecolor('#1a1f2e')
        ax.set_ylabel(ylabel, fontsize=9, color='#aaa')
        ax.tick_params(colors='#666', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#333')

    line_wb, = ax_wait.plot([], [], color=C_BASELINE, lw=1.5, label='Baseline')
    line_wo, = ax_wait.plot([], [], color=C_OPT, lw=1.5, label='Optimized')
    line_qb, = ax_queue.plot([], [], color=C_BASELINE, lw=1.5, label='Baseline')
    line_qo, = ax_queue.plot([], [], color=C_OPT, lw=1.5, label='Optimized')
    ax_wait.legend(loc='upper left', fontsize=7)
    ax_queue.legend(loc='upper left', fontsize=7)

    # Title and controls text
    title_text = fig.suptitle("", fontsize=13, fontweight='bold', color='white', y=0.96)
    controls_text = fig.text(0.5, 0.99,
        "[1/2/3] Scenario  [+/-] Speed  [R] Reset  [I] Incident  [Space] Pause  [Q] Quit",
        ha='center', fontsize=8, color='#888')

    # Status text (bottom left)
    status_text = fig.text(0.05, 0.01, "", fontsize=8, color='#666')

    def reset_simulation():
        """Reset both simulators and clear history."""
        state['step'] = 0
        state['incident'] = False
        state['total_departed_base'] = 0
        state['total_departed_opt'] = 0
        sim_base.reset()
        sim_opt.reset()
        for k in history:
            history[k].clear()

    # Keyboard handler
    def on_key(event):
        if event.key == '1':
            state['scenario'] = 'am_rush'
            reset_simulation()
        elif event.key == '2':
            state['scenario'] = 'pm_rush'
            reset_simulation()
        elif event.key == '3':
            state['scenario'] = 'midday'
            reset_simulation()
        elif event.key == 'r':
            reset_simulation()
        elif event.key in ['+', '=']:
            state['speed_idx'] = min(state['speed_idx'] + 1, len(SPEED_LEVELS) - 1)
        elif event.key == '-':
            state['speed_idx'] = max(state['speed_idx'] - 1, 0)
        elif event.key == 'i':
            state['incident'] = not state['incident']
            if state['incident']:
                sim_opt.trigger_incident(0, 'north', 0.3)
                sim_base.trigger_incident(0, 'north', 0.3)
            else:
                sim_opt.clear_incident(0, 'north')
                sim_base.clear_incident(0, 'north')
        elif event.key == ' ':
            state['paused'] = not state['paused']
        elif event.key in ['q', 'escape']:
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(frame):
        if state['paused']:
            return []

        cfg = SCENARIOS[state['scenario']]
        hour, dow = cfg['hour'], cfg['dow']
        speed = SPEED_LEVELS[state['speed_idx']]

        # Simulate (more steps per frame at higher speeds)
        steps_per_frame = max(1, int(3 * speed))
        for _ in range(steps_per_frame):
            state['step'] += 1
            rates = predict_rates(model, scaler, hour, dow, num_intersections=4)
            sim_base.step(arrival_rates=rates)
            sim_opt.step(arrival_rates=rates)

            if state['step'] % 30 == 0:
                sim_base.update_lights(baseline.compute_timings(iids))
                ql = {i.id: {d: len(q) for d, q in i.queues.items()} for i in sim_opt.intersections}
                sim_opt.update_lights(optimizer.compute_timings(iids, predicted_rates=rates, queue_lengths=ql))

        # Track throughput
        state['total_departed_base'] = sim_base.total_departed
        state['total_departed_opt'] = sim_opt.total_departed

        # Get incidents for display
        incidents = {0: {'north': state['incident']}} if state['incident'] else None

        # Draw grids
        draw_grid(ax_base, sim_base, "BASELINE (Fixed 30/30)", C_BASELINE, LightState, incidents)
        draw_grid(ax_opt, sim_opt, "OPTIMIZED (Webster + ML)", C_OPT, LightState, incidents)

        # Improvement badge
        mb, mo = sim_base.get_metrics(), sim_opt.get_metrics()
        if mb['avg_wait_time'] > 0.1:
            imp = (mb['avg_wait_time'] - mo['avg_wait_time']) / mb['avg_wait_time'] * 100
        else:
            imp = 0
        imp_col = '#2ecc71' if imp >= 15 else '#f39c12' if imp >= 0 else '#e74c3c'
        imp_text = f"+{imp:.0f}%" if imp >= 0 else f"{imp:.0f}%"
        ax_opt.text(0.5, 1.4, f"{imp_text} wait time", color=imp_col, fontsize=11,
                    ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#1a1f2e', ec=imp_col, lw=1.5))

        # Throughput comparison
        if state['total_departed_base'] > 0:
            tp_imp = (state['total_departed_opt'] - state['total_departed_base']) / state['total_departed_base'] * 100
            tp_col = '#2ecc71' if tp_imp > 0 else '#e74c3c'
            tp_text = f"+{tp_imp:.0f}%" if tp_imp >= 0 else f"{tp_imp:.0f}%"
            ax_opt.text(0.5, 1.25, f"{tp_text} throughput ({state['total_departed_opt']} vs {state['total_departed_base']})",
                        color=tp_col, fontsize=8, ha='center')

        # Update title
        speed_str = f"{speed}x" if speed != 1 else ""
        status = " [PAUSED]" if state['paused'] else ""
        incident_status = " | ⚠ INCIDENT" if state['incident'] else ""
        speed_display = f" [{speed_str}]" if speed_str else ""
        title_text.set_text(f"Smart Traffic Light Optimisation — {cfg['label']}{speed_display}{incident_status}{status}")

        # Update status text
        sim_time = state['step']  # Each step is ~1 second
        mins, secs = divmod(sim_time, 60)
        status_text.set_text(f"Sim Time: {mins:02d}:{secs:02d} | Step: {state['step']} | Speed: {speed}x")

        # Update history
        history['t'].append(state['step'])
        history['wb'].append(mb['avg_wait_time'])
        history['wo'].append(mo['avg_wait_time'])
        history['qb'].append(mb['total_queue'])
        history['qo'].append(mo['total_queue'])

        # Update charts
        if len(history['t']) > 1:
            t = list(history['t'])
            line_wb.set_data(t, list(history['wb']))
            line_wo.set_data(t, list(history['wo']))
            line_qb.set_data(t, list(history['qb']))
            line_qo.set_data(t, list(history['qo']))

            ax_wait.set_xlim(min(t), max(t))
            ax_wait.set_ylim(0, max(max(history['wb']), max(history['wo']), 1) * 1.1)
            ax_queue.set_xlim(min(t), max(t))
            ax_queue.set_ylim(0, max(max(history['qb']), max(history['qo']), 1) * 1.1)

        return []

    print("\n  Controls:")
    print("    [1/2/3] Switch scenario (AM Rush / PM Rush / Midday)")
    print("    [+/=]   Speed up simulation")
    print("    [-]     Slow down simulation")
    print("    [R]     Reset simulation")
    print("    [I]     Toggle incident on Int 0")
    print("    [Space] Pause/Resume")
    print("    [Q]     Quit\n")

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
