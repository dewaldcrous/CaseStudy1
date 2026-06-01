"""
Smart Traffic Light Optimisation — Standalone Demo
===================================================

Interactive demo with GUI controls:
  - Scenario buttons (AM Rush / PM Rush / Midday)
  - Speed slider (0.5x to 10x)
  - Incident toggle button
  - Reset and Pause buttons
  - Time of day clock display
  - Per-intersection vehicle counts by origin color
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
from matplotlib.widgets import Button, Slider
from collections import deque

# Colors
C_BG = "#0e1117"
C_ROAD = "#2a3441"
C_GREEN = "#00ff00"
C_RED = "#ff0000"
C_BASELINE = "#e74c3c"
C_OPT = "#2ecc71"

# Vehicle colors by origin intersection
ORIGIN_COLORS = {
    0: "#9b59b6",  # Purple - Int 0 (top-left)
    1: "#ffffff",  # White - Int 1 (top-right)
    2: "#2c3e50",  # Dark - Int 2 (bottom-left)
    3: "#f39c12",  # Orange - Int 3 (bottom-right)
}

SCENARIOS = {
    "am_rush": {"hour": 8, "dow": 0, "label": "AM Rush (8 AM)", "key": "1"},
    "pm_rush": {"hour": 17, "dow": 0, "label": "PM Rush (5 PM)", "key": "2"},
    "midday": {"hour": 12, "dow": 0, "label": "Midday (12 PM)", "key": "3"},
}

SPEED_LEVELS = [0.5, 1, 2, 3, 5, 10]


def get_intersection_counts(sim):
    """Get vehicle counts per intersection by origin.

    For each intersection, counts how many vehicles are currently queued there,
    broken down by which intersection they originally came from (origin).

    Returns dict: {int_id: {0: count, 1: count, 2: count, 3: count, 'total': count}}
    """
    counts = {i.id: {0: 0, 1: 0, 2: 0, 3: 0, 'total': 0} for i in sim.intersections}

    # Count vehicles in queues at each intersection
    for inter in sim.intersections:
        for d, q in inter.queues.items():
            for v in q:
                # Origin = first intersection in route (where vehicle entered network)
                origin = v.route[0] if hasattr(v, 'route') and v.route else inter.id
                if origin in counts[inter.id]:
                    counts[inter.id][origin] += 1
                counts[inter.id]['total'] += 1

    return counts


def draw_grid(ax, sim, title, title_color, LightState, incidents=None):
    """Draw a 2x2 intersection grid with traffic lights and vehicles.

    Args:
        ax: Matplotlib axes to draw on
        sim: TrafficSimulator instance with intersection state
        title: Grid title (e.g., "BASELINE" or "OPTIMIZED")
        title_color: Color for the title text
        LightState: Enum for GREEN/RED light states
        incidents: Optional dict of {int_id: {direction: bool}} for incident markers
    """
    ax.clear()
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')

    m = sim.get_metrics()
    ax.set_title(f"{title}\nQueue: {m['total_queue']} | Wait: {m['avg_wait_time']:.1f}s",
                 color=title_color, fontsize=10, fontweight='bold', pad=8)

    positions = {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}

    # Get per-intersection counts
    counts = get_intersection_counts(sim)

    # Roads
    rw = 0.15
    for x in [0, 1]:
        ax.add_patch(patches.Rectangle((x - rw/2, -0.5), rw, 2.0, fc=C_ROAD, ec='none'))
    for y in [0, 1]:
        ax.add_patch(patches.Rectangle((-0.5, y - rw/2), 2.0, rw, fc=C_ROAD, ec='none'))

    # Draw each intersection
    for inter in sim.intersections:
        x, y = positions[inter.id]

        # Check for incident
        has_incident = incidents and inter.id in incidents and any(incidents[inter.id].values())
        box_color = '#ff6b6b' if has_incident else '#4fc3f7'

        # Intersection box
        ax.add_patch(patches.Rectangle((x-0.08, y-0.08), 0.16, 0.16,
                                        fc='#1a1f2e', ec=box_color, lw=2))

        # Incident marker
        if has_incident:
            ax.text(x, y+0.12, "⚠", color='#ff6b6b', fontsize=10, ha='center', va='center')

        # Traffic lights
        for d, (dx, dy) in [('north', (0, 0.06)), ('south', (0, -0.06)),
                            ('east', (0.06, 0)), ('west', (-0.06, 0))]:
            c = C_GREEN if inter.light_states[d] == LightState.GREEN else C_RED
            ax.add_patch(patches.Circle((x+dx, y+dy), 0.02, fc=c, ec='white', lw=0.5))

        # Intersection ID
        ax.text(x, y, str(inter.id), color='#4fc3f7', ha='center', va='center',
                fontsize=9, fontweight='bold')

        # Draw queued vehicles (limit to 6 per direction for performance)
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

            # Show overflow count
            if len(q) > 6:
                ox, oy = x + dx * 0.5, y + dy * 0.5
                ax.text(ox, oy, f"+{len(q)-6}", color='#ff0', fontsize=7,
                        ha='center', va='center', fontweight='bold')

        # === Per-intersection count badge ===
        c = counts[inter.id]

        # Badge position (corners of the grid)
        badge_positions = {
            0: (-0.45, 1.35),   # Int 0: top-left corner
            1: (1.45, 1.35),   # Int 1: top-right corner
            2: (-0.45, -0.35), # Int 2: bottom-left corner
            3: (1.45, -0.35),  # Int 3: bottom-right corner
        }
        bx, by = badge_positions[inter.id]

        # Draw badge background
        ax.add_patch(patches.FancyBboxPatch((bx-0.18, by-0.22), 0.36, 0.44,
                     boxstyle="round,pad=0.02", fc='#1a1f2e', ec='#4fc3f7', lw=1.5, alpha=0.95))

        # Total count header
        ax.text(bx, by+0.12, f"{c['total']}", color='#4fc3f7', fontsize=10,
                ha='center', va='center', fontweight='bold')

        # Per-origin counts with colored indicators (2x2 mini grid)
        # Row 1: Int 0 (purple), Int 1 (white)
        # Row 2: Int 2 (dark), Int 3 (orange)
        mini_positions = [
            (-0.07, -0.02, 0),  # Purple - Int 0
            (0.07, -0.02, 1),   # White - Int 1
            (-0.07, -0.12, 2),  # Dark - Int 2
            (0.07, -0.12, 3),   # Orange - Int 3
        ]
        for mx, my, oid in mini_positions:
            # Colored dot
            ax.add_patch(patches.Circle((bx + mx - 0.04, by + my), 0.025,
                         fc=ORIGIN_COLORS[oid], ec='white', lw=0.5))
            # Count number
            ax.text(bx + mx + 0.02, by + my, str(c[oid]), color='#fff', fontsize=7,
                    ha='left', va='center', fontweight='bold')

    # Draw traveling vehicles (vehicles between intersections)
    for v in getattr(sim, 'traveling_vehicles', []):
        if not hasattr(v, 'route') or v.current_route_idx < 1:
            continue
        progress = min((sim.time - v.travel_start_time) / 2.0, 1.0)
        p1, p2 = positions[v.route[v.current_route_idx - 1]], positions[v.route[v.current_route_idx]]
        vx = p1[0] + (p2[0] - p1[0]) * progress
        vy = p1[1] + (p2[1] - p1[1]) * progress
        col = ORIGIN_COLORS.get(v.route[0], '#888')
        ax.add_patch(patches.Circle((vx, vy), 0.025, fc=col, ec='white', lw=0.5))


def format_time(hour, step):
    """Convert hour and simulation step to clock time string."""
    total_seconds = step
    minutes = (total_seconds // 60) % 60
    sim_hour = (hour + total_seconds // 3600) % 24
    am_pm = "AM" if sim_hour < 12 else "PM"
    display_hour = sim_hour % 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:{minutes:02d} {am_pm}"


def main():
    """Run the interactive traffic simulation demo."""
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
        'speed': 1.0,
        'total_departed_base': 0,
        'total_departed_opt': 0,
    }

    # History (smaller buffer for performance)
    history = {k: deque(maxlen=100) for k in ['t', 'wb', 'wo', 'qb', 'qo']}

    # Figure - adjusted layout to prevent overlap
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 9), facecolor=C_BG)

    # Main grid layout with more space at top
    gs = GridSpec(3, 2, figure=fig, height_ratios=[3, 1, 0.3],
                  left=0.05, right=0.95, top=0.88, bottom=0.12, wspace=0.15, hspace=0.3)

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

    # Title - moved down to y=0.93 to prevent overlap
    title_text = fig.suptitle("", fontsize=12, fontweight='bold', color='white', y=0.93)
    clock_text = fig.text(0.95, 0.93, "", fontsize=10, color='#4fc3f7', ha='right', fontweight='bold')

    # Status text
    status_text = fig.text(0.5, 0.01, "", fontsize=8, color='#666', ha='center')

    # ==================== GUI CONTROLS ====================
    btn_color = '#2a3441'
    btn_hover = '#3d4f61'
    btn_active = '#4fc3f7'

    # Scenario buttons
    ax_am = fig.add_axes([0.05, 0.04, 0.08, 0.04])
    ax_pm = fig.add_axes([0.14, 0.04, 0.08, 0.04])
    ax_mid = fig.add_axes([0.23, 0.04, 0.08, 0.04])

    btn_am = Button(ax_am, 'AM Rush', color=btn_active, hovercolor=btn_hover)
    btn_pm = Button(ax_pm, 'PM Rush', color=btn_color, hovercolor=btn_hover)
    btn_mid = Button(ax_mid, 'Midday', color=btn_color, hovercolor=btn_hover)

    scenario_buttons = {'am_rush': (btn_am, ax_am), 'pm_rush': (btn_pm, ax_pm), 'midday': (btn_mid, ax_mid)}

    # Speed slider
    ax_speed = fig.add_axes([0.38, 0.045, 0.18, 0.025])
    speed_slider = Slider(ax_speed, 'Speed', 0.5, 10, valinit=1.0, valstep=0.5,
                          color='#4fc3f7', initcolor='none')
    ax_speed.set_facecolor('#1a1f2e')

    # Control buttons
    ax_incident = fig.add_axes([0.60, 0.04, 0.1, 0.04])
    ax_reset = fig.add_axes([0.71, 0.04, 0.08, 0.04])
    ax_pause = fig.add_axes([0.80, 0.04, 0.08, 0.04])

    btn_incident = Button(ax_incident, 'Incident', color=btn_color, hovercolor=btn_hover)
    btn_reset = Button(ax_reset, 'Reset', color=btn_color, hovercolor=btn_hover)
    btn_pause = Button(ax_pause, 'Pause', color=btn_color, hovercolor=btn_hover)

    # Legend
    fig.text(0.5, 0.085,
        "Origin: ● Purple=Int0  ● White=Int1  ● Dark=Int2  ● Orange=Int3",
        fontsize=7, color='#888', ha='center')

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
        btn_incident.color = btn_color
        ax_incident.set_facecolor(btn_color)

    def update_scenario_buttons():
        """Update button colors to show active scenario."""
        for sc, (btn, ax) in scenario_buttons.items():
            if sc == state['scenario']:
                btn.color = btn_active
                ax.set_facecolor(btn_active)
            else:
                btn.color = btn_color
                ax.set_facecolor(btn_color)

    def on_am_click(event):
        state['scenario'] = 'am_rush'
        reset_simulation()
        update_scenario_buttons()

    def on_pm_click(event):
        state['scenario'] = 'pm_rush'
        reset_simulation()
        update_scenario_buttons()

    def on_mid_click(event):
        state['scenario'] = 'midday'
        reset_simulation()
        update_scenario_buttons()

    def on_speed_change(val):
        state['speed'] = val

    def on_incident_click(event):
        state['incident'] = not state['incident']
        if state['incident']:
            sim_opt.trigger_incident(0, 'north', 0.3)
            sim_base.trigger_incident(0, 'north', 0.3)
            btn_incident.color = '#ff6b6b'
            ax_incident.set_facecolor('#ff6b6b')
        else:
            sim_opt.clear_incident(0, 'north')
            sim_base.clear_incident(0, 'north')
            btn_incident.color = btn_color
            ax_incident.set_facecolor(btn_color)

    def on_reset_click(event):
        reset_simulation()

    def on_pause_click(event):
        state['paused'] = not state['paused']
        if state['paused']:
            btn_pause.label.set_text('Resume')
            btn_pause.color = '#f39c12'
            ax_pause.set_facecolor('#f39c12')
        else:
            btn_pause.label.set_text('Pause')
            btn_pause.color = btn_color
            ax_pause.set_facecolor(btn_color)

    # Connect callbacks
    btn_am.on_clicked(on_am_click)
    btn_pm.on_clicked(on_pm_click)
    btn_mid.on_clicked(on_mid_click)
    speed_slider.on_changed(on_speed_change)
    btn_incident.on_clicked(on_incident_click)
    btn_reset.on_clicked(on_reset_click)
    btn_pause.on_clicked(on_pause_click)

    # Keyboard shortcuts
    def on_key(event):
        if event.key == '1':
            on_am_click(None)
        elif event.key == '2':
            on_pm_click(None)
        elif event.key == '3':
            on_mid_click(None)
        elif event.key == 'r':
            on_reset_click(None)
        elif event.key == 'i':
            on_incident_click(None)
        elif event.key == ' ':
            on_pause_click(None)
        elif event.key in ['q', 'escape']:
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(frame):
        if state['paused']:
            return []

        cfg = SCENARIOS[state['scenario']]
        hour, dow = cfg['hour'], cfg['dow']
        speed = state['speed']

        # Simulate - fewer steps at high speed for responsiveness
        steps_per_frame = max(1, int(2 * speed))
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
        ax_opt.text(0.5, 1.5, f"{imp_text} wait time", color=imp_col, fontsize=10,
                    ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#1a1f2e', ec=imp_col, lw=1.5))

        # Throughput comparison
        if state['total_departed_base'] > 0:
            tp_imp = (state['total_departed_opt'] - state['total_departed_base']) / state['total_departed_base'] * 100
            tp_col = '#2ecc71' if tp_imp > 0 else '#e74c3c'
            tp_text = f"+{tp_imp:.0f}%" if tp_imp >= 0 else f"{tp_imp:.0f}%"
            ax_opt.text(0.5, 1.35, f"{tp_text} throughput ({state['total_departed_opt']} vs {state['total_departed_base']})",
                        color=tp_col, fontsize=7, ha='center')

        # Update title
        incident_status = " | ⚠ INCIDENT" if state['incident'] else ""
        status = " [PAUSED]" if state['paused'] else ""
        title_text.set_text(f"Smart Traffic Light Optimisation — {cfg['label']}{incident_status}{status}")

        # Update clock
        time_str = format_time(cfg['hour'], state['step'])
        clock_text.set_text(f"🕐 {time_str}")

        # Update status
        sim_time = state['step']
        mins, secs = divmod(sim_time, 60)
        status_text.set_text(f"Elapsed: {mins:02d}:{secs:02d} | Departed: {state['total_departed_opt']} | Speed: {speed:.1f}x")

        # Update history (every 3rd frame for performance)
        if state['step'] % 3 == 0:
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

    print("\n  Demo loaded! Use buttons or keyboard shortcuts:")
    print("    [1/2/3] Scenario | [R] Reset | [I] Incident | [Space] Pause | [Q] Quit\n")

    # Animation with slightly longer interval for better responsiveness
    ani = FuncAnimation(fig, update, interval=80, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
