"""
Smart Traffic Light Optimisation — Interactive Dashboard
=========================================================

Run:
    streamlit run dashboard.py

Opens a browser tab with:
  - Live 2×2 intersection grid (Plotly) — shows light states, queue sizes, vehicle counts
  - Live metrics charts — wait time, queue length, throughput (baseline vs optimized)
  - Control panel — scenario selector, speed slider, start/pause, incident button

No GIF needed — the simulation runs live in the browser.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os
import time as time_mod
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config must be first Streamlit call ──────────────────────
st.set_page_config(
    page_title="Traffic Light Optimisation",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy-load heavy imports once ──────────────────────────────────
@st.cache_resource(show_spinner="Training ML model…")
def load_model():
    """Train model once and cache — survives page reruns."""
    from src.ml_model import generate_training_data, add_features, train_and_compare
    df    = generate_training_data(days=30, num_intersections=4)
    df_fe = add_features(df)
    trained, scalers, metrics_df = train_and_compare(df_fe)
    best_name  = metrics_df.iloc[0]["Model"]
    best_model = trained[best_name]
    best_scaler= scalers[best_name]
    return best_model, best_scaler, best_name, metrics_df


# ── Colours ───────────────────────────────────────────────────────
C_GREEN    = "#2ecc71"
C_RED      = "#e74c3c"
C_YELLOW   = "#f39c12"
C_BASELINE = "#e74c3c"
C_OPT      = "#2ecc71"
C_BG       = "#0e1117"
C_PANEL    = "#1a1f2e"


# ── Build Plotly intersection grid ────────────────────────────────
def build_grid_figure(sim, title: str) -> go.Figure:
    """
    Build a Plotly figure showing the 2×2 intersection grid.
    Each intersection shows:
      - Circle (grey) for the intersection node
      - 4 coloured dots showing current light state per direction
      - Bar on each approach showing queue length
      - Text labels for queue counts
    """
    from src.simulator import LightState

    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        margin=dict(l=20, r=20, t=50, b=20),
        title=dict(text=title, font=dict(color="white", size=13)),
        xaxis=dict(range=[-0.7, 1.7], showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(range=[-0.7, 1.7], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x"),
        showlegend=False,
    )

    # Draw road grid
    for x in [0, 1]:
        fig.add_shape(type="line", x0=x, y0=-0.5, x1=x, y1=1.5,
                      line=dict(color="#444", width=8))
    for y in [0, 1]:
        fig.add_shape(type="line", x0=-0.5, y0=y, x1=1.5, y1=y,
                      line=dict(color="#444", width=8))

    # Draw each intersection
    for inter in sim.intersections:
        cx, cy = inter.position
        ql     = inter.queue_lengths()
        m      = sim.get_metrics()

        lstate_col = {
            LightState.GREEN:  C_GREEN,
            LightState.RED:    C_RED,
            LightState.YELLOW: C_YELLOW,
        }

        # Intersection circle
        fig.add_shape(type="circle",
                      x0=cx - 0.1, y0=cy - 0.1,
                      x1=cx + 0.1, y1=cy + 0.1,
                      fillcolor="#555", line_color="white", line_width=1)
        fig.add_annotation(x=cx, y=cy, text=str(inter.id),
                           showarrow=False, font=dict(color="white", size=12))

        # Light state dots + queue bars per direction
        offsets = {
            "north": (cx,        cy + 0.2,   cx - 0.05, cy + 0.25, "v"),
            "south": (cx,        cy - 0.2,   cx - 0.05, cy - 0.25, "v"),
            "east":  (cx + 0.2,  cy,         cx + 0.25, cy - 0.05, "h"),
            "west":  (cx - 0.2,  cy,         cx - 0.25, cy - 0.05, "h"),
        }
        for direction, (dx, dy, bx, by, orient) in offsets.items():
            col = lstate_col[inter.light_states[direction]]
            q   = ql[direction]
            qcap = min(q, 15)   # cap bar at 15 for display

            # Light dot
            fig.add_shape(type="circle",
                          x0=dx-0.05, y0=dy-0.05,
                          x1=dx+0.05, y1=dy+0.05,
                          fillcolor=col, line_color="white", line_width=0.5)

            # Queue bar
            bar_len = qcap * 0.025
            if orient == "v":
                sign = 1 if direction == "north" else -1
                fig.add_shape(type="rect",
                              x0=bx, y0=by,
                              x1=bx + 0.1, y1=by + sign * bar_len,
                              fillcolor=col, opacity=0.6,
                              line_width=0)
            else:
                sign = 1 if direction == "east" else -1
                fig.add_shape(type="rect",
                              x0=bx, y0=by,
                              x1=bx + sign * bar_len, y1=by + 0.1,
                              fillcolor=col, opacity=0.6,
                              line_width=0)

            # Queue count label
            fig.add_annotation(
                x=bx + (0.05 if orient == "v" else (0.05 * sign)),
                y=by + (0.05 if orient == "h" else 0.02),
                text=str(q), showarrow=False,
                font=dict(color="white", size=9),
            )

        # Incident warning
        if sim._incidents.get(inter.id):
            fig.add_annotation(
                x=cx, y=cy + 0.45,
                text="⚠ INCIDENT", showarrow=False,
                font=dict(color=C_YELLOW, size=11, family="Arial Bold"),
            )

    return fig


# ── Build live metrics figure ─────────────────────────────────────
def build_metrics_figure(history_base: dict, history_opt: dict) -> go.Figure:
    """Three-panel subplot: wait time, queue, throughput."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Avg Wait Time (s)", "Total Queue", "Throughput (veh/s)"],
        vertical_spacing=0.12,
    )

    t = history_base["time"]
    pairs = [
        ("avg_wait_time", 1),
        ("total_queue",   2),
        ("throughput",    3),
    ]
    for key, row in pairs:
        fig.add_trace(go.Scatter(
            x=t, y=history_base[key],
            name="Fixed baseline", line=dict(color=C_BASELINE, width=1.5),
            showlegend=(row == 1),
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=history_opt[key],
            name="Webster optimized", line=dict(color=C_OPT, width=1.5),
            showlegend=(row == 1),
        ), row=row, col=1)

    fig.update_layout(
        paper_bgcolor=C_BG, plot_bgcolor=C_PANEL,
        font=dict(color="white"),
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation="h", y=1.05,
                    font=dict(color="white")),
        height=420,
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#333", row=i, col=1)
        fig.update_yaxes(gridcolor="#333", row=i, col=1)
        if i == 3:
            fig.update_xaxes(title_text="Simulation time (s)", row=i, col=1)

    return fig


# ── Simulation state init ─────────────────────────────────────────
def init_state(scenario: str, model, scaler):
    """Initialise or reset simulator state in session_state."""
    from src.simulator import TrafficSimulator
    from src.optimizer import FixedTimingController, WebsterOptimizer
    from src.ml_model  import predict_rates

    scenario_cfg = {
        "am_rush":  {"hour": 8,  "dow": 0},
        "full_day": {"hour": 6,  "dow": 0},
        "incident": {"hour": 8,  "dow": 0},
    }
    cfg = scenario_cfg.get(scenario, scenario_cfg["am_rush"])

    sim_b = TrafficSimulator(num_intersections=4, seed=42)
    sim_o = TrafficSimulator(num_intersections=4, seed=42)
    bctrl = FixedTimingController()
    wopt  = WebsterOptimizer()
    iids  = [i.id for i in sim_b.intersections]

    rates0 = predict_rates(model, scaler, cfg["hour"], cfg["dow"],
                           num_intersections=4)
    sim_b.update_lights(bctrl.compute_timings(iids))
    sim_o.update_lights(wopt.compute_timings(iids, predicted_rates=rates0))

    empty_hist = {k: [] for k in ("time", "avg_wait_time",
                                   "total_queue", "throughput")}
    st.session_state.update(
        sim_base   = sim_b,
        sim_opt    = sim_o,
        baseline   = bctrl,
        optimizer  = wopt,
        iids       = iids,
        hour0      = cfg["hour"],
        dow        = cfg["dow"],
        step       = 0,
        running    = False,
        hist_base  = empty_hist.copy(),
        hist_opt   = {k: [] for k in empty_hist},
    )


# ── Advance one animation frame ───────────────────────────────────
def advance(speed: int, model, scaler) -> None:
    """Run `speed` sim-seconds, update histories."""
    from src.optimizer import FixedTimingController, WebsterOptimizer
    from src.ml_model  import predict_rates

    s   = st.session_state
    for _ in range(speed):
        current_hour = int((s.hour0 * 3600 + s.step) % 86400 / 3600)
        rates = predict_rates(model, scaler, current_hour, s.dow,
                              num_intersections=4)
        if s.step % 60 == 0:
            ql_o = s.sim_opt.get_queue_lengths()
            s.sim_base.update_lights(s.baseline.compute_timings(s.iids))
            s.sim_opt.update_lights(
                s.optimizer.compute_timings(s.iids,
                    predicted_rates=rates, queue_lengths=ql_o)
            )
        s.sim_base.step(rates)
        s.sim_opt.step(rates)
        s.step += 1

    mb = s.sim_base.get_metrics()
    mo = s.sim_opt.get_metrics()
    for key in ("time", "avg_wait_time", "total_queue", "throughput"):
        s.hist_base[key].append(mb[key])
        s.hist_opt[key].append(mo[key])


# ══════════════════════════════════════════════════════════════════
#  Main dashboard layout
# ══════════════════════════════════════════════════════════════════

# Header
st.markdown(
    """
    <h1 style='color:white; margin-bottom:4px'>
        🚦 Smart Traffic Light Optimisation
    </h1>
    <p style='color:#aaa; margin-top:0'>
        Webster (1958) adaptive controller vs fixed 30s/30s baseline
        &nbsp;·&nbsp; LightGBM demand forecasting
        &nbsp;·&nbsp; 4-intersection network
    </p>
    """,
    unsafe_allow_html=True,
)

# Load model (cached)
model, scaler, best_name, metrics_df = load_model()

# ── Sidebar controls ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    scenario = st.selectbox(
        "Scenario",
        options=["am_rush", "full_day", "incident"],
        format_func=lambda x: {
            "am_rush":  "🌅 Monday 8 AM Rush",
            "full_day": "🕐 Full Day Cycle (6am→)",
            "incident": "⚠️ AM Rush + Incident",
        }[x],
    )

    speed = st.slider("Simulation speed (steps/frame)", 1, 20, 5)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Start", width="stretch")
    with col2:
        pause_btn = st.button("⏸ Pause", width="stretch")

    reset_btn = st.button("↺ Reset", width="stretch")

    st.markdown("---")
    st.markdown("### 🚨 Incident controls")
    st.caption("Block an approach (accident / road closure)")

    inc_int = st.selectbox("Intersection", [0, 1, 2, 3])
    inc_dir = st.selectbox("Direction", ["north", "south", "east", "west"])
    inc_cap = st.slider("Capacity remaining (%)", 0, 100, 15) / 100.0

    col3, col4 = st.columns(2)
    with col3:
        trig_btn = st.button("⚠ Trigger", width="stretch")
    with col4:
        clr_btn  = st.button("✓ Clear", width="stretch")

    st.markdown("---")
    st.markdown("### 📊 Model")
    st.caption(f"**Best model**: {best_name}")
    st.dataframe(
        metrics_df[["Model", "Test MAE (veh/s)", "Test R2"]].rename(
            columns={"Test MAE (veh/s)": "MAE", "Test R2": "R²"}
        ).set_index("Model"),
        width="stretch",
    )

# ── Initialise state if needed ────────────────────────────────────
if "sim_base" not in st.session_state or reset_btn:
    init_state(scenario, model, scaler)

# ── Handle button events ──────────────────────────────────────────
if start_btn:
    st.session_state.running = True
if pause_btn:
    st.session_state.running = False
if trig_btn:
    st.session_state.sim_base.trigger_incident(inc_int, inc_dir, inc_cap)
    st.session_state.sim_opt.trigger_incident(inc_int, inc_dir, inc_cap)
    st.toast(f"⚠ Incident: {inc_dir} at intersection {inc_int} — {int(inc_cap*100)}% capacity", icon="⚠️")
if clr_btn:
    st.session_state.sim_base.clear_incident(inc_int, inc_dir)
    st.session_state.sim_opt.clear_incident(inc_int, inc_dir)
    st.toast(f"✓ Incident cleared: {inc_dir} at intersection {inc_int}", icon="✅")

# ── Auto-trigger incident for incident scenario ───────────────────
if (scenario == "incident"
        and "sim_opt" in st.session_state
        and st.session_state.step == 150
        and not st.session_state.sim_opt._incidents):
    st.session_state.sim_base.trigger_incident(0, "north", 0.15)
    st.session_state.sim_opt.trigger_incident(0, "north", 0.15)

if (scenario == "incident"
        and "sim_opt" in st.session_state
        and st.session_state.step == 240
        and st.session_state.sim_opt._incidents):
    st.session_state.sim_base.clear_incident(0)
    st.session_state.sim_opt.clear_incident(0)

# ── Advance simulation if running ────────────────────────────────
if st.session_state.get("running", False):
    advance(speed, model, scaler)

# ── Layout: grid | metrics ────────────────────────────────────────
s = st.session_state
mb = s.sim_base.get_metrics() if "sim_base" in s else {}
mo = s.sim_opt.get_metrics()  if "sim_opt"  in s else {}

# KPI strip
k1, k2, k3, k4 = st.columns(4)
bwait = mb.get("avg_wait_time", 0)
owait = mo.get("avg_wait_time", 0)
imp   = (bwait - owait) / max(bwait, 0.01) * 100
t_str = f"{s.get('step', 0)}s"
k1.metric("Simulation time", t_str)
k2.metric("Baseline wait", f"{bwait:.1f}s")
k3.metric("Optimized wait", f"{owait:.1f}s",
          delta=f"{owait - bwait:.1f}s vs baseline",
          delta_color="inverse")
k4.metric("Improvement", f"{max(imp,0):.1f}%",
          delta="✅ ≥ 20% target" if imp >= 20 else "⚠ below target")

st.markdown("---")

col_grid, col_charts = st.columns([1, 1])

with col_grid:
    grid_placeholder = st.empty()
    with grid_placeholder:
        if "sim_opt" in s:
            fig_grid = build_grid_figure(
                s.sim_opt,
                f"Webster Optimized — t={s.get('step',0)}s",
            )
            st.plotly_chart(fig_grid, width="stretch", key="grid")
        else:
            st.info("Press ▶ Start to begin simulation")

with col_charts:
    charts_placeholder = st.empty()
    with charts_placeholder:
        if "hist_base" in s and s["hist_base"]["time"]:
            fig_m = build_metrics_figure(s.hist_base, s.hist_opt)
            st.plotly_chart(fig_m, width="stretch", key="metrics")
        else:
            st.info("Charts will appear once simulation starts")

# ── Auto-rerun while running ──────────────────────────────────────
if st.session_state.get("running", False):
    time_mod.sleep(0.08)   # ~12 fps
    st.rerun()
