# Smart Traffic Light Optimisation — Senior Quant Case Study

A machine-learning–driven adaptive traffic signal controller for a **connected 2×2 intersection network** that reduces average vehicle wait time by **~28%** (up to **57% during PM Rush**) and achieves **+17% throughput** during rush hours.

---

## Quick Start — Key Documents

| Document | Purpose |
|----------|---------|
| **[`demo.html`](demo.html)** | Interactive visualization — run in browser, see vehicles routing through network |
| **[`project_overview.html`](project_overview.html)** | Comprehensive technical documentation with all figures and explanations |
| **[`presenter_guide.html`](presenter_guide.html)** | Presentation talking points, anticipated questions, and demo walkthrough |

**Open `demo.html` in any browser** to see the connected network simulation in action.

---

## Demo Options Comparison

| | **demo.html** | **TrafficDemo.exe** |
|---|---|---|
| **Pros** | Zero install, runs in any browser | Uses actual Python ML models |
| | Full UI controls (buttons, sliders) | No Python/dependencies required |
| | Smaller file size (~50KB) | Identical algorithms to main.py |
| | Easy to share/embed | Smooth matplotlib animation |
| **Cons** | JS port of algorithms (not Python models) | Larger file (~150MB) |
| | Requires modern browser | Keyboard-only controls |
| | No direct model file access | Windows-only |

**Both produce equivalent results** — same Webster formula, same 70/30 ML-queue blend, same ~28% improvement. The HTML version ports the trained model coefficients to JavaScript; the executable runs the actual LightGBM model.

---

## Results Summary

| Metric | Value |
|--------|-------|
| Avg wait reduction (all scenarios) | **~28%** |
| Best scenario (PM Rush) | **+57%** |
| AM Rush throughput improvement | **+17%** |
| Best ML model | **LightGBM** (R² = 0.985, Gini = 0.94) |
| Network topology | **4 connected intersections** (2×2 grid) |

### Connected Network Model

Vehicles enter the 2×2 grid at edge approaches, route through adjacent intersections, and exit at the opposite edge:

```
        North Edge (spawn/exit)
              ↓   ↓
      +-----[Int 0]---[Int 1]-----+
      |        |         |        |
West  →        ↓         ↓        ← East
Edge  →        |         |        ← Edge
      |        |         |        |
      +-----[Int 2]---[Int 3]-----+
              ↑   ↑
        South Edge (spawn/exit)
```

**Vehicle colors by origin:** Purple (Int 0), White (Int 1), Black (Int 2), Orange (Int 3)

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the interactive demo (recommended)
#    Just open demo.html in any browser - no server needed

# 3. Run standalone Python visualization (smooth animation)
python run_demo.py              # AM Rush (default)
python run_demo.py pm_rush      # PM Rush
python run_demo.py full_day     # Full day cycle
# Or use the batch file:
run_demo.bat

# 4. Run the full ML pipeline + benchmarks
python main.py

# 5. Run Streamlit dashboard (interactive controls)
streamlit run dashboard.py
```

### Build Executable (Optional)

```bash
# Create standalone .exe (no Python required)
pip install pyinstaller
python build_exe.py

# Output: dist/TrafficDemo.exe
```

Requires Python 3.10+.

---

## Project Structure

```
solution/
├── demo.html              # ← Interactive browser visualization (START HERE)
├── project_overview.html  # ← Full technical documentation
├── presenter_guide.html   # ← Presentation guide
├── run_demo.py            # Standalone Python demo (smooth animation)
├── run_demo.bat           # Batch launcher for Python demo
├── build_exe.py           # Build standalone .exe with PyInstaller
├── main.py                # ML pipeline orchestrator
├── dashboard.py           # Streamlit dashboard
├── ml_model.js            # ML predictions for demo.html
├── src/
│   ├── simulator.py       # Connected network simulation
│   ├── ml_model.py        # LightGBM traffic prediction
│   ├── optimizer.py       # Webster's formula + controllers
│   ├── live_viz.py        # Matplotlib animation
│   └── model_validation.py
├── models/                # Saved ML models
└── figures/               # Generated charts
```

---

## Key Architecture

### ML → Optimizer Pipeline

```
ML predicts 4 rates per intersection:
  north: 0.14, south: 0.12, east: 0.05, west: 0.04

Aggregate by axis (max):
  ns_pred = max(0.14, 0.12) = 0.14
  ew_pred = max(0.05, 0.04) = 0.05

Blend with queue feedback (70/30):
  ns_demand = 0.7 × ns_pred + 0.3 × queue_ns

Flow ratios for Webster:
  y_ns = ns_demand / 0.5  (saturation = 0.5 veh/s)
```

### Independent Intersection Optimization

Each intersection runs its own Webster optimizer independently — no green wave coordination. This is a documented limitation with potential for future improvement.

---

## Scenario Benchmarks

| Scenario | Wait Δ | Throughput Δ | Notes |
|----------|--------|--------------|-------|
| PM Rush | **+57%** | +12% | Under capacity — both metrics improve |
| Midday | +36% | +5% | Balanced demand |
| Weekend | +29% | +3% | Lower volume |
| AM Rush | -8% | **+17%** | Over-capacity — tradeoff explained below |

### AM Rush Tradeoff

AM Rush arrival rates (0.56 veh/s) exceed saturation capacity (0.5 veh/s). The optimizer aggressively pushes NS vehicles downstream, causing cascading congestion (Int 0 → Int 2). Result: +17% throughput but higher wait times. The baseline's 50/50 split throttles flow, keeping wait times lower but processing fewer vehicles.

---

## References

See `project_overview.html` for full citations including Webster (1958), LightGBM (Ke et al., 2017), SHAP (Lundberg & Lee, 2017).
