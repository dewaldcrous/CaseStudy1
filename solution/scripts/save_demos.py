"""
Save all three demo animations as GIFs.
Uses Agg (non-interactive) backend so no display window is required.
Run: python save_demos.py

Produces (in figures/ folder):
  figures/fig_live_demo_am_rush.gif    -- Monday 8am rush hour
  figures/fig_live_demo_full_day.gif   -- Full 24-hour weekday cycle
  figures/fig_live_demo_incident.gif   -- AM rush + accident + recovery
"""

import matplotlib
matplotlib.use("Agg")   # must be set before importing pyplot

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.live_viz import run_live_demo

scenarios = [
    ("am_rush",  300, 4,  "Monday 8 AM rush — NS/EW asymmetry demo"),
    ("full_day", 480, 10, "Full weekday cycle — optimizer adapts to all regimes"),
    ("incident", 300, 4,  "AM rush + accident at t=150s + recovery at t=240s"),
]

for scenario, duration, speed, desc in scenarios:
    print(f"\n{'='*60}")
    print(f"  Saving: {scenario}  ({desc})")
    print(f"{'='*60}")
    run_live_demo(
        scenario     = scenario,
        duration_s   = duration,
        speed_factor = speed,
        save_gif     = True,
    )

print("\nAll GIFs saved to figures/:")
for scenario, _, _, _ in scenarios:
    fname = f"figures/fig_live_demo_{scenario}.gif"
    size  = os.path.getsize(fname) / 1024 if os.path.exists(fname) else 0
    print(f"  {fname:<40}  {size:.0f} KB")
