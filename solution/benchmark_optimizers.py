"""
Benchmark all optimization algorithms against each other.

Runs the same AM rush scenario with each optimizer and compares results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.simulator import TrafficSimulator
from src.optimizer import (
    FixedTimingController, WebsterOptimizer, MaxPressureOptimizer,
    ProportionalOptimizer, PureMLOptimizer, PureQueueOptimizer,
    OPTIMIZER_REGISTRY
)
from src.ml_model import DIRECTIONS, INTERSECTION_FACTORS, DIRECTION_FACTORS

# ================================================================== #
#  Simulation parameters
# ================================================================== #

NUM_INTERSECTIONS = 4
SIM_DURATION = 3600       # 1 hour
TIMING_UPDATE_EVERY = 60  # seconds
SCENARIO_HOUR = 8         # AM rush
SCENARIO_DOW = 0          # Monday


def get_arrival_rates(hour, dow, num_intersections=4):
    """Generate arrival rates using the same rules as ml_model.py."""
    rates = {}
    for iid in range(num_intersections):
        rates[iid] = {}
        ifact = INTERSECTION_FACTORS[iid % len(INTERSECTION_FACTORS)]

        for direction in DIRECTIONS:
            dfact = DIRECTION_FACTORS[direction]
            is_ns = direction in ("north", "south")

            if dow >= 5:  # Weekend
                base = 0.06 + 0.04 * np.exp(-((hour - 13) ** 2) / 18)
                dm = 1.0
            else:  # Weekday
                morning = 0.20 * np.exp(-((hour - 8.0) ** 2) / 2.0)
                evening = 0.16 * np.exp(-((hour - 17.5) ** 2) / 3.0)
                base = 0.015 + morning + evening
                am_w = np.exp(-((hour - 8.0) ** 2) / 2.0)
                pm_w = np.exp(-((hour - 17.5) ** 2) / 3.0)
                ns_mult = 1.0 + 2.0 * am_w - 0.5 * pm_w
                ew_mult = 1.0 + 1.5 * pm_w - 0.3 * am_w
                dm = ns_mult if is_ns else ew_mult

            rates[iid][direction] = max(0.005, base * ifact * dfact * dm)

    return rates


def run_simulation(optimizer, duration=SIM_DURATION, verbose=False):
    """Run a simulation with the given optimizer."""
    sim = TrafficSimulator(num_intersections=NUM_INTERSECTIONS)
    intersection_ids = list(range(NUM_INTERSECTIONS))

    for t in range(duration):
        # Get current hour
        hour = (SCENARIO_HOUR + t // 3600) % 24

        # Get arrival rates
        rates = get_arrival_rates(hour, SCENARIO_DOW, NUM_INTERSECTIONS)

        # Update timings every TIMING_UPDATE_EVERY seconds
        if t % TIMING_UPDATE_EVERY == 0:
            queue_lengths = sim.get_queue_lengths()
            timings = optimizer.compute_timings(
                intersection_ids,
                predicted_rates=rates,
                queue_lengths=queue_lengths
            )
            sim.set_timings(timings)

        # Step simulation
        sim.step(rates)

    # Return metrics
    metrics = sim.get_metrics()
    return {
        "avg_wait": metrics.get("avg_wait_time", 0),
        "total_departed": metrics.get("total_departed", 0),
        "final_queue": metrics.get("total_queue", 0),
    }


def main():
    print("=" * 70)
    print("  OPTIMIZER BENCHMARK: AM Rush Hour (8 AM Monday)")
    print("=" * 70)
    print()

    # Define optimizers to test
    optimizers = {
        "Fixed 30/30 (Baseline)": FixedTimingController(),
        "Webster (70% ML)":       WebsterOptimizer(ml_weight=0.7),
        "Webster (50% ML)":       WebsterOptimizer(ml_weight=0.5),
        "Webster (90% ML)":       WebsterOptimizer(ml_weight=0.9),
        "Pure ML (100%)":         PureMLOptimizer(),
        "Pure Queue (0%)":        PureQueueOptimizer(),
        "Max Pressure":           MaxPressureOptimizer(),
        "Proportional (K=0.5)":   ProportionalOptimizer(gain=0.5),
        "Proportional (K=1.0)":   ProportionalOptimizer(gain=1.0),
    }

    results = {}

    print("Running simulations...")
    print()

    for name, optimizer in optimizers.items():
        print(f"  Running: {name}...", end=" ", flush=True)
        metrics = run_simulation(optimizer)
        results[name] = metrics
        print(f"Done (wait: {metrics['avg_wait']:.2f}s)")

    # Calculate improvements
    baseline_wait = results["Fixed 30/30 (Baseline)"]["avg_wait"]

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()
    print(f"{'Optimizer':<28} {'Avg Wait':>10} {'Improvement':>12} {'Departed':>10} {'Queue':>8}")
    print("-" * 70)

    # Sort by improvement
    sorted_results = sorted(
        results.items(),
        key=lambda x: (baseline_wait - x[1]["avg_wait"]) / max(baseline_wait, 0.01),
        reverse=True
    )

    for name, metrics in sorted_results:
        wait = metrics["avg_wait"]
        imp = (baseline_wait - wait) / max(baseline_wait, 0.01) * 100
        imp_str = f"+{imp:.1f}%" if imp >= 0 else f"{imp:.1f}%"
        status = "PASS" if imp >= 20 else ("---" if name.startswith("Fixed") else "FAIL")

        print(f"{name:<28} {wait:>8.2f}s {imp_str:>10}  {status}  {metrics['total_departed']:>6} {metrics['final_queue']:>6}")

    print("-" * 70)
    print()

    # Find best
    best_name, best_metrics = sorted_results[0]
    best_imp = (baseline_wait - best_metrics["avg_wait"]) / max(baseline_wait, 0.01) * 100

    print(f"BEST OPTIMIZER: {best_name}")
    print(f"  - Wait time: {best_metrics['avg_wait']:.2f}s (vs {baseline_wait:.2f}s baseline)")
    print(f"  - Improvement: {best_imp:.1f}%")
    print()

    # Summary insights
    print("KEY INSIGHTS:")
    print("-" * 40)

    # Compare Webster blends
    w70 = results["Webster (70% ML)"]["avg_wait"]
    w50 = results["Webster (50% ML)"]["avg_wait"]
    w90 = results["Webster (90% ML)"]["avg_wait"]
    print(f"  - Webster 70% ML: {w70:.2f}s")
    print(f"  - Webster 50% ML: {w50:.2f}s")
    print(f"  - Webster 90% ML: {w90:.2f}s")
    best_blend = min([(w70, 70), (w50, 50), (w90, 90)], key=lambda x: x[0])
    print(f"  -> Best blend: {best_blend[1]}% ML ({best_blend[0]:.2f}s)")
    print()

    # Compare algorithm families
    reactive_avg = np.mean([
        results["Pure Queue (0%)"]["avg_wait"],
        results["Max Pressure"]["avg_wait"],
        results["Proportional (K=0.5)"]["avg_wait"],
    ])
    predictive = results["Pure ML (100%)"]["avg_wait"]
    hybrid = results["Webster (70% ML)"]["avg_wait"]

    print(f"  - Reactive-only algorithms avg: {reactive_avg:.2f}s")
    print(f"  - Pure ML prediction: {predictive:.2f}s")
    print(f"  - Hybrid (Webster 70%): {hybrid:.2f}s")

    if hybrid < reactive_avg and hybrid < predictive:
        print("  -> HYBRID approach (ML + queue) performs best!")
    elif predictive < reactive_avg:
        print("  -> Prediction-based approaches outperform reactive-only.")
    else:
        print("  -> Reactive approaches competitive with prediction.")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
