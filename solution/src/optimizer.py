"""
Traffic light timing controllers.

Multiple controllers are implemented for comparison:

  FixedTimingController
    The "dumb" baseline used by most real-world legacy systems.
    Every intersection gets the same fixed green split regardless of demand.
    Rule: 30 s NS green / 30 s EW green = 60 s cycle.
    Problem: wastes half the cycle on the light direction when
             demand is heavily asymmetric (e.g., AM rush is NS-dominant).

  WebsterOptimizer
    Uses Webster's (1958) optimal cycle formula to set green splits
    from ML-predicted demand + real-time queue feedback.
    Outperforms fixed timing whenever demand is asymmetric OR
    whenever the fixed 60 s cycle is too long for current volume.

  MaxPressureOptimizer
    Academic algorithm that allocates green proportional to queue "pressure".
    Provably stable but reactive-only (no prediction).

  ProportionalOptimizer
    Simple P-controller that adjusts from 30/30 based on queue imbalance.
    Easy to understand and implement.

  PureMLOptimizer
    Uses only ML predictions with no queue feedback.
    Good when predictions are accurate, slow to react to sudden changes.

  PureQueueOptimizer
    Reactive only, uses queue lengths with no ML prediction.
    Responds to current state but has no foresight.
"""

import numpy as np
from typing import Dict, List, Optional


DIRECTIONS        = ["north", "south", "east", "west"]
SATURATION_FLOW   = 0.5    # veh/s — max discharge rate when green (1800 veh/hr)
LOST_TIME_PER_PHASE = 2.0  # s — startup delay + clearance per phase change
MIN_GREEN         = 10.0   # s — minimum green (pedestrian safety floor)
MAX_CYCLE         = 120.0  # s — upper cap on cycle length
MIN_CYCLE         = 24.0   # s — 2 phases × MIN_GREEN + 2 × LOST_TIME


class FixedTimingController:
    """
    Baseline: identical fixed green split at every intersection.

    WHY IT'S SUBOPTIMAL
    -------------------
    With a 30/30 split and 0.5 veh/s saturation flow:
      Capacity per direction = 0.5 × (30/60) = 0.25 veh/s

    During an 8 AM rush where NS demand = 0.22 veh/s and EW = 0.05 veh/s:
      - NS gets 30 s of green → exactly enough, but red = 30 s → avg wait 7.5 s
      - EW gets 30 s of green → wildly over-provisioned; those 30 s are wasted
        instead of being given to NS, which reduces NS red time and wait.

    Webster's optimal split for the same demand: NS = 75% green, EW = 25%.
    With a 32 s cycle: NS red = 8 s → avg wait 2 s. That's the gain.
    """

    def __init__(self, ns_green: float = 30.0, ew_green: float = 30.0) -> None:
        self.ns_green = ns_green
        self.ew_green = ew_green

    def compute_timings(
        self,
        intersection_ids: List[int],
        **_kwargs,               # absorbs predicted_rates / queue_lengths
    ) -> Dict[int, Dict[str, float]]:
        return {
            iid: {"ns_green": self.ns_green, "ew_green": self.ew_green}
            for iid in intersection_ids
        }


class WebsterOptimizer:
    """
    Webster's (1958) optimal cycle length formula.

    MATHEMATICAL BASIS
    ------------------
    Webster minimised total intersection delay by finding the cycle length C
    that trades off two opposing costs:
      - Longer C → less lost time per vehicle (fewer phase switches)
      - Shorter C → shorter red periods → less wait per vehicle

    The optimal solution is:
        C* = (1.5 * L + 5) / (1 - Y)

    Where:
        L = total lost time = n_phases × LOST_TIME_PER_PHASE
            (time "wasted" on startup lag and amber clearance each cycle)

        Y = sum of y_i for each phase i
            y_i = q_i / s   (critical flow ratio for phase i)
            q_i = predicted arrival rate on the busiest approach in phase i
            s   = saturation flow rate (max discharge when green)

        Y physically means: "what fraction of green time is actually needed?"
        If Y = 0.6 the intersection is 60% loaded. Y ≥ 1.0 → over-capacity.

    GREEN SPLIT RULE
    ----------------
    Allocate effective green proportionally to flow ratios:
        g_i = (C* - L) × y_i / Y

    REAL-TIME BLEND
    ---------------
    Pure ML prediction is accurate but has a lag (updated once per minute).
    We blend with queue depth to react instantly to sudden demand spikes:
        effective_demand = α × ML_rate + (1-α) × queue_signal
        queue_signal = (queue_depth / ref_depth) × saturation_flow

    With α=0.7 the controller is primarily prediction-driven but
    extends green when a queue has already built up.
    """

    def __init__(self, n_phases: int = 2, ml_weight: float = 0.7) -> None:
        self.n_phases  = n_phases
        self.L         = n_phases * LOST_TIME_PER_PHASE   # total lost time
        self.ml_weight = ml_weight

    def compute_timings(
        self,
        intersection_ids: List[int],
        predicted_rates: Optional[Dict[int, Dict[str, float]]] = None,
        queue_lengths:   Optional[Dict[int, Dict[str, int]]]   = None,
    ) -> Dict[int, Dict[str, float]]:
        return {
            iid: self._one_intersection(iid, predicted_rates, queue_lengths)
            for iid in intersection_ids
        }

    def _one_intersection(self, iid, predicted_rates, queue_lengths):
        rates  = (predicted_rates or {}).get(iid, {d: 0.05 for d in DIRECTIONS})
        queues = (queue_lengths   or {}).get(iid, {d: 0    for d in DIRECTIONS})

        # --- Step 1: ML-predicted demand per phase ---
        # Take the maximum of the two approaches in each phase
        # (the critical approach determines how much green the phase needs)
        ns_pred = max(rates.get("north", 0.05), rates.get("south", 0.05))
        ew_pred = max(rates.get("east",  0.05), rates.get("west",  0.05))

        # --- Step 2: Queue-derived virtual demand signal ---
        # Deep queues mean recent arrivals exceeded capacity;
        # normalise by a reference depth of 20 vehicles.
        ns_q = (queues.get("north", 0) + queues.get("south", 0)) / 20.0 * SATURATION_FLOW
        ew_q = (queues.get("east",  0) + queues.get("west",  0)) / 20.0 * SATURATION_FLOW

        # --- Step 3: Blend prediction and queue signal ---
        a = self.ml_weight
        ns = max(a * ns_pred + (1 - a) * ns_q, 0.005)
        ew = max(a * ew_pred + (1 - a) * ew_q, 0.005)

        # --- Step 4: Compute flow ratios y_i = demand / saturation ---
        y_ns = ns / SATURATION_FLOW
        y_ew = ew / SATURATION_FLOW
        Y    = min(y_ns + y_ew, 0.97)   # cap at 0.97 to keep C finite

        # --- Step 5: Webster's optimal cycle length ---
        C = (1.5 * self.L + 5.0) / (1.0 - Y)
        C = float(np.clip(C, MIN_CYCLE, MAX_CYCLE))

        # --- Step 6: Green splits proportional to flow ratios ---
        effective_green = C - self.L
        g_ns = effective_green * y_ns / (y_ns + y_ew)
        g_ew = effective_green * y_ew / (y_ns + y_ew)

        # --- Step 7: Enforce minimum green (safety + pedestrian) ---
        if g_ns < MIN_GREEN:
            g_ew -= (MIN_GREEN - g_ns)
            g_ns  = MIN_GREEN
        if g_ew < MIN_GREEN:
            g_ns -= (MIN_GREEN - g_ew)
            g_ew  = MIN_GREEN

        return {
            "ns_green": float(np.clip(g_ns, MIN_GREEN, C - MIN_GREEN)),
            "ew_green": float(np.clip(g_ew, MIN_GREEN, C - MIN_GREEN)),
        }


class MaxPressureOptimizer:
    """
    Max Pressure algorithm — allocates green proportional to queue "pressure".

    ACADEMIC BACKGROUND
    -------------------
    Max Pressure is a well-studied algorithm in traffic control theory.
    It's provably stable (queues won't grow unbounded if demand < capacity).

    The idea: give green to the direction with highest "pressure", where
    pressure = queue_length × potential_discharge_rate.

    LIMITATION
    ----------
    Purely reactive — no prediction. Can't anticipate demand spikes.
    """

    def __init__(self, n_phases: int = 2) -> None:
        self.n_phases = n_phases
        self.L = n_phases * LOST_TIME_PER_PHASE

    def compute_timings(
        self,
        intersection_ids: List[int],
        predicted_rates: Optional[Dict[int, Dict[str, float]]] = None,
        queue_lengths:   Optional[Dict[int, Dict[str, int]]]   = None,
    ) -> Dict[int, Dict[str, float]]:
        return {
            iid: self._one_intersection(iid, queue_lengths)
            for iid in intersection_ids
        }

    def _one_intersection(self, iid, queue_lengths):
        queues = (queue_lengths or {}).get(iid, {d: 0 for d in DIRECTIONS})

        # Calculate "pressure" for each phase
        ns_queue = queues.get("north", 0) + queues.get("south", 0)
        ew_queue = queues.get("east", 0) + queues.get("west", 0)
        total_queue = max(ns_queue + ew_queue, 1)

        # Allocate green proportional to queue pressure
        ns_frac = ns_queue / total_queue
        ew_frac = ew_queue / total_queue

        # Fixed 60s cycle, split by pressure
        cycle = 60.0
        eff_green = cycle - self.L
        g_ns = eff_green * ns_frac
        g_ew = eff_green * ew_frac

        # Enforce minimums
        if g_ns < MIN_GREEN:
            g_ew -= (MIN_GREEN - g_ns)
            g_ns = MIN_GREEN
        if g_ew < MIN_GREEN:
            g_ns -= (MIN_GREEN - g_ew)
            g_ew = MIN_GREEN

        return {
            "ns_green": float(np.clip(g_ns, MIN_GREEN, cycle - MIN_GREEN)),
            "ew_green": float(np.clip(g_ew, MIN_GREEN, cycle - MIN_GREEN)),
        }


class ProportionalOptimizer:
    """
    Simple proportional controller based on queue imbalance.

    CONCEPT
    -------
    Start from a 30/30 baseline and adjust proportionally to queue difference:
        g_ns = 30 + K × (ns_queue - ew_queue)
        g_ew = 30 + K × (ew_queue - ns_queue)

    Where K is the proportional gain (default 0.5).

    ADVANTAGE
    ---------
    Very simple, easy to understand and tune. No complex math.

    LIMITATION
    ----------
    Purely reactive, no prediction. Can oscillate with high K values.
    """

    def __init__(self, gain: float = 0.5) -> None:
        self.K = gain
        self.base_green = 30.0

    def compute_timings(
        self,
        intersection_ids: List[int],
        predicted_rates: Optional[Dict[int, Dict[str, float]]] = None,
        queue_lengths:   Optional[Dict[int, Dict[str, int]]]   = None,
    ) -> Dict[int, Dict[str, float]]:
        return {
            iid: self._one_intersection(iid, queue_lengths)
            for iid in intersection_ids
        }

    def _one_intersection(self, iid, queue_lengths):
        queues = (queue_lengths or {}).get(iid, {d: 0 for d in DIRECTIONS})

        ns_queue = queues.get("north", 0) + queues.get("south", 0)
        ew_queue = queues.get("east", 0) + queues.get("west", 0)

        # Adjust from base proportionally to queue difference
        g_ns = self.base_green + self.K * (ns_queue - ew_queue)
        g_ew = self.base_green + self.K * (ew_queue - ns_queue)

        # Clamp to valid range
        g_ns = float(np.clip(g_ns, MIN_GREEN, MAX_CYCLE / 2))
        g_ew = float(np.clip(g_ew, MIN_GREEN, MAX_CYCLE / 2))

        return {"ns_green": g_ns, "ew_green": g_ew}


class PureMLOptimizer(WebsterOptimizer):
    """
    Webster optimizer with ml_weight=1.0 (no queue feedback).

    Uses only ML predictions to set timings. Good when predictions
    are accurate, but slow to react to sudden changes not in training data.
    """

    def __init__(self, n_phases: int = 2) -> None:
        super().__init__(n_phases=n_phases, ml_weight=1.0)


class PureQueueOptimizer(WebsterOptimizer):
    """
    Webster optimizer with ml_weight=0.0 (no ML prediction).

    Purely reactive controller based on current queue lengths.
    Responds immediately to demand but has no foresight.
    """

    def __init__(self, n_phases: int = 2) -> None:
        super().__init__(n_phases=n_phases, ml_weight=0.0)


# ================================================================== #
#  Convenience: get optimizer by name
# ================================================================== #

OPTIMIZER_REGISTRY = {
    "fixed":        FixedTimingController,
    "webster":      WebsterOptimizer,
    "webster_70":   lambda: WebsterOptimizer(ml_weight=0.7),
    "webster_50":   lambda: WebsterOptimizer(ml_weight=0.5),
    "pure_ml":      PureMLOptimizer,
    "pure_queue":   PureQueueOptimizer,
    "max_pressure": MaxPressureOptimizer,
    "proportional": ProportionalOptimizer,
}


def get_optimizer(name: str):
    """Get optimizer instance by name."""
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}. "
                         f"Available: {list(OPTIMIZER_REGISTRY.keys())}")
    factory = OPTIMIZER_REGISTRY[name]
    return factory() if callable(factory) else factory
