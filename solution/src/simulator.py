"""
Traffic Simulator — built on the starter code base
====================================================

The starter code (problem1_traffic/traffic_simulator.py) provided:
  LightState    — RED / GREEN / YELLOW enum
  Vehicle       — id, position, destination, speed, wait_time
  Intersection  — id, position, light_states, timer
  TrafficSimulator skeleton — spawn_vehicle, step, update_lights,
                              get_state, get_metrics (all TODO)
  TrafficDataGenerator — generate_daily_patterns, add_weather_effects

This file implements all of those TODOs using queue-based physics,
extends the data structures with per-approach vehicle queues, and
adds the timing logic needed by the Webster optimizer.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


# ================================================================== #
#  Starter code base structures (kept verbatim from the skeleton)     #
# ================================================================== #

class LightState(Enum):
    """Traffic light states — from starter code."""
    RED    = 0
    GREEN  = 1
    YELLOW = 2


@dataclass
class Vehicle:
    """
    Represents a vehicle queued at an intersection approach.
    From starter code — extended with arrived_at for wait-time tracking.

    position   : (x, y) coordinate of the approach lane this vehicle is in
    destination: (x, y) of the next intersection this vehicle is heading to
    speed      : discharge speed when light is green (vehicles/second)
    wait_time  : cumulative seconds spent waiting at a RED light
    arrived_at : simulation time (s) when the vehicle joined the queue
    """
    id: int
    position: Tuple[float, float]
    destination: Tuple[float, float]
    speed: float
    wait_time: float = 0.0
    arrived_at: float = 0.0


@dataclass
class Intersection:
    """
    Represents a traffic intersection.
    From starter code — extended with per-approach vehicle queues and
    phase-duration tracking.

    light_states : direction → LightState (which directions are green/red)
    queues       : direction → list of Vehicle objects waiting at that approach
    timer        : seconds elapsed in the current phase
    ns_green_dur : how many seconds the NS phase should last (set by optimizer)
    ew_green_dur : how many seconds the EW phase should last (set by optimizer)
    """
    id: int
    position: Tuple[float, float]
    light_states: Dict[str, LightState] = field(default_factory=lambda: {
        'north': LightState.GREEN,
        'south': LightState.GREEN,
        'east':  LightState.RED,
        'west':  LightState.RED,
    })
    timer: float = 0.0
    # Per-approach vehicle queues (not in starter code — added here)
    queues: Dict[str, List[Vehicle]] = field(default_factory=lambda: {
        d: [] for d in ('north', 'south', 'east', 'west')
    })
    # Phase durations — updated by update_lights() / Webster optimizer
    ns_green_dur: float = 30.0
    ew_green_dur: float = 30.0
    # Currently serving which phase?
    _current_phase: str = 'NS'   # 'NS' or 'EW'

    def green_dirs(self) -> List[str]:
        """Directions currently showing GREEN."""
        return [d for d, s in self.light_states.items()
                if s == LightState.GREEN]

    def red_dirs(self) -> List[str]:
        """Directions currently showing RED."""
        return [d for d, s in self.light_states.items()
                if s == LightState.RED]

    def queue_lengths(self) -> Dict[str, int]:
        """Number of vehicles waiting per direction."""
        return {d: len(q) for d, q in self.queues.items()}


# ================================================================== #
#  Physics constants                                                   #
# ================================================================== #

SATURATION_RATE = 0.5   # vehicles/second discharged per green approach
DT              = 1.0   # simulation timestep (seconds)
VEHICLE_SPEED   = 10.0  # m/s — used for position tracking in animation


# ================================================================== #
#  TrafficSimulator — implements all starter-code TODOs               #
# ================================================================== #

class TrafficSimulator:
    """
    Simulates traffic flow through a network of intersections.

    Implements the starter-code skeleton methods:
      spawn_vehicle()  — Poisson arrivals each timestep
      step()           — advances simulation by DT seconds
      update_lights()  — applies optimizer's green-time decisions
      get_state()      — extracts ML feature vector
      get_metrics()    — returns avg_wait, throughput, queue total

    Physics model (queue-based, same approach as starter code suggests):
      - Vehicles arrive at each approach as a Poisson process
      - When a direction is GREEN, vehicles depart at SATURATION_RATE
      - Every queued vehicle accumulates wait_time each second it waits
      - Phase transitions happen when the current phase timer expires
    """

    def __init__(self, num_intersections: int = 4, grid_size: int = 2,
                 seed: int = 42):
        np.random.seed(seed)
        self._seed           = seed
        self._vehicle_id_ctr = 0
        self.grid_size       = grid_size
        self.intersections   = self._create_intersection_grid(num_intersections)
        self.vehicles        = []          # all active vehicles across network
        self.time            = 0.0
        self.total_departed  = 0
        self.total_wait_time = 0.0

        # Incident tracking — {intersection_id: {direction: capacity_factor}}
        # e.g. {0: {'north': 0.2}} means north approach at int 0 is 80% blocked
        self._incidents: Dict[int, Dict[str, float]] = {}

    # ---------------------------------------------------------------- #
    #  Starter code method: _create_intersection_grid                   #
    # ---------------------------------------------------------------- #

    def _create_intersection_grid(self, n: int) -> List[Intersection]:
        """
        Create a 2×2 grid of intersections.
        Positions are on a unit grid: (col, row).
        """
        side = max(1, int(np.ceil(np.sqrt(n))))
        intersections = []
        for i in range(n):
            col, row = i % side, i // side
            intersections.append(Intersection(
                id=i,
                position=(float(col), float(row)),
                light_states={
                    'north': LightState.GREEN,
                    'south': LightState.GREEN,
                    'east':  LightState.RED,
                    'west':  LightState.RED,
                },
            ))
        return intersections

    # ---------------------------------------------------------------- #
    #  Starter code method: spawn_vehicle  (implemented)                #
    # ---------------------------------------------------------------- #

    def spawn_vehicle(self, arrival_rates: Dict[int, Dict[str, float]]) -> None:
        """
        Spawn new vehicles based on Poisson arrival rates.

        For each (intersection, direction) pair, draw a Poisson random
        variable with the predicted arrival rate × DT.  Each drawn
        vehicle becomes a Vehicle object added to that approach's queue.

        arrival_rates : {intersection_id: {direction: rate_in_veh_per_s}}
                        Comes from the ML model's predict_rates() output.
        """
        for inter in self.intersections:
            iid = inter.id
            for direction, queue in inter.queues.items():
                rate = arrival_rates.get(iid, {}).get(direction, 0.03)

                # Apply incident capacity reduction if active
                cap = self._incidents.get(iid, {}).get(direction, 1.0)
                rate *= cap

                n_arrivals = int(np.random.poisson(rate * DT))

                # Approach position (slightly offset from intersection centre)
                cx, cy  = inter.position
                offsets = {'north': (cx, cy - 0.4),
                           'south': (cx, cy + 0.4),
                           'east':  (cx + 0.4, cy),
                           'west':  (cx - 0.4, cy)}
                pos = offsets[direction]
                dest = inter.position   # vehicle is heading through this intersection

                for _ in range(n_arrivals):
                    v = Vehicle(
                        id          = self._vehicle_id_ctr,
                        position    = pos,
                        destination = dest,
                        speed       = VEHICLE_SPEED,
                        arrived_at  = self.time,
                    )
                    self._vehicle_id_ctr += 1
                    queue.append(v)
                    self.vehicles.append(v)

    # ---------------------------------------------------------------- #
    #  Starter code method: step  (implemented)                         #
    # ---------------------------------------------------------------- #

    def step(self, arrival_rates: Dict[int, Dict[str, float]]) -> None:
        """
        Advance the simulation by DT seconds.

        Implements the starter-code TODO:
        1. Spawn vehicles (Poisson arrivals)
        2. Discharge vehicles from GREEN approaches (saturation flow)
        3. Accumulate wait time for all queued vehicles
        4. Update phase timers and switch phases when due
        """
        # 1. Arrivals
        self.spawn_vehicle(arrival_rates)

        for inter in self.intersections:
            # 2. Departures — green approaches discharge at SATURATION_RATE
            for direction in inter.green_dirs():
                queue = inter.queues[direction]
                if queue:
                    # Stochastic discharge: 0.5 veh/s → 0 or 1 per second
                    can_depart = int(SATURATION_RATE * DT + np.random.random())
                    n_depart   = min(len(queue), can_depart)
                    departed   = queue[:n_depart]
                    inter.queues[direction] = queue[n_depart:]

                    # Record wait times and remove from active vehicle list
                    for v in departed:
                        self.total_wait_time += v.wait_time
                        self.total_departed  += 1
                        if v in self.vehicles:
                            self.vehicles.remove(v)

            # 3. Accumulate wait time for vehicles still in RED queues
            for direction in inter.red_dirs():
                for v in inter.queues[direction]:
                    v.wait_time += DT

            # Also accumulate for vehicles in GREEN queue waiting behind others
            for direction in inter.green_dirs():
                for v in inter.queues[direction]:
                    v.wait_time += DT

            # 4. Phase timer and transition
            inter.timer += DT
            phase_dur = (inter.ns_green_dur if inter._current_phase == 'NS'
                         else inter.ew_green_dur)

            if inter.timer >= phase_dur:
                inter.timer = 0.0
                if inter._current_phase == 'NS':
                    inter._current_phase = 'EW'
                    inter.light_states = {
                        'north': LightState.RED,
                        'south': LightState.RED,
                        'east':  LightState.GREEN,
                        'west':  LightState.GREEN,
                    }
                else:
                    inter._current_phase = 'NS'
                    inter.light_states = {
                        'north': LightState.GREEN,
                        'south': LightState.GREEN,
                        'east':  LightState.RED,
                        'west':  LightState.RED,
                    }

        self.time += DT

    # ---------------------------------------------------------------- #
    #  Starter code method: update_lights  (implemented)                #
    # ---------------------------------------------------------------- #

    def update_lights(self,
                      light_timings: Dict[int, Dict[str, float]]) -> None:
        """
        Update traffic light timing based on the optimizer's decisions.

        Implements the starter-code TODO: 'This is where your optimization comes in!'

        light_timings : {intersection_id: {'ns_green': float, 'ew_green': float}}
                        Output of WebsterOptimizer.compute_timings() or
                        FixedTimingController.compute_timings().
        """
        for inter in self.intersections:
            t = light_timings.get(inter.id, {})
            inter.ns_green_dur = t.get('ns_green', 30.0)
            inter.ew_green_dur = t.get('ew_green', 30.0)

    # ---------------------------------------------------------------- #
    #  Starter code method: get_state  (implemented)                    #
    # ---------------------------------------------------------------- #

    def get_state(self) -> np.ndarray:
        """
        Extract the current simulation state as a feature vector for ML.

        Implements the starter-code TODO: 'Extract features for ML model.
        Example features: queue lengths, waiting times, time of day, etc.'

        Returns a flat numpy array:
          [q_north_0, q_south_0, q_east_0, q_west_0,   ← int 0 queue lengths
           q_north_1, ...                                ← int 1 queue lengths
           ...
           avg_wait_0, avg_wait_1, ...]                 ← per-intersection avg wait
        """
        features = []
        for inter in self.intersections:
            qlens = inter.queue_lengths()
            features.extend([
                qlens['north'], qlens['south'],
                qlens['east'],  qlens['west'],
            ])
            # Average wait time of vehicles currently in this intersection's queues
            all_vehicles = [v for q in inter.queues.values() for v in q]
            avg_wait = (np.mean([v.wait_time for v in all_vehicles])
                        if all_vehicles else 0.0)
            features.append(avg_wait)

        return np.array(features, dtype=float)

    # ---------------------------------------------------------------- #
    #  Starter code method: get_metrics  (implemented)                  #
    # ---------------------------------------------------------------- #

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Implements the starter-code TODO: 'Compute average wait time,
        throughput, etc.'

        Returns dict with keys matching the starter-code skeleton:
          avg_wait_time  : total vehicle-seconds waited / total departed
          throughput     : vehicles departed per second (overall rate)
          total_queue    : vehicles currently queued across all approaches
          total_departed : cumulative count of vehicles that cleared
          time           : current simulation time (s)
        """
        avg_wait = (self.total_wait_time / self.total_departed
                    if self.total_departed > 0 else 0.0)

        total_queue = sum(
            len(q)
            for inter in self.intersections
            for q in inter.queues.values()
        )

        return {
            'avg_wait_time':  avg_wait,
            'throughput':     self.total_departed / max(self.time, 1.0),
            'total_queue':    total_queue,
            'total_departed': self.total_departed,
            'time':           self.time,
        }

    # ---------------------------------------------------------------- #
    #  Public helpers                                                    #
    # ---------------------------------------------------------------- #

    def get_queue_lengths(self) -> Dict[int, Dict[str, int]]:
        """Return queue lengths per intersection per direction."""
        return {inter.id: inter.queue_lengths()
                for inter in self.intersections}

    def set_timings(self, timings: Dict[int, Dict[str, float]]) -> None:
        """Alias for update_lights() — used by existing main.py calls."""
        self.update_lights(timings)

    def trigger_incident(self, intersection_id: int, direction: str,
                         capacity_factor: float = 0.2) -> None:
        """
        Simulate an unexpected event (accident, road closure).

        Sets the effective arrival capacity on one approach to
        capacity_factor × normal, simulating a blockage.

        intersection_id : which intersection is affected
        direction       : 'north' / 'south' / 'east' / 'west'
        capacity_factor : 0.0 = total closure; 0.2 = 80% blocked
        """
        if intersection_id not in self._incidents:
            self._incidents[intersection_id] = {}
        self._incidents[intersection_id][direction] = capacity_factor

    def clear_incident(self, intersection_id: int,
                       direction: Optional[str] = None) -> None:
        """Remove an incident — road reopens to full capacity."""
        if direction:
            self._incidents.get(intersection_id, {}).pop(direction, None)
        else:
            self._incidents.pop(intersection_id, None)

    def reset(self) -> None:
        """Reset the simulator to its initial state."""
        self.__init__(
            num_intersections=len(self.intersections),
            grid_size=self.grid_size,
            seed=self._seed,
        )


# ================================================================== #
#  TrafficDataGenerator — extends starter-code class                  #
# ================================================================== #

class TrafficDataGenerator:
    """
    Generate synthetic traffic data for ML training.

    Extends the starter-code TrafficDataGenerator with properly
    implemented methods that use SyntheticDataGenerator from
    synthetic_data.py as the base, then layer on realistic patterns.

    The starter code provided the method signatures and the intent:
      generate_daily_patterns()  — two-peak weekday pattern
      add_weather_effects()      — rain/cloud multiplier

    This implementation fills in the TODO bodies.
    """

    @staticmethod
    def generate_daily_patterns(days: int = 30) -> np.ndarray:
        """
        Generate realistic hourly traffic patterns (vehicles arriving per hour).

        Implements the starter-code TODO:
        - Morning rush (7-9am)    ← Gaussian peak centred at 8:00
        - Evening rush (5-7pm)    ← Gaussian peak centred at 17:30
        - Weekend patterns         ← lower, flat midday Gaussian
        - Special events           ← captured via log-normal noise

        Returns hourly vehicle counts (float) for `days` days.
        """
        rng    = np.random.default_rng(42)
        hours  = days * 24
        result = np.zeros(hours)

        for i in range(hours):
            h   = i % 24
            dow = (i // 24) % 7
            is_weekend = dow >= 5

            if is_weekend:
                # Weekend: flat midday Gaussian
                base = 30 + 20 * np.exp(-((h - 13) ** 2) / 18)
            else:
                # Weekday: two-peak Gaussian (Rule A from data generator)
                morning = 80 * np.exp(-((h - 8.0)  ** 2) / 2.0)
                evening = 64 * np.exp(-((h - 17.5) ** 2) / 3.0)
                base    = 6 + morning + evening

            # Day-to-day noise (Rule F)
            noise    = float(rng.lognormal(0.0, 0.08))
            result[i] = max(1.0, base * noise)

        return result

    @staticmethod
    def add_weather_effects(traffic: np.ndarray,
                            weather_data: np.ndarray = None) -> np.ndarray:
        """
        Modify traffic based on weather conditions.

        Implements the starter-code TODO: 'Add weather impact (rain -> more traffic)'

        weather_data : optional array of weather codes (0=clear, 1=cloudy, 2=rain)
                       per hour.  If None, randomly generated.
        Returns traffic array modified by weather multipliers.
        """
        rng = np.random.default_rng(99)
        n   = len(traffic)
        mult_map = {0: 1.00, 1: 1.05, 2: 1.25}   # clear / cloudy / rain

        if weather_data is None:
            # Draw one weather type per day (55% clear, 30% cloudy, 15% rain)
            weather_data = np.zeros(n, dtype=int)
            for day in range(n // 24 + 1):
                wtype = rng.choice([0, 1, 2], p=[0.55, 0.30, 0.15])
                start = day * 24
                end   = min(start + 24, n)
                weather_data[start:end] = wtype

        multipliers = np.array([mult_map.get(int(w), 1.0) for w in weather_data])
        return traffic * multipliers
