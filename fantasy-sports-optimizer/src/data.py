"""
Data generation and feature engineering for the Fantasy Sports Team Optimizer.

This module provides two public classes:

- :class:`SyntheticDataGenerator` — produces a reproducible pool of 200 synthetic
  soccer players with 10 weeks of historical fantasy points.
- :class:`FeatureEngineer` — derives rolling statistics, trend signals, and position
  encodings from the raw player DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Position configuration (shared between generator and feature engineer)
# ---------------------------------------------------------------------------

#: Probability weights for each position (GK 10%, DEF 35%, MID 35%, FWD 20%).
POSITION_PROBS: dict[str, float] = {
    "GK": 0.10,
    "DEF": 0.35,
    "MID": 0.35,
    "FWD": 0.20,
}

#: Mean salary per position used for the truncated-normal salary draw.
POSITION_SALARY_MEANS: dict[str, float] = {
    "GK": 5.0,
    "DEF": 6.0,
    "MID": 7.5,
    "FWD": 8.0,
}

#: Poisson base rate (λ per week) per position before quality scaling.
POSITION_BASE_RATES: dict[str, int] = {
    "GK": 4,
    "DEF": 5,
    "MID": 6,
    "FWD": 7,
}

#: Salary standard deviation used for all positions.
SALARY_SIGMA: float = 1.5

#: Salary clip bounds [min, max].
SALARY_CLIP: tuple[float, float] = (4.0, 13.0)

#: Quality uniform distribution bounds [min, max].
QUALITY_BOUNDS: tuple[float, float] = (0.5, 2.0)

#: Number of real-world teams.
N_TEAMS: int = 20

#: Minimum and maximum players per team (inclusive).
TEAM_SIZE_BOUNDS: tuple[int, int] = (5, 15)

#: Number of historical weeks generated per player.
N_WEEKS: int = 10


class SyntheticDataGenerator:
    """Generate a reproducible pool of synthetic fantasy soccer players.

    All randomness is driven by a single :class:`numpy.random.Generator`
    created from the caller-supplied *seed*, so the output is fully
    deterministic for a given seed value.

    Example::

        gen = SyntheticDataGenerator()
        df = gen.generate_player_stats(n_players=200, seed=42)
    """

    def generate_player_stats(
        self,
        n_players: int = 200,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate a DataFrame of synthetic player statistics.

        Parameters
        ----------
        n_players:
            Total number of players to generate.  Defaults to 200.
        seed:
            Integer seed passed to :func:`numpy.random.default_rng` for full
            reproducibility.  Defaults to 42.

        Returns
        -------
        pandas.DataFrame
            One row per player with the following columns:

            * ``player_id`` — 0-indexed integer identifier.
            * ``name`` — string label ``"Player_0"`` … ``"Player_{n-1}"``.
            * ``position`` — one of ``"GK"``, ``"DEF"``, ``"MID"``, ``"FWD"``.
            * ``team`` — one of ``"Team_0"`` … ``"Team_19"``.
            * ``salary`` — float in ``[4.0, 13.0]``.
            * ``quality`` — float in ``[0.5, 2.0]``.
            * ``points_week_0`` … ``points_week_9`` — non-negative integers
              drawn from a Poisson distribution.

        Notes
        -----
        * Position distribution: GK 10 %, DEF 35 %, MID 35 %, FWD 20 %.
        * Salary is drawn from a truncated normal (σ = 1.5) centred on the
          position mean and clipped to ``[4.0, 13.0]``.
        * Quality is drawn from ``Uniform(0.5, 2.0)``.
        * Weekly points are drawn from ``Poisson(λ = quality × base_rate)``
          where base rates are GK 4, DEF 5, MID 6, FWD 7.
        * Team assignment uses a Dirichlet-weighted probability vector and is
          re-sampled until every team has between 5 and 15 players.
        """
        rng = np.random.default_rng(seed)

        positions_list = list(POSITION_PROBS.keys())
        position_probs = np.array([POSITION_PROBS[p] for p in positions_list])

        # ------------------------------------------------------------------
        # 1. Positions
        # ------------------------------------------------------------------
        positions = rng.choice(positions_list, size=n_players, p=position_probs)

        # ------------------------------------------------------------------
        # 2. Salaries — truncated normal per position, clipped to [4, 13]
        # ------------------------------------------------------------------
        salaries = np.empty(n_players, dtype=float)
        for pos in positions_list:
            mask = positions == pos
            n_pos = int(mask.sum())
            if n_pos == 0:
                continue
            raw = rng.normal(
                loc=POSITION_SALARY_MEANS[pos],
                scale=SALARY_SIGMA,
                size=n_pos,
            )
            salaries[mask] = np.clip(raw, SALARY_CLIP[0], SALARY_CLIP[1])

        # ------------------------------------------------------------------
        # 3. Quality — Uniform(0.5, 2.0)
        # ------------------------------------------------------------------
        qualities = rng.uniform(QUALITY_BOUNDS[0], QUALITY_BOUNDS[1], size=n_players)

        # ------------------------------------------------------------------
        # 4. Weekly points — Poisson(λ = quality × base_rate)
        # ------------------------------------------------------------------
        weekly_points = np.empty((n_players, N_WEEKS), dtype=int)
        for i, (pos, qual) in enumerate(zip(positions, qualities)):
            lam = qual * POSITION_BASE_RATES[pos]
            weekly_points[i] = rng.poisson(lam=lam, size=N_WEEKS)

        # ------------------------------------------------------------------
        # 5. Team assignment — Dirichlet-weighted, re-sample until valid
        # ------------------------------------------------------------------
        teams = self._assign_teams(rng, n_players)

        # ------------------------------------------------------------------
        # 6. Assemble DataFrame
        # ------------------------------------------------------------------
        data: dict = {
            "player_id": np.arange(n_players, dtype=int),
            "name": [f"Player_{i}" for i in range(n_players)],
            "position": positions,
            "team": teams,
            "salary": salaries,
            "quality": qualities,
        }
        for w in range(N_WEEKS):
            data[f"points_week_{w}"] = weekly_points[:, w]

        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_teams(rng: np.random.Generator, n_players: int) -> list[str]:
        """Assign players to teams using a Dirichlet-weighted draw.

        The assignment is re-sampled until every team has between
        :data:`TEAM_SIZE_BOUNDS` ``[0]`` and :data:`TEAM_SIZE_BOUNDS` ``[1]``
        players (inclusive).

        Parameters
        ----------
        rng:
            The seeded random generator to use for all draws.
        n_players:
            Total number of players to assign.

        Returns
        -------
        list[str]
            A list of length *n_players* where each element is a team label
            of the form ``"Team_k"`` (k ∈ 0 … 19).
        """
        team_labels = [f"Team_{k}" for k in range(N_TEAMS)]
        min_size, max_size = TEAM_SIZE_BOUNDS

        while True:
            # Draw a Dirichlet-weighted probability vector over teams
            alpha = rng.dirichlet(np.ones(N_TEAMS))
            team_indices = rng.choice(N_TEAMS, size=n_players, p=alpha)

            # Validate: every team must have between min_size and max_size players
            counts = np.bincount(team_indices, minlength=N_TEAMS)
            if counts.min() >= min_size and counts.max() <= max_size:
                return [team_labels[idx] for idx in team_indices]


class FeatureEngineer:
    """Derive engineered features from raw player statistics.

    Transforms a raw player DataFrame (as produced by
    :class:`SyntheticDataGenerator`) into a feature-augmented DataFrame
    suitable for model training.

    Example::

        fe = FeatureEngineer()
        df_features = fe.transform(df, target_week=9)
    """

    #: The 11 feature columns added by :meth:`transform`.
    FEATURE_COLUMNS: list[str] = [
        "rolling_avg_3",
        "rolling_std_3",
        "ewma_3",
        "salary",
        "pos_GK",
        "pos_DEF",
        "pos_MID",
        "pos_FWD",
        "player_quality_score",
        "form_trend",
        "value_per_cost",
    ]

    def transform(
        self,
        df: pd.DataFrame,
        target_week: int = 9,
    ) -> pd.DataFrame:
        """Add 11 engineered feature columns to *df* and return the result.

        Parameters
        ----------
        df:
            Raw player DataFrame as returned by
            :meth:`SyntheticDataGenerator.generate_player_stats`.
        target_week:
            The week index being predicted (0-indexed).  Features are
            computed from the three weeks immediately preceding this week.
            Defaults to 9.

        Returns
        -------
        pandas.DataFrame
            A copy of *df* augmented with exactly the 11 feature columns
            listed in :attr:`FEATURE_COLUMNS`.  No NaN values will be
            present in any feature column.

        Notes
        -----
        * Rolling features are computed over
          ``points_week_{target_week-3}`` … ``points_week_{target_week-1}``.
        * ``form_trend`` is the slope of a linear fit over the last 3 weeks.
        * ``player_quality_score`` = player all-time mean / position-group
          all-time mean.
        * Missing rolling features (< 3 weeks available) are imputed with
          the position-group mean, then the global mean as a fallback.
        * ``value_per_cost = rolling_avg_3 / salary``; set to 0 if
          ``salary ≤ 0``.
        """
        df = df.copy()

        # ------------------------------------------------------------------
        # Identify the three preceding week columns
        # ------------------------------------------------------------------
        week_cols = [f"points_week_{w}" for w in range(N_WEEKS)]
        available_weeks = [c for c in week_cols if c in df.columns]

        # Weeks strictly before target_week
        preceding = [
            f"points_week_{w}"
            for w in range(target_week)
            if f"points_week_{w}" in df.columns
        ]
        last_3 = preceding[-3:] if len(preceding) >= 3 else preceding

        # ------------------------------------------------------------------
        # Rolling statistics (row-wise over last_3 columns)
        # ------------------------------------------------------------------
        if last_3:
            pts_matrix = df[last_3].values.astype(float)
            df["rolling_avg_3"] = pts_matrix.mean(axis=1)
            df["rolling_std_3"] = pts_matrix.std(axis=1, ddof=0)
            # EWMA with span=3 — weights [1/6, 2/6, 3/6] for oldest→newest
            n_cols = pts_matrix.shape[1]
            alpha = 2.0 / (3 + 1)  # span=3 → alpha=0.5
            # Build weights for available columns (oldest first)
            weights = np.array(
                [(1 - alpha) ** (n_cols - 1 - k) for k in range(n_cols)]
            )
            weights /= weights.sum()
            df["ewma_3"] = pts_matrix @ weights
        else:
            df["rolling_avg_3"] = np.nan
            df["rolling_std_3"] = np.nan
            df["ewma_3"] = np.nan

        # ------------------------------------------------------------------
        # form_trend — slope of linear fit over last 3 weeks
        # ------------------------------------------------------------------
        if len(last_3) >= 2:
            x = np.arange(len(last_3), dtype=float)
            pts_matrix = df[last_3].values.astype(float)
            # Vectorised polyfit slope: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
            n = len(x)
            sum_x = x.sum()
            sum_x2 = (x ** 2).sum()
            denom = n * sum_x2 - sum_x ** 2
            if denom != 0:
                sum_y = pts_matrix.sum(axis=1)
                sum_xy = pts_matrix @ x
                df["form_trend"] = (n * sum_xy - sum_x * sum_y) / denom
            else:
                df["form_trend"] = 0.0
        else:
            df["form_trend"] = np.nan

        # ------------------------------------------------------------------
        # player_quality_score — player all-time mean / position-group mean
        # ------------------------------------------------------------------
        all_week_cols = [c for c in available_weeks if c in df.columns]
        if all_week_cols:
            player_mean = df[all_week_cols].mean(axis=1)
            pos_group_mean = player_mean.groupby(df["position"]).transform("mean")
            # Avoid division by zero
            df["player_quality_score"] = np.where(
                pos_group_mean != 0, player_mean / pos_group_mean, 1.0
            )
        else:
            df["player_quality_score"] = 1.0

        # ------------------------------------------------------------------
        # One-hot encode position
        # ------------------------------------------------------------------
        for pos in ["GK", "DEF", "MID", "FWD"]:
            df[f"pos_{pos}"] = (df["position"] == pos).astype(int)

        # ------------------------------------------------------------------
        # value_per_cost
        # ------------------------------------------------------------------
        salary = df["salary"].values.astype(float)
        rolling_avg = df["rolling_avg_3"].values.astype(float)
        df["value_per_cost"] = np.where(salary > 0, rolling_avg / salary, 0.0)

        # ------------------------------------------------------------------
        # Impute NaN values in rolling features
        # ------------------------------------------------------------------
        rolling_features = ["rolling_avg_3", "rolling_std_3", "ewma_3", "form_trend"]
        for feat in rolling_features:
            if df[feat].isna().any():
                # Position-group mean imputation
                pos_means = df.groupby("position")[feat].transform("mean")
                df[feat] = df[feat].fillna(pos_means)
                # Global mean fallback
                global_mean = df[feat].mean()
                if pd.isna(global_mean):
                    global_mean = 0.0
                df[feat] = df[feat].fillna(global_mean)

        # Re-compute value_per_cost after imputation (rolling_avg_3 may have changed)
        rolling_avg = df["rolling_avg_3"].values.astype(float)
        df["value_per_cost"] = np.where(salary > 0, rolling_avg / salary, 0.0)

        # Final NaN guard — replace any remaining NaN with 0
        for feat in self.FEATURE_COLUMNS:
            if df[feat].isna().any():
                df[feat] = df[feat].fillna(0.0)

        return df
