"""Fraud-signal engineering.

Each signal is a small pure function operating on a daily
``(artist, title, date, streams)`` DataFrame so it can be unit-tested in
isolation. ``engineer_signals`` composes them and honours a scoring ``mode``:

* ``retrospective`` — batch audit of historical data; may use look-ahead
  signals (e.g. a spike followed by a next-day collapse).
* ``live`` — scoring a partial series up to "today"; look-ahead signals are
  disallowed so a day's features never depend on the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

GROUP = ["artist", "title"]


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw chart rows to one streams total per (artist, title, date)."""
    daily = df.groupby(["artist", "title", "date"], as_index=False)["streams"].sum()
    daily = daily.sort_values(["artist", "title", "date"]).reset_index(drop=True)
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily["day_of_week"] = daily["date"].dt.dayofweek
    return daily


def add_velocity(daily: pd.DataFrame) -> pd.DataFrame:
    """Stream-velocity spike: today's streams vs the trailing rolling average."""
    g = daily.groupby(GROUP)["streams"]
    daily["prev_streams"] = g.shift(1)
    daily["pct_change"] = daily["streams"].div(daily["prev_streams"].replace(0, np.nan)) - 1
    daily["rolling_7d_avg"] = g.transform(
        lambda x: x.rolling(config.ROLLING_WINDOW, min_periods=1).mean()
    )
    daily["spike_ratio"] = daily["streams"] / daily["rolling_7d_avg"].replace(0, np.nan)
    return daily


def add_drop_after_spike(daily: pd.DataFrame) -> pd.DataFrame:
    """LOOK-AHEAD signal: a spike immediately followed by a collapse.

    Uses the *next* day's streams, so it is valid only for retrospective
    analysis and must never be used for live scoring.
    """
    nxt = daily.groupby(GROUP)["streams"].shift(-1)
    daily["drop_after_spike"] = (
        (daily["spike_ratio"] > config.SPIKE_RATIO_FLAG)
        & (nxt < daily["rolling_7d_avg"])
    ).astype(float)
    return daily


def add_seasonal(daily: pd.DataFrame, mode: str = "retrospective") -> pd.DataFrame:
    """Seasonal correction: streams vs the artist's average for that month.

    Retrospective uses the full-month mean (peeks within the month — fine for a
    batch audit). Live uses a *causal* expanding mean of the artist's same-month
    streams up to each date, so a day's signal never depends on the future.
    """
    if mode == "live":
        d = daily.sort_values(["artist", "month", "date"])
        cum = d.groupby(["artist", "month"])["streams"].cumsum()
        cnt = d.groupby(["artist", "month"]).cumcount() + 1
        monthly_avg = (cum / cnt).reindex(daily.index)
    else:
        monthly_avg = daily.groupby(["artist", "month"])["streams"].transform("mean")
    daily["seasonal_spike"] = daily["streams"] / monthly_avg.replace(0, np.nan)
    return daily


def add_plateau(daily: pd.DataFrame) -> pd.DataFrame:
    """Plateau detection: unnaturally low variance (bots stream at flat levels)."""
    g = daily.groupby(GROUP)["streams"]
    daily["rolling_std"] = g.transform(
        lambda x: x.rolling(config.ROLLING_WINDOW, min_periods=3).std()
    )
    daily["rolling_cv"] = daily["rolling_std"] / daily["rolling_7d_avg"].replace(0, np.nan)
    return daily


def add_weekend_ratio(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekend-vs-weekday mean streams per track.

    Humans skew toward weekends; bots run flat 24/7. Reimplemented with a clean
    ``unstack`` instead of the original nested-index lambda, which was fragile
    and hard to reason about.
    """
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    means = (
        daily.groupby(GROUP + ["is_weekend"])["streams"].mean().unstack("is_weekend")
    )
    weekday = means.get(0)
    weekend = means.get(1)
    if weekday is None or weekend is None:
        daily["weekend_ratio"] = 0.0
        return daily
    ratio = (weekend / (weekday + 1)).rename("weekend_ratio")
    daily = daily.merge(ratio, on=GROUP, how="left")
    daily["weekend_ratio"] = daily["weekend_ratio"].fillna(0.0)
    return daily


def engineer_signals(daily: pd.DataFrame, mode: str = "retrospective",
                     weekend: bool = False) -> pd.DataFrame:
    """Run the full signal stack for a given mode and drop unusable rows."""
    daily = add_velocity(daily)
    daily = add_seasonal(daily, mode)
    daily = add_plateau(daily)
    if mode == "retrospective":
        daily = add_drop_after_spike(daily)
    else:
        # Look-ahead signal is unavailable when scoring live data.
        daily["drop_after_spike"] = 0.0
    if weekend:
        daily = add_weekend_ratio(daily)
    daily = daily.dropna(
        subset=["pct_change", "spike_ratio", "seasonal_spike", "rolling_cv"]
    )
    return daily.reset_index(drop=True)


def build_feature_matrix(daily: pd.DataFrame, mode: str = "retrospective") -> pd.DataFrame:
    """Select the model features for ``mode`` and sanitise inf/NaN."""
    feats = config.features_for(mode)
    X = daily[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


# ── Explainability ──────────────────────────────────────────────────────
# Which human-readable reason each flagged day fired on. Vectorised so it is
# cheap at daily scale.
REASON_COLUMNS = {
    "r_velocity_spike": "velocity_spike",
    "r_seasonal_spike": "seasonal_spike",
    "r_plateau": "plateau_low_variance",
    "r_drop_after_spike": "drop_after_spike",
}


def add_reason_flags(daily: pd.DataFrame, flagged_col: str = "ensemble_bot") -> pd.DataFrame:
    """Add per-day boolean reason columns for records the ensemble flagged."""
    flagged = daily[flagged_col] == 1
    daily["r_velocity_spike"] = (
        (daily["spike_ratio"] > config.SPIKE_RATIO_FLAG) & flagged
    ).astype(int)
    daily["r_seasonal_spike"] = (
        (daily["seasonal_spike"] > config.SEASONAL_SPIKE_FLAG) & flagged
    ).astype(int)
    daily["r_plateau"] = (
        (daily["rolling_cv"] > 0) & (daily["rolling_cv"] < config.LOW_CV_FLAG) & flagged
    ).astype(int)
    drop = daily["drop_after_spike"] if "drop_after_spike" in daily else 0
    daily["r_drop_after_spike"] = ((drop > 0) & flagged).astype(int)
    return daily


def summarise_reasons(row: pd.Series) -> str:
    """Turn per-group reason counts into a ranked, human-readable string."""
    counts = {label: row.get(col, 0) for col, label in REASON_COLUMNS.items()}
    ranked = [label for label, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True) if n > 0]
    return ", ".join(ranked[:3]) if ranked else "—"
