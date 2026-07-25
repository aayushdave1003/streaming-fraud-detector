"""Unit tests for signal engineering, the ensemble model, and confidence.

These cover exactly the logic that silently breaks under refactors: rolling /
groupby signals, the look-ahead boundary between retrospective and live modes,
collab splitting, confidence monotonicity, and model persistence.
"""
import numpy as np
import pandas as pd
import pytest

import config
import signals
import run_pipeline
from models import EnsembleDetector


def make_track(streams, artist="A", title="T", start="2020-01-01"):
    """Build a raw one-track frame with consecutive daily dates."""
    dates = pd.date_range(start, periods=len(streams), freq="D")
    return pd.DataFrame({
        "artist": artist, "title": title, "date": dates,
        "streams": np.asarray(streams, dtype=float),
    })


# ── Aggregation & basic signals ─────────────────────────────────────────
def test_aggregate_daily_sums_duplicates():
    df = pd.DataFrame({
        "artist": ["A", "A"], "title": ["T", "T"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "streams": [100.0, 50.0],
    })
    daily = signals.aggregate_daily(df)
    assert len(daily) == 1
    assert daily.loc[0, "streams"] == 150.0


def test_velocity_spike_detected():
    # 13 flat days then a 10x spike.
    s = [100] * 13 + [1000]
    daily = signals.engineer_signals(signals.aggregate_daily(make_track(s)))
    spike_row = daily.iloc[-1]
    assert spike_row["spike_ratio"] > config.SPIKE_RATIO_FLAG
    assert daily["spike_ratio"].iloc[:-1].max() < config.SPIKE_RATIO_FLAG


def test_plateau_has_low_cv():
    # Perfectly constant streams -> coefficient of variation ~ 0.
    daily = signals.engineer_signals(signals.aggregate_daily(make_track([500] * 20)))
    assert daily["rolling_cv"].dropna().max() < config.LOW_CV_FLAG


def test_seasonal_spike_within_month():
    s = [100] * 20 + [1500]  # all January
    daily = signals.engineer_signals(signals.aggregate_daily(make_track(s)))
    assert daily.iloc[-1]["seasonal_spike"] > config.SEASONAL_SPIKE_FLAG


def test_weekend_ratio_reflects_weekend_skew():
    # 2020-01-04/05 are Sat/Sun; make weekends much larger.
    df = make_track([100, 100, 100, 1000, 1000, 100, 100], start="2020-01-01")
    daily = signals.add_weekend_ratio(signals.aggregate_daily(df))
    assert daily["weekend_ratio"].iloc[0] > 1.0


# ── Look-ahead boundary ─────────────────────────────────────────────────
def test_live_mode_has_no_drop_after_spike():
    s = [100] * 13 + [1000, 100]
    daily = signals.engineer_signals(signals.aggregate_daily(make_track(s)), mode="live")
    assert (daily["drop_after_spike"] == 0).all()


def test_retrospective_mode_uses_future():
    s = [100] * 13 + [1000, 100]  # spike then collapse
    daily = signals.engineer_signals(signals.aggregate_daily(make_track(s)), mode="retrospective")
    assert daily["drop_after_spike"].sum() >= 1


def test_live_features_are_causal():
    """A past day's causal features must not change when future days are added."""
    s = [100, 120, 90, 300, 110, 105, 400, 95, 130, 100, 115, 250, 100, 108]
    full = signals.engineer_signals(signals.aggregate_daily(make_track(s)), mode="live")
    truncated = signals.engineer_signals(
        signals.aggregate_daily(make_track(s[:-1])), mode="live"
    )
    causal = ["pct_change", "spike_ratio", "rolling_cv"]
    merged = full.merge(truncated, on="date", suffixes=("_full", "_trunc"))
    for col in causal:
        a = merged[f"{col}_full"].to_numpy()
        b = merged[f"{col}_trunc"].to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"{col} changed when future was added"


# ── Collab splitting ────────────────────────────────────────────────────
def test_collab_split_regex():
    out = pd.Series(["Drake & Future", "Post Malone, Swae Lee", "SZA feat. Kendrick"]) \
        .str.split(config.COLLAB_SPLIT_REGEX, regex=True)
    assert out.iloc[0] == ["Drake", "Future"]
    assert out.iloc[1] == ["Post Malone", "Swae Lee"]
    assert out.iloc[2] == ["SZA", "Kendrick"]


# ── Confidence ──────────────────────────────────────────────────────────
def test_confidence_monotonic_in_flagged_share():
    total = pd.Series([100, 100])
    flagged = pd.Series([10, 90])
    spike = pd.Series([5.0, 5.0])
    drop = pd.Series([2, 2])
    conf = run_pipeline._confidence(flagged, total, spike, drop)
    assert conf.iloc[1] > conf.iloc[0]


def test_confidence_bounded():
    conf = run_pipeline._confidence(
        pd.Series([100]), pd.Series([100]), pd.Series([100.0]), pd.Series([100])
    )
    assert 0 <= conf.iloc[0] <= 100


# ── Ensemble model ──────────────────────────────────────────────────────
def test_ensemble_flags_injected_anomalies():
    rng = np.random.default_rng(0)
    normal = rng.normal(100, 5, size=(300, 5))
    spikes = np.tile([5000, 50, 40, 30, 0.9], (8, 1))  # obvious outliers
    X = np.vstack([normal, spikes])
    det = EnsembleDetector(contamination=0.05)
    _, _, ensemble = det.fit_predict_batch(X)
    assert ensemble.sum() > 0
    # At least some of the injected outliers should be caught.
    assert ensemble[-8:].sum() >= 1


def test_model_persistence_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(200, 5))
    det = EnsembleDetector(contamination=0.05)
    det.fit_predict_batch(X)
    path = tmp_path / "model.joblib"
    det.save(str(path))

    loaded = EnsembleDetector.load(str(path))
    new = rng.normal(0, 1, size=(10, 5))
    preds = loaded.predict(new)
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_predict_requires_fit():
    with pytest.raises(RuntimeError):
        EnsembleDetector().predict(np.zeros((3, 5)))
