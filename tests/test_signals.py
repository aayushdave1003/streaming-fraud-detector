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
    causal = ["pct_change", "spike_ratio", "rolling_cv", "seasonal_spike"]
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


# ── Holiday-seasonality confounder ──────────────────────────────────────
def _artist_row(**kw):
    base = dict(bot_streams=0.0, holiday_bot_streams=0.0, total_streams=1000.0,
                flagged_days=0, holiday_flagged_days=0, bot_pct=0.0,
                confidence=50.0, note="")
    base.update(kw)
    return base


def test_holiday_confounder_excuses_pure_seasonal_artist():
    # Almost all flags fall in the holiday window -> reclassified as legit.
    df = pd.DataFrame([_artist_row(
        bot_streams=800.0, holiday_bot_streams=800.0, total_streams=1000.0,
        flagged_days=10, holiday_flagged_days=10, bot_pct=80.0, confidence=90.0,
    )])
    out = run_pipeline.apply_holiday_confounder(df)
    assert out.loc[0, "bot_pct"] == 0.0          # holiday bot streams removed
    assert out.loc[0, "confidence"] < 90.0        # confidence down-weighted
    assert "Holiday" in out.loc[0, "note"]
    assert out.loc[0, "holiday_flag_share"] == 1.0


def test_holiday_confounder_leaves_year_round_artist_untouched():
    # A one-off December flag amid year-round activity -> low share -> untouched.
    df = pd.DataFrame([_artist_row(
        bot_streams=500.0, holiday_bot_streams=50.0, total_streams=1000.0,
        flagged_days=20, holiday_flagged_days=1, bot_pct=50.0, confidence=70.0,
    )])
    out = run_pipeline.apply_holiday_confounder(df)
    assert out.loc[0, "bot_pct"] == 50.0
    assert out.loc[0, "confidence"] == 70.0
    assert out.loc[0, "note"] == ""


# ── Release-window confounder ───────────────────────────────────────────
def _flagged_track(artist, title, n, flagged_idx):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    eb = [1 if i in flagged_idx else 0 for i in range(n)]
    return pd.DataFrame({"artist": artist, "title": title, "date": dates,
                         "ensemble_bot": eb, "streams": [100.0] * n})


def test_release_confounder_clears_launch_ramp():
    # All flags land in the first 14 days on chart -> reclassified as a launch ramp.
    out = run_pipeline.apply_release_confounder(_flagged_track("A", "Launch", 30, [0, 1, 2, 3]))
    assert out["ensemble_bot"].sum() == 0
    assert (out["release_adjusted"] == 1).all()


def test_release_confounder_keeps_sustained_flags():
    # Flags well outside the release window -> left untouched.
    out = run_pipeline.apply_release_confounder(_flagged_track("A", "Sustained", 30, [20, 21, 22, 23]))
    assert out["ensemble_bot"].sum() == 4
    assert (out["release_adjusted"] == 0).all()


def test_release_confounder_below_threshold_keeps_all():
    # Only 1 of 5 flags in-window (share 0.2 < 0.6) -> nothing cleared.
    out = run_pipeline.apply_release_confounder(_flagged_track("A", "Mixed", 30, [3, 18, 19, 20, 21]))
    assert out["ensemble_bot"].sum() == 5


# ── Injected-anomaly benchmark ──────────────────────────────────────────
def test_inject_labels_and_boosts():
    import benchmark
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    daily = signals.aggregate_daily(
        pd.DataFrame({"artist": "A", "title": "T", "date": dates, "streams": [100.0] * 60})
    )
    inj, chosen = benchmark.inject(daily, n_tracks=1, window=5, boost=4.0, rng=np.random.default_rng(0))
    assert inj["is_injected"].sum() == 5           # exactly one 5-day window labeled
    assert chosen == [("A", "T")]
    assert (inj.loc[inj["is_injected"] == 1, "streams"] == 400.0).all()   # boosted 4x
    assert (inj.loc[inj["is_injected"] == 0, "streams"] == 100.0).all()   # rest untouched


def test_inject_plateau_is_flat_elevated():
    import benchmark
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    daily = signals.aggregate_daily(
        pd.DataFrame({"artist": "A", "title": "T", "date": dates, "streams": [100.0] * 60})
    )
    inj, _ = benchmark.inject(daily, 1, 5, 4.0, np.random.default_rng(0), pattern="plateau")
    win = inj.loc[inj["is_injected"] == 1, "streams"]
    assert win.nunique() == 1                     # flat -> low variance (plateau signal)
    assert win.iloc[0] == pytest.approx(400.0)    # 4x the 100-mean baseline


# ── Hand-labeling harness ───────────────────────────────────────────────
def test_labeling_score_metrics():
    import labeling
    # bot&flagged=tp, bot¬flagged=fn, legit¬flagged=tn, legit&flagged=fp, unsure excluded
    df = pd.DataFrame({
        "label": ["bot", "bot", "legit", "legit", "unsure"],
        "flagged_days": [3, 0, 0, 2, 5],
        "bot_pct": [10.0, 0.0, 0.0, 1.0, 8.0],
    })
    m = labeling._score_labels(df)
    assert m["labeled"] == 4
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
