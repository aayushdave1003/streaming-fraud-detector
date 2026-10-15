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


# ── One-off event-spike confounder ──────────────────────────────────────
def _artist_span(**kw):
    base = dict(flagged_days=10, bot_streams=800.0, total_streams=1000.0, bot_pct=80.0,
                confidence=50.0, note="")
    base.update(kw)
    return base


def test_oneoff_confounder_clears_isolated_burst():
    # A 20-day flagged burst inside a 4-year career -> event, not fraud.
    d = pd.Timestamp("2018-06-01")
    df = pd.DataFrame([_artist_span(
        first_active=pd.Timestamp("2016-01-01"), last_active=pd.Timestamp("2020-01-01"),
        first_flag=d, last_flag=d + pd.Timedelta(days=20),
    )])
    out = run_pipeline.apply_oneoff_confounder(df)
    assert out.loc[0, "bot_pct"] == 0.0
    assert out.loc[0, "confidence"] < 50.0
    assert "One-off" in out.loc[0, "note"]


def test_oneoff_confounder_keeps_recurring():
    # Flags span ~2 years -> sustained/recurring, left untouched (e.g. BTS comebacks).
    df = pd.DataFrame([_artist_span(
        bot_pct=50.0, bot_streams=500.0,
        first_active=pd.Timestamp("2018-01-01"), last_active=pd.Timestamp("2021-01-01"),
        first_flag=pd.Timestamp("2018-06-01"), last_flag=pd.Timestamp("2020-06-01"),
    )])
    out = run_pipeline.apply_oneoff_confounder(df)
    assert out.loc[0, "bot_pct"] == 50.0


def test_oneoff_confounder_keeps_short_career():
    # Short burst but also a short career (<1yr) -> can't call it a one-off event.
    d = pd.Timestamp("2019-06-01")
    df = pd.DataFrame([_artist_span(
        first_active=pd.Timestamp("2019-05-01"), last_active=pd.Timestamp("2019-07-01"),
        first_flag=d, last_flag=d + pd.Timedelta(days=10),
    )])
    out = run_pipeline.apply_oneoff_confounder(df)
    assert out.loc[0, "bot_pct"] == 80.0


def test_oneoff_confounder_keeps_sparse_flags():
    # Only 2 flagged days -> too sparse to be a sustained event burst; protects
    # the small real signal on sparse data (e.g. BTS's 2 flagged US/Global days).
    d = pd.Timestamp("2018-06-01")
    df = pd.DataFrame([_artist_span(
        flagged_days=2, bot_pct=30.0,
        first_active=pd.Timestamp("2016-01-01"), last_active=pd.Timestamp("2020-01-01"),
        first_flag=d, last_flag=d,
    )])
    out = run_pipeline.apply_oneoff_confounder(df)
    assert out.loc[0, "bot_pct"] == 30.0


# ── Track-level confounder parity ───────────────────────────────────────
# The holiday and one-off confounders used to run on the artist table only, so
# fraud_results_tracks.csv shipped uncorrected: a Christmas record flagged every
# Nov-Dec sat at the top of the track suspect list while the artist table had
# already excused it. These pin both levels to the same correction.

def _daily_for_track_level(dates, streams, flags, holiday_months=(11, 12)):
    """Minimal daily frame carrying every column track_level aggregates."""
    d = pd.DataFrame({
        "artist": "A", "title": "T",
        "date": pd.to_datetime(list(dates)),
        "streams": np.asarray(streams, dtype=float),
        "ensemble_bot": np.asarray(flags, dtype=int),
    })
    d["month"] = d["date"].dt.month
    d["bot_stream_val"] = d["streams"] * d["ensemble_bot"]
    d["spike_ratio"] = 2.5
    d["rolling_cv"] = 0.3
    d["drop_after_spike"] = 0.0
    d["release_adjusted"] = 0
    d["holiday_window"] = d["month"].isin(holiday_months).astype(int)
    d["holiday_bot_stream_val"] = d["bot_stream_val"] * d["holiday_window"]
    d["holiday_flag"] = d["ensemble_bot"] * d["holiday_window"]
    for c in signals.REASON_COLUMNS:
        d[c] = 0
    return d


def test_track_level_applies_holiday_confounder():
    # A Christmas record: every flagged day falls in December.
    dates = pd.date_range("2019-12-20", periods=10, freq="D").tolist()
    dates += pd.date_range("2020-06-01", periods=10, freq="D").tolist()
    flags = [1] * 10 + [0] * 10
    out = run_pipeline.track_level(_daily_for_track_level(dates, [1000] * 20, flags))
    assert out.loc[0, "bot_pct"] == 0.0, "holiday-window bot streams must be reclassified"
    assert out.loc[0, "note"] == config.HOLIDAY_NOTE


def test_track_level_keeps_year_round_flags():
    # Flags spread across the year -> not seasonality; must survive untouched.
    dates = pd.date_range("2020-01-01", periods=24, freq="MS").tolist()
    flags = [1, 0] * 12
    out = run_pipeline.track_level(_daily_for_track_level(dates, [1000] * 24, flags))
    assert out.loc[0, "bot_pct"] > 0.0
    assert out.loc[0, "note"] != config.HOLIDAY_NOTE


def test_track_level_applies_oneoff_confounder():
    # Long chart life, all flags inside one short burst -> an event, not bots.
    dates = pd.date_range("2019-01-01", periods=800, freq="D")
    flags = np.zeros(800, dtype=int)
    flags[400:410] = 1          # 10 flagged days in a 10-day span
    out = run_pipeline.track_level(
        _daily_for_track_level(dates, [1000] * 800, flags, holiday_months=()))
    assert out.loc[0, "bot_pct"] == 0.0, "one-off event burst must be cleared"
    assert out.loc[0, "note"] == config.ONEOFF_NOTE


def test_track_level_matches_artist_level_on_same_data():
    # Same underlying daily data must produce the same verdict at both levels.
    # Sized to clear artist_level's qualification filters (>=90 days, >=5M
    # streams, >=2 years active) so both frames are non-empty.
    dates = pd.date_range("2019-01-01", periods=730, freq="D")
    flags = np.zeros(730, dtype=int)
    holiday = dates.month.isin([11, 12])
    flags[np.flatnonzero(holiday)[:40]] = 1      # flags only ever in Nov-Dec
    daily = _daily_for_track_level(dates, [100_000] * 730, flags)
    daily["year"] = daily["date"].dt.year

    track = run_pipeline.track_level(daily)
    artist = run_pipeline.artist_level(daily)
    assert not artist.empty, "fixture must clear the qualification filters"
    assert track.loc[0, "bot_pct"] == artist.loc[0, "bot_pct"]
    assert track.loc[0, "note"] == artist.loc[0, "note"] == config.HOLIDAY_NOTE


# ── One-off confounder: burst clustering ────────────────────────────────
# The rule used to measure first-flag-to-last-flag, so two unrelated flagged
# days years apart read as one enormous "event" and the filter missed exactly
# the cases it was built for (Jessie J's China-TV spike measured 1,663 days at
# all-regions scope). It now tests the dominant burst.

def _burst_row(**kw):
    base = dict(flagged_days=10, bot_streams=800.0, total_streams=1000.0, bot_pct=80.0,
                confidence=50.0, note="",
                first_active=pd.Timestamp("2016-01-01"), last_active=pd.Timestamp("2020-01-01"),
                first_flag=pd.Timestamp("2018-06-01"), last_flag=pd.Timestamp("2018-06-10"))
    base.update(kw)
    return base


def test_oneoff_clears_dominant_burst_despite_distant_straggler():
    # 9 of 10 flags in one 9-day burst; one straggler 3 years later. The old
    # global-span test saw ~1100 days and kept it; the burst test clears it.
    df = pd.DataFrame([_burst_row(
        first_flag=pd.Timestamp("2016-06-01"), last_flag=pd.Timestamp("2019-07-01"),
        max_burst_days=9, max_burst_span_days=9, n_flag_bursts=2,
    )])
    out = run_pipeline.apply_oneoff_confounder(df)
    assert out.loc[0, "bot_pct"] == 0.0
    assert "One-off" in out.loc[0, "note"]


def test_oneoff_keeps_flags_spread_across_many_bursts():
    # No single burst dominates (biggest holds 3 of 20) -> recurring, not an
    # event. This is the BTS-across-comebacks case.
    df = pd.DataFrame([_burst_row(
        flagged_days=20, max_burst_days=3, max_burst_span_days=4, n_flag_bursts=7,
    )])
    out = run_pipeline.apply_oneoff_confounder(df)
    assert out.loc[0, "bot_pct"] == 80.0
    assert out.loc[0, "note"] == ""


def test_oneoff_keeps_single_long_burst():
    # One burst, but it runs far longer than the event window -> a sustained
    # campaign, which is what the detector is for.
    df = pd.DataFrame([_burst_row(
        flagged_days=120, max_burst_days=120,
        max_burst_span_days=config.ONEOFF_WINDOW_DAYS + 60, n_flag_bursts=1,
    )])
    out = run_pipeline.apply_oneoff_confounder(df)
    assert out.loc[0, "bot_pct"] == 80.0


def test_flag_burst_stats_splits_on_gap():
    # Two clusters 200 days apart must register as two bursts, the larger
    # holding 5 days.
    a = pd.date_range("2019-01-01", periods=5, freq="D")
    b = pd.date_range("2019-08-01", periods=3, freq="D")
    daily = pd.DataFrame({
        "artist": "A", "title": "T",
        "date": a.append(b), "ensemble_bot": 1,
    })
    out = run_pipeline.flag_burst_stats(daily, ["artist", "title"])
    assert out.loc[0, "n_flag_bursts"] == 2
    assert out.loc[0, "max_burst_days"] == 5
    assert out.loc[0, "max_burst_span_days"] == 4


def test_flag_burst_stats_contiguous_is_one_burst():
    daily = pd.DataFrame({
        "artist": "A", "title": "T",
        "date": pd.date_range("2019-01-01", periods=12, freq="D"), "ensemble_bot": 1,
    })
    out = run_pipeline.flag_burst_stats(daily, ["artist", "title"])
    assert out.loc[0, "n_flag_bursts"] == 1
    assert out.loc[0, "max_burst_days"] == 12


def test_flag_burst_stats_handles_no_flags():
    daily = pd.DataFrame({
        "artist": "A", "title": "T",
        "date": pd.date_range("2019-01-01", periods=5, freq="D"), "ensemble_bot": 0,
    })
    assert run_pipeline.flag_burst_stats(daily, ["artist", "title"]).empty
