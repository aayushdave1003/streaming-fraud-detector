"""End-to-end stream-fraud detection pipeline — the single entry point.

    python run_pipeline.py                      # full run on the curated parquet
    python run_pipeline.py --data charts_combined.csv   # run straight off raw csv
    python run_pipeline.py --sample 50000       # quick dev run
    python run_pipeline.py --contamination 0.05 # override the anomaly rate
    python run_pipeline.py --mode live          # drop look-ahead signals
    python run_pipeline.py --score new_day.csv  # score NEW data with saved model

Outputs: fraud_results_tracks.csv, fraud_results_artists.csv, fraud_analysis.png,
and (with --save-model) model.joblib.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import signals
from models import EnsembleDetector


# ── Loading ─────────────────────────────────────────────────────────────
def load_data(path: str, sample: int | None = None) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["date"])
    if "region" in df.columns:
        df = df[df["region"].isin(config.REGIONS)].copy()
    df = df.dropna(subset=["streams"])
    if sample:
        df = df.head(sample)
    return df


# ── Aggregation helpers ─────────────────────────────────────────────────
_REASON_AGG = {c: (c, "sum") for c in signals.REASON_COLUMNS}


def _confidence(flagged_days, total_days, max_spike, drop_count):
    return (
        (flagged_days / total_days.clip(1)) * config.CONF_FLAGGED_SHARE_W
        + (max_spike.clip(0, 10) / 10) * config.CONF_SPIKE_W
        + (drop_count / total_days.clip(1)) * config.CONF_DROP_W
    ).clip(0, 100).round(1)


def apply_holiday_confounder(artist: pd.DataFrame) -> pd.DataFrame:
    """Systematic holiday-seasonality correction (replaces the name allowlist).

    For artists whose flagged days are overwhelmingly in Nov–Dec, reclassify the
    holiday-window bot streams as legitimate (drops bot_pct), scale confidence
    down, and annotate. Conservative by design via a high share threshold.
    """
    denom = artist["flagged_days"].replace(0, np.nan)
    share = (artist["holiday_flagged_days"] / denom).fillna(0)
    artist["holiday_flag_share"] = share.round(2)
    mask = (share >= config.HOLIDAY_FLAG_SHARE_THRESHOLD) & (artist["flagged_days"] > 0)
    artist.loc[mask, "bot_streams"] = (
        artist.loc[mask, "bot_streams"] - artist.loc[mask, "holiday_bot_streams"]
    ).clip(lower=0)
    artist.loc[mask, "bot_pct"] = (
        artist.loc[mask, "bot_streams"] / artist.loc[mask, "total_streams"] * 100
    ).round(1)
    artist.loc[mask, "confidence"] = (artist.loc[mask, "confidence"] * config.HOLIDAY_CONF_MULT).round(1)
    artist.loc[mask, "note"] = config.HOLIDAY_NOTE
    return artist


def track_level(daily: pd.DataFrame) -> pd.DataFrame:
    track = daily.groupby(signals.GROUP).agg(
        total_streams=("streams", "sum"),
        bot_streams=("bot_stream_val", "sum"),
        flagged_days=("ensemble_bot", "sum"),
        total_days=("ensemble_bot", "count"),
        max_spike_ratio=("spike_ratio", "max"),
        avg_rolling_cv=("rolling_cv", "mean"),
        drop_after_spike_count=("drop_after_spike", "sum"),
        **_REASON_AGG,
    ).reset_index()
    track["bot_pct"] = (track["bot_streams"] / track["total_streams"] * 100).round(1)
    track["confidence"] = _confidence(
        track["flagged_days"], track["total_days"],
        track["max_spike_ratio"], track["drop_after_spike_count"],
    )
    track["flag_reasons"] = track.apply(signals.summarise_reasons, axis=1)
    return track.drop(columns=list(signals.REASON_COLUMNS))


def artist_level(daily: pd.DataFrame) -> pd.DataFrame:
    cols = ["artist", "title", "streams", "ensemble_bot", "bot_stream_val", "year",
            "spike_ratio", "drop_after_spike", "holiday_bot_stream_val", "holiday_flag",
            *signals.REASON_COLUMNS]
    dc = daily[cols].copy().reset_index(drop=True)
    dc["artist_split"] = dc["artist"].str.split(config.COLLAB_SPLIT_REGEX, regex=True)
    dc = dc.explode("artist_split")
    dc["artist_split"] = dc["artist_split"].str.strip()
    dc = dc[dc["artist_split"].str.len() > 0].reset_index(drop=True)

    artist = dc.groupby("artist_split").agg(
        total_streams=("streams", "sum"),
        bot_streams=("bot_stream_val", "sum"),
        flagged_days=("ensemble_bot", "sum"),
        total_days=("ensemble_bot", "count"),
        years_active=("year", "nunique"),
        max_spike_ratio=("spike_ratio", "max"),
        drop_events=("drop_after_spike", "sum"),
        holiday_bot_streams=("holiday_bot_stream_val", "sum"),
        holiday_flagged_days=("holiday_flag", "sum"),
        **_REASON_AGG,
    ).reset_index().rename(columns={"artist_split": "artist"})

    artist["bot_pct"] = (artist["bot_streams"] / artist["total_streams"] * 100).round(1)
    artist["confidence"] = _confidence(
        artist["flagged_days"], artist["total_days"],
        artist["max_spike_ratio"], artist["drop_events"],
    )
    artist["flag_reasons"] = artist.apply(signals.summarise_reasons, axis=1)
    artist = artist.drop(columns=list(signals.REASON_COLUMNS))

    # Qualification filters (drop thin/low-volume artists and holiday catalog).
    artist = artist[
        (artist["total_days"] >= config.MIN_DAYS)
        & (artist["total_streams"] >= config.MIN_TOTAL_STREAMS)
        & (artist["years_active"] >= config.MIN_YEARS_ACTIVE)
        & (~artist["artist"].isin(config.HOLIDAY_ARTISTS))
    ].copy()

    # Confounder: down-weight known death/tribute spikes.
    artist["note"] = ""
    for name in config.DEATH_SPIKES:
        mask = artist["artist"] == name
        if mask.any():
            artist.loc[mask, "confidence"] = (artist.loc[mask, "confidence"] * config.DEATH_SPIKE_CONF_MULT).round(1)
            artist.loc[mask, "note"] = config.DEATH_SPIKE_NOTE

    # Confounder: systematic holiday-seasonality correction.
    artist = apply_holiday_confounder(artist)
    return artist.drop(columns=["holiday_bot_streams", "holiday_flagged_days"])


# ── Plot ────────────────────────────────────────────────────────────────
def make_plot(artist: pd.DataFrame, path: str) -> None:
    top = artist.sort_values("bot_pct", ascending=False).head(10)
    if top.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Spotify Bot Stream Detection — Ensemble Model (IF + LOF)", fontsize=14)
    colors = ["#e74c3c" if c > 60 else "#f39c12" if c > 30 else "#1DB954" for c in top["confidence"]]
    axes[0].barh(top["artist"], top["bot_pct"], color=colors)
    axes[0].set_title("Bot % — color = confidence (red=high, yellow=med, green=low)")
    axes[0].set_xlabel("Bot Stream %")
    axes[0].invert_yaxis()
    axes[1].scatter(artist["bot_pct"], artist["confidence"], alpha=0.5, color="steelblue", s=20)
    axes[1].set_title("Bot % vs Confidence Score")
    axes[1].set_xlabel("Bot %")
    axes[1].set_ylabel("Confidence Score")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ── Orchestration ───────────────────────────────────────────────────────
def run(data: str = config.CURATED_PARQUET, mode: str = "retrospective",
        sample: int | None = None, contamination: float = config.CONTAMINATION,
        output_dir: str = ".", save_model: bool = True, quiet: bool = False):
    def say(*a):
        if not quiet:
            print(*a)

    say(f"📂 Loading {data} ...")
    df = load_data(data, sample)
    say(f"✅ Rows (US/Global): {len(df):,}")

    daily = signals.aggregate_daily(df)
    daily = signals.engineer_signals(daily, mode=mode)
    say(f"🔧 Signals engineered ({mode} mode) — {len(daily):,} daily records")

    X = signals.build_feature_matrix(daily, mode=mode)
    det = EnsembleDetector(contamination=contamination)
    iso_bot, lof_bot, ensemble = det.fit_predict_batch(X)
    daily["iso_bot"], daily["lof_bot"], daily["ensemble_bot"] = iso_bot, lof_bot, ensemble
    daily["bot_stream_val"] = daily["streams"] * daily["ensemble_bot"]
    daily = signals.add_reason_flags(daily, "ensemble_bot")
    daily["holiday_window"] = daily["month"].isin(config.HOLIDAY_MONTHS).astype(int)
    daily["holiday_bot_stream_val"] = daily["bot_stream_val"] * daily["holiday_window"]
    daily["holiday_flag"] = daily["ensemble_bot"] * daily["holiday_window"]
    say(f"🤖 IF flagged {iso_bot.sum():,} | LOF {lof_bot.sum():,} | ensemble {ensemble.sum():,}")

    track = track_level(daily)
    artist = artist_level(daily)

    os.makedirs(output_dir, exist_ok=True)
    track_path = os.path.join(output_dir, os.path.basename(config.RESULTS_TRACKS))
    artist_path = os.path.join(output_dir, os.path.basename(config.RESULTS_ARTISTS))
    track.to_csv(track_path, index=False)
    artist.to_csv(artist_path, index=False)
    make_plot(artist, os.path.join(output_dir, os.path.basename(config.PLOT_PATH)))
    if save_model:
        det.save(os.path.join(output_dir, os.path.basename(config.MODEL_PATH)))

    say(f"✅ Wrote {artist_path} ({len(artist):,} artists), {track_path} ({len(track):,} tracks)")
    if not quiet and not artist.empty:
        top = artist.sort_values("bot_pct", ascending=False).head(15)
        say("\n📊 Top suspicious artists:")
        say(top[["artist", "total_streams", "bot_pct", "confidence", "flag_reasons"]].to_string(index=False))
    return artist, track


def incremental_score(new_data: str, model_path: str = config.MODEL_PATH,
                      mode: str | None = None) -> pd.DataFrame:
    """Score NEW raw chart data with a previously-saved model (no refit).

    ``mode`` defaults to whatever the saved model was trained on — a model
    trained with the look-ahead ``drop_after_spike`` feature must be scored the
    same way, so the feature matrices line up.
    """
    det = EnsembleDetector.load(model_path)
    if mode is None:
        mode = "retrospective" if det.feature_names and "drop_after_spike" in det.feature_names else "live"
    df = load_data(new_data)
    daily = signals.aggregate_daily(df)
    daily = signals.engineer_signals(daily, mode=mode)
    X = signals.build_feature_matrix(daily, mode=mode)
    daily["ensemble_bot"] = det.predict(X)
    return daily


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=config.CURATED_PARQUET)
    p.add_argument("--mode", choices=["retrospective", "live"], default=None,
                   help="Signal mode. Default: retrospective for training; "
                        "inferred from the saved model for --score.")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--contamination", type=float, default=config.CONTAMINATION)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--score", metavar="NEW_DATA", default=None,
                   help="Score NEW chart data with the saved model instead of training.")
    args = p.parse_args()

    if args.score:
        scored = incremental_score(args.score, mode=args.mode)
        print(f"✅ Scored {len(scored):,} daily records — flagged {int(scored['ensemble_bot'].sum()):,}")
        return
    run(data=args.data, mode=args.mode or "retrospective", sample=args.sample,
        contamination=args.contamination, output_dir=args.output_dir,
        save_model=not args.no_save_model)


if __name__ == "__main__":
    main()
