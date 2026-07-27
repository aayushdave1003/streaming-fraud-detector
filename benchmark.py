"""Injected-anomaly benchmark — a controllable ground-truth proxy.

`evaluate.py` checks the detector against 8 artist-level documented purges;
this complements it with a *day-level* recall metric: plant known synthetic bot
boosts into real chart series and measure how many the ensemble recovers, swept
across boost magnitudes.

⚠️ CAVEAT: boosts are injected into REAL data that already contains organic
anomalies, so **recall** (detection rate of the planted bots) is the clean
metric; **precision** is a loose bound — organic anomalies count as false
positives against the injected-only labels. We report raw ensemble flags
(pre-confounder) since this measures the core detector, not the business filters.

    python benchmark.py
    python benchmark.py --boosts 1.5 2 3 5 8 --n-tracks 300 --window 5
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import signals
from models import EnsembleDetector
from run_pipeline import load_data

INJECT_OFFSET = 15  # start boosts past the release window so they aren't confounded


def inject(daily: pd.DataFrame, n_tracks: int, window: int, boost: float, rng):
    """Multiply a random `window`-day slice of `n_tracks` tracks by `boost`.

    Returns the modified frame with an ``is_injected`` label column and the list
    of chosen (artist, title) keys. Uses a fresh rng seeded identically per call
    so the *same* tracks/windows are chosen across boost levels (clean sweep).
    """
    daily = daily.copy()
    daily["is_injected"] = 0
    groups = {k: v for k, v in daily.groupby(signals.GROUP).groups.items()
              if len(v) >= window + INJECT_OFFSET + 5}
    keys = list(groups.keys())
    if not keys:
        return daily, []
    n = min(n_tracks, len(keys))
    chosen = []
    for i in rng.choice(len(keys), size=n, replace=False):
        idx = list(groups[keys[i]])
        start = int(rng.integers(INJECT_OFFSET, len(idx) - window))
        win = idx[start:start + window]
        daily.loc[win, "streams"] = daily.loc[win, "streams"] * boost
        daily.loc[win, "is_injected"] = 1
        chosen.append(keys[i])
    return daily, chosen


def score(daily: pd.DataFrame, contamination: float) -> pd.DataFrame:
    """Run the real signal + ensemble stack; return raw ensemble flags."""
    daily = signals.engineer_signals(daily.copy(), mode="retrospective")
    X = signals.build_feature_matrix(daily, mode="retrospective")
    _, _, ensemble = EnsembleDetector(contamination=contamination).fit_predict_batch(X)
    daily["ensemble_bot"] = ensemble
    return daily


def run_benchmark(data=config.CURATED_PARQUET, boosts=None, n_tracks=300, window=5,
                  seed=42, contamination=config.CONTAMINATION, sample=None) -> pd.DataFrame:
    boosts = boosts or [1.5, 2.0, 3.0, 5.0, 8.0]
    base = signals.aggregate_daily(load_data(data, sample))
    rows = []
    for b in boosts:
        inj, _ = inject(base, n_tracks, window, b, np.random.default_rng(seed))
        scored = score(inj, contamination)
        m = scored["is_injected"] == 1
        f = scored["ensemble_bot"] == 1
        inj_tracks = scored[m].drop_duplicates(signals.GROUP)
        det_tracks = scored[m & f].drop_duplicates(signals.GROUP)
        rows.append({
            "boost": b,
            "injected_days": int(m.sum()),
            "day_recall": round(float((f & m).sum() / max(1, m.sum())), 3),
            "injected_tracks": len(inj_tracks),
            "track_recall": round(len(det_tracks) / max(1, len(inj_tracks)), 3),
            "precision_loose": round(float((f & m).sum() / max(1, f.sum())), 3),
        })
        print(f"  boost {b:>4}x: day-recall {rows[-1]['day_recall']:.3f}, "
              f"track-recall {rows[-1]['track_recall']:.3f} "
              f"({len(det_tracks)}/{len(inj_tracks)} tracks)")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path="benchmark_injection.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["boost"], df["day_recall"], "o-", color="#1DB954", label="day-level recall")
    ax.plot(df["boost"], df["track_recall"], "s--", color="#2747b0", label="track-level recall")
    ax.set_xlabel("injected boost (× normal streams)")
    ax.set_ylabel("recall of planted bots")
    ax.set_ylim(0, 1.02)
    ax.set_title("Injected-anomaly recall vs. boost magnitude")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=config.CURATED_PARQUET)
    p.add_argument("--boosts", nargs="+", type=float, default=None)
    p.add_argument("--n-tracks", type=int, default=300)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args()
    df = run_benchmark(args.data, args.boosts, args.n_tracks, args.window, args.seed, sample=args.sample)
    print("\n" + df.to_string(index=False))
    df.to_csv("benchmark_injection.csv", index=False)
    plot(df)


if __name__ == "__main__":
    main()
