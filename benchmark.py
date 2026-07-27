"""Injected-anomaly benchmark — a controllable ground-truth proxy.

`evaluate.py` checks the detector against 8 artist-level documented purges; this
complements it with a *day-level* recall metric: plant known synthetic bot
patterns into real chart series and measure how many the ensemble recovers.

Two sweeps:
  1. boost sweep — recall vs. boost magnitude, for each injection pattern.
  2. contamination sweep — recall vs. the anomaly-rate knob, making the
     precision/recall tradeoff explicit (ties back to tune.py).

Two injection patterns:
  * spike   — multiply a window by `boost` (pump; hits velocity + drop-after).
  * plateau — replace a window with a flat elevated level (bot-characteristic;
              also hits the low-variance plateau signal the spike misses).

⚠️ CAVEAT: injected into REAL data that already holds organic anomalies, so
**recall** (detection of planted bots) is the clean metric; **precision** is a
loose bound (organic anomalies count as false positives). Scores raw ensemble
flags (the core detector, pre-confounder).

    python benchmark.py
    python benchmark.py --boosts 2 3 5 8 --patterns spike plateau
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
DEFAULT_BOOSTS = [1.5, 2.0, 3.0, 5.0, 8.0]
DEFAULT_CONTAMS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]


def inject(daily, n_tracks, window, boost, rng, pattern="spike"):
    """Plant `n_tracks` bot windows and label them with ``is_injected``.

    Selection + window start are drawn identically regardless of pattern (no
    pattern-dependent rng draws), so the same tracks/windows are used across
    patterns and boosts for a clean comparison.
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
        if pattern == "plateau":
            baseline = float(daily.loc[idx, "streams"].mean())
            daily.loc[win, "streams"] = boost * baseline          # flat, low-variance
        else:  # spike
            daily.loc[win, "streams"] = daily.loc[win, "streams"] * boost
        daily.loc[win, "is_injected"] = 1
        chosen.append(keys[i])
    return daily, chosen


def score(daily, contamination):
    """Run the real signal + ensemble stack; return raw ensemble flags."""
    daily = signals.engineer_signals(daily.copy(), mode="retrospective")
    X = signals.build_feature_matrix(daily, mode="retrospective")
    _, _, ensemble = EnsembleDetector(contamination=contamination).fit_predict_batch(X)
    daily["ensemble_bot"] = ensemble
    return daily


def metrics(scored):
    m = scored["is_injected"] == 1
    f = scored["ensemble_bot"] == 1
    inj_tracks = scored[m].drop_duplicates(signals.GROUP)
    det_tracks = scored[m & f].drop_duplicates(signals.GROUP)
    return {
        "injected_days": int(m.sum()),
        "day_recall": round(float((f & m).sum() / max(1, m.sum())), 3),
        "injected_tracks": len(inj_tracks),
        "track_recall": round(len(det_tracks) / max(1, len(inj_tracks)), 3),
        "precision_loose": round(float((f & m).sum() / max(1, f.sum())), 3),
        "flagged_frac": round(float(f.sum() / max(1, len(scored))), 4),
    }


def boost_sweep(base, patterns, boosts, n_tracks, window, seed, contamination):
    rows = []
    for pattern in patterns:
        for b in boosts:
            inj, _ = inject(base, n_tracks, window, b, np.random.default_rng(seed), pattern)
            r = metrics(score(inj, contamination)); r.update(pattern=pattern, boost=b)
            rows.append(r)
            print(f"  [{pattern:>7}] boost {b:>4}x: track-recall {r['track_recall']:.3f}, "
                  f"day-recall {r['day_recall']:.3f}")
    return pd.DataFrame(rows)


def contamination_sweep(base, boost, pattern, contams, n_tracks, window, seed):
    rows = []
    for c in contams:
        inj, _ = inject(base, n_tracks, window, boost, np.random.default_rng(seed), pattern)
        r = metrics(score(inj, c)); r.update(contamination=c)
        rows.append(r)
        print(f"  contamination {c:>5}: track-recall {r['track_recall']:.3f}, "
              f"flagged {r['flagged_frac']:.3%}, precision {r['precision_loose']:.3f}")
    return pd.DataFrame(rows)


def plot_boost(df, path="benchmark_boost.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"spike": "#1DB954", "plateau": "#2747b0"}
    for pattern, g in df.groupby("pattern"):
        ax.plot(g["boost"], g["track_recall"], "o-", color=colors.get(pattern, "#888"),
                label=f"{pattern} (track recall)")
    ax.set_xlabel("injected boost (× normal streams)")
    ax.set_ylabel("track-level recall of planted bots")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.2); ax.legend()
    ax.set_title("Injected-anomaly recall vs. boost magnitude")
    fig.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
    print(f"✅ Saved {path}")


def plot_contamination(df, path="benchmark_contamination.png"):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["contamination"], df["track_recall"], "o-", color="#1DB954", label="track recall")
    ax1.set_xlabel("contamination (anomaly-rate knob)")
    ax1.set_ylabel("track-level recall", color="#1DB954"); ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(df["contamination"], df["flagged_frac"] * 100, "s--", color="#e74c3c", label="flagged % of days")
    ax2.set_ylabel("flagged % of all days (false-positive load)", color="#e74c3c")
    ax1.axvline(config.CONTAMINATION, color="gray", ls=":", label=f"default {config.CONTAMINATION}")
    ax1.grid(alpha=0.2); fig.suptitle("Recall vs. contamination — the precision/recall tradeoff")
    fig.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)
    print(f"✅ Saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=config.CURATED_PARQUET)
    p.add_argument("--boosts", nargs="+", type=float, default=None)
    p.add_argument("--patterns", nargs="+", default=["spike", "plateau"], choices=["spike", "plateau"])
    p.add_argument("--contams", nargs="+", type=float, default=None)
    p.add_argument("--contam-boost", type=float, default=5.0, help="boost held fixed in the contamination sweep")
    p.add_argument("--n-tracks", type=int, default=300)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args()

    base = signals.aggregate_daily(load_data(args.data, args.sample))

    print("── Boost sweep (contamination fixed at default) ──")
    bs = boost_sweep(base, args.patterns, args.boosts or DEFAULT_BOOSTS,
                     args.n_tracks, args.window, args.seed, config.CONTAMINATION)
    bs.to_csv("benchmark_boost.csv", index=False)
    plot_boost(bs)

    print("\n── Contamination sweep (plateau, boost fixed) ──")
    cs = contamination_sweep(base, args.contam_boost, "plateau",
                             args.contams or DEFAULT_CONTAMS, args.n_tracks, args.window, args.seed)
    cs.to_csv("benchmark_contamination.csv", index=False)
    plot_contamination(cs)

    print("\n" + bs.to_string(index=False))
    print("\n" + cs.to_string(index=False))


if __name__ == "__main__":
    main()
