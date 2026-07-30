"""Sensitivity analysis for the ``contamination`` hyper-parameter.

``contamination`` is the single most consequential knob: it *defines* what
fraction of records the models call anomalous, so the headline "bot %" is
partly an artifact of it. This runs the pipeline at several values and reports
how the flagged fraction and the documented-purge artists' ranks respond, so
the default (config.CONTAMINATION) is a justified choice rather than a guess.

    python tune.py                       # default sweep on the curated parquet
    python tune.py --values 0.01 0.03 0.05 0.08 --sample 200000
"""
from __future__ import annotations

import argparse
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
import evaluate as ev
import run_pipeline

DEFAULT_VALUES = [0.01, 0.02, 0.03, 0.05, 0.08]


def sweep(data: str = config.CURATED_PARQUET, values=None, sample=None):
    values = values or DEFAULT_VALUES
    rows = []
    for c in values:
        with tempfile.TemporaryDirectory() as tmp:
            artist, track = run_pipeline.run(
                data=data, contamination=c, sample=sample, output_dir=tmp,
                save_model=False, quiet=True,
            )
            flagged_frac = float(track["flagged_days"].sum() / track["total_days"].sum()) if len(track) else 0.0
            apath = os.path.join(tmp, os.path.basename(config.RESULTS_ARTISTS))
            report = ev.evaluate(apath, config.REAL_WORLD, write_report=False, quiet=True)
            m = report["metrics"]
            rows.append({
                "contamination": c,
                "n_artists": m["n_artists"],
                "flagged_fraction": round(flagged_frac, 4),
                "median_pct_positives": m.get("median_percentile_of_positives"),
                "precision_at_10": m.get("precision_at_10"),
                "recall_at_25": m.get("recall_at_25"),
                "roc_auc": m.get("roc_auc_bot_pct"),
            })
            print(f"  contamination={c}: flagged={flagged_frac:.3%}, "
                  f"median_pct_positives={m.get('median_percentile_of_positives')}, "
                  f"auc={m.get('roc_auc_bot_pct')}")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path: str = "tune_contamination.png") -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df["contamination"], df["flagged_fraction"] * 100, "o-", color="#e74c3c", label="flagged %")
    ax1.set_xlabel("contamination")
    ax1.set_ylabel("flagged % of records", color="#e74c3c")
    ax2 = ax1.twinx()
    if df["median_pct_positives"].notna().any():
        ax2.plot(df["contamination"], df["median_pct_positives"], "s--", color="#1DB954",
                 label="median percentile of documented artists")
        ax2.set_ylabel("median percentile of documented artists", color="#1DB954")
    ax1.axvline(config.CONTAMINATION, color="gray", ls=":", label=f"default={config.CONTAMINATION}")
    fig.suptitle("Contamination sensitivity")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=config.CURATED_PARQUET)
    p.add_argument("--values", nargs="+", type=float, default=None)
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args()
    df = sweep(args.data, args.values, args.sample)
    print("\n" + df.to_string(index=False))
    df.to_csv("tune_contamination.csv", index=False)
    plot(df)


if __name__ == "__main__":
    main()
