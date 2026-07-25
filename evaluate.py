"""Validate detector output against documented real-world stream purges.

We treat artists with a *verified* documented purge (``real_world_reports.csv``)
as positives and ask: does the detector rank them near the top of its
suspicion list?

⚠️ IMPORTANT CAVEAT. The chart dataset is 2017–2021; the documented purges are
mostly 2025 and are artist-level, not track/day-level. So this is a weak,
directional check — "did artists later purged already show elevated bot
signals historically?" — not a clean supervised benchmark. Metrics are
rank-based for that reason, and absolute precision/recall should be read with
the temporal mismatch in mind.

    python evaluate.py
    python evaluate.py --results fraud_results_artists.csv
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

import config

_PAREN = re.compile(r"\s*\(.*?\)\s*$")


def normalize(name: str) -> str:
    """'BTS (Jimin)' -> 'BTS', 'BLACKPINK (Rosé)' -> 'BLACKPINK'."""
    return _PAREN.sub("", str(name)).strip()


def positives_present(results: pd.DataFrame, real_world: pd.DataFrame):
    """Verified documented artists that also appear in the results table."""
    verified = real_world[real_world["verified"] == True]  # noqa: E712
    documented = {normalize(a) for a in verified["artist"]}
    present_artists = {normalize(a) for a in results["artist"]}
    hits = sorted(documented & present_artists)
    missing = sorted(documented - present_artists)
    return hits, missing, documented


def evaluate(results_path: str = config.RESULTS_ARTISTS,
             real_world_path: str = config.REAL_WORLD,
             score_col: str = "bot_pct") -> dict:
    results = pd.read_csv(results_path)
    real_world = pd.read_csv(real_world_path)
    results["_norm"] = results["artist"].map(normalize)
    # Collapse collab-split duplicates to the best score per normalized artist.
    ranked = (results.sort_values(score_col, ascending=False)
              .drop_duplicates("_norm")
              .reset_index(drop=True))
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["percentile"] = (1 - ranked["rank"] / len(ranked)) * 100

    hits, missing, documented = positives_present(ranked, real_world)
    n = len(ranked)

    per_artist = []
    for a in hits:
        row = ranked[ranked["_norm"] == a].iloc[0]
        per_artist.append({
            "artist": a,
            "rank": int(row["rank"]),
            "percentile": round(float(row["percentile"]), 1),
            "bot_pct": float(row.get("bot_pct", float("nan"))),
            "confidence": float(row.get("confidence", float("nan"))),
        })

    pos_ranks = {ranked[ranked["_norm"] == a]["rank"].iloc[0] for a in hits}
    metrics = {"n_artists": n, "documented_verified": len(documented),
               "documented_present": len(hits), "documented_absent": missing}
    for k in (10, 25, 50):
        if n == 0:
            continue
        topk = set(range(1, min(k, n) + 1))
        tp = len(pos_ranks & topk)
        metrics[f"precision_at_{k}"] = round(tp / min(k, n), 3)
        metrics[f"recall_at_{k}"] = round(tp / len(hits), 3) if hits else 0.0

    if per_artist:
        pcts = [p["percentile"] for p in per_artist]
        metrics["median_percentile_of_positives"] = round(sorted(pcts)[len(pcts) // 2], 1)

    # Optional AUC (documented artists as positives across the full ranking).
    try:
        from sklearn.metrics import roc_auc_score
        y = ranked["_norm"].isin(hits).astype(int)
        if y.nunique() == 2:
            metrics["roc_auc_bot_pct"] = round(float(roc_auc_score(y, ranked[score_col])), 3)
    except Exception:
        pass

    report = {"caveat": "Chart data 2017-2021 vs purges ~2025; directional only.",
              "score_col": score_col, "metrics": metrics, "positives": per_artist}

    with open(config.EVAL_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    m = report["metrics"]
    print("=" * 66)
    print("  DETECTOR EVALUATION vs DOCUMENTED REAL-WORLD PURGES")
    print("=" * 66)
    print(f"  ⚠️  {report['caveat']}")
    print(f"  Ranking artists by: {report['score_col']}")
    print(f"  Artists ranked: {m['n_artists']:,}")
    print(f"  Documented (verified) artists: {m['documented_verified']} "
          f"— {m['documented_present']} present, {m['documented_absent']} absent")
    if m.get("documented_absent"):
        print(f"    absent (not in 2017-2021 charts): {', '.join(m['documented_absent'])}")
    print("-" * 66)
    for k in (10, 25, 50):
        if f"precision_at_{k}" in m:
            print(f"  precision@{k:<3} {m[f'precision_at_{k}']:.3f}    recall@{k:<3} {m[f'recall_at_{k}']:.3f}")
    if "median_percentile_of_positives" in m:
        print(f"  median percentile of documented artists: {m['median_percentile_of_positives']}")
    if "roc_auc_bot_pct" in m:
        print(f"  ROC-AUC ({report['score_col']} as score): {m['roc_auc_bot_pct']}")
    print("-" * 66)
    print("  Documented artists found, with their rank:")
    for p in report["positives"]:
        print(f"    #{p['rank']:<5} {p['artist']:<20} bot%={p['bot_pct']:<6} "
              f"conf={p['confidence']:<6} (top {100 - p['percentile']:.1f}%)")
    print("=" * 66)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=config.RESULTS_ARTISTS)
    p.add_argument("--real-world", default=config.REAL_WORLD)
    p.add_argument("--score-col", default="bot_pct", choices=["bot_pct", "confidence"])
    args = p.parse_args()
    evaluate(args.results, args.real_world, args.score_col)


if __name__ == "__main__":
    main()
