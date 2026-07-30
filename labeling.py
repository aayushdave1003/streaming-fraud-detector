"""Human-labeling harness for REAL track-level ground truth.

The injected-anomaly benchmark (`benchmark.py`) is synthetic, and `evaluate.py`'s
documented purges are artist-level and temporally offset. A genuine track-level
benchmark needs human judgment — so this exports the detector's candidates for
review and scores the detector against whatever labels a human fills in.

    python labeling.py export     # -> labels_template.csv (candidates + context)
    python labeling.py score      # score detector vs a filled-in labels.csv

Workflow: fill the `label` column of `labels_template.csv` with **bot** / **legit**
(leave blank or `unsure` to skip), save it as `labels.csv`, then run `score`.

Nothing here fabricates labels — the whole point is that a human supplies them.
The export deliberately mixes high-suspicion tracks with a random low-suspicion
control sample, so the filled set can measure both precision *and* recall.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config

TEMPLATE = "labels_template.csv"
LABELS = "labels.csv"
CONTEXT_COLS = ["artist", "title", "total_streams", "bot_pct", "confidence",
                "flagged_days", "flag_reasons", "note"]


def export(results_path: str = config.RESULTS_TRACKS, n_top: int = 60,
           n_control: int = 40, seed: int = 42, out: str = TEMPLATE) -> pd.DataFrame:
    """Write a review sheet: the top suspicious tracks + a random control sample."""
    tracks = pd.read_csv(results_path)
    for c in CONTEXT_COLS:
        if c not in tracks.columns:
            tracks[c] = ""
    top = tracks.sort_values("bot_pct", ascending=False).head(n_top)
    rest = tracks.drop(top.index)
    control = rest.sample(n=min(n_control, len(rest)), random_state=seed) if len(rest) else rest.head(0)
    sheet = pd.concat([top, control]).drop_duplicates(["artist", "title"])
    sheet = sheet[CONTEXT_COLS].copy()
    sheet.insert(0, "label", "")           # human fills: bot | legit | unsure
    sheet.to_csv(out, index=False)
    print(f"✅ Wrote {out}: {len(sheet)} tracks to review "
          f"({len(top)} high-suspicion + {len(control)} control).")
    print("   Fill the `label` column with bot / legit, save as labels.csv, then: python labeling.py score")
    return sheet


def _score_labels(merged: pd.DataFrame) -> dict:
    """Precision/recall/F1 of the detector (flagged_days>0) vs human labels."""
    lab = merged["label"].astype(str).str.strip().str.lower()
    keep = lab.isin(["bot", "legit"])
    m = merged[keep].copy()
    y_true = (lab[keep] == "bot").astype(int).to_numpy()
    y_pred = (m["flagged_days"].fillna(0) > 0).astype(int).to_numpy()
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {"labeled": int(keep.sum()), "bot": int(y_true.sum()), "legit": int((y_true == 0).sum()),
           "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
           "accuracy": round((tp + tn) / max(1, len(m)), 3)}
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_true)) == 2:
            out["roc_auc_bot_pct"] = round(float(roc_auc_score(y_true, m["bot_pct"].fillna(0))), 3)
    except Exception:
        pass
    return out


def score(labels_path: str = LABELS, results_path: str = config.RESULTS_TRACKS) -> dict | None:
    if not os.path.exists(labels_path):
        print(f"No {labels_path} yet. Run `python labeling.py export`, fill the `label` "
              f"column (bot/legit) of {TEMPLATE}, save it as {labels_path}, then re-run.")
        return None
    labels = pd.read_csv(labels_path)
    tracks = pd.read_csv(results_path)
    merged = labels.merge(tracks[["artist", "title", "flagged_days", "bot_pct"]],
                          on=["artist", "title"], how="left", suffixes=("", "_r"))
    # prefer the fresh results columns if the template's went stale
    for c in ("flagged_days", "bot_pct"):
        if f"{c}_r" in merged:
            merged[c] = merged[f"{c}_r"]
    m = _score_labels(merged)
    if m["labeled"] == 0:
        print("No usable labels yet — fill the `label` column with bot / legit.")
        return m
    print("=" * 52)
    print("  DETECTOR vs HAND LABELS (track-level, real ground truth)")
    print("=" * 52)
    print(f"  labeled: {m['labeled']}  ({m['bot']} bot / {m['legit']} legit)")
    print(f"  precision {m['precision']}   recall {m['recall']}   f1 {m['f1']}   acc {m['accuracy']}")
    print(f"  confusion: tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")
    if "roc_auc_bot_pct" in m:
        print(f"  ROC-AUC (bot_pct): {m['roc_auc_bot_pct']}")
    print("=" * 52)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--results", default=config.RESULTS_TRACKS)
    e.add_argument("--n-top", type=int, default=60); e.add_argument("--n-control", type=int, default=40)
    s = sub.add_parser("score"); s.add_argument("--labels", default=LABELS)
    s.add_argument("--results", default=config.RESULTS_TRACKS)
    args = p.parse_args()
    if args.cmd == "export":
        export(args.results, args.n_top, args.n_control)
    else:
        score(args.labels, args.results)


if __name__ == "__main__":
    main()
