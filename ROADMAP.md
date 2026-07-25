# Roadmap

Product roadmap for the detector itself (not repo housekeeping). This pass
implemented every item below; each is checked off with a pointer to where it
now lives.

## Tier 1 — Credibility (accuracy & trust)

- [x] **Validate against ground truth.** `evaluate.py` scores the detector
  against the documented purges in `real_world_reports.csv` using rank-based
  metrics (precision/recall@k, median percentile, ROC-AUC), with an explicit
  temporal caveat (chart data is 2017–2021; purges are ~2025).
  *Result on the current data: ROC-AUC ≈ 0.80 by `bot_pct`; Kendrick Lamar,
  BTS, and BLACKPINK all rank above the median; Michael Smith is correctly
  absent (independent artist, never charted).*
- [x] **Per-flag explainability.** The pipeline tags every flagged day with the
  signals that fired (`signals.add_reason_flags`) and rolls them up into a
  `flag_reasons` column on both result tables; the dashboard shows a
  "Why flagged" column.
- [x] **Fix look-ahead / split retrospective vs. live.** `signals.engineer_signals`
  takes a `mode`. `retrospective` may use the look-ahead `drop_after_spike`
  signal; `live` cannot. A unit test asserts a past day's causal features do
  not change when future days are appended.
- [x] **Justify / tune the anomaly rate.** All constants moved to `config.py`.
  `tune.py` sweeps `contamination` and reports the effect on flagged fraction
  and documented-artist ranks (`tune_contamination.png/.csv`).
  *Finding: lower contamination (0.01–0.02) ranks the documented artists
  higher (median percentile ~91, AUC ~0.89) than the 0.03 default (~78 / 0.80).
  The default trades precision for coverage; 0.02 is a reasonable tightening if
  precision matters more.*

## Tier 2 — A pipeline, not 3 scripts

- [x] **ETL is committed.** `etl.py` builds the curated dataset from the raw
  Spotify Charts CSV in bounded-memory chunks — the previously machine-only
  build step is now reproducible.
- [x] **Deliver on "streaming".** Models persist via joblib
  (`EnsembleDetector.save/load`); `run_pipeline.incremental_score` scores new
  data out-of-sample with a novelty LOF, no refit. Scoring mode is inferred
  from the saved model's feature set so matrices line up.
- [x] **Refactor duplication.** Signal + model logic live in `signals.py` /
  `models.py`; `run_pipeline.py` is the single entry point. `fraud_analysis.py`
  (v1) was removed; `fraud_analysis_v2.py` is now a thin deprecation shim.
- [x] **Columnar storage.** ETL writes Parquet; the pipeline and dashboard
  prefer it over re-parsing the multi-GB CSV.

## Tier 3 — Product hardening

- [x] **Analyst threshold controls.** The dashboard has a confidence slider that
  re-filters live.
- [x] **Name-surfacing guardrail.** `app.mask_low_confidence` hides the identity
  of low-confidence flags behind an anonymous placeholder (toggle + threshold in
  the sidebar) so unproven anomalies aren't publicly named.
- [x] **Unit tests.** `tests/test_signals.py` covers the signals, the look-ahead
  boundary, collab splitting, confidence monotonicity, ensemble detection, and
  model-persistence round-trip. `pytest` — 14 passing.
- [~] **Systematic confounder handling.** Partial. The holiday catalog and
  death/tribute spikes are handled in `config.py` + `run_pipeline.artist_level`,
  and are now centralized/testable — but release-week and viral/playlist spikes
  are still not modeled. See "Next" below.

## Next (not yet done)

- Release-week and viral/playlist spike confounders (needs an external signal).
- Track/day-level ground truth for a stronger benchmark than artist-level.
- A true causal seasonal signal (the monthly-average term still uses
  within-month information; fine for retrospective audits, not for strict live
  scoring).
- Backfill genres via the Spotify API (`fetch_genres.py`) instead of the offline
  curated lists in `assign_genres.py`.
