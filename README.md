# 🎵 Spotify Stream Fraud Detector

An unsupervised **ML pipeline for detecting bot-driven stream manipulation** on Spotify chart
data, with an interactive Streamlit dashboard. It engineers behavioral fraud signals from daily
stream counts, runs an **ensemble of anomaly-detection models** (Isolation Forest + Local Outlier
Factor), estimates the royalty impact of suspected fake streams, and cross-references the findings
against publicly documented, real-world stream-purge incidents.

> ⚠️ **Disclaimer — read first.** This is an experimental, educational anomaly-detection project.
> A high "bot %" is a *statistical signal that a stream pattern is unusual*, **not** an accusation
> that an artist or label bought fake streams. Legitimate events (viral moments, playlist adds,
> tribute spikes after a death, holiday seasonality) also produce anomalies. The pipeline explicitly
> down-weights several known confounders, but flagged results should be treated as leads to
> investigate, never as conclusions.

---

## What it does

1. **Engineers fraud signals** from daily per-track stream counts (velocity spikes, abrupt
   post-spike drop-offs, seasonal corrections, weekend/weekday patterns, and low-variance
   "plateau" streaming that bots tend to produce).
2. **Detects anomalies** with two unsupervised models and keeps only the streams **both** models
   independently flag (a conservative ensemble to cut false positives).
3. **Aggregates to track and artist level**, computes a bot-stream percentage and a 0–100
   **confidence score**, and post-processes known confounders (holiday catalog, artist-death
   tribute spikes).
4. **Estimates royalty impact** at Spotify's ~$0.004/stream average payout.
5. **Visualizes everything** in a Streamlit dashboard — genre browser, per-artist stream timelines
   with highlighted spikes, track breakdowns, a royalty calculator, and a panel of verified
   real-world purge incidents.

---

## Pipeline architecture

```
                 ┌─────────────────────────────┐
  Spotify chart  │  charts_combined.csv         │   (large source data — gitignored)
  data (daily) → │  artist, title, date, region,│
                 │  streams                      │
                 └──────────────┬───────────────┘
                                │
                       run_pipeline.py  ← main pipeline
                                │
        ┌───────────────────────┴───────────────────────┐
        │  Signal engineering (per artist+title, daily): │
        │   1. Stream-velocity spike  (streams / 7d avg) │
        │   2. Abrupt drop-off after spike               │
        │   3. Seasonal correction (vs monthly avg)      │
        │   4. Weekend-vs-weekday ratio                  │
        │   5. Plateau / low rolling coefficient of var. │
        └───────────────────────┬───────────────────────┘
                                │  StandardScaler
                ┌───────────────┴───────────────┐
       Isolation Forest                 Local Outlier Factor
       (contamination 3%)               (n_neighbors 20, 3%)
                └───────────────┬───────────────┘
                        ENSEMBLE: flag only if BOTH agree
                                │
              ┌─────────────────┴─────────────────┐
       track aggregation                   artist aggregation
       + confidence score                 (collabs split, catalog
                                          & death-spike filters)
              │                                   │
     fraud_results_tracks.csv          fraud_results_artists.csv
                        │  +  fraud_analysis.png  │
                        └─────────────┬───────────┘
                                      │
     artist_genres.csv ──────────────┤   real_world_reports.csv
     (assign_genres.py /             │   (real_world_data.py)
      fetch_genres.py)               │
                                      ▼
                                   app.py
                          Streamlit dashboard
```

---

## Repository layout

The core pipeline is a set of small, testable modules:

| Module | Role |
|--------|------|
| `config.py` | **Every tunable in one place** — regions, contamination, thresholds, confidence weights, confounder lists, paths. |
| `signals.py` | **Signal engineering** as pure, unit-testable functions with `retrospective` vs. `live` modes (look-ahead is disallowed live). |
| `models.py` | `EnsembleDetector` — Isolation Forest + LOF, flag only if both agree; joblib-persistable; out-of-sample scoring for new data. |
| `etl.py` | Chunked, bounded-memory build of the curated Parquet from the raw charts CSV. |
| `run_pipeline.py` | **The single entry point.** Load → signals → ensemble → track/artist stats + confidence + `flag_reasons` + confounder post-processing. Also `--score` for incremental scoring. |
| `evaluate.py` | Validate output against documented purges (`real_world_reports.csv`) — precision/recall@k, percentile, ROC-AUC (see caveat). |
| `tune.py` | Contamination sensitivity sweep → `tune_contamination.png/.csv`. |
| `app.py` | **Streamlit dashboard** — genre browser, stream timelines, royalty calculator, explainability, confidence controls, name-surfacing guardrail. |
| `tests/test_signals.py` | `pytest` suite (14 tests) for signals, look-ahead boundary, model persistence. |

Supporting scripts: `streaming_fraud_starter.py` (synthetic-data prototype),
`real_world_data.py` (curated incidents), `assign_genres.py` / `fetch_genres.py`
(genre tagging, offline / Spotify API), `spotify_fetch.py`, `debug_spotify.py`.

> `fraud_analysis.py` (v1) was retired and `fraud_analysis_v2.py` is now a thin
> shim that forwards to `run_pipeline.py`. See [ROADMAP.md](ROADMAP.md).

### Committed outputs (so the dashboard runs out of the box)
`fraud_results_artists.csv`, `fraud_results_tracks.csv`, `artist_genres.csv`,
`real_world_reports.csv`, `eval_report.json`, `tune_contamination.csv`, and the
generated `.png` charts. The curated `charts_us_global.parquet` and the trained
`model.joblib` (~90 MB) are gitignored — rebuild them with `etl.py` /
`run_pipeline.py`.

---

## Data

The large source datasets are **intentionally not committed** (the raw Spotify Charts CSV is
multiple GB and exceeds GitHub's 100 MB per-file limit). The pipeline reads a
`charts_combined.csv` with at least these columns:

```
artist, title, date, region, streams
```

This is derived from the public **Spotify Charts** daily dataset (available on Kaggle). To
reproduce end-to-end, obtain that dataset, filter/combine it into `charts_combined.csv` with the
columns above, and place it in the project root. The small result CSVs checked into the repo let
you explore the dashboard without the raw data.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/aayushdave1003/streaming-fraud-detector.git
cd streaming-fraud-detector

# 2. Create a virtual environment (Python 3.11+; developed on 3.13)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Spotify API access for genre/audio-feature scripts
cp .env.example .env
#   then edit .env with your Spotify app credentials
```

Get Spotify credentials from the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
The `.env` file is gitignored — **never commit real credentials.**

---

## Running it

```bash
# 0. (once) build the curated, columnar dataset from the raw charts CSV
python etl.py                       # charts_combined.csv -> charts_us_global.parquet

# 1. run the main ensemble pipeline
python run_pipeline.py              # full run on the curated parquet
python run_pipeline.py --sample 50000          # quick dev run
python run_pipeline.py --contamination 0.02    # override the anomaly rate
python run_pipeline.py --mode live             # drop look-ahead signals

# 2. validate against documented real-world purges
python evaluate.py

# 3. justify the contamination choice (sensitivity sweep)
python tune.py

# 4. score NEW chart data with the saved model (no refit)
python run_pipeline.py --score new_day.csv

# tag genres (offline, no API)  ...or online via Spotify API
python assign_genres.py            # / python fetch_genres.py

# curated real-world incidents table
python real_world_data.py

# run the tests
pytest -q

# launch the dashboard (degrades gracefully without the big data file)
streamlit run app.py
```

No raw data? The committed result CSVs let the dashboard and `evaluate.py` /
`tune.py` run without the multi-GB source; only steps 0–1 need it.

---

## How the detection works

**Signals** are computed per `(artist, title)` over time, so a track is compared against *its own*
history rather than a global average:

- **Velocity spike** — `streams / rolling_7d_avg`. Sudden multiples of the recent average.
- **Drop-after-spike** — a spike immediately followed by a collapse below the average (bots switch off).
- **Seasonal spike** — streams vs the artist's monthly average, to separate genuine seasonality.
- **Weekend ratio** — humans skew toward weekends; flat 24/7 activity is suspicious.
- **Plateau (rolling CV)** — unnaturally low variance in daily streams.

Features are standardized, then **Isolation Forest** (`contamination=0.03`) and **Local Outlier
Factor** (`n_neighbors=20`) each flag ~3% of records. A record is called a "bot day" only when
**both** models agree — a deliberately conservative ensemble.

**Confidence score (0–100)** blends: share of a track's days flagged, peak spike magnitude, and how
often spikes were followed by drop-offs.

**Confounder handling.** Christmas-catalog artists are excluded, artists need ≥90 days of data and
≥2 active years to qualify, and known artist-death tribute spikes (e.g. Mac Miller, Juice WRLD,
XXXTENTACION) have their confidence sharply reduced and annotated — because grief-driven streaming
looks anomalous but isn't fraud.

---

## Tech stack

Python · pandas · NumPy · scikit-learn (IsolationForest, LocalOutlierFactor) · joblib (model
persistence) · PyArrow (Parquet) · Streamlit · Plotly · matplotlib/seaborn · Spotipy (Spotify Web
API) · pytest.

---

## License

No license is specified yet. Until one is added, this code is provided for educational and
demonstrative purposes; the author retains all rights. The real-world incident data is compiled
from public reporting (Rolling Stone, Billboard, Music Business Worldwide, and others as cited in
`real_world_data.py`).
