"""Central configuration for the stream-fraud detection pipeline.

Every tunable the pipeline depends on lives here so the behaviour is auditable
and adjustable in one place instead of being scattered as magic numbers across
scripts (roadmap Tier 1: "justify / tune the anomaly rate").
"""
from __future__ import annotations

# ── Data selection ──────────────────────────────────────────────────────
REGIONS = ["United States", "Global"]   # chart regions the pipeline analyses
ROLLING_WINDOW = 7                        # days, for rolling mean / std

# ── Anomaly models ──────────────────────────────────────────────────────
RANDOM_STATE = 42
CONTAMINATION = 0.02          # assumed anomaly fraction (both IF and LOF); see tune.py
                             # (sweep: 0.02 ranks documented purge artists higher than 0.03)
IF_ESTIMATORS = 200
LOF_NEIGHBORS = 20

# Features fed to the models. Retrospective mode adds the look-ahead
# `drop_after_spike` signal; live mode omits it (see signals.py).
BASE_FEATURES = ["streams", "pct_change", "spike_ratio", "seasonal_spike", "rolling_cv"]
RETROSPECTIVE_ONLY_FEATURES = ["drop_after_spike"]


def features_for(mode: str) -> list[str]:
    """Feature list for a given scoring mode.

    ``retrospective`` may use future-looking signals; ``live`` may not.
    """
    if mode == "retrospective":
        return BASE_FEATURES + RETROSPECTIVE_ONLY_FEATURES
    return list(BASE_FEATURES)


# ── Signal thresholds (drive both flagging heuristics and explainability) ─
SPIKE_RATIO_FLAG = 2.0        # streams > 2x rolling avg  == velocity spike
SEASONAL_SPIKE_FLAG = 2.0     # streams > 2x monthly avg  == seasonal spike
LOW_CV_FLAG = 0.15            # rolling coefficient of variation below this == plateau

# ── Confidence score weights (must sum to 100) ──────────────────────────
CONF_FLAGGED_SHARE_W = 40     # share of a track's days that were flagged
CONF_SPIKE_W = 30            # peak spike magnitude
CONF_DROP_W = 30            # how often spikes were followed by a drop-off

# ── Artist-level qualification filters ──────────────────────────────────
MIN_DAYS = 90
MIN_TOTAL_STREAMS = 5_000_000
MIN_YEARS_ACTIVE = 2

# ── Confounders ─────────────────────────────────────────────────────────
# Christmas-catalog acts whose seasonal spikes are legitimate, not fraud.
HOLIDAY_ARTISTS = [
    "Brenda Lee", "Bing Crosby", "Bobby Helms", "Perry Como", "Nat King Cole",
    "Andy Williams", "Burl Ives", "Dean Martin", "Mariah Carey",
    "Vince Guaraldi Trio", "Frank Sinatra", "Darlene Love", "The Ronettes",
    "José Feliciano", "Gene Autry", "Carpenters", "Ella Fitzgerald",
    "Tony Bennett", "Chris Rea",
]

# Artists with grief/tribute streaming spikes after a death — anomalous but
# not fraud. Confidence for these is multiplied down and annotated.
DEATH_SPIKES = {
    "Nipsey Hussle": "2019-03-31",
    "Mac Miller": "2018-09-07",
    "Juice WRLD": "2019-12-08",
    "Pop Smoke": "2020-02-19",
    "Lil Peep": "2017-11-15",
    "XXXTentacion": "2018-06-18",
}
DEATH_SPIKE_CONF_MULT = 0.4
DEATH_SPIKE_NOTE = "⚠️ Death/tribute spike detected — results may be inflated"

# Systematic holiday-seasonality confounder (generalizes HOLIDAY_ARTISTS above).
# When an artist's flagged days are overwhelmingly concentrated in Nov–Dec, the
# "anomaly" is almost certainly Christmas seasonality, not fraud: reclassify the
# holiday-window bot streams as legitimate and down-weight confidence. The high
# share threshold keeps this conservative — a one-off December fraud by an
# otherwise year-round artist has a low holiday share and is not excused.
HOLIDAY_MONTHS = [11, 12]
HOLIDAY_FLAG_SHARE_THRESHOLD = 0.6
HOLIDAY_CONF_MULT = 0.3
HOLIDAY_NOTE = "🎄 Holiday-seasonal spike — likely legitimate, not fraud"

# ── Collab splitting (artist-level attribution) ─────────────────────────
COLLAB_SPLIT_REGEX = r",\s*|\s*&\s*|\s*ft\.?\s*|\s*feat\.?\s*"

# ── Economics ───────────────────────────────────────────────────────────
SPOTIFY_RATE = 0.004  # avg USD paid per stream (rough industry figure)

# ── Paths ───────────────────────────────────────────────────────────────
RAW_CHARTS = "charts_combined.csv"            # large source (gitignored)
CURATED_PARQUET = "charts_us_global.parquet"  # curated columnar (gitignored)
RESULTS_ARTISTS = "fraud_results_artists.csv"
RESULTS_TRACKS = "fraud_results_tracks.csv"
REAL_WORLD = "real_world_reports.csv"
PLOT_PATH = "fraud_analysis.png"
MODEL_PATH = "model.joblib"
EVAL_REPORT = "eval_report.json"
