"""Deprecated entry point — kept only for backwards compatibility.

The pipeline has been refactored into modules:

    config.py       — all tunables in one place
    signals.py      — testable signal engineering (retrospective vs live)
    models.py       — EnsembleDetector (IF + LOF), persistable, out-of-sample scoring
    run_pipeline.py — the single entry point (this file just forwards to it)

Run `python run_pipeline.py` instead. See README.md.
"""
from run_pipeline import run

if __name__ == "__main__":
    print("⚠️  fraud_analysis_v2.py is deprecated — forwarding to run_pipeline.run().")
    run()
