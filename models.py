"""Ensemble anomaly detector: Isolation Forest + Local Outlier Factor.

A record is flagged only when **both** models independently call it an
outlier — a deliberately conservative ensemble that trades recall for far
fewer false positives.

Two scoring paths:

* ``fit_predict_batch`` — in-sample flags for a full historical batch (matches
  the original pipeline's behaviour).
* ``predict`` — out-of-sample flags for *new* data using the persisted models,
  enabling incremental/live scoring without refitting on the whole history.

The detector is picklable via :func:`save` / :func:`load` (joblib), so training
and scoring can be separate steps.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import config


class EnsembleDetector:
    def __init__(self,
                 contamination: float = config.CONTAMINATION,
                 random_state: int = config.RANDOM_STATE,
                 if_estimators: int = config.IF_ESTIMATORS,
                 lof_neighbors: int = config.LOF_NEIGHBORS):
        self.contamination = contamination
        self.random_state = random_state
        self.if_estimators = if_estimators
        self.lof_neighbors = lof_neighbors
        self.scaler = StandardScaler()
        self.iso = IsolationForest(
            n_estimators=if_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        # novelty=True LOF can score data it was not fitted on (live path).
        self.lof_novelty = LocalOutlierFactor(
            n_neighbors=lof_neighbors, contamination=contamination, novelty=True
        )
        self.fitted = False
        self.feature_names = None  # captured at fit time; drives out-of-sample scoring

    @staticmethod
    def _as_array(X) -> np.ndarray:
        return np.asarray(X, dtype=float)

    def fit_predict_batch(self, X):
        """Fit on X and return in-sample ``(iso_bot, lof_bot, ensemble)`` flags.

        Also fits a novelty LOF so the persisted model can later score new data.
        """
        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)
        Xs = self.scaler.fit_transform(self._as_array(X))
        iso_bot = (self.iso.fit_predict(Xs) == -1).astype(int)
        # In-sample LOF (novelty=False) reproduces the original batch behaviour.
        lof_batch = LocalOutlierFactor(
            n_neighbors=self.lof_neighbors, contamination=self.contamination
        )
        lof_bot = (lof_batch.fit_predict(Xs) == -1).astype(int)
        # Separately fit the novelty LOF for out-of-sample scoring.
        self.lof_novelty.fit(Xs)
        self.fitted = True
        ensemble = ((iso_bot == 1) & (lof_bot == 1)).astype(int)
        return iso_bot, lof_bot, ensemble

    def predict(self, X):
        """Out-of-sample ensemble flags for NEW data (requires a fitted model)."""
        if not self.fitted:
            raise RuntimeError("EnsembleDetector must be fitted or loaded before predict().")
        if hasattr(X, "columns") and self.feature_names is not None:
            X = X[self.feature_names]  # align columns/order to training
        Xs = self.scaler.transform(self._as_array(X))
        iso_bot = (self.iso.predict(Xs) == -1).astype(int)
        lof_bot = (self.lof_novelty.predict(Xs) == -1).astype(int)
        return ((iso_bot == 1) & (lof_bot == 1)).astype(int)

    def save(self, path: str = config.MODEL_PATH) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = config.MODEL_PATH) -> "EnsembleDetector":
        return joblib.load(path)
