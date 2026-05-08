from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class DeltaHighRegressor(BaseEstimator, RegressorMixin):
    """Regressor that predicts next high as today's high plus a learned residual."""

    def __init__(self, base_estimator, anchor_col: str = "high", residual_clip: float | None = None):
        self.base_estimator = base_estimator
        self.anchor_col = anchor_col
        self.residual_clip = residual_clip

    def fit(self, X, y):
        residual_target = y - X[self.anchor_col]
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, residual_target)
        self.feature_names_in_ = getattr(self.estimator_, "feature_names_in_", X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def predict(self, X):
        residual = self.estimator_.predict(X)
        if self.residual_clip is not None:
            residual = np.clip(residual, -self.residual_clip, self.residual_clip)
        return X[self.anchor_col].to_numpy() + residual


class PersistenceHighRegressor(BaseEstimator, RegressorMixin):
    """Baseline-compatible estimator that predicts tomorrow's high as today's high."""

    def __init__(self, anchor_col: str = "high"):
        self.anchor_col = anchor_col

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def predict(self, X):
        return X[self.anchor_col].to_numpy()
