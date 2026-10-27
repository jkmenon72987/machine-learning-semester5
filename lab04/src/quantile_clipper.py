
"""Training-data-fitted numerical quantile clipping transformer."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TrainingQuantileClipper(BaseEstimator, TransformerMixin):
    """Clip values to the 1st and 99th percentiles learned during fit."""

    def __init__(self, lower_quantile=1.0, upper_quantile=99.0):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)

        self.lower_bounds_ = np.nanpercentile(
            values, self.lower_quantile, axis=0
        )
        self.upper_bounds_ = np.nanpercentile(
            values, self.upper_quantile, axis=0
        )

        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        values = np.asarray(X, dtype=float)

        return np.clip(
            values,
            self.lower_bounds_,
            self.upper_bounds_,
        )

