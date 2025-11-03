"""Topological Anomaly Detectors."""

# Author: Martin Royer
from typing import Optional, Union

import numbers
import warnings
import numpy as np

from sklearn.base import _fit_context, TransformerMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from tdaad.utils.remapping_functions import score_flat_fast_remapping
from tdaad.topological_embedding import TopologicalEmbedding
from tdaad.utils.local_elliptic_envelope import EllipticEnvelope


class TopologicalAnomalyDetector(EllipticEnvelope, TransformerMixin):
    """
    Anomaly detection for multivariate time series using topological embeddings and robust covariance estimation.

    This detector extracts topological features from sliding windows of time series data and
    uses a robust Mahalanobis distance (via EllipticEnvelope) to score anomalies.

    Read more in the :ref:`User Guide <topological_anomaly_detection>`.

    Parameters
    ----------
    window_size : int, default=100
        Sliding window size for extracting time series subsequences.

    step : int, default=5
        Step size between windows.

    tda_max_dim : int, default=1
        Maximum homology dimension used for topological feature extraction.

    n_centers_by_dim : int, default=5
        Number of k-means centers per topological dimension (for vectorization).

    support_fraction : float or None, default=None
        Proportion of data to use for robust covariance estimation. If None, computed automatically.

    contamination : float, default=0.1
        Proportion of anomalies in the data, used to compute decision threshold.

    random_state : int, RandomState instance, or None, default=42
        Controls randomness of the topological embedding and robust estimator.

    Attributes
    ----------
    topological_embedding_ : object
        TopologicalEmbedding transformer object that is fitted at `fit`.

    Examples
    --------
    >>> n_timestamps = 1000
    >>> n_sensors = 20
    >>> import pandas as pd
    >>> timestamps = pd.to_datetime('2024-01-01', utc=True) + pd.Timedelta(1, 'h') * np.arange(n_timestamps)
    >>> X = pd.DataFrame(np.random.random(size=(n_timestamps, n_sensors)), index=timestamps)
    >>> X.iloc[n_timestamps//2:,:10] = -X.iloc[n_timestamps//2:,10:20]
    >>> detector = TopologicalAnomalyDetector(n_centers_by_dim=2, tda_max_dim=1).fit(X)
    >>> anomaly_scores = detector.score_samples(X)
    >>> decision = detector.decision_function(X)
    >>> anomalies = detector.predict(X)
    """

    _parameter_constraints: dict = {
        "window_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "step": [Interval(numbers.Integral, 1, None, closed="left")],
        "tda_max_dim": [Interval(numbers.Integral, 0, 2, closed="both")],
        "n_centers_by_dim": [Interval(numbers.Integral, 2, None, closed="left")],
        "support_fraction": [Interval(numbers.Real, 0, 1, closed="right"), None],
        "contamination": [Interval(numbers.Real, 0, 0.5, closed="both")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        window_size: int = 100,
        step: int = 5,
        tda_max_dim: int = 1,
        n_centers_by_dim: int = 5,
        support_fraction: Optional[float] = None,
        contamination: float = 0.1,
        random_state: Optional[Union[int, np.random.RandomState]] = 42,
    ):
        super().__init__(
            support_fraction=support_fraction,
            contamination=contamination,
            random_state=random_state,
        )
        self.window_size = window_size
        self.step = step
        self.tda_max_dim = tda_max_dim
        self.n_centers_by_dim = n_centers_by_dim

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the TopologicalAnomalyDetector model.

        Args
        ----
            X : {array-like, sparse matrix} of shape (n_timestamps, n_sensors)
                Multiple time series to transform, where `n_timestamps` is the number of timestamps
                in the series X, and `n_sensors` is the number of sensors.
            y : Ignored
                Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.topological_embedding_ = TopologicalEmbedding(
            window_size=self.window_size,
            step=self.step,
            n_centers_by_dim=self.n_centers_by_dim,
            tda_max_dim=self.tda_max_dim,
        )
        embedding = self.topological_embedding_.fit_transform(X)
        try:
            super().fit(embedding)
        except ValueError:
            warnings.warn(f"Failed with support_fraction={self.support_fraction}. ")

        self._update_padding(X)
        raw_scores = super().score_samples(embedding)
        self.dist_ = score_flat_fast_remapping(
            raw_scores,
            window_size=self.window_size,
            stride=self.step,
            padding_length=self.padding_length_,
        )
        self.offset_ = np.percentile(self.dist_, 100.0 * self.contamination)
        return self

    def _update_padding(self, X):
        imax = (X.shape[0] - self.window_size) // self.step
        self.padding_length_ = X.shape[0] - (self.step * imax + self.window_size)

    def _raw_score_samples(self, X):
        """Compute raw Mahalanobis scores (pre-remapping)."""
        check_is_fitted(self, "topological_embedding_")
        self._update_padding(X)
        embedding = self.topological_embedding_.transform(X)
        return super().score_samples(embedding)

    def score_samples(self, X, y=None):
        """Compute anomaly scores from topological features."""
        raw_scores = self._raw_score_samples(X)
        return score_flat_fast_remapping(
            raw_scores,
            window_size=self.window_size,
            stride=self.step,
            padding_length=self.padding_length_,
        )

    def decision_function(self, X):
        """Return the distance to the decision boundary."""
        return self.offset_ - self.score_samples(X)

    def predict(self, X):
        """Predict inliers (1) and outliers (-1) using learned threshold."""
        scores = self.score_samples(X)
        return np.where(scores < self.offset_, -1, 1)

    def transform(self, X):
        """Alias for score_samples. Returns anomaly scores."""
        return self.score_samples(X)
