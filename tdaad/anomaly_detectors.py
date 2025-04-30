"""Topological Anomaly Detectors."""

# Author: Martin Royer

from typing import Sequence
from numbers import Integral

import pandas as pd

from sklearn.utils._param_validation import Interval
from sklearn.base import _fit_context, TransformerMixin
from tdaad.utils.remapping_functions import score_flat_fast_remapping
from tdaad.topological_embedding import TopologicalEmbedding
from tdaad.utils.local_elliptic_envelope import EllipticEnvelope


class TopologicalAnomalyDetector(EllipticEnvelope, TransformerMixin):
    """Object for detecting anomaly base on Topological Embedding and sklearn.covariance.EllipticEnvelope.

    This object analyzes multiple time series data through the following operations:
    - run a sliding window algorithm and represent each time series window with topological features,
        see :ref:`Topological Embedding <topological_embedding>`,

    - use a MinCovDet algorithm to robustly estimate the data mean and covariance in the embedding space,
        and use these to derive an embedding mahalanobis distance and associated outlier detection procedure,
        see :ref:`Elliptic Envelope <elliptic_envelope>`.

    After fitting, it is able to produce an anomaly score from a time series describing normal / abnormal time segments.
    (the lower, the more abnormal)
    The predict method (inherited from EllipticEnvelope) allows to transform that score into
    binary normal / anomaly labels.

    Read more in the :ref:`User Guide <topological_anomaly_detection>`.

    Parameters
    ----------
    window_size : int, default=40
        Size of the sliding window algorithm to extract subsequences as input to named_pipeline.
    step : int, default=5
        Size of the sliding window steps between each window.
    tda_max_dim : int, default=2
        The maximum dimension of the topological feature extraction.
    n_centers_by_dim : int, default=5
        The number of centroids to generate by dimension for vectorizing topological features.
        The resulting embedding will have total dimension =< tda_max_dim * n_centers_by_dim.
        The resulting embedding dimension might be smaller because of the KMeans algorithm in the Archipelago step.
    support_fraction : float, default=None
        The proportion of points to be included in the support of the raw
        MCD estimate. If None, the minimum value of support_fraction will
        be used within the algorithm: `[n_sample + n_features + 1] / 2`.
        Range is (0, 1).
    contamination : float, default=0.1
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Range is (0, 0.5]. Only matters for computing the decision function.
    random_state : int, RandomState instance or None, default=None
        Determines the pseudo random number generator for shuffling
        the data. Pass an int for reproducible results across multiple function
        calls.

    Attributes
    ----------
    topological_embedding_ : object
        TopologicalEmbedding transformer object that is fitted at `fit`.

    Examples
    --------
    >>> import numpy as np
    >>> n_timestamps = 1000
    >>> n_sensors = 20
    >>> timestamps = pd.to_datetime('2024-01-01', utc=True) + pd.Timedelta(1, 'h') * np.arange(n_timestamps)
    >>> X = pd.DataFrame(np.random.random(size=(n_timestamps, n_sensors)), index=timestamps)
    >>> X.iloc[n_timestamps//2:,:10] = -X.iloc[n_timestamps//2:,10:20]
    >>> detector = TopologicalAnomalyDetector(n_centers_by_dim=2, tda_max_dim=1).fit(X)
    >>> anomaly_scores = detector.score_samples(X)
    >>> decision = detector.decision_function(X)
    >>> anomalies = detector.predict(X)
    """

    required_properties: Sequence[str] = ["multiple_time_series"]

    _parameter_constraints: dict = {
        **EllipticEnvelope._parameter_constraints,
        "window_size": [Interval(Integral, left=2, right=None, closed="left")],
        "step": [Interval(Integral, left=1, right=None, closed="left")],
        "tda_max_dim": [Interval(Integral, left=0, right=3, closed="left")],
        "n_centers_by_dim": [Interval(Integral, left=1, right=None, closed="left")],
    }

    def __init__(
            self,
            window_size: int = 100,
            step: int = 5,
            tda_max_dim: int = 2,
            n_centers_by_dim: int = 5,
            support_fraction: float = None,
            contamination: float = 0.1,
            random_state: int = 42,
    ):
        super().__init__(
            support_fraction=support_fraction,
            contamination=contamination,
            random_state=random_state
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
        if not hasattr(X, "index"):
            X = pd.DataFrame(data=X, index=range(X.shape[0]))
        embedding = self.topological_embedding_.fit_transform(X)
        try:
            super().fit(embedding)
        except ValueError as e:
            print(f"Catching {e=}, will increase support fraction.")
            self.support_fraction = 1
            super().fit(embedding)
            self.support_fraction = None
        return self

    def _warped_score_samples(self, X, y=None):  # this exists to retrieve scores before remapping
        """Compute the negative Mahalanobis distances associated with the TopologicalEmbedding representation of X.

        Args
        ----
            X : {array-like, sparse matrix} of shape (n_timestamps, n_sensors)
                Multiple time series to transform, where `n_timestamps` is the number of timestamps
                in the series X, and `n_sensors` is the number of sensors.
            y : Ignored
                Not used, present for API consistency by convention.

        Returns
        -------
        negative_mahal_distances : pandas.DataFrame of shape (n_samples,)
            Opposite of the Mahalanobis distances.
        """
        if not hasattr(X, "index"):
            X = pd.DataFrame(data=X, index=range(X.shape[0]))

        imax = (X.shape[0] - self.window_size) // self.step
        self.padding_length_ = X.shape[0] - (self.step * imax + self.window_size)
        print(f"{X.shape[0]=}, {self.window_size=}, {self.step=}, so running {self.padding_length_=}...")

        embedding = self.topological_embedding_.transform(X=X)
        return super().score_samples(embedding)

    def score_samples(self, X, y=None):
        """Compute the negative Mahalanobis distances associated with the TopologicalEmbedding representation of X.

        Args
        ----
            X : {array-like, sparse matrix} of shape (n_timestamps, n_sensors)
                Multiple time series to transform, where `n_timestamps` is the number of timestamps
                in the series X, and `n_sensors` is the number of sensors.
            y : Ignored
                Not used, present for API consistency by convention.

        Returns
        -------
        negative_mahal_distances : ndarray of shape (n_samples,)
            Opposite of the Mahalanobis distances.
        """
        warped_score_samples = self._warped_score_samples(X)

        unwarped_scores = score_flat_fast_remapping(warped_score_samples, window_size=self.window_size,
                                                    stride=self.step, padding_length=self.padding_length_)
        # print(f"...yields {remapped_scores.shape[0]=}")

        return unwarped_scores


TopologicalAnomalyDetector.transform = TopologicalAnomalyDetector.score_samples
