import warnings
import time
from collections import defaultdict
import logging
from joblib import Parallel, delayed

import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin

from tdaad._feature_functions import (
    feature_variance,
    feature_derivative_std,
    feature_entropy,
    feature_autocorr,
    feature_spectral_entropy,
    feature_rms,
    feature_channel_correlation,
    feature_dominant_freq,
    feature_skewness,
    feature_kurtosis,
    feature_total_persistence_corr,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Dictionary of available features
FEATURE_FUNCTIONS = {
    "variance": feature_variance,
    "derivative_std": feature_derivative_std,
    "entropy": feature_entropy,
    "autocorr": feature_autocorr,
    "spectral_entropy": feature_spectral_entropy,
    "rms": feature_rms,
    "channel_corr": feature_channel_correlation,
    "dominant_freq": feature_dominant_freq,
    "skewness": feature_skewness,
    "kurtosis": feature_kurtosis,
    "totalpers": feature_total_persistence_corr,
}


class ResolutionSelector(BaseEstimator, TransformerMixin):
    """
    Selects resolution (window size(s)) for downstream algorithms.

    Parameters
    ----------
    resolution : int, str, list of int, or dict
        - int: use fixed resolution
        - "adaptive_global": select top-k global resolutions using default candidates
        - "adaptive_local": select top-k local resolutions using default candidates
        - list of int: adaptive global resolution selection over provided candidates
        - dict: {"global": [...]} or {"local": [...]} — candidates + strategy

    k : int, default=2
        Number of top resolutions to select (adaptive only).

    n_jobs : int, default=-1
        Number of parallel jobs for adaptive selectors.

    feature_selector_kwargs : dict
        Additional kwargs for the resolution selector classes.

    Methods
    -------
    fit(X, y=None):
        Fits selector if adaptive.

    transform(X):
        Returns:
            - int if fixed window size
            - list of int for global adaptive
            - list of list of int for local adaptive
    """

    def __init__(
        self,
        candidates,
        k=2,
        n_jobs=-1,
        feature_selector_kwargs=None,
    ):
        self.candidates = candidates
        self.k = k
        self.n_jobs = n_jobs
        self.feature_selector_kwargs = feature_selector_kwargs or {}

    def fit(self, X, y=None):
        if hasattr(X, "values"):
            X = X.values

        if isinstance(self.candidates, int):
            return self  # fixed window size

        default_candidates = [64, 128, 256]

        # Decide strategy and candidates
        if self.candidates == "adaptive_global":
            strategy = "global"
            candidates = default_candidates
        elif self.candidates == "adaptive_local":
            strategy = "local"
            candidates = default_candidates
        elif isinstance(self.candidates, list):
            strategy = "global"
            candidates = self.candidates
        elif isinstance(self.candidates, dict):
            if "global" in self.candidates:
                strategy = "global"
                candidates = self.candidates["global"]
            elif "local" in self.candidates:
                strategy = "local"
                candidates = self.candidates["local"]
            else:
                raise ValueError("Dict window_size must have 'global' or 'local' key.")
        else:
            raise ValueError(f"Invalid window_size value: {self.candidates}")

        # Initialize appropriate selector
        finder_cls = (
            GlobalResolutionsFinder if strategy == "global" else LocalResolutionsFinder
        )
        self.finder_ = finder_cls(
            resolutions=candidates,
            k=self.k,
            n_jobs=self.n_jobs,
            **self.feature_selector_kwargs,
        )
        self.finder_.fit(X)
        return self

    def transform(self, X):
        if isinstance(self.candidates, int):
            return self.candidates
        else:
            best_resolutions, scores = self.finder_.transform(X)
            return best_resolutions


class BaseResolutionsFinder(BaseEstimator, TransformerMixin):
    """
    Base class for resolution finding that extract and score
    time-domain features from sliding windows over multiple candidate resolutions.

    Parameters
    ----------
    resolutions : list of int
        Candidate window sizes (resolutions) to evaluate.

    k : int, default=2
        Number of top resolutions to find.

    score_method : str, default='combined'
        Method to aggregate feature vectors into a single score.
        Options: 'mean', 'variance', or 'combined' (mean * variance).

    n_jobs : int, default=-1
        Number of parallel jobs to run for feature extraction or scoring.

    features : list of str, optional
        List of feature names to use. Defaults to all available in FEATURE_FUNCTIONS.
    """

    def __init__(
        self,
        resolutions,
        k=2,
        feat_time_threshold=1e-3,
        score_method="combined",
        n_jobs=-1,
        features=None,
    ):
        self.resolutions = resolutions
        self.k = k
        self.feat_time_threshold = feat_time_threshold
        self.score_method = score_method
        self.n_jobs = n_jobs
        self.features = features or list(FEATURE_FUNCTIONS.keys())

    def _to_array(self, X):
        return X.values if hasattr(X, "values") else X

    def _time_features(self, window):
        times = {}
        enabled = {}

        for name in self.features:
            func = FEATURE_FUNCTIONS.get(name)
            if not func:
                warnings.warn(f"Unknown feature: {name}")
                continue
            try:
                start = time.perf_counter()
                func(window)
                elapsed = time.perf_counter() - start
                times[name] = elapsed
                enabled[name] = elapsed <= self.feat_time_threshold
            except Exception as e:
                times[name] = float("inf")
                enabled[name] = False
                warnings.warn(f"Feature '{name}' failed: {e}")

        logger.info("Feature timing (seconds):")
        for name, t in sorted(times.items(), key=lambda x: x[1]):
            status = "✅" if t <= self.feat_time_threshold else "❌ (disabled)"
            logger.info(f"  {name:<16}: {t:.6f} {status}")

        self._enabled_features = {
            name: is_enabled for name, is_enabled in enabled.items() if is_enabled
        }

    def _extract_window(self, X, t, w):
        hw = w // 2
        if t - hw < 0 or t + hw > X.shape[0]:
            return None
        return X[t - hw : t + hw]

    def _calibrate_on_first_valid_window(self, X):
        for w in self.resolutions:
            window = self._extract_window(X, w // 2, w)
            if window is None:
                continue
            self._time_features(window)
            break

    def fit(self, X, y=None):
        X = self._to_array(X)
        self._calibrate_on_first_valid_window(X)

        enabled = [k for k, v in self._enabled_features.items() if v]
        if not enabled:
            raise RuntimeError(
                "BaseResolutionsFinder: All features were disabled after calibration. "
                "Check your input data, feature functions, or 'feat_time_threshold'."
            )
        logger.info(f"Enabled features: {enabled}")
        return self

    def _extract_features(self, window):
        return np.array(
            [
                FEATURE_FUNCTIONS[name](window)
                for name in self._enabled_features
                if self._enabled_features[name]
            ]
        )

    def _parallel_score_matrix(self, X):
        if not self._enabled_features:
            raise NotFittedError("Call `.fit(X)` before using `.transform(X)`.")

        X = self._to_array(X)
        n_samples = X.shape[0]
        half_ws = {w: w // 2 for w in self.resolutions}

        logger.info(
            f"Starting resolution scoring: X.shape={X.shape}, "
            f"resolutions={self.resolutions}, score_method='{self.score_method}', k={self.k}"
        )

        jobs = [
            (t, w)
            for w in self.resolutions
            for t in range(half_ws[w], n_samples - half_ws[w])
        ]

        def compute(t, w):
            window = self._extract_window(X, t, w)
            if window is None:
                return (t, w, None)
            features = self._extract_features(window)
            score = features.mean()
            return (t, w, score)

        results = Parallel(n_jobs=self.n_jobs)(delayed(compute)(t, w) for t, w in jobs)

        scores_by_t = defaultdict(list)
        scores_by_w = defaultdict(list)
        valid_count = 0

        for t, w, score in results:
            if score is None:
                continue
            scores_by_t[t].append((score, w))
            scores_by_w[w].append(score)
            valid_count += 1

        logger.info(f"Completed scoring over {valid_count} valid (t, w) windows.")
        return scores_by_t, scores_by_w


class GlobalResolutionsFinder(BaseResolutionsFinder):
    """
    Finds top-k global resolutions (window sizes) based on scoring
    feature vectors extracted across all sliding windows.

    The scoring aggregates features computed at every valid window position
    for each resolution, producing a global score per resolution, from which
    the top-k resolutions are found.

    Parameters
    ----------
    Inherits all parameters from BaseResolutionsFinder.
    """

    def fit(self, X, y=None):
        return super().fit(X, y)

    def transform(self, X):
        if hasattr(X, "values"):
            X = X.values
        _, scores_by_w = self._parallel_score_matrix(X)

        results = []
        for w, scores in scores_by_w.items():
            scores = np.array(scores)
            if self.score_method == "mean":
                agg = scores.mean()
            elif self.score_method == "variance":
                agg = scores.var()
            elif self.score_method == "combined":
                agg = scores.mean() * scores.var()
            logger.debug(f"Resolution {w}: aggregated score = {agg:.4f}")
            results.append((w, agg))

        top_k = sorted(results, key=lambda x: x[1], reverse=True)[: self.k]
        logger.info(f"Top-{self.k} resolutions (global): {[w for w, _ in top_k]}")
        return [w for w, _ in top_k], results


class LocalResolutionsFinder(BaseResolutionsFinder):
    """
    Finds the top-k local resolutions (window sizes) per time index by computing
    features on sliding windows centered at each time point for each candidate resolution.

    For each time index, scores are computed for all candidate resolutions, and
    the top-k resolutions with highest scores are returned, resulting in a sequence
    of local best resolutions over time.

    Parameters
    ----------
    Inherits all parameters from BaseResolutionsFinder.
    """

    def fit(self, X, y=None):
        return super().fit(X, y)

    def transform(self, X):
        if hasattr(X, "values"):
            X = X.values
        scores_by_t, _ = self._parallel_score_matrix(X)

        n_samples = X.shape[0]
        top_k_per_time = []
        for t in range(n_samples):
            top_k = sorted(scores_by_t.get(t, []), key=lambda x: x[0], reverse=True)[
                : self.k
            ]
            top_k_per_time.append([w for _, w in top_k])

        logger.info(f"Computed top-{self.k} resolutions per time index (local).")
        return top_k_per_time, scores_by_t
