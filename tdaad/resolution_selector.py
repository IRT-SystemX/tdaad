import warnings
import time
from collections import defaultdict
import logging
from joblib import Parallel, delayed
import numpy as np
from scipy.signal import welch
from numba import njit
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

        candidates = [64, 128, 256]
        # Decide strategy and candidates
        if self.candidates == "adaptive_global":
            strategy = "global"
        elif self.candidates == "adaptive_local":
            strategy = "local"
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
        selector_cls = (
            GlobalResolutionsFinder if strategy == "global" else LocalResolutionsFinder
        )
        self.selector_ = selector_cls(
            resolutions=candidates,
            k=self.k,
            n_jobs=self.n_jobs,
            **self.feature_selector_kwargs,
        )
        self.selector_.fit(X)
        return self

    def transform(self, X):
        if isinstance(self.candidates, int):
            return self.candidates
        else:
            return self.selector_.transform(X)


@njit
def entropy_fast(x, bins=10):
    hist = np.zeros(bins, dtype=np.float64)
    bin_edges = np.linspace(np.min(x), np.max(x), bins + 1)
    n = x.shape[0]

    # Compute histogram manually (since np.histogram not supported by njit)
    for i in range(n):
        xi = x[i]
        # Find the right bin (linear scan since bins small)
        for b in range(bins):
            if bin_edges[b] <= xi < bin_edges[b + 1]:
                hist[b] += 1
                break
        else:
            # If xi == max(x), put it in last bin
            if xi == bin_edges[-1]:
                hist[bins - 1] += 1

    # Normalize histogram
    for i in range(bins):
        hist[i] /= n

    s = 0.0
    for h in hist:
        if h > 0:
            s -= h * np.log(h)
    return s


@njit
def autocorr_fast(x):
    n = x.shape[0]
    mean_x = 0.0
    for i in range(n):
        mean_x += x[i]
    mean_x /= n

    numerator = 0.0
    denominator = 0.0
    for i in range(n - 1):
        numerator += (x[i] - mean_x) * (x[i + 1] - mean_x)
    for i in range(n):
        denominator += (x[i] - mean_x) ** 2

    if denominator == 0:
        return 0.0
    return numerator / denominator


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

    include_variance : bool, default=True
        Whether to include variance feature.

    include_derivative_std : bool, default=True
        Whether to include standard deviation of derivatives feature.

    include_entropy : bool, default=True
        Whether to include entropy feature.

    include_autocorr : bool, default=False
        Whether to include autocorrelation features.

    include_spectral_entropy : bool, default=False
        Whether to include spectral entropy feature.
    """

    def __init__(
        self,
        resolutions,
        k=2,
        score_method="combined",
        n_jobs=-1,
        include_variance=True,
        include_derivative_std=True,
        include_entropy=False,
        include_autocorr=False,
        include_spectral_entropy=False,
    ):
        self.resolutions = resolutions
        self.k = k
        self.score_method = score_method
        self.n_jobs = n_jobs
        self.include_variance = include_variance
        self.include_derivative_std = include_derivative_std
        self.include_entropy = include_entropy
        self.include_autocorr = include_autocorr
        self.include_spectral_entropy = include_spectral_entropy

    def calibrate_features(self, X, sample_size=1):
        """
        Measure feature timing on a few random windows to auto-disable slow features
        before parallel transform.
        """
        if hasattr(X, "values"):
            X = X.values
        if not hasattr(self, "_enabled_features"):
            self._enabled_features = {}

        n = X.shape[0]
        for _ in range(sample_size):
            t = np.random.randint(20, n - 20)
            w = self.resolutions[0]
            hw = w // 2
            window = X[t - hw : t + hw]
            self._time_features(window)
            break  # only run once for now

    def _extract_features(self, window):
        """
        Extracts time-domain features from a given window.
        Auto-disables slow features after first measurement.
        """
        if not hasattr(self, "_enabled_features"):
            raise NotFittedError(
                "This resolution selector is not fitted yet. "
                "Call `.fit(X)` before using `.transform(X)`."
            )
        features = []
        if self._enabled_features.get("variance", False):
            features.append(np.var(window))
        if self._enabled_features.get("entropy", False):
            features.append(entropy_fast(window[:, 0]))
        if self._enabled_features.get("derivative_std", False):
            deriv = np.diff(window, axis=0)
            features.append(np.std(deriv))
        if self._enabled_features.get("autocorr", False):
            features.append(autocorr_fast(window[:, 0]))
        if self._enabled_features.get("spectral_entropy", False):
            freqs, psd = welch(window[:, 0], nperseg=len(window))
            psd = psd / np.sum(psd)
            se = -np.sum(psd * np.log(psd + 1e-8))
            features.append(se)
        return np.array(features)

    def _time_features(self, window):
        """
        Measure time per feature, auto-disable slow ones.
        """
        times = {}
        enabled = {}

        def timed(name, func):
            start = time.perf_counter()
            try:
                func()
                elapsed = time.perf_counter() - start
                times[name] = elapsed
                enabled[name] = True
            except Exception as e:
                times[name] = float("inf")
                enabled[name] = False
                warnings.warn(f"Feature '{name}' failed: {e}")

        # Time each feature
        timed("variance", lambda: np.var(window))
        timed(
            "entropy",
            lambda: -np.sum(
                np.histogram(window, bins=10, density=True)[0]
                * np.log(np.histogram(window, bins=10, density=True)[0] + 1e-8)
            ),
        )
        timed("derivative_std", lambda: np.std(np.diff(window, axis=0)))
        timed(
            "autocorr", lambda: np.correlate(window[:, 0], window[:, 0], mode="full")[0]
        )
        timed(
            "spectral_entropy",
            lambda: __import__("scipy.signal").signal.welch(
                window[:, 0], nperseg=len(window)
            ),
        )

        fastest = min(t for t in times.values() if t > 0)
        threshold = fastest * 10

        logger.info("Feature timing (seconds):")
        for name, t in sorted(times.items(), key=lambda x: x[1]):
            status = "✅" if t <= threshold else "❌ (disabled)"
            logger.info(f"  {name:<16}: {t:.6f} {status}")
            enabled[name] = enabled[name] and (t <= threshold)

        self._enabled_features = enabled

    def _parallel_score_matrix(self, X):
        """
        Compute features for all valid (t, w) pairs in parallel.

        Returns
        -------
        dict[t] -> list of (w, score)
        dict[w] -> list of feature vectors
        """
        n_samples = X.shape[0]
        resolutions = self.resolutions
        half_ws = {w: w // 2 for w in resolutions}

        logger.info(
            f"Starting resolution scoring: X.shape={X.shape}, "
            f"{resolutions=}, score_method='{self.score_method}', k={self.k}"
        )

        def compute(t, w):
            hw = half_ws[w]
            if t - hw < 0 or t + hw > n_samples:
                return (t, w, None)  # invalid
            window = X[t - hw : t + hw]
            features = self._extract_features(window)
            if self.score_method == "mean":
                score = features.mean()
            elif self.score_method == "variance":
                score = features.var()
            elif self.score_method == "combined":
                score = features.mean() * features.var()
            else:
                raise ValueError(f"Invalid score_method: {self.score_method}")
            return (t, w, score)

        # Build valid (t, w) jobs
        jobs = [
            (t, w)
            for w in resolutions
            for t in range(half_ws[w], n_samples - half_ws[w])
        ]

        results = Parallel(n_jobs=self.n_jobs)(delayed(compute)(t, w) for t, w in jobs)

        # Aggregate
        scores_by_t = defaultdict(list)
        scores_by_w = defaultdict(list)
        valid_count = 0
        for item in results:
            if item is None:
                continue
            t, w, score = item
            scores_by_t[t].append((score, w))
            scores_by_w[w].append(score)
            valid_count += 1

        logger.info(f"Completed scoring over {valid_count} valid (t, w) windows.")
        return scores_by_t, scores_by_w

    def fit(self, X, y=None):
        if hasattr(X, "values"):
            X = X.values
        if not hasattr(self, "_enabled_features"):
            # Use first valid window to calibrate features
            for w in self.resolutions:
                hw = w // 2
                if X.shape[0] > 2 * hw:
                    window = X[hw:-hw][:w]  # a valid centered window
                    self._time_features(window)
                    break
        enabled = [k for k, v in self._enabled_features.items() if v]
        logger.info(f"Enabled features: {enabled}")
        return self


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
        return [w for w, _ in top_k]


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
        return top_k_per_time
