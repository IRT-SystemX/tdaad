from collections import defaultdict
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import entropy
from scipy.signal import welch


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

    include_entropy : bool, default=True
        Whether to include entropy feature.

    include_derivative_std : bool, default=True
        Whether to include standard deviation of derivatives feature.

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
        include_entropy=True,
        include_derivative_std=True,
        include_autocorr=False,
        include_spectral_entropy=False,
    ):
        self.resolutions = resolutions
        self.k = k
        self.score_method = score_method
        self.n_jobs = n_jobs
        self.include_variance = include_variance
        self.include_entropy = include_entropy
        self.include_derivative_std = include_derivative_std
        self.include_autocorr = include_autocorr
        self.include_spectral_entropy = include_spectral_entropy

    def _extract_features(self, window):
        feats = []
        if self.include_variance:
            feats.append(np.var(window, axis=0).mean())
        if self.include_entropy:
            ent = [
                entropy(np.histogram(window[:, i], bins=10, density=True)[0] + 1e-12)
                for i in range(window.shape[1])
            ]
            feats.append(np.mean(ent))
        if self.include_derivative_std:
            feats.append(np.std(np.diff(window, axis=0), axis=0).mean())
        if self.include_autocorr:
            ac_vals = []
            for i in range(window.shape[1]):
                x = window[:, i] - np.mean(window[:, i])
                if len(x) >= 3:
                    ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
                    ac2 = np.corrcoef(x[:-2], x[2:])[0, 1]
                    ac_vals.extend([ac1, ac2])
            feats.append(np.mean(ac_vals) if ac_vals else 0.0)
        if self.include_spectral_entropy:
            se_vals = []
            for i in range(window.shape[1]):
                f, Pxx = welch(window[:, i], nperseg=len(window))
                Pxx += 1e-12
                Pxx /= Pxx.sum()
                se_vals.append(-np.sum(Pxx * np.log2(Pxx)))
            feats.append(np.mean(se_vals))
        return np.array(feats)

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
        return self

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
        return self

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
