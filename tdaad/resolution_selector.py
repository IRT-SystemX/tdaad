from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import entropy
from scipy.signal import welch


class BaseResolutionSelector(BaseEstimator, TransformerMixin):
    """
    Base class for resolution selectors that extract and score
    time-domain features from sliding windows over multiple candidate resolutions.

    Parameters
    ----------
    window_sizes : list of int
        Candidate window sizes (resolutions) to evaluate.

    k : int, default=2
        Number of top resolutions to select.

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
        window_sizes,
        k=2,
        score_method="combined",
        n_jobs=-1,
        include_variance=True,
        include_entropy=True,
        include_derivative_std=True,
        include_autocorr=False,
        include_spectral_entropy=False,
    ):
        self.window_sizes = window_sizes
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


class GlobalResolutionSelector(BaseResolutionSelector):
    """
    Selects top-k global resolutions (window sizes) based on scoring
    feature vectors extracted across all sliding windows.

    The scoring aggregates features computed at every valid window position
    for each resolution, producing a global score per resolution, from which
    the top-k resolutions are selected.

    Parameters
    ----------
    Inherits all parameters from BaseResolutionSelector.
    """

    def fit(self, X, y=None):
        return self

    def _score_resolution(self, X, w):
        half_w = w // 2
        n_samples = X.shape[0]
        features = []
        for t in range(half_w, n_samples - half_w):
            window = X[t - half_w : t + half_w]
            f = self._extract_features(window)
            features.append(f)
        if not features:
            return (w, -np.inf)
        features = np.vstack(features)
        if self.score_method == "mean":
            score = features.mean()
        elif self.score_method == "variance":
            score = features.var()
        elif self.score_method == "combined":
            score = features.mean() * features.var()
        else:
            raise ValueError(f"Invalid score_method: {self.score_method}")
        return (w, score)

    def transform(self, X):
        if hasattr(X, "values"):
            X = X.values
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._score_resolution)(X, w) for w in self.window_sizes
        )
        top_k = sorted(results, key=lambda x: x[1], reverse=True)[: self.k]
        return [w for w, _ in top_k]


class LocalResolutionSelector(BaseResolutionSelector):
    """
    Selects the top-k local resolutions (window sizes) per time index by computing
    features on sliding windows centered at each time point for each candidate resolution.

    For each time index, scores are computed for all candidate resolutions, and
    the top-k resolutions with highest scores are returned, resulting in a sequence
    of local best resolutions over time.

    Parameters
    ----------
    Inherits all parameters from BaseResolutionSelector.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "values"):
            X = X.values
        n_samples = X.shape[0]
        n_channels = X.shape[1]
        k = self.k
        window_sizes = self.window_sizes

        # Precompute half window sizes for convenience
        half_ws = {w: w // 2 for w in window_sizes}

        # For each time index, compute features for all window sizes
        def features_at_t(t):
            feats_per_resolution = []
            for w in window_sizes:
                hw = half_ws[w]
                if t - hw < 0 or t + hw > n_samples:
                    feats_per_resolution.append(
                        (-np.inf, w)
                    )  # invalid window, score=-inf
                    continue
                window = X[t - hw : t + hw]
                f = self._extract_features(window)
                # score features as mean/var/combined
                if self.score_method == "mean":
                    score = f.mean()
                elif self.score_method == "variance":
                    score = f.var()
                elif self.score_method == "combined":
                    score = f.mean() * f.var()
                else:
                    raise ValueError(f"Invalid score_method: {self.score_method}")
                feats_per_resolution.append((score, w))
            # pick top k scores (score, w) tuples
            feats_per_resolution.sort(key=lambda x: x[0], reverse=True)
            return [w for _, w in feats_per_resolution[:k]]

        # Parallel over time indices
        top_k_per_time = Parallel(n_jobs=self.n_jobs)(
            delayed(features_at_t)(t) for t in range(n_samples)
        )
        return top_k_per_time
