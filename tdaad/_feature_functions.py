import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.fft import fft

from numba import njit

from ripser import ripser


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


def feature_variance(window):
    return np.var(window, axis=0).mean()  # mean over channels


def feature_derivative_std(window):
    return np.std(np.diff(window, axis=0), axis=0).mean()


def feature_entropy(window, bins=10):
    return np.mean([entropy_fast(window[:, i], bins) for i in range(window.shape[1])])


def feature_spectral_entropy(window):
    entropies = []
    for i in range(window.shape[1]):
        freqs, psd = welch(window[:, i], nperseg=len(window))
        psd /= np.sum(psd)
        entropies.append(-np.sum(psd * np.log(psd + 1e-8)))
    return np.mean(entropies)


def feature_autocorr(window):
    return np.mean([autocorr_fast(window[:, i]) for i in range(window.shape[1])])


def feature_rms(window):
    return np.sqrt(np.mean(window**2, axis=0)).mean()


def feature_channel_correlation(window):
    if window.shape[1] < 2:
        return 0.0  # not enough channels
    corr_matrix = np.corrcoef(window.T)
    # Take upper triangle without diagonal
    upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    return np.mean(np.abs(upper))  # mean absolute correlation


def feature_dominant_freq(window):
    dom_freqs = []
    for i in range(window.shape[1]):
        signal = window[:, i]
        spectrum = np.abs(fft(signal))
        dom_freq = np.argmax(spectrum[1:])  # skip DC component
        dom_freqs.append(dom_freq)
    return np.mean(dom_freqs)


def feature_skewness(window):
    return np.mean(skew(window, axis=0))


def feature_kurtosis(window):
    return np.mean(kurtosis(window, axis=0))


def feature_total_persistence_corr(window, maxdim=1, p=1):
    """
    Computes total persistence using 1 - correlation as a distance matrix
    between channels in a multivariate time window.

    Parameters
    ----------
    window : np.ndarray of shape (n_samples, n_channels)
        Time series window.

    dim : int, default=1
        Homology dimension for persistent homology (0, 1, etc).

    p : int, default=1
        Power to raise each persistence value before summing.

    Returns
    -------
    float
        Total persistence.
    """
    # Compute channel-to-channel correlation
    target = 1.0 - np.corrcoef(window.T)
    dgms = ripser(target, distance_matrix=True, maxdim=maxdim)["dgms"]
    total_persistence = 0.0
    for dim in range(maxdim):
        dgm = dgms[dim]
        if dgm.size > 0:
            dgm = dgm[~np.isinf(dgm[:, 1]), :]
            pers_dim = dgm[:, 1] - dgm[:, 0]
            total_persistence += np.sum(pers_dim**p)
    return total_persistence
