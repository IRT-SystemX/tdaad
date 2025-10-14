import numpy as np
from scipy.signal import welch
from numba import njit


def feature_variance(window):
    return np.var(window)


def feature_derivative_std(window):
    return np.std(np.diff(window, axis=0))


@njit
def entropy_fast(window, bins=10):
    x = window[:, 0]
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


def feature_spectral_entropy(window):
    freqs, psd = welch(window[:, 0], nperseg=len(window))
    psd = psd / np.sum(psd)
    return -np.sum(psd * np.log(psd + 1e-8))


@njit
def autocorr_fast(window):
    x = window[:, 0]
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
