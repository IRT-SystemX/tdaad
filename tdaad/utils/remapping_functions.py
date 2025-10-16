"""Remapping Functions."""

# Author: Martin Royer

import numpy as np


def score_flat_fast_remapping(scores, window_size, stride, padding_length=0):
    """
    Remap window-level anomaly scores to a flat sequence of per-time-step scores.

    Parameters
    ----------
    scores : array-like of shape (n_windows,)
        Anomaly scores for each window. Can be a pandas Series or NumPy array.

    window_size : int
        Size of the sliding window.

    stride : int
        Step size between windows.

    padding_length : int, optional (default=0)
        Extra length to pad the output array (typically at the end of a signal).

    Returns
    -------
    remapped_scores : np.ndarray of shape (n_timestamps + padding_length,)
        Flattened anomaly scores with per-timestep resolution. NaN values (from
        positions not covered by any window) are replaced with 0.
    """
    # Ensure scores is a NumPy array
    if hasattr(scores, "values"):
        scores = scores.values

    n_windows = len(scores)

    # Compute begin and end indices for each window
    begins = np.arange(n_windows) * stride
    ends = begins + window_size

    # Output length based on last window + padding
    total_length = ends[-1] + padding_length
    remapped_scores = np.full(total_length, np.nan)

    # Find all unique intersection points between windows
    intersections = np.unique(np.concatenate((begins, ends)))

    # For each interval between two intersections, find overlapping windows and sum their scores
    for left, right in zip(intersections[:-1], intersections[1:]):
        overlapping = (begins <= left) & (right <= ends)
        if np.any(overlapping):
            remapped_scores[left:right] = np.nansum(scores[overlapping])

    # Replace NaNs (unscored positions) with 0
    np.nan_to_num(remapped_scores, copy=False)

    return remapped_scores
