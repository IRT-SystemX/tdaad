"""Remapping Functions."""

# Author: Martin Royer

import numpy as np


def score_flat_fast_remapping(scores, window_size, stride, padding_length=0):
    """Univariate score remapping without window indices and with precomputed padding_length."""
    if hasattr(scores, "index"):
        scores = scores.values
    begins = np.array([i * stride for i in range(scores.shape[0])])
    ends = begins + window_size

    remapped_scores = np.full(shape=stride * (scores.shape[0] - 1) + window_size + padding_length, fill_value=np.nan)

    # iterate over window intersection points, interpolate (vote) in-between intersection and next_intersection
    intersections = np.unique(np.r_[begins, ends])
    for intersection, next_intersection in zip(intersections[:-1], intersections[1:]):
        window_indices = np.flatnonzero((begins <= intersection) & (next_intersection < ends + 1))
        remapped_scores[intersection:next_intersection] = np.nansum(scores[window_indices])    # or nanmedian

    # replace unknown values with 0 = normal (for the padding at the end)
    np.nan_to_num(remapped_scores, copy=False)
    return remapped_scores
