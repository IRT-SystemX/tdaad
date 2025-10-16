"""Window Functions."""

# Author: Martin Royer

import hashlib
import numpy as np
import pandas as pd

from joblib import Parallel, delayed


def hash_window(window: np.ndarray) -> str:
    """Hash encoding of sliding window index."""
    return hashlib.sha1(np.ascontiguousarray(window).view(np.uint8)).hexdigest()


def sliding_window_3D_view(data, window_size, step):
    """
    Create a 3D sliding window view over a 2D array without copying data.

    This function returns overlapping sliding windows from a 2D input array
    using NumPy's `as_strided` for memory-efficient view creation. The resulting
    3D array has shape `(num_windows, window_size, num_features)`, where each
    window contains `window_size` rows from the original data, spaced by `step`.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array of shape (num_rows, num_features).
    window_size : int
        Number of consecutive rows to include in each window.
    step : int
        Step size (stride) between successive windows.

    Returns
    -------
    np.ndarray
        3D array of shape (num_windows, window_size, num_features), where each
        entry is a view into the original `data`.

    Notes
    -----
    - This function uses `np.lib.stride_tricks.as_strided`, which does not copy
      the data. Be cautious when modifying the output array.
    - The number of windows returned is calculated as:
      floor((num_rows - window_size) / step) + 1
    """
    num_rows, num_features = data.shape

    shape = (num_rows - window_size + 1, window_size, num_features)
    strides = (data.strides[0], data.strides[0], data.strides[1])

    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return windows[::step]


def sliding_window_ppl_pp(data, func, window_size=120, step=5, n_jobs=-1):
    """
    Apply a processing function to sliding windows over time series data in parallel.

    This function slices a 2D time series (Pandas DataFrame) into overlapping windows,
    applies a user-defined function (`func`) to each window in parallel, and returns
    the aggregated results as a DataFrame indexed by a hash of each window.

    Parameters
    ----------
    data : pd.DataFrame
        Input 2D time series data with shape (num_rows, num_features). Must be indexable
        and convertible to a NumPy array.
    func : callable
        Function to apply to each window. It should accept a NumPy array of shape
        (window_size, num_features) and return a result (e.g., scalar, dict, or Series).
    step : int, optional (default=5)
        Step size (stride) between successive windows.
    window_size : int, optional (default=120)
        Number of consecutive rows to include in each sliding window.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs to run. Passed to `joblib.Parallel`.
        Use -1 to utilize all available CPUs.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row corresponds to a window. The index is a unique hash of the
        window content (via `hash_window`), and each row contains the result of `func(w)`.

    Notes
    -----
    - Requires the helper function `_sliding_window_3D_view()` to create window views.
    - Requires a `hash_window()` function that generates a unique, hashable ID for a window.
    - Function assumes that `func(w)` returns something convertible to a dictionary-like format
      (e.g., dict, Series) for use with `pd.DataFrame.from_dict`.

    Example
    -------
    >>> def mean_window(w):
    ...     return {'mean': w.mean()}
    >>> result = sliding_window_ppl_pp(X, func=mean_window, window_size=10, step=2)
    >>> print(result.head())
    """
    windows = sliding_window_3D_view(data.to_numpy(), window_size, step)

    results = Parallel(n_jobs=n_jobs)(
        delayed(lambda wdw: (hash_window(wdw), func(wdw)))(w) for w in windows
    )

    post_result = pd.DataFrame.from_dict(dict(results), orient="index")
    return post_result
