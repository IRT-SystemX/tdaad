"""Window Functions."""

# Author: Martin Royer

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
import hashlib


def hash_window(window: np.ndarray) -> str:
    """Hash encoding of sliding window index."""
    return hashlib.sha1(window.view(np.uint8)).hexdigest()


def _sliding_window_3D_view(data, window_size, step):
    num_rows, num_features = data.shape
    # max_windows = (num_rows - window_size) // step + 1

    shape = (num_rows - window_size + 1, window_size, num_features)
    strides = (data.strides[0], data.strides[0], data.strides[1])

    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return windows[::step]


def sliding_window_ppl_pp(
    data, pipeline, step=5, window_size=120, parallel=True, n_jobs=-1
):
    """Applies a pipeline to timeseries data chunks using the Sliding Window algorithm.

    @param data: pd.DataFrame with index to apply named_pipeline to.
    @param window_size: size of the sliding window algorithm to extract subsequences as input to named_pipeline.
    @param step: size of the sliding window steps between each window.
    @param pipeline: pipeline (sequence of operators that have a `name` attribute) to apply to each window.
    @param parallel: boolean to use joblib.Parallel library on the sliding window algorithm.
    @param n_jobs: int to set the joblib.Parallel maximum number of concurrently running jobs.
    @return: pd.DataFrame that maps data to the result of applying named_pipeline to window view of data.
    """
    results = []
    windows = _sliding_window_3D_view(data.to_numpy(), window_size, step)

    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(lambda wdw: (hash_window(wdw), pipeline.transform(wdw)))(w)
            for w in windows
        )
    else:
        results = [(hash_window(w), pipeline.transform(w)) for w in windows]
    post_result = pd.DataFrame.from_dict(dict(results), orient="index")
    return post_result
