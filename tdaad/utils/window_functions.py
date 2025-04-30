"""Window Functions."""

# Author: Martin Royer

import numpy as np
import pandas as pd

from tqdm import tqdm


def sliding_window_ppl(data, pipeline, step=5, window_size=120):
    """Applies a pipeline to timeseries data chunks using the Sliding Window algorithm.

    @param data: pd.DataFrame with index to apply named_pipeline to.
    @param window_size: size of the sliding window algorithm to extract subsequences as input to named_pipeline.
    @param step: size of the sliding window steps between each window.
    @param pipeline: pipeline (sequence of operators that have a `name` attribute) to apply to each window.
    @return: pd.DataFrame that maps data to the result of applying named_pipeline to window view of data.
    """

    swv = np.lib.stride_tricks.sliding_window_view(data.index, window_size)[::step, :]
    result = {}
    for window in tqdm(swv):
        result[str(tuple(window))] = pipeline.transform(data.loc[window].copy())
    post_result = pd.DataFrame.from_dict(result, orient="index")
    return post_result
