"""Topological Embedding Transformers."""

# Author: Martin Royer

from functools import partial
import multiprocessing

from joblib import Parallel, delayed

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

from gudhi.representations.vector_methods import Atol

from tdaad.utils.tda_functions import transform_to_persistence_diagram


class PandasAtol(Atol):
    """
    ATOL vectorization with pandas-compatible input handling.

    This subclass converts pandas Series or DataFrame inputs to NumPy arrays
    before delegating to the base Atol implementation, avoiding warnings
    caused by np.concatenate on pandas objects.
    """

    def fit(self, X, y=None, sample_weight=None):
        if hasattr(X, "values"):
            X = X.values
        return super().fit(X, y=y, sample_weight=sample_weight)


def effective_n_jobs(default=-1):
    """Return a safe n_jobs based on whether we're already inside a worker process."""
    try:
        # If we're in the main process, multiprocessing.current_process().name == 'MainProcess'
        if multiprocessing.current_process().name != "MainProcess":
            return 1  # inside pool → avoid nested parallelism
    except Exception:
        pass
    return default


def sliding_window_ppl_pp(
    data,
    func,
    window_size=120,
    step=5,
    n_jobs=-1,
):
    """
    Apply a processing function to sliding windows over time series data in parallel.

    Each window is identified by its starting index in the original DataFrame,
    providing uniqueness and full traceability.

    Parameters
    ----------
    data : pd.DataFrame
        Input 2D time series data with shape (num_rows, num_features).
    func : callable
        Function applied to each window. Receives a NumPy array of shape
        (window_size, num_features).
    window_size : int, optional
        Number of consecutive rows per window.
    step : int, optional
        Stride between successive windows.
    n_jobs : int, optional
        Number of parallel jobs (joblib).

    Returns
    -------
    pd.DataFrame
        One row per window. Index = window start index in `data`.
    """

    values = data.to_numpy()
    n_rows = len(data)

    # Compute window start positions
    start_indices = range(0, n_rows - window_size + 1, step)

    def process_window(start_idx):
        window = values[start_idx : start_idx + window_size]
        return start_idx, func(window)

    results = Parallel(n_jobs=n_jobs)(delayed(process_window)(i) for i in start_indices)

    # Convert results to DataFrame
    result_dict = dict(results)
    result_df = pd.DataFrame.from_dict(result_dict, orient="index")

    result_df.index.name = "window_start"

    return result_df


class TopologicalEmbedding(BaseEstimator, TransformerMixin):
    """Topological embedding for multiple time series.

    Slices time series into smaller time series windows, forms an affinity matrix on each window
    and applies a Rips procedure to produce persistence diagrams for each affinity
    matrix. Then uses Atol [ref:Atol] on each dimension through the
    gudhi.representation.Archipelago representation to produce topological vectorization.

    Read more in the :ref:`User Guide <topological_embedding>`.

    Parameters
    ----------
    window_size : int, default=40
        Size of the sliding window algorithm to extract subsequences as input to named_pipeline.
    step : int, default=5
        Size of the sliding window steps between each window.
    n_centers_by_dim : int, default=5
        The number of centroids to generate by dimension for vectorizing topological features.
        The resulting embedding will have total dimension =< tda_max_dim * n_centers_by_dim.
        The resulting embedding dimension might be smaller because of the KMeans algorithm in the Archipelago step.
    tda_max_dim : int, default=2
        The maximum dimension of the topological feature extraction.

    Examples
    ----------
    >>> n_timestamps = 100
    >>> n_sensors = 5
    >>> timestamps = pd.to_datetime('2024-01-01', utc=True) + pd.Timedelta(1, 'h') * np.arange(n_timestamps)
    >>> X = pd.DataFrame(np.random.random(size=(n_timestamps, n_sensors)), index=timestamps)
    >>> TopologicalEmbedding(n_centers_by_dim=2, tda_max_dim=1).fit_transform(X)
    """

    def __init__(
        self,
        window_size: int = 40,
        step: int = 5,
        tda_max_dim: int = 2,
        n_centers_by_dim: int = 5,
        parallel="auto",
    ):
        self.window_size = window_size
        self.step = step
        self.tda_max_dim = tda_max_dim
        self.n_centers_by_dim = n_centers_by_dim
        self.parallel = parallel
        if parallel == "auto":
            n_jobs = effective_n_jobs(default=-1)
        elif parallel is True:
            n_jobs = -1
        else:
            n_jobs = 1
        self.n_jobs = n_jobs

    def _build_pipeline(self):
        steps = []
        steps.append(("Standard scaler", StandardScaler()))
        func = partial(transform_to_persistence_diagram, tda_max_dim=self.tda_max_dim)
        steps.append(
            (
                "Sliding persistence diagram transformer",
                FunctionTransformer(
                    func=sliding_window_ppl_pp,
                    kw_args={
                        "window_size": self.window_size,
                        "step": self.step,
                        "func": func,
                        "n_jobs": self.n_jobs,
                    },
                ),
            )
        )
        steps.append(
            (
                "Archipelago",
                ColumnTransformer(
                    [
                        (
                            f"Atol{i}",
                            PandasAtol(
                                quantiser=KMeans(
                                    n_clusters=self.n_centers_by_dim,
                                    random_state=202312,
                                    n_init="auto",
                                )
                            ),
                            i,
                        )
                        for i in range(self.tda_max_dim + 1)
                    ]
                ),
            )
        )
        return Pipeline(steps).set_output(transform="pandas")

    def fit(self, X, y=None):
        """
        Fit the internal pipeline to the data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input feature matrix.

        y : array-like, optional
            Target values (not used here, but accepted for compatibility with sklearn).

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        return self

    def transform(self, X):
        """
        Apply transformations to the input data using the fitted pipeline.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data to transform.

        Returns
        -------
        X_transformed : array-like or DataFrame
            Transformed data.
        """
        return self.pipeline_.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Returns
        -------
        X_transformed : array-like
        """
        return self.fit(X, y).transform(X)
