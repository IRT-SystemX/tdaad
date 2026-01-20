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

    Converts pandas inputs to NumPy arrays before calling the base
    implementation to avoid warnings caused by np.concatenate on pandas objects.
    """

    def fit(self, X, y=None, sample_weight=None):
        if hasattr(X, "values"):
            X = X.values
        return super().fit(X, y=y, sample_weight=sample_weight)


def resolve_n_jobs(parallel):
    """
    Resolve n_jobs safely, avoiding nested parallelism.
    """
    if parallel == "auto":
        try:
            if multiprocessing.current_process().name != "MainProcess":
                return 1
        except Exception:
            pass
        return -1

    if parallel is True:
        return -1

    return 1


def sliding_window_apply(
    data: pd.DataFrame,
    window_fn,
    window_size: int,
    step: int,
    n_jobs: int,
) -> pd.DataFrame:
    """
    Apply a processing function to sliding windows over time series data in parallel.

    Returns a DataFrame indexed by window start index.
    """
    values = data.to_numpy()
    n_rows = len(data)

    start_indices = range(0, n_rows - window_size + 1, step)

    def process_window(start_idx):
        window = values[start_idx : start_idx + window_size]
        return start_idx, window_fn(window)

    results = Parallel(n_jobs=n_jobs)(delayed(process_window)(i) for i in start_indices)

    result_df = pd.DataFrame.from_dict(dict(results), orient="index")
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
    tda_max_dim : int, default=2
        The maximum dimension of the topological feature extraction.
    n_centers_by_dim : int, default=5
        The number of centroids to generate by dimension for vectorizing topological features.
        The resulting embedding will have total dimension =< tda_max_dim * n_centers_by_dim.
        The resulting embedding dimension might be smaller because of the KMeans algorithm in the Archipelago step.

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

    def _build_pipeline(self):
        n_jobs = resolve_n_jobs(self.parallel)
        persistence_fn = partial(
            transform_to_persistence_diagram,
            tda_max_dim=self.tda_max_dim,
        )

        window_transformer = FunctionTransformer(
            func=sliding_window_apply,
            kw_args=dict(
                window_fn=persistence_fn,
                window_size=self.window_size,
                step=self.step,
                n_jobs=n_jobs,
            ),
        )

        archipelago = ColumnTransformer(
            transformers=[
                (
                    f"atol_dim_{i}",
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
        )

        pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("sliding_persistence", window_transformer),
                ("archipelago", archipelago),
            ]
        )

        return pipeline.set_output(transform="pandas")

    def fit(self, X, y=None):
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline_.transform(X)
