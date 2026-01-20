"""Topological Embedding Transformers."""

# Author: Martin Royer
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

from gudhi.representations.vector_methods import Atol
from gudhi.sklearn.rips_persistence import RipsPersistence


def numpy_data_to_similarity(X, filter_nan=True):
    r"""Transforms numpy matrix X into similarity matrix :math:`1-\mathbf{Corr}(X)`."""
    target = 1 - np.corrcoef(X, rowvar=False)
    # this filters when a variable is constant -> nan on all rows
    nanrowcols = np.isnan(target).all(axis=0) if filter_nan else ~target.any(axis=0)
    return target[~nanrowcols][:, ~nanrowcols]


class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    """
    Slice a 2D numpy array into overlapping windows.

    Output: list of 2D numpy arrays, one per window.
    """

    def __init__(self, window_size=40, step=5):
        self.window_size = window_size
        self.step = step

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_rows = X.shape[0]
        self.window_index_ = list(range(0, n_rows - self.window_size + 1, self.step))
        return [X[i : i + self.window_size] for i in self.window_index_]


class TopologicalEmbedding(BaseEstimator, TransformerMixin):
    """
    Topological embedding for multivariate time series using sliding windows,
    persistent homology (Rips), and ATOL vectorization.

    Pipeline:
        Sliding windows -> similarity -> RipsPersistence -> ColumnTransformer(Atol)
    """

    def __init__(
        self,
        window_size: int = 40,
        step: int = 5,
        tda_max_dim: int = 2,
        n_centers_by_dim: int = 5,
        filter_nan: bool = True,
    ):
        self.window_size = window_size
        self.step = step
        self.tda_max_dim = tda_max_dim
        self.n_centers_by_dim = n_centers_by_dim
        self.filter_nan = filter_nan

    def _build_pipeline(self):
        # FunctionTransformer to convert windows -> distance/similarity matrices
        similarity_fn = FunctionTransformer(
            func=lambda X_list: [
                numpy_data_to_similarity(x, filter_nan=self.filter_nan) for x in X_list
            ]
        )

        # Batched RipsPersistence
        rips_transformer = RipsPersistence(
            homology_dimensions=range(self.tda_max_dim + 1),
            input_type="lower distance matrix",
        )

        # ColumnTransformer: one Atol per homology dimension
        archipelago_transformer = ColumnTransformer(
            transformers=[
                (
                    f"atol_dim_{i}",
                    Atol(
                        quantiser=KMeans(
                            n_clusters=self.n_centers_by_dim,
                            random_state=42,
                            n_init="auto",
                        )
                    ),
                    i,
                )
                for i in range(self.tda_max_dim + 1)
            ]
        )

        # Full sklearn pipeline
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "windows",
                    SlidingWindowTransformer(
                        window_size=self.window_size, step=self.step
                    ),
                ),
                ("similarity", similarity_fn),
                ("rips", rips_transformer),
                ("archipelago", archipelago_transformer),
            ]
        )

        return pipeline

    def fit(self, X, y=None):
        """
        Fit the full pipeline to the data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input multivariate time series.

        y : Ignored
        """
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        return self

    def transform(self, X):
        """
        Apply the fitted pipeline to the input data.

        Returns
        -------
        X_transformed : np.ndarray
            Topological feature embedding.
        """
        return self.pipeline_.transform(X)
