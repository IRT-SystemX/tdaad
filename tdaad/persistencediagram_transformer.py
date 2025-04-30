"""Persistence Diagram Transformers."""

# Author: Martin Royer

import numpy as np
from operator import itemgetter

from gudhi.sklearn.rips_persistence import RipsPersistence
from sklearn.preprocessing import FunctionTransformer

from tdaad.utils.local_pipeline import LocalPipeline


def _data_to_similarity(X, filter_nan=True):
    r"""Transforms dataframe X into similarity matrix :math:`1-\mathbf{Corr}(X)`."""
    target = 1 - X.corr().to_numpy()
    nanrowcols = np.isnan(target).all(axis=0) if filter_nan else ~target.any(
        axis=0)  # this filters when a variable is constant -> nan on all rows
    return target[~nanrowcols, :][:, ~nanrowcols]


def wrap_in_list(X):
    """ wrapper because RipsPersistence.transform expects a list"""
    return [X]


class PersistenceDiagramTransformer(LocalPipeline):
    """Persistence Diagram Transformer for point cloud.

    For a given point cloud, form a similarity matrix and apply a RipsPersistence procedure
    to produce topological descriptors in the form of persistence diagrams.

    Read more in the :ref: `User Guide <persistence_diagrams>`.

    Parameters:
        tda_max_dim : int, default=2
            The maximum dimension of the topological feature extraction.

    Example
    -------
    >>> n_timestamps = 100
    >>> n_sensors = 5
    >>> timestamps = pd.to_datetime('2024-01-01', utc=True) + pd.Timedelta(1, 'h') * np.arange(n_timestamps)
    >>> X = pd.DataFrame(np.random.random(size=(n_timestamps, n_sensors)), index=timestamps)
    >>> PersistenceDiagramTransformer().fit_transform(X)
    """

    def __init__(self, tda_max_dim=2):
        self.tda_max_dim = tda_max_dim
        similarity_transformer = FunctionTransformer(func=_data_to_similarity)
        similarity_transformer.name = r"1-$\mathbf{Corr}(X)$"
        list_encapsulate_transformer = FunctionTransformer(func=wrap_in_list)
        list_encapsulate_transformer.name = ""
        rips_transformer = RipsPersistence(homology_dimensions=range(
            tda_max_dim + 1), input_type='lower distance matrix')
        rips_transformer.name = ""
        list_popper_transformer = FunctionTransformer(func=itemgetter(0))
        list_popper_transformer.name = ""

        steps = [
            ("similarity_step", similarity_transformer),
            ("list_encapsulate", list_encapsulate_transformer),
            ("rips_step", rips_transformer),
            ("list_popper", list_popper_transformer),
        ]
        super().__init__(steps=steps)

    def fit_transform(self, X, y=None, **fit_params):
        """Transforms data X into a list of persistence diagrams arranged in order of homology dimension.

        Args:
            X : {array-like, sparse matrix} of shape (n_timestamps, n_sensors)
                Multiple time series to transform, where `n_timestamps` is the number of timestamps
                in the series X, and `n_sensors` is the number of sensors.
            y : Ignored
                Not used, present for API consistency by convention.

            **fit_params : Ignored
                Not used, present for API consistency.

        Nb: this function can be removed, but is here so that returns can be explicited.

        Returns:
        --------
        by_dim_arrays: list of persistence diagrams [pd_0, pd_1, ...] arranged in order of homology dimension.
            a persistence diagram pd_i is a ndarray of shape {n_i, 2} where n_i is the number of homological
            features in dimension i found in the similarity matrix of the data.

        """
        by_dim_arrays = self.fit(X=X, y=y, **fit_params).transform(X)
        return by_dim_arrays
