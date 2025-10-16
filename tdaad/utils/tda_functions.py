"""Persistence Diagram Transformers."""

# Author: Martin Royer

import numpy as np

from gudhi.sklearn.rips_persistence import RipsPersistence


def _numpy_data_to_similarity(X, filter_nan=True):
    r"""Transforms numpy matrix X into similarity matrix :math:`1-\mathbf{Corr}(X)`."""
    target = 1 - np.corrcoef(X, rowvar=False)
    # this filters when a variable is constant -> nan on all rows
    nanrowcols = np.isnan(target).all(axis=0) if filter_nan else ~target.any(axis=0)
    return target[~nanrowcols][:, ~nanrowcols]


def transform_to_persistence_diagram(X, tda_max_dim=0):
    """Persistence Diagram Transformer for point cloud.

    For a given point cloud, form a similarity matrix and apply a RipsPersistence procedure
    to produce topological descriptors in the form of persistence diagrams.

    Read more in the :ref: `User Guide <persistence_diagrams>`.

    Parameters:
        tda_max_dim : int, default=0
            The maximum dimension of the topological feature extraction.

    Example
    -------
    >>> n_timestamps = 100
    >>> n_sensors = 5
    >>> import numpy as np
    >>> np.corrcoef(X)
    >>> import pandas as pd
    >>> timestamps = pd.to_datetime('2024-01-01', utc=True) + pd.Timedelta(1, 'h') * np.arange(n_timestamps)
    >>> X = pd.DataFrame(np.random.random(size=(n_timestamps, n_sensors)), index=timestamps)
    >>> PersistenceDiagramTransformer().fit_transform(X.to_numpy())
    """
    sim_target = [_numpy_data_to_similarity(X)]
    rips_transformer = RipsPersistence(
        homology_dimensions=range(tda_max_dim + 1),
        input_type="lower distance matrix",
    )
    rips_target = rips_transformer.transform(sim_target)
    return rips_target[0]
