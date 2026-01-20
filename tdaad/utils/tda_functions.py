"""Persistence Diagram Transformers."""

# Author: Martin Royer

import numpy as np

from gudhi.sklearn.rips_persistence import RipsPersistence


def numpy_data_to_similarity(X, filter_nan=True):
    r"""Transforms numpy matrix X into similarity matrix :math:`1-\mathbf{Corr}(X)`."""
    target = 1 - np.corrcoef(X, rowvar=False)
    # this filters when a variable is constant -> nan on all rows
    nanrowcols = np.isnan(target).all(axis=0) if filter_nan else ~target.any(axis=0)
    return target[~nanrowcols][:, ~nanrowcols]


def transform_to_persistence_diagram(X, tda_max_dim=0):
    """For a given point cloud, form a similarity matrix and apply a RipsPersistence procedure
    to produce topological descriptors in the form of persistence diagrams.

    no longer used in topological embedding
    """
    sim_target = numpy_data_to_similarity(X)
    rips_transformer = RipsPersistence(
        homology_dimensions=range(tda_max_dim + 1),
        input_type="lower distance matrix",
    )
    return rips_transformer.fit_transform([sim_target])[0]
