"""Pandas Elliptic Envelope."""

# Author: Martin Royer

import pandas as pd

from sklearn.utils.validation import check_is_fitted
from sklearn.covariance import EllipticEnvelope


def pandas_mahalanobis(self, X):
    """Compute the negative Mahalanobis distances of embedding matrix X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The embedding matrix.

    Returns
    -------
    negative_mahal_distances : pandas.DataFrame of shape (n_samples,)
        Opposite of the Mahalanobis distances.
    """
    return pd.DataFrame(index=X.index, data=self.mahalanobis(X))


def pandas_score_samples(self, X):
    """Compute the negative Mahalanobis distances.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix.

    Returns
    -------
    negative_mahal_distances : array-like of shape (n_samples,)
        Opposite of the Mahalanobis distances.
    """
    check_is_fitted(self)
    return -pandas_mahalanobis(self, X)


EllipticEnvelope.score_samples = pandas_score_samples
