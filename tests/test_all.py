import numpy as np
import pandas as pd

import itertools

from tdaad.utils.tda_functions import transform_to_persistence_diagram
from tdaad.topological_embedding import TopologicalEmbedding
from tdaad.anomaly_detectors import TopologicalAnomalyDetector


def test_transform_to_persistence_diagram():
    """Test for PersistenceDiagramTransformer functionalities."""
    n_timestamps = 100
    n_sensors = 5
    timestamps = pd.to_datetime("2024-01-01", utc=True) + pd.Timedelta(
        1, "h"
    ) * np.arange(n_timestamps)
    X = pd.DataFrame(np.random.random(size=(n_timestamps, n_sensors)), index=timestamps)

    # testing that the diagrams are in R^2
    assert transform_to_persistence_diagram(X)[0].shape[1] == 2
    # testing the transform functionality
    assert isinstance(transform_to_persistence_diagram(X)[0], np.ndarray)
    # testing the tda_max_dim functionality
    assert len(transform_to_persistence_diagram(X, tda_max_dim=0)) == 1
    assert len(transform_to_persistence_diagram(X, tda_max_dim=1)) == 2

    return


def test_topologicalembedding():
    """Test for TopologicalEmbedding functionalities."""
    n_timestamps = 100
    n_sensors = 5
    timestamps = pd.to_datetime("2024-01-01", utc=True) + pd.Timedelta(
        1, "h"
    ) * np.arange(n_timestamps)
    X = pd.DataFrame(np.random.random(size=(n_timestamps, n_sensors)), index=timestamps)

    n_centers_by_dim = 2
    tda_max_dim = 1
    te = TopologicalEmbedding(n_centers_by_dim=n_centers_by_dim, tda_max_dim=1).fit(X)

    # testing the transform functionality
    assert isinstance(te.transform(X), pd.DataFrame)
    # testing the n_centers_by_dim, tda_max_dim functionalities
    assert te.transform(X).shape[1] == n_centers_by_dim * (tda_max_dim + 1)
    return


def test_topologicalanomalydetector():
    """Test for TopologicalAnomalyDetector functionalities."""
    n_timestamps = 1000
    n_sensors = 20
    timestamps = pd.to_datetime("2024-01-01", utc=True) + pd.Timedelta(
        1, "h"
    ) * np.arange(n_timestamps)
    X = pd.DataFrame(np.random.random(size=(n_timestamps, n_sensors)), index=timestamps)
    X.iloc[n_timestamps // 2 :, :10] = -X.iloc[n_timestamps // 2 :, 10:20]

    detector = TopologicalAnomalyDetector().fit(X)
    anomaly_scores = detector.score_samples(X)
    # testing the scoring functionality
    assert isinstance(anomaly_scores, np.ndarray)
    for window_size, step in itertools.product([25, 28], [2, 3, 5, 20]):
        detector = TopologicalAnomalyDetector(
            n_centers_by_dim=2, tda_max_dim=1, window_size=window_size, step=step
        ).fit(X)
        anomaly_scores = detector.score_samples(X)
        # testing the remapping functionality
        assert len(anomaly_scores) == n_timestamps
    decision = detector.decision_function(X)
    # testing the decision functionality
    assert decision[n_timestamps // 2] < 0
    anomalies = detector.predict(X)
    # testing the predict functionality
    assert anomalies[n_timestamps // 2] == -1
    return
