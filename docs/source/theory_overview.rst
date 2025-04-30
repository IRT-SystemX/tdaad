.. _theory_overview:


💡 Theory overview
===================

.. _topological_data_analysis:

Topological Data Analysis is a recent and fast growing field providing topological and geometric tools to infer features for complex data.
See an introduction in [CM21]_.

.. _topological_anomaly_detection:

Topological Anomaly Detection in this module:
    - run a sliding window algorithm and represent each time series window with topological features,
        see :ref:`Topological Embedding <topological_embedding>`,
    - use a MinCovDet algorithm to robustly estimate the data mean and covariance in the embedding space,
        and use these to derive an embedding mahalanobis distance and associated outlier detection procedure,
        see :ref:`Elliptic Envelope <elliptic_envelope>`.

This library is the implementation result of the TADA algorithm introduced in [CLR24]_.

.. _topological_embedding:

For more details on the way to produce a temporal topological embedding, please refer to [CLR24]_.


.. _elliptic_envelope:

`Elliptic Envelope. <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope>`_

Essentially once you estimate the mean and covariance of a set of vectors, assuming a Gaussian multivariate span you have a natural envelope of said span using the mahalanobis distance. Elliptic Envelope is a sklearn tool that derives that score.


📑 References
==============

.. [CM21] `F. Chazal, B. Michel. An introduction to Topological Data Analysis: fundamental and practical aspects for data scientists. Frontiers in AI, 2021 <https://www.frontiersin.org/articles/10.3389/frai.2021.667963/full>`_.

.. [CLR24] `F. Chazal, C. Levrard and M. Royer. Topological Analysis for Detecting Anomalies (TADA) in dependent sequences: application to Time Series. Journal of Machine Learning Research, 2024 <http://jmlr.org/papers/v25/24-0853.html>`_.
