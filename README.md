
<div align="center">
	<img src="_static/Logo_ConfianceAI.png" width="20%" alt="ConfianceAI Logo" />
    <h1 style="font-size: large; font-weight: bold;">TDAAD</h1>
</div>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.12-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.13-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MPL-2">
    </a>
    <a href="_static/pylint/pylint.txt">
        <img src="_static/pylint/pylint.svg" alt="Pylint Score">
    </a>
    <a href="_static/flake8/index.html">
        <img src="_static/flake8/flake8.svg" alt="Flake8 Report">
    </a>
	<a href="_static/coverage/index.html">
        <img src="_static/coverage/coverage.svg" alt="Coverage report">
    </a>


</div>
<br>

# Topological Data Analysis for Anomaly Detection module

This package named `tdaad` is a Python module for detecting anomalies in multiple time series.

## 🚀 Install

To install and use the component you can create a Python virtual environment as follows:
```bash
pip install virtualenv
virtualenv myenv
source myenv/bin/activate
```

Then you can clone the git repo and install from within the folder:
```bash
git clone https://github.com/IRT-SystemX/tdaad.git
cd tdaad
pip install .
```

## 🎮 Basic usage

### Context

`tdaad` provides machine learning algorithms for analyzing timeseries data through the lense of Topological Data
Analysis, and deriving anomaly scores.

The targeted input is an object `X` representing a _multiple time series_ with variables columns and timestamps lines.
We use the term _multiple time series_ to describe a set of univariate timeseries that describe a system or object.
Note that the package does not handle analysis of a single univariate timeseries.

The main idea of this package is to analyze time series with topological methods. It is done in three essential steps:

1. cut the timeseries into chunks using a sliding window algorithm,
2. represent each timeseries window with topological features,
3. estimate the empirical covariance of those topological features to derive an anomaly detection procedure.

The combination of steps 1. and 2. are performed by an object called `TopologicalEmbedding`,
and the method can be understood as a representation learning method.

Step 3. is a standard step of anomaly detection procedure based on vectorized data.

The combined result is the `TopologicalAnomalyDetector` object.


### Main features: `TopologicalEmbedding` and `TopologicalAnomalyDetector`

As the package is based upon, inspired by and compatible with reknown `scikit-learn`
python library, the representation learning and anomaly detection learning are performed in the most standard way.

```
   from tdaad.topological_embedding import TopologicalEmbedding
   embedding = TopologicalEmbedding().fit_transform(X)
```

Based on this representation, an empirical covariance is learned and an
elliptic envelope is calculated with the associated mahalanobis distance, allowing for derivation of an anomaly score.
All of this is performed within the `TopologicalAnomalyDetector` object, so that to perform an anomaly detection process
one only needs the following:


```
   from tdaad.anomaly_detectors import TopologicalAnomalyDetector
   detector = TopologicalAnomalyDetector().fit(X)
   anomaly_scores = detector.score_samples(X)
```

## 🔀 Improved usage

### Inputs

For now the component is designed to handle inputs `X` in the form of a
DataFrame with variables as columns and timestamps as lines.

### Main outputs

The main output is an `anomaly_score` from the `TopologicalAnomalyDetector` object,
in the form of a univariate `numpy.ndarray`. The scores are not bounded, the lower scores correspond to the more abnormal portions of the timeseries, according to the topological representation that was constructed.

Another key output is that topological representation or embedding, result of the `transform` of the `TopologicalEmbedding` object: a multivariate `pandas.DataFrame` that "encodes" the timeseries into vectorial (and topological) representation.


### Algorithm main parameters

The aforementioned `TopologicalAnomalyDetector` objects are ready-to-use as such,
but should be manually tuned for better results. Key parameters include:
+ `window_size` for the sliding window algorithm, important for capturing phenomenons at a certain scale. Ideally one would want at least `window_size=100`.
+ `n_centers_by_dim` determining the size of the embedding, that one wants as small as possible at risk of missing key information. Perhaps start with value 2 or 3, and if the features are not relevant it is worth going to 10 to 20.
+ `tda_max_dim` the maximum topological dimension to create features. One should start with dimension 0, 1 or 2. The computation time increases sensibly with this parameter.

### Good to know

- For this algorithm to run smoothly, the number of sensors involved can go to 100 without too much problem.
- It seems possible to adapt the algorithm to an online version if one stays around this order of magnitude. The way to go would be 1. find a reasonable "normal-regime" period and `window_size` to train the detector, then run it on streaming data.
- Because of its design, the tda detectors would be invariant to noises or transformation that are invariant through correlation (e.g. multiplication), but also to label shifts (as TDA is invariant to label permutation).
- This detector is good to use in a context where a complex system (e.g. train on a railway, drone fleet, factory...) has various sensors information that can be used to build inference on the state of the system (working / anomalous => needs attending).

## Changelog
Stay up-to-date with the changes and improvements made to TDAAD in our changelog. Each release provides a summary of new features, fixes, and enhancements.
Check out our [changelog](CHANGELOG.md) to see what's new and improved!


## Document generation

To regenerate the documentation, rerun the following commands from the project root, adapting if
necessary:

```
pip install -r docs/docs_requirements.txt -r requirements.txt
sphinx-apidoc -o docs/source/generated tdaad
sphinx-build -M html docs/source docs/build -W --keep-going
```

## License

MPL 2.0