import logging
import numpy as np

from tdaad.resolution_selector import (
    GlobalResolutionsFinder,
    LocalResolutionsFinder,
    ResolutionSelector,
)

FEATURES = ["skewness", "kurtosis", "totalpers"]

logging.basicConfig(level=logging.INFO)

X = np.random.randn(10000, 3)

finder = GlobalResolutionsFinder(
    resolutions=[32, 64, 128, 256, 512],
    k=3,
    score_method="combined",
    n_jobs=4,
    features=FEATURES,
)

top_resolutions, scores = finder.fit_transform(X)
print(f"Top-k global {top_resolutions=}, {scores=}")


finder = LocalResolutionsFinder(
    resolutions=[32, 64, 128, 256],
    k=2,
    score_method="combined",
    n_jobs=4,
    features=FEATURES,
)

local_topk, scores_by_t = finder.fit_transform(X)
# local_topk is a list of length n_samples, each item a list of top-k resolutions at that time
print(f"Top-k local {local_topk[100]=}, {scores_by_t=}")


for feature in FEATURES:
    selector = ResolutionSelector(
        candidates=[10, 50, 100, 500, 1000],  # or fixed int, or list of candidates
        k=2,
        n_jobs=-1,
        feature_selector_kwargs={"features": [feature], "feat_time_threshold": 1e-1},
    )

    selector.fit(X)
    best_resolutions = selector.transform(X)
    print(f"Selector with {feature=}, selected {best_resolutions=}")

    # selector.finder_.fit_transform(X)
    # scores = selector.finder_._parallel_score_matrix(X)

    # from tdaad._feature_functions import feature_total_persistence_corr

    # window = X[:50, :]
    # feature_total_persistence_corr(window)
