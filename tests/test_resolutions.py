import logging
import numpy as np

from tdaad.resolution_selector import (
    GlobalResolutionsFinder,
    LocalResolutionsFinder,
    ResolutionSelector,
    FEATURE_FUNCTIONS,
)

logging.basicConfig(level=logging.INFO)

X = np.random.randn(10000, 3)

selector = GlobalResolutionsFinder(
    resolutions=[32, 64, 128, 256, 512],
    k=3,
    score_method="combined",
    n_jobs=4,
)

top_resolutions = selector.fit_transform(X)
print("Top-k global resolutions:", top_resolutions)


selector = LocalResolutionsFinder(
    resolutions=[32, 64, 128, 256],
    k=2,
    score_method="combined",
    n_jobs=4,
)

local_topk = selector.fit_transform(X)
# local_topk is a list of length n_samples, each item a list of top-k resolutions at that time
print(f"Top-k local resolutions at {local_topk[100]=}")


for feature in FEATURE_FUNCTIONS.keys():
    selector = ResolutionSelector(
        candidates=[10, 50, 100, 500, 1000],  # or fixed int, or list of candidates
        k=2,
        n_jobs=-1,
        feature_selector_kwargs={"features": [feature]},
    )

    selector.fit(X)
    best_resolutions = selector.transform(X)
    print(f"Selector with {feature=}, selected {best_resolutions=}")
