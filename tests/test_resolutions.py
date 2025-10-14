import numpy as np

from tdaad.resolution_selector import (
    GlobalResolutionsFinder,
    LocalResolutionsFinder,
    ResolutionSelector,
)

X = np.random.randn(10000, 3)

selector = GlobalResolutionsFinder(
    resolutions=[32, 64, 128, 256, 512],
    k=3,
    score_method="combined",
    n_jobs=4,
    include_variance=True,
    include_entropy=True,
    include_derivative_std=True,
)

top_resolutions = selector.fit_transform(X)
print("Top-k global resolutions:", top_resolutions)


selector = LocalResolutionsFinder(
    resolutions=[32, 64, 128, 256],
    k=2,
    score_method="combined",
    n_jobs=4,
    include_variance=True,
    include_entropy=True,
    include_derivative_std=True,
)

local_topk = selector.fit_transform(X)
# local_topk is a list of length n_samples, each item a list of top-k resolutions at that time
print(f"Top-k local resolutions at {local_topk[100]=}")


selector = ResolutionSelector(
    candidates=[2, 50, 500],  # or fixed int, or list of candidates
    k=2,
    n_jobs=4,
    feature_selector_kwargs={"include_entropy": True},
)

selector.fit(X)
best_resolutions = selector.transform(X)
print("Selected window size(s):", best_resolutions)
