import numpy as np

from tdaad.resolution_selector import GlobalResolutionSelector, LocalResolutionSelector

X = np.random.randn(10000, 3)

selector = GlobalResolutionSelector(
    window_sizes=[32, 64, 128, 256, 512],
    k=3,
    score_method="combined",
    n_jobs=4,
    include_variance=True,
    include_entropy=True,
    include_derivative_std=True,
)

top_resolutions = selector.fit_transform(X)
print("Top-k global resolutions:", top_resolutions)


selector = LocalResolutionSelector(
    window_sizes=[32, 64, 128, 256],
    k=2,
    score_method="combined",
    n_jobs=4,
    include_variance=True,
    include_entropy=True,
    include_derivative_std=True,
)

local_topk = selector.fit_transform(X)
# local_topk is a list of length n_samples, each item a list of top-k resolutions at that time
print(local_topk[100])  # e.g. [64, 32]
