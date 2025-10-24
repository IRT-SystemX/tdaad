import os
from pathlib import Path
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def load_datasets(path):
    return [
        pd.read_csv(path / file, header=0, index_col=0, compression="gzip")
        for file in os.listdir(path)
        if "gz" in file
    ]


N_DATASETS = 10
DATA_DIR = Path("~/SVN/DATASETS/gudhi-data/timeseries/topological_wheels/").expanduser()
RESULTS_BASE = Path("../../results/TopologicalWheels").expanduser()

datasets = load_datasets(DATA_DIR)


i, j = N_DATASETS - 2, N_DATASETS - 1
X_train, y_train = datasets[i].iloc[:, :-1], datasets[i].iloc[:, -1]
X_test, y_test = datasets[j].iloc[:, :-1], datasets[j].iloc[:, -1]


from tdaad._feature_functions import feature_total_persistence_corr


resolutions = [10, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 600, 800]
X = X_train.values

X = np.random.randn(10000, 64)
n_samples = X.shape[0]
half_ws = {w: w // 2 for w in resolutions}


jobs = [(t, w) for w in resolutions for t in range(half_ws[w], n_samples - half_ws[w])]


def _extract_window(X, t, w):
    hw = w // 2
    if t - hw < 0 or t + hw > X.shape[0]:
        return None
    return X[t - hw : t + hw]


def compute(t, w):
    window = _extract_window(X, t, w)
    if window is None:
        return (t, w, None)
    total_pers_01 = feature_total_persistence_corr(window)
    total_pers_0 = feature_total_persistence_corr(window, maxdim=0)
    total_pers_1 = total_pers_01 - total_pers_0
    return (t, w, total_pers_0, total_pers_1, total_pers_01)


results = Parallel(n_jobs=-1)(delayed(compute)(t, w) for t, w in jobs)

df = pd.DataFrame(
    results, columns=["time", "w", "total_pers_0", "total_pers_1", "total_pers_01"]
)
df["time"] = pd.to_datetime(df["time"])


# Compute total persistence wrt time per w
plt.figure(figsize=(10, 6))
for w_value, group in df.groupby("w"):
    plt.plot(group["time"], group["total_pers_01"], marker="o", label=f"w={w_value}")

plt.xlabel("Time")
plt.ylabel("Total persistence")
plt.title("Total persistenceover Time by w")
plt.legend(title="w")
plt.grid(True)
plt.tight_layout()
plt.show()


# Compute mean feature per w
mean_totalpers_0 = df.groupby("w")["total_pers_0"].mean().reset_index()
mean_totalpers_1 = df.groupby("w")["total_pers_1"].mean().reset_index()
mean_totalpers_01 = df.groupby("w")["total_pers_01"].mean().reset_index()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(
    mean_totalpers_0["w"],
    mean_totalpers_0["total_pers_0"],
    color="skyblue",
    marker="o",
    label="PH0",
)
plt.plot(
    mean_totalpers_1["w"],
    mean_totalpers_1["total_pers_1"],
    color="green",
    marker="o",
    label="PH1",
)
plt.plot(
    mean_totalpers_01["w"],
    mean_totalpers_01["total_pers_01"],
    color="red",
    marker="x",
    label="PH0 + PH1",
)

plt.legend()
plt.xlabel("Window size")
plt.ylabel("Mean total persistence")
plt.title("Mean total persistence by window size")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
