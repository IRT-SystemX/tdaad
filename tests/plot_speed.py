import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../benchmark_results.csv")

# Prepare your dataframes
df_timeout = df[df["Runtime"].isnull()]
df_success = df[df["Runtime"].notnull()]

TIMEOUT_LIMIT = 30  # seconds
# For timeouts, assign a dummy runtime for plotting (slightly above timeout)
df_timeout_plot = df_timeout.copy()
df_timeout_plot["Runtime"] = TIMEOUT_LIMIT + 1

# Plot settings
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Left: Runtime vs Timesteps (T)
sns.lineplot(
    data=df_success,
    x="T",
    y="Runtime",
    hue="Threads",
    style="W",
    markers=True,
    dashes=False,
    ax=axes[0],
    legend="brief",
    err_style=None,
)

# Plot timeouts as red crosses on the left plot
if not df_timeout_plot.empty:
    axes[0].scatter(
        df_timeout_plot["T"],
        df_timeout_plot["Runtime"],
        color="red",
        marker="x",
        s=100,
        label="Timeout",
        zorder=5,
    )

axes[0].set_title("Runtime vs Timesteps (T)")
axes[0].set_xlabel("Timesteps (T)")
axes[0].set_ylabel("Runtime (seconds)")
axes[0].legend(title="Threads / Window / Timeout")

# Right: Runtime vs Dimension (D)
sns.lineplot(
    data=df_success,
    x="D",
    y="Runtime",
    hue="Threads",
    style="W",
    markers=True,
    dashes=False,
    ax=axes[1],
    legend=False,
    err_style=None,
)

# Plot timeouts as red crosses on the right plot
if not df_timeout_plot.empty:
    axes[1].scatter(
        df_timeout_plot["D"],
        df_timeout_plot["Runtime"],
        color="red",
        marker="x",
        s=100,
        label="Timeout",
        zorder=5,
    )

axes[1].set_title("Runtime vs Dimension (D)")
axes[1].set_xlabel("Dimension (D)")
axes[1].set_ylabel("Runtime (seconds)")

plt.suptitle("Benchmark Runtime Analysis (including Timeouts)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("benchmark_runtime_plots.png")
plt.show()
