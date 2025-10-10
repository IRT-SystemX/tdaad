import time
from memory_profiler import memory_usage
import numpy as np
from joblib import Parallel, delayed
from itertools import product
import os
import pandas as pd

from tdaad.anomaly_detectors import TopologicalAnomalyDetector


# --------- Config ---------
DIMENSIONS = [10, 50, 100, 200]
TIMESTEPS = [1000, 5000, 10_000]
WINDOW_SIZES = [10, 50, 100, 500]
THREAD_COUNTS = [1, 4]
TIMEOUT_SECONDS = 30
REPEATS = 1


# --------- Data Generator ---------
def generate_data(T: int, D: int):
    return np.random.randn(T, D)


# --------- Dummy Algorithm (Replace with your own) ---------


def your_algorithm(data, window_size, num_threads):
    T, D = data.shape
    results = (
        TopologicalAnomalyDetector(
            tda_max_dim=1, window_size=window_size, n_jobs=num_threads
        )
        .fit(data)
        .transform(data)
    )
    return results


# --------- Timeout Worker ---------
def _timeout_worker(queue, func, args, timeout):
    try:
        start = time.time()
        mem = memory_usage((func, args), interval=0.1, timeout=timeout, max_usage=True)
        runtime = time.time() - start
        queue.put((runtime, mem))
    except Exception as e:
        queue.put(e)


from concurrent.futures import ThreadPoolExecutor, TimeoutError


def run_with_timeout(func, args=(), timeout=30):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            start = time.time()
            future.result(timeout=timeout)
            runtime = time.time() - start
            return runtime  # ✅ return float
        except TimeoutError:
            return None
        except Exception as e:
            raise e


# --------- Single Benchmark Run ---------
def run_benchmark_case(D, T, W, threads):
    data = generate_data(T, D)
    try:
        runtime = run_with_timeout(
            your_algorithm, args=(data, W, threads), timeout=TIMEOUT_SECONDS
        )
        if runtime is None:
            print(f"❌ Timeout: D={D}, T={T}, W={W}, Threads={threads}")
            return (D, T, W, threads, None)
        else:
            print(f"✅ OK: D={D}, T={T}, W={W}, Threads={threads} -> {runtime:.2f}s")
            return (D, T, W, threads, runtime)
    except Exception as e:
        print(f"❌ Error: D={D}, T={T}, W={W}, Threads={threads} -> {e}")
        return (D, T, W, threads, None)


# --------- MAIN ---------
if __name__ == "__main__":
    test_cases = list(product(DIMENSIONS, TIMESTEPS, WINDOW_SIZES, THREAD_COUNTS))
    test_cases *= REPEATS

    # Estimate safe number of parallel tests
    max_cores = os.cpu_count() or 4
    est_threads_per_case = max(THREAD_COUNTS)
    max_parallel_tests = max(1, max_cores // est_threads_per_case)

    print(
        f"🔬 Running {len(test_cases)} benchmark tests using {max_parallel_tests} workers..."
    )

    results = Parallel(n_jobs=max_parallel_tests)(
        delayed(run_benchmark_case)(D, T, W, threads)
        for (D, T, W, threads) in test_cases
    )

    df = pd.DataFrame(results, columns=["D", "T", "W", "Threads", "Runtime"])
    df.to_csv("benchmark_results.csv", index=False)

    print("✅ Benchmarking complete! Results saved to benchmark_results.csv")
