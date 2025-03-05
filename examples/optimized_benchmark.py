"""
Optimized benchmark comparing HPDA and pandas for specific operations.

This script focuses on operations where HPDA would show advantages:
1. Large datasets
2. Operations that benefit from parallelism
3. Complex computations
"""

import hpda
import pandas as pd
import numpy as np
import time
import random
import psutil
import os
import gc

# Process for memory measurement
process = psutil.Process(os.getpid())

def measure_memory_usage():
    """Measure current memory usage in MB."""
    gc.collect()
    return process.memory_info().rss / (1024 * 1024)

def time_operation(name, func, warmup=False):
    """Measure operation time and memory usage."""
    gc.collect()
    
    initial_memory = measure_memory_usage()
    start_time = time.time()
    result = func()
    elapsed = time.time() - start_time
    peak_memory = measure_memory_usage() - initial_memory
    
    if not warmup:
        print(f"  {name}: {elapsed:.4f}s, {peak_memory:.2f}MB")
    
    return elapsed, peak_memory, result

def benchmark_statistical_ops(size=100000):
    """Benchmark statistical operations on pre-created dataframes."""
    print(f"\nBenchmarking Statistical Operations (size={size})...")
    
    # Create dataframes ahead of time
    print("  Creating dataframes (not part of benchmark time)...")
    # For pandas, use numpy arrays directly
    pandas_df = pd.DataFrame({
        "A": np.random.randint(0, 1000, size),
        "B": np.random.random(size) * 100,
        "C": np.random.random(size) * 100,
        "D": np.random.random(size) * 100,
    })
    
    # For HPDA, convert to lists (this is a limitation of our current implementation)
    hpda_df = hpda.DataFrame({
        "A": [int(x) for x in np.random.randint(0, 1000, size)],
        "B": [float(x) for x in np.random.random(size) * 100],
        "C": [float(x) for x in np.random.random(size) * 100],
        "D": [float(x) for x in np.random.random(size) * 100],
    })
    
    # Set parallel execution for HPDA
    hpda_df.set_execution_policy(hpda.ExecutionPolicy.Parallel)
    hpda_df.set_num_threads(4)
    
    # Warmup
    time_operation("Pandas warmup", lambda: pandas_df.mean(), warmup=True)
    time_operation("HPDA warmup", lambda: hpda_df.mean(), warmup=True)
    
    # Measure mean
    print("Mean operation:")
    pandas_mean_time, pandas_mean_mem, _ = time_operation("Pandas", lambda: pandas_df.mean())
    hpda_mean_time, hpda_mean_mem, _ = time_operation("HPDA", lambda: hpda_df.mean())
    
    if hpda_mean_time > 0:
        speedup = pandas_mean_time / hpda_mean_time
        print(f"  Speedup: {speedup:.2f}x")
    
    if hpda_mean_mem > 0:
        mem_reduction = pandas_mean_mem / hpda_mean_mem
        print(f"  Memory reduction: {mem_reduction:.2f}x")
    
    # Measure standard deviation
    print("\nStandard deviation operation:")
    pandas_std_time, pandas_std_mem, _ = time_operation("Pandas", lambda: pandas_df.std())
    hpda_std_time, hpda_std_mem, _ = time_operation("HPDA", lambda: hpda_df.std())
    
    if hpda_std_time > 0:
        speedup = pandas_std_time / hpda_std_time
        print(f"  Speedup: {speedup:.2f}x")
    
    if hpda_std_mem > 0:
        mem_reduction = pandas_std_mem / hpda_std_mem
        print(f"  Memory reduction: {mem_reduction:.2f}x")
    
    return pandas_df, hpda_df

def benchmark_sort(pandas_df, hpda_df):
    """Benchmark sort operation on pre-created dataframes."""
    print("\nBenchmarking Sort Operation...")
    
    # Warmup
    time_operation("Pandas warmup", lambda: pandas_df.sort_values(by="B"), warmup=True)
    time_operation("HPDA warmup", lambda: hpda_df.sort_values("B"), warmup=True)
    
    # Measure sort
    pandas_sort_time, pandas_sort_mem, _ = time_operation("Pandas", lambda: pandas_df.sort_values(by="B"))
    hpda_sort_time, hpda_sort_mem, _ = time_operation("HPDA", lambda: hpda_df.sort_values("B"))
    
    if hpda_sort_time > 0:
        speedup = pandas_sort_time / hpda_sort_time
        print(f"  Speedup: {speedup:.2f}x")
    
    if hpda_sort_mem > 0:
        mem_reduction = pandas_sort_mem / hpda_sort_mem
        print(f"  Memory reduction: {mem_reduction:.2f}x")

def benchmark_groupby(size=100000):
    """Benchmark groupby operation on pre-created dataframes with categorical data."""
    print(f"\nBenchmarking GroupBy Operation (size={size})...")
    
    # Create dataframes with categorical data
    print("  Creating dataframes (not part of benchmark time)...")
    categories = ["A", "B", "C", "D", "E"]
    
    pandas_df = pd.DataFrame({
        "category": np.random.choice(categories, size),
        "value1": np.random.random(size) * 100,
        "value2": np.random.random(size) * 100,
    })
    
    hpda_df = hpda.DataFrame({
        "category": [random.choice(categories) for _ in range(size)],
        "value1": [random.random() * 100 for _ in range(size)],
        "value2": [random.random() * 100 for _ in range(size)],
    })
    
    # Set parallel execution for HPDA
    hpda_df.set_execution_policy(hpda.ExecutionPolicy.Parallel)
    hpda_df.set_num_threads(4)
    
    # Warmup
    time_operation("Pandas warmup", lambda: pandas_df.groupby("category").mean(), warmup=True)
    time_operation("HPDA warmup", lambda: hpda_df.groupby("category").mean(), warmup=True)
    
    # Measure groupby
    pandas_gb_time, pandas_gb_mem, _ = time_operation("Pandas", lambda: pandas_df.groupby("category").mean())
    hpda_gb_time, hpda_gb_mem, _ = time_operation("HPDA", lambda: hpda_df.groupby("category").mean())
    
    if hpda_gb_time > 0:
        speedup = pandas_gb_time / hpda_gb_time
        print(f"  Speedup: {speedup:.2f}x")
    
    if hpda_gb_mem > 0:
        mem_reduction = pandas_gb_mem / hpda_gb_mem
        print(f"  Memory reduction: {mem_reduction:.2f}x")

def main():
    print("HPDA vs Pandas Optimized Benchmark")
    print("================================")
    
    # Conduct benchmark with dataset of 100,000 rows
    pandas_df, hpda_df = benchmark_statistical_ops(100000)
    
    # Benchmark sort and groupby
    benchmark_sort(pandas_df, hpda_df)
    benchmark_groupby(100000)
    
    print("\nLarger Dataset Benchmark (500,000 rows)")
    print("================================")
    
    # Conduct benchmark with dataset of 500,000 rows
    pandas_df_large, hpda_df_large = benchmark_statistical_ops(500000)
    
    # Benchmark sort and groupby with larger dataset
    benchmark_sort(pandas_df_large, hpda_df_large)
    benchmark_groupby(500000)
    
    print("\nBenchmark completed.")

if __name__ == "__main__":
    main()