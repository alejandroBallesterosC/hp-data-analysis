"""
Advanced GroupBy Benchmark comparing HPDA and pandas for large datasets.

This script focuses specifically on testing groupby operations with:
1. Different dataset sizes
2. Different numbers of unique groups
3. Different execution policies
"""

import hpda
import pandas as pd
import numpy as np
import time
import random
import psutil
import os
import gc
import matplotlib.pyplot as plt

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

def benchmark_groupby_unique_values(sizes=[100000, 500000, 1000000], 
                                    unique_values_counts=[5, 20, 100]):
    """Benchmark groupby with different numbers of unique categorical values."""
    results = {
        'sizes': sizes,
        'unique_values': unique_values_counts,
        'pandas_time': {},
        'hpda_time': {},
        'pandas_memory': {},
        'hpda_memory': {},
        'speedup': {}
    }
    
    for n_unique in unique_values_counts:
        results['pandas_time'][n_unique] = []
        results['hpda_time'][n_unique] = []
        results['pandas_memory'][n_unique] = []
        results['hpda_memory'][n_unique] = []
        results['speedup'][n_unique] = []
    
    print(f"\nBenchmarking GroupBy with Different Numbers of Unique Values...")
    
    for size in sizes:
        print(f"\nDataset size: {size}")
        
        for n_unique in unique_values_counts:
            print(f"\n  Number of unique values: {n_unique}")
            unique_categories = [f"Cat_{i}" for i in range(n_unique)]
            
            # Create pandas dataframe
            pandas_categories = np.random.choice(unique_categories, size)
            pandas_values = np.random.random(size) * 100
            pandas_df = pd.DataFrame({
                "category": pandas_categories,
                "value1": pandas_values,
                "value2": np.random.random(size) * 100
            })
            
            # Create HPDA dataframe
            hpda_categories = [random.choice(unique_categories) for _ in range(size)]
            hpda_values = [float(v) for v in pandas_values]  # Keep the same values for fair comparison
            hpda_df = hpda.DataFrame({
                "category": hpda_categories,
                "value1": hpda_values,
                "value2": [random.random() * 100 for _ in range(size)]
            })
            
            # Set parallel execution for HPDA
            hpda_df.set_execution_policy(hpda.ExecutionPolicy.Parallel)
            hpda_df.set_num_threads(4)
            
            # Warmup
            time_operation("Pandas warmup", lambda: pandas_df.groupby("category").mean(), warmup=True)
            time_operation("HPDA warmup", lambda: hpda_df.groupby("category").mean(), warmup=True)
            
            # Measure groupby
            pandas_time, pandas_memory, _ = time_operation("Pandas", lambda: pandas_df.groupby("category").mean())
            hpda_time, hpda_memory, _ = time_operation("HPDA", lambda: hpda_df.groupby("category").mean())
            
            speedup = pandas_time / hpda_time if hpda_time > 0 else float('inf')
            print(f"  Speedup: {speedup:.2f}x")
            
            if hpda_memory > 0:
                mem_reduction = pandas_memory / hpda_memory
                print(f"  Memory reduction: {mem_reduction:.2f}x")
            
            # Store results
            results['pandas_time'][n_unique].append(pandas_time)
            results['hpda_time'][n_unique].append(hpda_time)
            results['pandas_memory'][n_unique].append(pandas_memory)
            results['hpda_memory'][n_unique].append(hpda_memory)
            results['speedup'][n_unique].append(speedup)
    
    return results

def plot_groupby_results(results):
    """Plot benchmark results."""
    # Create figure with multiple subplots
    fig, axes = plt.subplots(len(results['unique_values']), 2, figsize=(12, 5 * len(results['unique_values'])))
    
    for i, n_unique in enumerate(results['unique_values']):
        # Time comparison
        ax1 = axes[i, 0]
        ax1.plot(results['sizes'], results['pandas_time'][n_unique], marker='o', label='pandas')
        ax1.plot(results['sizes'], results['hpda_time'][n_unique], marker='x', label='hpda')
        ax1.set_title(f"GroupBy Time - {n_unique} unique values")
        ax1.set_xlabel("Dataset Size")
        ax1.set_ylabel("Time (seconds)")
        ax1.legend()
        ax1.grid(True)
        
        # Speedup
        ax2 = axes[i, 1]
        ax2.plot(results['sizes'], results['speedup'][n_unique], marker='o', color='green')
        ax2.set_title(f"Speedup - {n_unique} unique values")
        ax2.set_xlabel("Dataset Size")
        ax2.set_ylabel("Speedup Factor (higher is better)")
        ax2.axhline(y=1.0, color='r', linestyle='--')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("groupby_benchmark_results.png")
    print("\nBenchmark plot saved to groupby_benchmark_results.png")

def main():
    print("HPDA vs Pandas GroupBy Advanced Benchmark")
    print("========================================")
    
    # Run benchmarks
    results = benchmark_groupby_unique_values(
        sizes=[100000, 500000, 1000000],
        unique_values_counts=[5, 50, 500]
    )
    
    # Plot results
    plot_groupby_results(results)
    
    print("\nBenchmark completed.")

if __name__ == "__main__":
    main()