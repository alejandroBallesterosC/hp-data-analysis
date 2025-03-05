"""
Benchmark HPDA against pandas for common operations.

This script compares the performance of our high-performance data analysis library
against pandas for various operations on large datasets, measuring both runtime
and memory usage.
"""

import hpda
import pandas as pd
import numpy as np
import time
import random
import psutil
import os
import gc
from tabulate import tabulate
import matplotlib.pyplot as plt

# Process for memory measurement
process = psutil.Process(os.getpid())

def measure_memory_usage():
    """Measure current memory usage in MB."""
    gc.collect()  # Force garbage collection to get accurate memory usage
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def benchmark_operation(operation_name, pandas_func, hpda_func, sizes, repeats=1):
    """
    Benchmark a specific operation for both pandas and HPDA.
    
    Args:
        operation_name: Name of the operation for reporting
        pandas_func: Function that runs the pandas operation
        hpda_func: Function that runs the HPDA operation
        sizes: List of dataset sizes to test
        repeats: Number of times to repeat each test for averaging
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "operation": operation_name,
        "sizes": sizes,
        "pandas_time": [],
        "hpda_time": [],
        "pandas_memory": [],
        "hpda_memory": [],
        "time_speedup": [],
        "memory_reduction": []
    }
    
    for size in sizes:
        print(f"Benchmarking {operation_name} with size {size}...")
        
        # Pandas timing
        pandas_times = []
        pandas_memories = []
        for _ in range(repeats):
            # Measure initial memory
            initial_memory = measure_memory_usage()
            
            # Run pandas operation
            start_time = time.time()
            pandas_func(size)
            elapsed = time.time() - start_time
            
            # Measure memory after operation
            final_memory = measure_memory_usage()
            memory_used = final_memory - initial_memory
            
            pandas_times.append(elapsed)
            pandas_memories.append(memory_used)
            
            # Clean up
            gc.collect()
        
        # Average pandas results
        avg_pandas_time = sum(pandas_times) / len(pandas_times)
        avg_pandas_memory = sum(pandas_memories) / len(pandas_memories)
        
        # HPDA timing
        hpda_times = []
        hpda_memories = []
        for _ in range(repeats):
            # Measure initial memory
            initial_memory = measure_memory_usage()
            
            # Run HPDA operation
            start_time = time.time()
            hpda_func(size)
            elapsed = time.time() - start_time
            
            # Measure memory after operation
            final_memory = measure_memory_usage()
            memory_used = final_memory - initial_memory
            
            hpda_times.append(elapsed)
            hpda_memories.append(memory_used)
            
            # Clean up
            gc.collect()
        
        # Average HPDA results
        avg_hpda_time = sum(hpda_times) / len(hpda_times)
        avg_hpda_memory = sum(hpda_memories) / len(hpda_memories)
        
        # Calculate speedup and memory reduction
        time_speedup = avg_pandas_time / avg_hpda_time if avg_hpda_time > 0 else float('inf')
        memory_reduction = avg_pandas_memory / avg_hpda_memory if avg_hpda_memory > 0 else float('inf')
        
        # Store results
        results["pandas_time"].append(avg_pandas_time)
        results["hpda_time"].append(avg_hpda_time)
        results["pandas_memory"].append(avg_pandas_memory)
        results["hpda_memory"].append(avg_hpda_memory)
        results["time_speedup"].append(time_speedup)
        results["memory_reduction"].append(memory_reduction)
        
        print(f"  Pandas: {avg_pandas_time:.4f}s, {avg_pandas_memory:.2f}MB")
        print(f"  HPDA: {avg_hpda_time:.4f}s, {avg_hpda_memory:.2f}MB")
        print(f"  Speedup: {time_speedup:.2f}x, Memory reduction: {memory_reduction:.2f}x")
    
    return results

def create_pandas_dataframe(size):
    """Create a pandas DataFrame with random data."""
    return pd.DataFrame({
        "A": np.random.randint(0, 1000, size),
        "B": np.random.random(size) * 100,
        "C": np.random.random(size) * 100,
        "D": np.random.random(size) * 100,
        "E": np.random.choice(["cat", "dog", "bird", "fish", "hamster"], size)
    })

def create_hpda_dataframe(size):
    """Create an HPDA DataFrame with random data."""
    return hpda.DataFrame({
        "A": [random.randint(0, 1000) for _ in range(size)],
        "B": [random.random() * 100 for _ in range(size)],
        "C": [random.random() * 100 for _ in range(size)],
        "D": [random.random() * 100 for _ in range(size)],
        "E": [random.choice(["cat", "dog", "bird", "fish", "hamster"]) for _ in range(size)]
    })

# Define benchmark operations
def benchmark_creation():
    """Benchmark DataFrame creation."""
    sizes = [10000, 25000, 50000]
    
    def pandas_creation(size):
        df = create_pandas_dataframe(size)
        return df
    
    def hpda_creation(size):
        df = create_hpda_dataframe(size)
        return df
    
    return benchmark_operation("DataFrame Creation", pandas_creation, hpda_creation, sizes)

def benchmark_statistical_ops():
    """Benchmark statistical operations (mean, std)."""
    sizes = [10000, 25000, 50000]
    
    def pandas_stats(size):
        df = create_pandas_dataframe(size)
        mean_vals = df[["A", "B", "C", "D"]].mean()
        std_vals = df[["A", "B", "C", "D"]].std()
        return mean_vals, std_vals
    
    def hpda_stats(size):
        df = create_hpda_dataframe(size)
        # Set parallel execution for HPDA
        df.set_execution_policy(hpda.ExecutionPolicy.Parallel)
        df.set_num_threads(4)
        mean_vals = df.mean()
        std_vals = df.std()
        return mean_vals, std_vals
    
    return benchmark_operation("Statistical Operations", pandas_stats, hpda_stats, sizes)

def benchmark_sorting():
    """Benchmark sorting operations."""
    sizes = [10000, 25000, 50000]
    
    def pandas_sort(size):
        df = create_pandas_dataframe(size)
        sorted_df = df.sort_values(by="B", ascending=False)
        return sorted_df
    
    def hpda_sort(size):
        df = create_hpda_dataframe(size)
        # Set parallel execution for HPDA
        df.set_execution_policy(hpda.ExecutionPolicy.Parallel)
        df.set_num_threads(4)
        sorted_df = df.sort_values("B", False)
        return sorted_df
    
    return benchmark_operation("Sorting", pandas_sort, hpda_sort, sizes)

def benchmark_groupby():
    """Benchmark groupby operations."""
    sizes = [10000, 25000, 50000]
    
    def pandas_groupby(size):
        df = create_pandas_dataframe(size)
        grouped = df.groupby("E").agg({"A": "mean", "B": "mean", "C": "mean", "D": "mean"})
        return grouped
    
    def hpda_groupby(size):
        df = create_hpda_dataframe(size)
        # Set parallel execution for HPDA
        df.set_execution_policy(hpda.ExecutionPolicy.Parallel)
        df.set_num_threads(4)
        grouped = df.groupby("E").mean()
        return grouped
    
    return benchmark_operation("GroupBy", pandas_groupby, hpda_groupby, sizes)

def plot_results(results):
    """Plot benchmark results."""
    # Create a figure with two rows of subplots (time and memory)
    fig, axes = plt.subplots(2, len(results), figsize=(15, 10))
    
    for i, result in enumerate(results):
        sizes = result["sizes"]
        
        # Time comparison
        axes[0, i].plot(sizes, result["pandas_time"], marker='o', label='pandas')
        axes[0, i].plot(sizes, result["hpda_time"], marker='x', label='hpda')
        axes[0, i].set_title(f"{result['operation']} - Time (s)")
        axes[0, i].set_xlabel("Dataset Size")
        axes[0, i].set_ylabel("Time (seconds)")
        axes[0, i].legend()
        axes[0, i].grid(True)
        
        # Memory comparison
        axes[1, i].plot(sizes, result["pandas_memory"], marker='o', label='pandas')
        axes[1, i].plot(sizes, result["hpda_memory"], marker='x', label='hpda')
        axes[1, i].set_title(f"{result['operation']} - Memory (MB)")
        axes[1, i].set_xlabel("Dataset Size")
        axes[1, i].set_ylabel("Memory (MB)")
        axes[1, i].legend()
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    plt.close()
    
    # Also create speedup charts
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    
    for i, result in enumerate(results):
        sizes = result["sizes"]
        
        # Speedup
        axes[i].plot(sizes, result["time_speedup"], marker='o', label='Time Speedup')
        axes[i].plot(sizes, result["memory_reduction"], marker='x', label='Memory Reduction')
        axes[i].set_title(f"{result['operation']} - Improvement Factor")
        axes[i].set_xlabel("Dataset Size")
        axes[i].set_ylabel("Factor (higher is better)")
        axes[i].axhline(y=1.0, color='r', linestyle='--')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig("benchmark_speedup.png")
    plt.close()

def print_results_table(results):
    """Print a table of benchmark results."""
    headers = ["Operation", "Size", "Pandas Time", "HPDA Time", "Speedup", 
               "Pandas Memory", "HPDA Memory", "Memory Reduction"]
    
    table_data = []
    
    for result in results:
        op_name = result["operation"]
        for i, size in enumerate(result["sizes"]):
            row = [
                op_name,
                size,
                f"{result['pandas_time'][i]:.4f}s",
                f"{result['hpda_time'][i]:.4f}s",
                f"{result['time_speedup'][i]:.2f}x",
                f"{result['pandas_memory'][i]:.2f}MB",
                f"{result['hpda_memory'][i]:.2f}MB",
                f"{result['memory_reduction'][i]:.2f}x"
            ]
            table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    """Run all benchmarks and report results."""
    print("HPDA vs Pandas Benchmark")
    print("=======================")
    print("Comparing performance for common operations on large datasets")
    print("Note: First run may include initialization overhead")
    print("\n")
    
    # Run benchmarks
    results = []
    
    print("=== DataFrame Creation Benchmark ===")
    creation_results = benchmark_creation()
    results.append(creation_results)
    
    print("\n=== Statistical Operations Benchmark ===")
    stats_results = benchmark_statistical_ops()
    results.append(stats_results)
    
    print("\n=== Sorting Benchmark ===")
    sorting_results = benchmark_sorting()
    results.append(sorting_results)
    
    print("\n=== GroupBy Benchmark ===")
    groupby_results = benchmark_groupby()
    results.append(groupby_results)
    
    # Print summary table
    print("\n\nBenchmark Results Summary")
    print("========================")
    print_results_table(results)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(results)
    print("Plots saved to benchmark_results.png and benchmark_speedup.png")

if __name__ == "__main__":
    main()