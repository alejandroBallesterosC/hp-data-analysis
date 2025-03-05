"""
Quick benchmark to compare HPDA against pandas on a few operations.
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

def benchmark_operation(name, pandas_func, hpda_func, repeats=1):
    """Run benchmark for a single operation."""
    print(f"\nBenchmarking {name}...")
    
    # Pandas
    pandas_times = []
    pandas_memories = []
    for _ in range(repeats):
        initial_memory = measure_memory_usage()
        start_time = time.time()
        pandas_func()
        elapsed = time.time() - start_time
        final_memory = measure_memory_usage()
        memory_used = final_memory - initial_memory
        
        pandas_times.append(elapsed)
        pandas_memories.append(memory_used)
        gc.collect()
    
    avg_pandas_time = sum(pandas_times) / len(pandas_times)
    avg_pandas_memory = sum(pandas_memories) / len(pandas_memories)
    
    # HPDA
    hpda_times = []
    hpda_memories = []
    for _ in range(repeats):
        initial_memory = measure_memory_usage()
        start_time = time.time()
        hpda_func()
        elapsed = time.time() - start_time
        final_memory = measure_memory_usage()
        memory_used = final_memory - initial_memory
        
        hpda_times.append(elapsed)
        hpda_memories.append(memory_used)
        gc.collect()
    
    avg_hpda_time = sum(hpda_times) / len(hpda_times)
    avg_hpda_memory = sum(hpda_memories) / len(hpda_memories)
    
    # Calculate improvements
    time_speedup = avg_pandas_time / avg_hpda_time if avg_hpda_time > 0 else float('inf')
    memory_reduction = avg_pandas_memory / avg_hpda_memory if avg_hpda_memory > 0 else float('inf')
    
    print(f"  Pandas: {avg_pandas_time:.4f}s, {avg_pandas_memory:.2f}MB")
    print(f"  HPDA: {avg_hpda_time:.4f}s, {avg_hpda_memory:.2f}MB")
    print(f"  Speedup: {time_speedup:.2f}x, Memory reduction: {memory_reduction:.2f}x")

def main():
    print("Quick HPDA vs Pandas Benchmark")
    print("============================")
    
    # Test size
    size = 10000
    
    # Create test data functions
    def create_pandas_df():
        return pd.DataFrame({
            "A": np.random.randint(0, 1000, size),
            "B": np.random.random(size) * 100,
            "C": np.random.random(size) * 100,
            "D": np.random.random(size) * 100,
        })
    
    def create_hpda_df():
        return hpda.DataFrame({
            "A": [random.randint(0, 1000) for _ in range(size)],
            "B": [random.random() * 100 for _ in range(size)],
            "C": [random.random() * 100 for _ in range(size)],
            "D": [random.random() * 100 for _ in range(size)],
        })
    
    # DataFrame creation
    benchmark_operation("DataFrame Creation", create_pandas_df, create_hpda_df)
    
    # Mean calculation
    pandas_df = create_pandas_df()
    hpda_df = create_hpda_df()
    hpda_df.set_execution_policy(hpda.ExecutionPolicy.Parallel)
    hpda_df.set_num_threads(4)
    
    benchmark_operation("Mean Calculation", 
                      lambda: pandas_df.mean(), 
                      lambda: hpda_df.mean())
    
    # Sorting
    benchmark_operation("Sorting", 
                      lambda: pandas_df.sort_values(by="B"), 
                      lambda: hpda_df.sort_values("B"))
    
    # GroupBy
    pandas_df["E"] = np.random.choice(["cat", "dog", "bird"], size)
    hpda_df_with_groups = create_hpda_df()
    hpda_df_with_groups.add_column("E", [random.choice(["cat", "dog", "bird"]) for _ in range(size)])
    hpda_df_with_groups.set_execution_policy(hpda.ExecutionPolicy.Parallel)
    hpda_df_with_groups.set_num_threads(4)
    
    benchmark_operation("GroupBy", 
                      lambda: pandas_df.groupby("E").mean(), 
                      lambda: hpda_df_with_groups.groupby("E").mean())
    
    print("\nBenchmark completed.")

if __name__ == "__main__":
    main()