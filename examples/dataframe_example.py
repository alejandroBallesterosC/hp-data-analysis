import hpda
import time
import numpy as np

def demo_basic_operations():
    print("Creating DataFrame with sample data...")
    
    # Create a sample DataFrame
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [10.1, 20.2, 30.3, 40.4, 50.5],
        "C": ["a", "b", "c", "d", "e"]
    }
    
    df = hpda.DataFrame(data)
    
    print("DataFrame created.")
    print(f"Shape: {df.shape()}")
    print(f"Columns: {df.columns()}")
    
    # Basic operations
    print("\nBasic operations:")
    print(f"Mean of column A: {df['A'].mean()}")
    print(f"Median of column B: {df['B'].median()}")
    print(f"Standard deviation of column B: {df['B'].std()}")
    
    # Print first few rows
    print("\nHead of DataFrame:")
    head_df = df.head(3)
    head_dict = head_df.to_dict()
    for col in head_dict:
        print(f"{col}: {head_dict[col]}")
    
    # Sort values
    print("\nSorted by column B (descending):")
    sorted_df = df.sort_values("B", False)
    sorted_dict = sorted_df.to_dict()
    for col in sorted_dict:
        print(f"{col}: {sorted_dict[col]}")
    
    return df

def demo_performance_scaling(size=100000):
    """
    Demonstrate performance scaling with different execution policies
    """
    print(f"\nPerformance test with {size} rows...")
    
    # Create large DataFrame with lists (not NumPy arrays)
    data = {
        "A": [np.random.randint(0, 1000) for _ in range(size)],
        "B": [np.random.random() * 100 for _ in range(size)],
        "C": [np.random.random() * 100 for _ in range(size)],
        "D": [np.random.random() * 100 for _ in range(size)],
        "E": [np.random.random() * 100 for _ in range(size)]
    }
    
    df = hpda.DataFrame(data)
    
    # Test sequential
    df.set_execution_policy(hpda.ExecutionPolicy.Sequential)
    start_time = time.time()
    result_seq = df.mean()
    seq_time = time.time() - start_time
    print(f"Sequential execution time: {seq_time:.4f} seconds")
    
    # Test parallel
    df.set_execution_policy(hpda.ExecutionPolicy.Parallel)
    # Set number of threads to 4
    df.set_num_threads(4)
    start_time = time.time()
    result_par = df.mean()
    par_time = time.time() - start_time
    print(f"Parallel execution time: {par_time:.4f} seconds")
    print(f"Speedup: {seq_time / par_time:.2f}x")
    
    # Since we're likely not running on a system with a GPU or cluster,
    # we'll just explain the capabilities rather than demonstrate
    print(f"\nAdditional execution policies available:")
    print("- GPU: Utilize GPU for even greater parallelism")
    print("- Distributed: Distribute workload across multiple machines")
    
    # Test sort performance
    print("\nTesting sort_values performance...")
    start_time = time.time()
    df.sort_values("B")
    sort_time = time.time() - start_time
    print(f"Sort execution time: {sort_time:.4f} seconds")
    
    return df

def demo_groupby():
    """
    Demonstrate groupby functionality
    """
    print("\nGroupBy demonstration...")
    
    # Create DataFrame with groups
    data = {
        "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "subcategory": ["X", "X", "Y", "Y", "X", "X", "Y", "Y"],
        "value": [10, 20, 30, 40, 50, 60, 70, 80]
    }
    
    df = hpda.DataFrame(data)
    
    # Single key groupby
    print("\nGroup by 'category':")
    gb_result = df.groupby("category").mean()
    gb_dict = gb_result.to_dict()
    for col in gb_dict:
        print(f"{col}: {gb_dict[col]}")
    
    # Multi-key groupby
    print("\nGroup by ['category', 'subcategory']:")
    gb_result = df.groupby(["category", "subcategory"]).mean()
    gb_dict = gb_result.to_dict()
    for col in gb_dict:
        print(f"{col}: {gb_dict[col]}")
    
    return df

def main():
    print("HP Data Analysis Library Demo")
    print("============================")
    
    # Basic operations
    df = demo_basic_operations()
    
    # GroupBy
    demo_groupby()
    
    # Performance scaling
    # Reduced size for demo purposes
    df_perf = demo_performance_scaling(50000)
    
    print("\nDemo completed.")

if __name__ == "__main__":
    main()