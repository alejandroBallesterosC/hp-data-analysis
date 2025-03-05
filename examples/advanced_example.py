import hpda
import time
import random
import string

def generate_random_string(length=5):
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

def demo_merge():
    """Demonstrate the merge functionality."""
    print("\nMerge Demonstration")
    print("==================")
    
    # Create dataframes for customers and orders
    customers = hpda.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    })
    
    orders = hpda.DataFrame({
        "order_id": [101, 102, 103, 104, 105, 106],
        "customer_id": [1, 2, 3, 1, 2, 6],  # Notice customer_id 6 has no match
        "amount": [150.5, 200.25, 340.0, 29.99, 150.0, 500.0]
    })
    
    print("Customers DataFrame:")
    customers_dict = customers.to_dict()
    for col in customers_dict:
        print(f"{col}: {customers_dict[col]}")
    
    print("\nOrders DataFrame:")
    orders_dict = orders.to_dict()
    for col in orders_dict:
        print(f"{col}: {orders_dict[col]}")
    
    # Inner join (default)
    inner_join = customers.merge(orders, "customer_id")
    print("\nInner Join Result:")
    inner_dict = inner_join.to_dict()
    for col in inner_dict:
        print(f"{col}: {inner_dict[col]}")
    
    # Left join
    left_join = customers.merge(orders, "customer_id", "left")
    print("\nLeft Join Result:")
    left_dict = left_join.to_dict()
    for col in left_dict:
        print(f"{col}: {left_dict[col]}")
    
    return customers, orders

def demo_large_groupby():
    """Demonstrate group by performance on a larger dataset."""
    print("\nLarge GroupBy Demonstration")
    print("==========================")
    
    # Create a larger dataset with product categories and sales
    size = 10000
    categories = ["Electronics", "Clothing", "Food", "Books", "Home"]
    regions = ["North", "South", "East", "West"]
    
    data = {
        "product_id": [i for i in range(size)],
        "category": [random.choice(categories) for _ in range(size)],
        "region": [random.choice(regions) for _ in range(size)],
        "sales": [random.uniform(10, 1000) for _ in range(size)],
        "units": [random.randint(1, 50) for _ in range(size)]
    }
    
    df = hpda.DataFrame(data)
    
    # Sequential execution
    df.set_execution_policy(hpda.ExecutionPolicy.Sequential)
    start_time = time.time()
    result_seq = df.groupby("category").mean()
    seq_time = time.time() - start_time
    print(f"Sequential groupby time: {seq_time:.4f} seconds")
    
    # Parallel execution
    df.set_execution_policy(hpda.ExecutionPolicy.Parallel)
    df.set_num_threads(4)
    start_time = time.time()
    result_par = df.groupby("category").mean()
    par_time = time.time() - start_time
    print(f"Parallel groupby time: {par_time:.4f} seconds")
    print(f"Speedup: {seq_time / par_time:.2f}x")
    
    # Show results
    result_dict = result_par.to_dict()
    print("\nGroupBy Results (Mean by Category):")
    for col in result_dict:
        print(f"{col}: {result_dict[col]}")
    
    # Multi-key groupby
    print("\nMulti-key GroupBy:")
    start_time = time.time()
    multi_result = df.groupby(["category", "region"]).mean()
    multi_time = time.time() - start_time
    print(f"Multi-key groupby time: {multi_time:.4f} seconds")
    
    multi_dict = multi_result.to_dict()
    print("\nSample of Multi-key GroupBy Results:")
    # Just show first few entries
    for col in multi_dict:
        print(f"{col}: {multi_dict[col][:5]}")
    
    return df

def main():
    print("HP Data Analysis Advanced Demo")
    print("==============================")
    
    demo_merge()
    demo_large_groupby()
    
    print("\nDemo completed.")

if __name__ == "__main__":
    main()