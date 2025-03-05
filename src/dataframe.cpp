#include "hpda/dataframe.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <thread>
#include <future>
#include <map>
#include <unordered_set>

namespace hpda {

// Helper functions
static bool compare_values(const DataValue& a, const DataValue& b);
static std::string value_to_string(const DataValue& val);

// DataFrame implementation
DataFrame::DataFrame(const std::unordered_map<std::string, Column>& data)
    : data_(data) {}

DataFrame::DataFrame(std::unordered_map<std::string, Column>&& data)
    : data_(std::move(data)) {}

size_t DataFrame::rows() const {
    if (data_.empty()) {
        return 0;
    }
    // All columns must have the same length
    return data_.begin()->second.size();
}

size_t DataFrame::cols() const {
    return data_.size();
}

std::vector<std::string> DataFrame::columns() const {
    std::vector<std::string> result;
    result.reserve(data_.size());
    for (const auto& [name, _] : data_) {
        result.push_back(name);
    }
    return result;
}

Series DataFrame::operator[](const std::string& column_name) const {
    auto it = data_.find(column_name);
    if (it == data_.end()) {
        throw std::out_of_range("Column '" + column_name + "' not found");
    }
    return Series(it->second, column_name);
}

void DataFrame::add_column(const std::string& name, const Column& data) {
    if (!data_.empty() && data.size() != rows()) {
        throw std::invalid_argument("Column size must match DataFrame row count");
    }
    data_[name] = data;
}

void DataFrame::add_column(const std::string& name, Column&& data) {
    if (!data_.empty() && data.size() != rows()) {
        throw std::invalid_argument("Column size must match DataFrame row count");
    }
    data_[name] = std::move(data);
}

void DataFrame::drop_column(const std::string& name) {
    auto it = data_.find(name);
    if (it == data_.end()) {
        throw std::out_of_range("Column '" + name + "' not found");
    }
    data_.erase(it);
}

DataFrame DataFrame::head(size_t n) const {
    if (n >= rows() || rows() == 0) {
        return *this;
    }

    std::unordered_map<std::string, Column> result_data;
    for (const auto& [name, col] : data_) {
        Column new_col(col.begin(), col.begin() + n);
        result_data[name] = std::move(new_col);
    }
    return DataFrame(std::move(result_data));
}

DataFrame DataFrame::tail(size_t n) const {
    if (n >= rows() || rows() == 0) {
        return *this;
    }

    size_t start_idx = rows() - n;
    std::unordered_map<std::string, Column> result_data;
    for (const auto& [name, col] : data_) {
        Column new_col(col.begin() + start_idx, col.end());
        result_data[name] = std::move(new_col);
    }
    return DataFrame(std::move(result_data));
}

// Template helper for reduction operations
template<typename Op>
Series DataFrame::apply_reduction(Op operation) const {
    Column result;
    result.reserve(cols());
    std::vector<std::string> column_names = columns();

    if (execution_policy_ == ExecutionPolicy::Parallel) {
        // Parallel execution using threads (simplified from std::execution::par)
        std::mutex result_mutex;
        std::vector<std::thread> threads;
        
        size_t chunk_size = std::max(size_t(1), column_names.size() / num_threads_);
        
        for (size_t i = 0; i < column_names.size(); i += chunk_size) {
            size_t end = std::min(i + chunk_size, column_names.size());
            
            threads.emplace_back([&, i, end]() {
                std::vector<std::pair<size_t, DataValue>> thread_results;
                
                for (size_t j = i; j < end; ++j) {
                    const auto& col_name = column_names[j];
                    const auto& col = data_.at(col_name);
                    DataValue val = operation(col);
                    thread_results.emplace_back(j, val);
                }
                
                std::lock_guard<std::mutex> lock(result_mutex);
                for (const auto& [idx, val] : thread_results) {
                    if (idx >= result.size()) {
                        result.resize(idx + 1);
                    }
                    result[idx] = val;
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } 
    else if (execution_policy_ == ExecutionPolicy::Distributed && !cluster_nodes_.empty()) {
        // Simplified distributed execution (real implementation would require network communication)
        size_t chunk_size = column_names.size() / cluster_nodes_.size();
        std::vector<std::future<std::vector<DataValue>>> futures;
        
        for (size_t i = 0; i < cluster_nodes_.size(); ++i) {
            size_t start_idx = i * chunk_size;
            size_t end_idx = (i == cluster_nodes_.size() - 1) ? column_names.size() : (i + 1) * chunk_size;
            
            futures.push_back(std::async(std::launch::async, [&, start_idx, end_idx]() {
                std::vector<DataValue> chunk_result;
                for (size_t j = start_idx; j < end_idx; ++j) {
                    const auto& col = data_.at(column_names[j]);
                    chunk_result.push_back(operation(col));
                }
                return chunk_result;
            }));
        }
        
        for (auto& future : futures) {
            auto chunk_result = future.get();
            result.insert(result.end(), chunk_result.begin(), chunk_result.end());
        }
    }
    else {
        // Sequential execution
        for (const auto& col_name : column_names) {
            const auto& col = data_.at(col_name);
            result.push_back(operation(col));
        }
    }
    
    return Series(std::move(result));
}

Series DataFrame::mean() const {
    return apply_reduction([](const Column& col) -> DataValue {
        double sum = 0.0;
        size_t count = 0;
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                sum += std::get<double>(val);
                count++;
            }
            else if (std::holds_alternative<int64_t>(val)) {
                sum += static_cast<double>(std::get<int64_t>(val));
                count++;
            }
        }
        
        return count > 0 ? sum / count : 0.0;
    });
}

Series DataFrame::median() const {
    return apply_reduction([](const Column& col) -> DataValue {
        std::vector<double> numeric_values;
        numeric_values.reserve(col.size());
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                numeric_values.push_back(std::get<double>(val));
            }
            else if (std::holds_alternative<int64_t>(val)) {
                numeric_values.push_back(static_cast<double>(std::get<int64_t>(val)));
            }
        }
        
        if (numeric_values.empty()) {
            return 0.0;
        }
        
        std::sort(numeric_values.begin(), numeric_values.end());
        size_t size = numeric_values.size();
        if (size % 2 == 0) {
            return (numeric_values[size/2 - 1] + numeric_values[size/2]) / 2.0;
        } else {
            return numeric_values[size/2];
        }
    });
}

Series DataFrame::std() const {
    // Calculate standard deviation for each column
    return apply_reduction([](const Column& col) -> DataValue {
        std::vector<double> numeric_values;
        numeric_values.reserve(col.size());
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                numeric_values.push_back(std::get<double>(val));
            }
            else if (std::holds_alternative<int64_t>(val)) {
                numeric_values.push_back(static_cast<double>(std::get<int64_t>(val)));
            }
        }
        
        if (numeric_values.empty()) {
            return 0.0;
        }
        
        double sum = std::accumulate(numeric_values.begin(), numeric_values.end(), 0.0);
        double mean = sum / numeric_values.size();
        
        double sq_sum = std::inner_product(
            numeric_values.begin(), numeric_values.end(), numeric_values.begin(), 0.0,
            std::plus<>(), [mean](double x, double y) { return (x - mean) * (y - mean); }
        );
        
        return std::sqrt(sq_sum / numeric_values.size());
    });
}

Series DataFrame::max() const {
    return apply_reduction([](const Column& col) -> DataValue {
        if (col.empty()) {
            return 0.0;
        }
        
        std::optional<double> max_numeric;
        std::optional<std::string> max_string;
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                double num = std::get<double>(val);
                if (!max_numeric || num > *max_numeric) {
                    max_numeric = num;
                }
            }
            else if (std::holds_alternative<int64_t>(val)) {
                double num = static_cast<double>(std::get<int64_t>(val));
                if (!max_numeric || num > *max_numeric) {
                    max_numeric = num;
                }
            }
            else if (std::holds_alternative<std::string>(val)) {
                const std::string& str = std::get<std::string>(val);
                if (!max_string || str > *max_string) {
                    max_string = str;
                }
            }
        }
        
        if (max_numeric) {
            return *max_numeric;
        } else if (max_string) {
            return *max_string;
        } else {
            return 0.0;  // Default for empty column
        }
    });
}

Series DataFrame::min() const {
    return apply_reduction([](const Column& col) -> DataValue {
        if (col.empty()) {
            return 0.0;
        }
        
        std::optional<double> min_numeric;
        std::optional<std::string> min_string;
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                double num = std::get<double>(val);
                if (!min_numeric || num < *min_numeric) {
                    min_numeric = num;
                }
            }
            else if (std::holds_alternative<int64_t>(val)) {
                double num = static_cast<double>(std::get<int64_t>(val));
                if (!min_numeric || num < *min_numeric) {
                    min_numeric = num;
                }
            }
            else if (std::holds_alternative<std::string>(val)) {
                const std::string& str = std::get<std::string>(val);
                if (!min_string || str < *min_string) {
                    min_string = str;
                }
            }
        }
        
        if (min_numeric) {
            return *min_numeric;
        } else if (min_string) {
            return *min_string;
        } else {
            return 0.0;  // Default for empty column
        }
    });
}

Series DataFrame::sum() const {
    return apply_reduction([](const Column& col) -> DataValue {
        double sum = 0.0;
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                sum += std::get<double>(val);
            }
            else if (std::holds_alternative<int64_t>(val)) {
                sum += static_cast<double>(std::get<int64_t>(val));
            }
        }
        
        return sum;
    });
}

// Helper for sort_values
static bool compare_values(const DataValue& a, const DataValue& b) {
    if (std::holds_alternative<double>(a) && std::holds_alternative<double>(b)) {
        return std::get<double>(a) < std::get<double>(b);
    }
    else if (std::holds_alternative<int64_t>(a) && std::holds_alternative<int64_t>(b)) {
        return std::get<int64_t>(a) < std::get<int64_t>(b);
    }
    else if (std::holds_alternative<double>(a) && std::holds_alternative<int64_t>(b)) {
        return std::get<double>(a) < static_cast<double>(std::get<int64_t>(b));
    }
    else if (std::holds_alternative<int64_t>(a) && std::holds_alternative<double>(b)) {
        return static_cast<double>(std::get<int64_t>(a)) < std::get<double>(b);
    }
    else if (std::holds_alternative<std::string>(a) && std::holds_alternative<std::string>(b)) {
        return std::get<std::string>(a) < std::get<std::string>(b);
    }
    else if (std::holds_alternative<bool>(a) && std::holds_alternative<bool>(b)) {
        return std::get<bool>(a) < std::get<bool>(b);
    }
    
    // Default ordering for mixed types based on type index
    return a.index() < b.index();
}

DataFrame DataFrame::sort_values(const std::string& column, bool ascending) const {
    if (data_.find(column) == data_.end()) {
        throw std::out_of_range("Column '" + column + "' not found");
    }
    
    // Create indices and sort them
    const auto& sort_col = data_.at(column);
    std::vector<size_t> indices(sort_col.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Use parallel sorting based on the execution policy
    if (execution_policy_ == ExecutionPolicy::Sequential) {
        // Sequential sorting
        if (ascending) {
            std::sort(indices.begin(), indices.end(), [&sort_col](size_t i1, size_t i2) {
                return compare_values(sort_col[i1], sort_col[i2]);
            });
        } else {
            std::sort(indices.begin(), indices.end(), [&sort_col](size_t i1, size_t i2) {
                return compare_values(sort_col[i2], sort_col[i1]);
            });
        }
    } 
    else if (execution_policy_ == ExecutionPolicy::Parallel || 
             execution_policy_ == ExecutionPolicy::Distributed) {
        // Implement parallel sorting with a divide-and-conquer approach
        const size_t n = indices.size();
        const size_t num_workers = (execution_policy_ == ExecutionPolicy::Parallel) 
                                 ? num_threads_ 
                                 : cluster_nodes_.size();
        
        // Use at least 2 threads even if only 1 was specified
        const size_t workers = std::max(size_t(2), num_workers);
        const size_t chunk_size = (n + workers - 1) / workers;
        
        std::vector<std::thread> threads;
        std::vector<std::vector<size_t>> sorted_chunks(workers);
        
        // Step 1: Sort chunks in parallel
        for (size_t t = 0; t < workers; ++t) {
            threads.emplace_back([&, t]() {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, n);
                
                if (start >= end) return;
                
                std::vector<size_t> chunk_indices(indices.begin() + start, indices.begin() + end);
                
                if (ascending) {
                    std::sort(chunk_indices.begin(), chunk_indices.end(), [&sort_col](size_t i1, size_t i2) {
                        return compare_values(sort_col[i1], sort_col[i2]);
                    });
                } else {
                    std::sort(chunk_indices.begin(), chunk_indices.end(), [&sort_col](size_t i1, size_t i2) {
                        return compare_values(sort_col[i2], sort_col[i1]);
                    });
                }
                
                sorted_chunks[t] = std::move(chunk_indices);
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Step 2: Merge sorted chunks using a merge heap
        if (ascending) {
            // Create a min-heap for merging
            using HeapEntry = std::pair<DataValue, std::pair<size_t, size_t>>;  // (value, (chunk_idx, pos_in_chunk))
            
            auto compare_heap = [](const HeapEntry& a, const HeapEntry& b) {
                return compare_values(b.first, a.first);  // Inverted for min-heap
            };
            
            std::vector<HeapEntry> heap;
            std::vector<size_t> chunk_positions(workers, 0);
            
            // Initialize heap with first element from each chunk
            for (size_t i = 0; i < workers; ++i) {
                if (!sorted_chunks[i].empty()) {
                    size_t idx = sorted_chunks[i][0];
                    heap.emplace_back(sort_col[idx], std::make_pair(i, 0));
                }
            }
            
            std::make_heap(heap.begin(), heap.end(), compare_heap);
            
            // Merge chunks
            std::vector<size_t> merged_indices;
            merged_indices.reserve(n);
            
            while (!heap.empty()) {
                std::pop_heap(heap.begin(), heap.end(), compare_heap);
                auto entry = heap.back();
                heap.pop_back();
                
                size_t chunk_idx = entry.second.first;
                size_t pos_in_chunk = entry.second.second;
                
                merged_indices.push_back(sorted_chunks[chunk_idx][pos_in_chunk]);
                
                // If we have more elements in this chunk, add the next one to the heap
                if (pos_in_chunk + 1 < sorted_chunks[chunk_idx].size()) {
                    size_t next_idx = sorted_chunks[chunk_idx][pos_in_chunk + 1];
                    heap.emplace_back(sort_col[next_idx], std::make_pair(chunk_idx, pos_in_chunk + 1));
                    std::push_heap(heap.begin(), heap.end(), compare_heap);
                }
            }
            
            indices = std::move(merged_indices);
        } else {
            // For descending order, use max-heap (similar as above but with inverted comparison)
            using HeapEntry = std::pair<DataValue, std::pair<size_t, size_t>>;
            
            auto compare_heap = [](const HeapEntry& a, const HeapEntry& b) {
                return compare_values(a.first, b.first);  // Normal comparison for max-heap for descending order
            };
            
            std::vector<HeapEntry> heap;
            std::vector<size_t> chunk_positions(workers, 0);
            
            for (size_t i = 0; i < workers; ++i) {
                if (!sorted_chunks[i].empty()) {
                    size_t idx = sorted_chunks[i][0];
                    heap.emplace_back(sort_col[idx], std::make_pair(i, 0));
                }
            }
            
            std::make_heap(heap.begin(), heap.end(), compare_heap);
            
            std::vector<size_t> merged_indices;
            merged_indices.reserve(n);
            
            while (!heap.empty()) {
                std::pop_heap(heap.begin(), heap.end(), compare_heap);
                auto entry = heap.back();
                heap.pop_back();
                
                size_t chunk_idx = entry.second.first;
                size_t pos_in_chunk = entry.second.second;
                
                merged_indices.push_back(sorted_chunks[chunk_idx][pos_in_chunk]);
                
                if (pos_in_chunk + 1 < sorted_chunks[chunk_idx].size()) {
                    size_t next_idx = sorted_chunks[chunk_idx][pos_in_chunk + 1];
                    heap.emplace_back(sort_col[next_idx], std::make_pair(chunk_idx, pos_in_chunk + 1));
                    std::push_heap(heap.begin(), heap.end(), compare_heap);
                }
            }
            
            indices = std::move(merged_indices);
        }
    }
    else if (execution_policy_ == ExecutionPolicy::GPU) {
        // Simulated GPU sorting (in a real implementation, this would offload to GPU)
        std::cout << "Simulating GPU sort execution" << std::endl;
        
        if (ascending) {
            std::sort(indices.begin(), indices.end(), [&sort_col](size_t i1, size_t i2) {
                return compare_values(sort_col[i1], sort_col[i2]);
            });
        } else {
            std::sort(indices.begin(), indices.end(), [&sort_col](size_t i1, size_t i2) {
                return compare_values(sort_col[i2], sort_col[i1]);
            });
        }
    }
    
    // Create new dataframe with sorted data using parallel construction
    std::unordered_map<std::string, Column> sorted_data;
    
    if (execution_policy_ == ExecutionPolicy::Parallel && rows() > 10000) {
        std::mutex map_mutex;
        std::vector<std::thread> threads;
        
        // Pre-allocate all columns to avoid mutex contention during push_back
        for (const auto& [name, col] : data_) {
            sorted_data[name] = Column(col.size());
        }
        
        const size_t chunk_size = (indices.size() + num_threads_ - 1) / num_threads_;
        
        for (int t = 0; t < num_threads_; ++t) {
            threads.emplace_back([&, t]() {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, indices.size());
                
                for (const auto& [name, col] : data_) {
                    auto& sorted_col = sorted_data[name];
                    for (size_t i = start; i < end; ++i) {
                        sorted_col[i] = col[indices[i]];
                    }
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Sequential construction for smaller datasets or non-parallel policies
        for (const auto& [name, col] : data_) {
            Column sorted_col(col.size());
            for (size_t i = 0; i < indices.size(); ++i) {
                sorted_col[i] = col[indices[i]];
            }
            sorted_data[name] = std::move(sorted_col);
        }
    }
    
    return DataFrame(std::move(sorted_data));
}

// Helper for merge
static std::string value_to_string(const DataValue& val) {
    if (std::holds_alternative<double>(val)) {
        return std::to_string(std::get<double>(val));
    }
    else if (std::holds_alternative<int64_t>(val)) {
        return std::to_string(std::get<int64_t>(val));
    }
    else if (std::holds_alternative<std::string>(val)) {
        return std::get<std::string>(val);
    }
    else if (std::holds_alternative<bool>(val)) {
        return std::get<bool>(val) ? "true" : "false";
    }
    return "";
}

DataFrame DataFrame::merge(const DataFrame& other, const std::string& on, 
                       const std::string& how) const {
    // Check if the 'on' column exists in both dataframes
    if (data_.find(on) == data_.end() || other.data_.find(on) == other.data_.end()) {
        throw std::invalid_argument("Join column not found in one or both DataFrames");
    }
    
    // Get references to the join columns
    const auto& left_key = data_.at(on);
    const auto& right_key = other.data_.at(on);
    
    // If the result is likely to be very large, we might want to use a different approach
    // For now, we'll improve the hash-based approach
    
    // Estimate result size to pre-allocate memory
    const size_t left_size = left_key.size();
    const size_t right_size = right_key.size();
    
    // Approximate number of matches based on join type
    // For many-to-many joins this is a rough estimate
    size_t estimated_result_size;
    if (how == "inner") {
        // For inner join, estimate based on smaller table size
        estimated_result_size = std::min(left_size, right_size);
    } else if (how == "left") {
        // For left join, result size is at least the size of the left table
        estimated_result_size = left_size;
    } else if (how == "right") {
        // For right join, result size is at least the size of the right table
        estimated_result_size = right_size;
    } else { // "outer" join
        // For outer join, result size is at most the sum of both tables
        estimated_result_size = left_size + right_size;
    }
    
    // Create a more efficient hash map for the right dataframe's keys
    // This uses direct DataValue comparison rather than string conversion
    
    // Define a custom hasher for DataValue
    struct DataValueHasher {
        std::size_t operator()(const DataValue& val) const {
            if (std::holds_alternative<double>(val)) {
                return std::hash<double>{}(std::get<double>(val));
            }
            else if (std::holds_alternative<int64_t>(val)) {
                return std::hash<int64_t>{}(std::get<int64_t>(val));
            }
            else if (std::holds_alternative<std::string>(val)) {
                return std::hash<std::string>{}(std::get<std::string>(val));
            }
            else if (std::holds_alternative<bool>(val)) {
                return std::hash<bool>{}(std::get<bool>(val));
            }
            return 0;
        }
    };
    
    // Define a custom equality comparator for DataValue
    struct DataValueEqual {
        bool operator()(const DataValue& a, const DataValue& b) const {
            if (a.index() != b.index()) return false;
            
            if (std::holds_alternative<double>(a)) {
                return std::get<double>(a) == std::get<double>(b);
            }
            else if (std::holds_alternative<int64_t>(a)) {
                return std::get<int64_t>(a) == std::get<int64_t>(b);
            }
            else if (std::holds_alternative<std::string>(a)) {
                return std::get<std::string>(a) == std::get<std::string>(b);
            }
            else if (std::holds_alternative<bool>(a)) {
                return std::get<bool>(a) == std::get<bool>(b);
            }
            return false;
        }
    };
    
    // Use a multimap to handle multiple matches
    std::unordered_multimap<DataValue, size_t, DataValueHasher, DataValueEqual> right_indices;
    right_indices.reserve(right_size);
    
    // Track which right rows have been matched (for RIGHT and OUTER joins)
    std::vector<bool> right_matched;
    if (how == "right" || how == "outer") {
        right_matched.resize(right_size, false);
    }
    
    // Build the hash map for the right dataframe's keys
    for (size_t i = 0; i < right_size; ++i) {
        right_indices.emplace(right_key[i], i);
    }
    
    // Prepare result dataframe with pre-allocated columns
    std::unordered_map<std::string, Column> result_data;
    
    // Pre-allocate columns for better memory efficiency
    const size_t column_reserve_size = std::min(estimated_result_size, size_t(1000000));  // Cap at 1M to avoid excessive allocation
    
    // Get all column names from both dataframes
    std::vector<std::string> left_columns;
    std::vector<std::string> right_columns;
    
    for (const auto& [name, _] : data_) {
        left_columns.push_back(name);
        result_data[name] = Column();
        result_data[name].reserve(column_reserve_size);
    }
    
    for (const auto& [name, _] : other.data_) {
        if (name != on && data_.find(name) == data_.end()) {
            right_columns.push_back(name);
            result_data[name] = Column();
            result_data[name].reserve(column_reserve_size);
        }
    }
    
    // Whether to use parallel processing for large joins
    bool use_parallel = (execution_policy_ == ExecutionPolicy::Parallel || 
                         execution_policy_ == ExecutionPolicy::Distributed) && 
                        left_size > 10000;
    
    // Perform the join operation
    if (use_parallel) {
        // For parallel joins, we need to process data in chunks
        const int num_workers = (execution_policy_ == ExecutionPolicy::Parallel) 
                            ? num_threads_ 
                            : cluster_nodes_.size();
                            
        const size_t chunk_size = (left_size + num_workers - 1) / num_workers;
        
        // We'll use thread-local result vectors to avoid contention
        struct ThreadLocalResults {
            std::unordered_map<std::string, Column> data;
            std::vector<size_t> matched_indices;  // For right/outer joins
        };
        
        std::vector<ThreadLocalResults> thread_results(num_workers);
        
        // Pre-allocate thread-local storage
        for (auto& tr : thread_results) {
            for (const auto& col_name : left_columns) {
                tr.data[col_name] = Column();
                tr.data[col_name].reserve(chunk_size * 2);  // Some buffer for matches
            }
            
            for (const auto& col_name : right_columns) {
                tr.data[col_name] = Column();
                tr.data[col_name].reserve(chunk_size * 2);
            }
            
            if (how == "right" || how == "outer") {
                tr.matched_indices.reserve(chunk_size);
            }
        }
        
        // Process chunks in parallel
        std::vector<std::thread> threads;
        for (int t = 0; t < num_workers; ++t) {
            threads.emplace_back([&, t]() {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, left_size);
                auto& local_result = thread_results[t];
                
                for (size_t i = start; i < end; ++i) {
                    auto range = right_indices.equal_range(left_key[i]);
                    bool match_found = range.first != range.second;
                    
                    // For LEFT and INNER joins, we need matches on the right
                    if ((match_found && (how == "inner" || how == "left" || how == "outer")) || 
                        (!match_found && (how == "left" || how == "outer"))) {
                        if (match_found) {
                            // For each matching row in the right dataframe
                            for (auto it = range.first; it != range.second; ++it) {
                                size_t right_idx = it->second;
                                
                                // Mark this right row as matched
                                if (how == "right" || how == "outer") {
                                    local_result.matched_indices.push_back(right_idx);
                                }
                                
                                // Add the left dataframe's values for this row
                                for (const auto& col_name : left_columns) {
                                    local_result.data[col_name].push_back(data_.at(col_name)[i]);
                                }
                                
                                // Add the right dataframe's values (excluding the join column)
                                for (const auto& col_name : right_columns) {
                                    local_result.data[col_name].push_back(other.data_.at(col_name)[right_idx]);
                                }
                            }
                        } else {
                            // No match found, but we include left row for LEFT/OUTER joins
                            // Add the left dataframe's values for this row
                            for (const auto& col_name : left_columns) {
                                local_result.data[col_name].push_back(data_.at(col_name)[i]);
                            }
                            
                            // Add null/default values for the right dataframe's columns
                            for (const auto& col_name : right_columns) {
                                const auto& col = other.data_.at(col_name);
                                // Add a default null value of the appropriate type
                                if (col.empty()) {
                                    local_result.data[col_name].emplace_back(0.0);
                                } else {
                                    DataValue null_value = col[0];
                                    if (std::holds_alternative<double>(null_value)) {
                                        local_result.data[col_name].emplace_back(0.0);
                                    } else if (std::holds_alternative<int64_t>(null_value)) {
                                        local_result.data[col_name].emplace_back(int64_t(0));
                                    } else if (std::holds_alternative<std::string>(null_value)) {
                                        local_result.data[col_name].emplace_back(std::string());
                                    } else if (std::holds_alternative<bool>(null_value)) {
                                        local_result.data[col_name].emplace_back(false);
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Merge thread-local results
        // First, calculate total result size to reserve memory
        size_t total_rows = 0;
        for (const auto& tr : thread_results) {
            if (!tr.data.empty() && !tr.data.begin()->second.empty()) {
                total_rows += tr.data.begin()->second.size();
            }
        }
        
        // Reserve space in result columns
        for (auto& [name, col] : result_data) {
            col.reserve(total_rows);
        }
        
        // Merge thread results into the final result
        for (const auto& tr : thread_results) {
            for (const auto& [name, col] : tr.data) {
                result_data[name].insert(result_data[name].end(), col.begin(), col.end());
            }
            
            // Collect matched right indices for RIGHT/OUTER joins
            if (how == "right" || how == "outer") {
                for (size_t idx : tr.matched_indices) {
                    right_matched[idx] = true;
                }
            }
        }
    }
    else {
        // Sequential join processing
        for (size_t i = 0; i < left_size; ++i) {
            auto range = right_indices.equal_range(left_key[i]);
            bool match_found = range.first != range.second;
            
            // For LEFT and INNER joins, we need matches on the right
            if ((match_found && (how == "inner" || how == "left" || how == "outer")) || 
                (!match_found && (how == "left" || how == "outer"))) {
                if (match_found) {
                    // For each matching row in the right dataframe
                    for (auto it = range.first; it != range.second; ++it) {
                        size_t right_idx = it->second;
                        
                        // Mark this right row as matched for RIGHT/OUTER joins
                        if (how == "right" || how == "outer") {
                            right_matched[right_idx] = true;
                        }
                        
                        // Add the left dataframe's values for this row
                        for (const auto& [name, col] : data_) {
                            result_data[name].push_back(col[i]);
                        }
                        
                        // Add the right dataframe's values (excluding the join column)
                        for (const auto& [name, col] : other.data_) {
                            if (name != on && data_.find(name) == data_.end()) {
                                result_data[name].push_back(col[right_idx]);
                            }
                        }
                    }
                } else {
                    // No match found, but we still need to include left row for LEFT/OUTER joins
                    // Add the left dataframe's values for this row
                    for (const auto& [name, col] : data_) {
                        result_data[name].push_back(col[i]);
                    }
                    
                    // Add null/default values for the right dataframe's columns
                    for (const auto& [name, col] : other.data_) {
                        if (name != on && data_.find(name) == data_.end()) {
                            // Add a default null value of the appropriate type
                            if (col.empty()) {
                                result_data[name].emplace_back(0.0);
                            } else {
                                DataValue null_value = col[0];
                                if (std::holds_alternative<double>(null_value)) {
                                    result_data[name].emplace_back(0.0);
                                } else if (std::holds_alternative<int64_t>(null_value)) {
                                    result_data[name].emplace_back(int64_t(0));
                                } else if (std::holds_alternative<std::string>(null_value)) {
                                    result_data[name].emplace_back(std::string());
                                } else if (std::holds_alternative<bool>(null_value)) {
                                    result_data[name].emplace_back(false);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // For RIGHT and OUTER joins, include unmatched right rows
    if (how == "right" || how == "outer") {
        for (size_t i = 0; i < right_size; ++i) {
            if (!right_matched[i]) {
                // Add null/default values for the left dataframe's columns
                for (const auto& [name, col] : data_) {
                    if (name == on) {
                        // For the join column, use the value from the right table
                        result_data[name].push_back(right_key[i]);
                    } else {
                        // For other columns, use default values
                        if (col.empty()) {
                            result_data[name].emplace_back(0.0);
                        } else {
                            DataValue null_value = col[0];
                            if (std::holds_alternative<double>(null_value)) {
                                result_data[name].emplace_back(0.0);
                            } else if (std::holds_alternative<int64_t>(null_value)) {
                                result_data[name].emplace_back(int64_t(0));
                            } else if (std::holds_alternative<std::string>(null_value)) {
                                result_data[name].emplace_back(std::string());
                            } else if (std::holds_alternative<bool>(null_value)) {
                                result_data[name].emplace_back(false);
                            }
                        }
                    }
                }
                
                // Add the right dataframe's values
                for (const auto& [name, col] : other.data_) {
                    if (name != on && data_.find(name) == data_.end()) {
                        result_data[name].push_back(col[i]);
                    }
                }
            }
        }
    }
    
    return DataFrame(std::move(result_data));
}

GroupByResult DataFrame::groupby(const std::string& column) const {
    return GroupByResult(*this, std::vector<std::string>{column});
}

GroupByResult DataFrame::groupby(const std::vector<std::string>& columns) const {
    return GroupByResult(*this, columns);
}

DataFrame DataFrame::resample(const std::string& rule, const std::string& time_column) const {
    // For simplicity, this is a placeholder implementation
    // A real implementation would need to handle datetime conversion and resampling rules
    throw std::runtime_error("Resample not implemented yet");
}

void DataFrame::set_execution_policy(ExecutionPolicy policy) {
    execution_policy_ = policy;
}

DataFrame::ExecutionPolicy DataFrame::get_execution_policy() const {
    return execution_policy_;
}

void DataFrame::set_num_threads(int threads) {
    if (threads < 1) {
        throw std::invalid_argument("Thread count must be at least 1");
    }
    num_threads_ = threads;
}

void DataFrame::set_cluster_nodes(const std::vector<std::string>& node_addresses) {
    cluster_nodes_ = node_addresses;
}

void DataFrame::distribute_workload(std::function<void(size_t, size_t)> task) const {
    // This is a simplified implementation of distributing work
    // A real implementation would handle network communication
    
    const size_t row_count = rows();
    if (execution_policy_ == ExecutionPolicy::Sequential || row_count == 0) {
        task(0, row_count);
        return;
    }
    
    if (execution_policy_ == ExecutionPolicy::Parallel) {
        // Use multiple threads on the local machine
        std::vector<std::thread> threads;
        const size_t chunk_size = (row_count + num_threads_ - 1) / num_threads_;
        
        for (int i = 0; i < num_threads_; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, row_count);
            if (start < end) {
                threads.emplace_back([task, start, end]() {
                    task(start, end);
                });
            }
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    else if (execution_policy_ == ExecutionPolicy::Distributed && !cluster_nodes_.empty()) {
        // Distribute across cluster nodes
        // This is just a placeholder - real implementation would need network communication
        std::cout << "Simulating distribution across " << cluster_nodes_.size() << " nodes" << std::endl;
        
        const size_t chunk_size = (row_count + cluster_nodes_.size() - 1) / cluster_nodes_.size();
        for (size_t i = 0; i < cluster_nodes_.size(); ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, row_count);
            if (start < end) {
                std::cout << "Node " << cluster_nodes_[i] << " processing rows " << start << " to " << end << std::endl;
                // In a real implementation, this would send the task to the node
                task(start, end);
            }
        }
    }
    else if (execution_policy_ == ExecutionPolicy::GPU) {
        // GPU parallelization placeholder
        std::cout << "Simulating GPU execution" << std::endl;
        // In a real implementation, this would offload the computation to a GPU
        task(0, row_count);
    }
}

// Series implementation
Series::Series(const Column& data, const std::string& name)
    : data_(data), name_(name) {}

Series::Series(Column&& data, const std::string& name)
    : data_(std::move(data)), name_(name) {}

size_t Series::size() const {
    return data_.size();
}

const std::string& Series::name() const {
    return name_;
}

const DataValue& Series::operator[](size_t index) const {
    return data_.at(index);
}

DataValue& Series::operator[](size_t index) {
    return data_.at(index);
}

double Series::mean() const {
    double sum = 0.0;
    size_t count = 0;
    
    for (const auto& val : data_) {
        if (std::holds_alternative<double>(val)) {
            sum += std::get<double>(val);
            count++;
        }
        else if (std::holds_alternative<int64_t>(val)) {
            sum += static_cast<double>(std::get<int64_t>(val));
            count++;
        }
    }
    
    return count > 0 ? sum / count : 0.0;
}

double Series::median() const {
    std::vector<double> numeric_values;
    numeric_values.reserve(data_.size());
    
    for (const auto& val : data_) {
        if (std::holds_alternative<double>(val)) {
            numeric_values.push_back(std::get<double>(val));
        }
        else if (std::holds_alternative<int64_t>(val)) {
            numeric_values.push_back(static_cast<double>(std::get<int64_t>(val)));
        }
    }
    
    if (numeric_values.empty()) {
        return 0.0;
    }
    
    std::sort(numeric_values.begin(), numeric_values.end());
    size_t size = numeric_values.size();
    if (size % 2 == 0) {
        return (numeric_values[size/2 - 1] + numeric_values[size/2]) / 2.0;
    } else {
        return numeric_values[size/2];
    }
}

double Series::std() const {
    std::vector<double> numeric_values;
    numeric_values.reserve(data_.size());
    
    for (const auto& val : data_) {
        if (std::holds_alternative<double>(val)) {
            numeric_values.push_back(std::get<double>(val));
        }
        else if (std::holds_alternative<int64_t>(val)) {
            numeric_values.push_back(static_cast<double>(std::get<int64_t>(val)));
        }
    }
    
    if (numeric_values.empty()) {
        return 0.0;
    }
    
    double sum = std::accumulate(numeric_values.begin(), numeric_values.end(), 0.0);
    double mean = sum / numeric_values.size();
    
    double sq_sum = std::inner_product(
        numeric_values.begin(), numeric_values.end(), numeric_values.begin(), 0.0,
        std::plus<>(), [mean](double x, double y) { return (x - mean) * (y - mean); }
    );
    
    return std::sqrt(sq_sum / numeric_values.size());
}

DataValue Series::max() const {
    if (data_.empty()) {
        return 0.0;
    }
    
    std::optional<double> max_numeric;
    std::optional<std::string> max_string;
    
    for (const auto& val : data_) {
        if (std::holds_alternative<double>(val)) {
            double num = std::get<double>(val);
            if (!max_numeric || num > *max_numeric) {
                max_numeric = num;
            }
        }
        else if (std::holds_alternative<int64_t>(val)) {
            double num = static_cast<double>(std::get<int64_t>(val));
            if (!max_numeric || num > *max_numeric) {
                max_numeric = num;
            }
        }
        else if (std::holds_alternative<std::string>(val)) {
            const std::string& str = std::get<std::string>(val);
            if (!max_string || str > *max_string) {
                max_string = str;
            }
        }
    }
    
    if (max_numeric) {
        return *max_numeric;
    } else if (max_string) {
        return *max_string;
    } else {
        return 0.0;  // Default for empty series
    }
}

DataValue Series::min() const {
    if (data_.empty()) {
        return 0.0;
    }
    
    std::optional<double> min_numeric;
    std::optional<std::string> min_string;
    
    for (const auto& val : data_) {
        if (std::holds_alternative<double>(val)) {
            double num = std::get<double>(val);
            if (!min_numeric || num < *min_numeric) {
                min_numeric = num;
            }
        }
        else if (std::holds_alternative<int64_t>(val)) {
            double num = static_cast<double>(std::get<int64_t>(val));
            if (!min_numeric || num < *min_numeric) {
                min_numeric = num;
            }
        }
        else if (std::holds_alternative<std::string>(val)) {
            const std::string& str = std::get<std::string>(val);
            if (!min_string || str < *min_string) {
                min_string = str;
            }
        }
    }
    
    if (min_numeric) {
        return *min_numeric;
    } else if (min_string) {
        return *min_string;
    } else {
        return 0.0;  // Default for empty series
    }
}

DataValue Series::sum() const {
    double sum = 0.0;
    
    for (const auto& val : data_) {
        if (std::holds_alternative<double>(val)) {
            sum += std::get<double>(val);
        }
        else if (std::holds_alternative<int64_t>(val)) {
            sum += static_cast<double>(std::get<int64_t>(val));
        }
    }
    
    return sum;
}

std::vector<double> Series::to_double() const {
    std::vector<double> result;
    result.reserve(data_.size());
    
    for (const auto& val : data_) {
        if (std::holds_alternative<double>(val)) {
            result.push_back(std::get<double>(val));
        }
        else if (std::holds_alternative<int64_t>(val)) {
            result.push_back(static_cast<double>(std::get<int64_t>(val)));
        }
        else {
            // For non-numeric values, use 0.0
            result.push_back(0.0);
        }
    }
    
    return result;
}

// GroupByResult implementation
GroupByResult::GroupByResult(const DataFrame& source, const std::vector<std::string>& keys)
    : source_(source), keys_(keys) {
    // Validate that all key columns exist
    for (const auto& key : keys) {
        bool found = false;
        for (const auto& col_name : source.columns()) {
            if (key == col_name) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::invalid_argument("GroupBy key column '" + key + "' not found");
        }
    }
}

DataFrame GroupByResult::mean() const {
    return perform_aggregation([](const Column& col) -> DataValue {
        double sum = 0.0;
        size_t count = 0;
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                sum += std::get<double>(val);
                count++;
            }
            else if (std::holds_alternative<int64_t>(val)) {
                sum += static_cast<double>(std::get<int64_t>(val));
                count++;
            }
        }
        
        return count > 0 ? sum / count : 0.0;
    });
}

DataFrame GroupByResult::median() const {
    return perform_aggregation([](const Column& col) -> DataValue {
        std::vector<double> numeric_values;
        numeric_values.reserve(col.size());
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                numeric_values.push_back(std::get<double>(val));
            }
            else if (std::holds_alternative<int64_t>(val)) {
                numeric_values.push_back(static_cast<double>(std::get<int64_t>(val)));
            }
        }
        
        if (numeric_values.empty()) {
            return 0.0;
        }
        
        std::sort(numeric_values.begin(), numeric_values.end());
        size_t size = numeric_values.size();
        if (size % 2 == 0) {
            return (numeric_values[size/2 - 1] + numeric_values[size/2]) / 2.0;
        } else {
            return numeric_values[size/2];
        }
    });
}

DataFrame GroupByResult::std() const {
    return perform_aggregation([](const Column& col) -> DataValue {
        std::vector<double> numeric_values;
        numeric_values.reserve(col.size());
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                numeric_values.push_back(std::get<double>(val));
            }
            else if (std::holds_alternative<int64_t>(val)) {
                numeric_values.push_back(static_cast<double>(std::get<int64_t>(val)));
            }
        }
        
        if (numeric_values.empty()) {
            return 0.0;
        }
        
        double sum = std::accumulate(numeric_values.begin(), numeric_values.end(), 0.0);
        double mean = sum / numeric_values.size();
        
        double sq_sum = std::inner_product(
            numeric_values.begin(), numeric_values.end(), numeric_values.begin(), 0.0,
            std::plus<>(), [mean](double x, double y) { return (x - mean) * (y - mean); }
        );
        
        return std::sqrt(sq_sum / numeric_values.size());
    });
}

DataFrame GroupByResult::sum() const {
    return perform_aggregation([](const Column& col) -> DataValue {
        double sum = 0.0;
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                sum += std::get<double>(val);
            }
            else if (std::holds_alternative<int64_t>(val)) {
                sum += static_cast<double>(std::get<int64_t>(val));
            }
        }
        
        return sum;
    });
}

DataFrame GroupByResult::max() const {
    return perform_aggregation([](const Column& col) -> DataValue {
        if (col.empty()) {
            return 0.0;
        }
        
        std::optional<double> max_numeric;
        std::optional<std::string> max_string;
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                double num = std::get<double>(val);
                if (!max_numeric || num > *max_numeric) {
                    max_numeric = num;
                }
            }
            else if (std::holds_alternative<int64_t>(val)) {
                double num = static_cast<double>(std::get<int64_t>(val));
                if (!max_numeric || num > *max_numeric) {
                    max_numeric = num;
                }
            }
            else if (std::holds_alternative<std::string>(val)) {
                const std::string& str = std::get<std::string>(val);
                if (!max_string || str > *max_string) {
                    max_string = str;
                }
            }
        }
        
        if (max_numeric) {
            return *max_numeric;
        } else if (max_string) {
            return *max_string;
        } else {
            return 0.0;  // Default for empty column
        }
    });
}

DataFrame GroupByResult::min() const {
    return perform_aggregation([](const Column& col) -> DataValue {
        if (col.empty()) {
            return 0.0;
        }
        
        std::optional<double> min_numeric;
        std::optional<std::string> min_string;
        
        for (const auto& val : col) {
            if (std::holds_alternative<double>(val)) {
                double num = std::get<double>(val);
                if (!min_numeric || num < *min_numeric) {
                    min_numeric = num;
                }
            }
            else if (std::holds_alternative<int64_t>(val)) {
                double num = static_cast<double>(std::get<int64_t>(val));
                if (!min_numeric || num < *min_numeric) {
                    min_numeric = num;
                }
            }
            else if (std::holds_alternative<std::string>(val)) {
                const std::string& str = std::get<std::string>(val);
                if (!min_string || str < *min_string) {
                    min_string = str;
                }
            }
        }
        
        if (min_numeric) {
            return *min_numeric;
        } else if (min_string) {
            return *min_string;
        } else {
            return 0.0;  // Default for empty column
        }
    });
}

DataFrame GroupByResult::count() const {
    return perform_aggregation([](const Column& col) -> DataValue {
        return static_cast<int64_t>(col.size());
    });
}

DataFrame GroupByResult::agg(const std::string& function) const {
    if (function == "mean") {
        return mean();
    } else if (function == "median") {
        return median();
    } else if (function == "std") {
        return std();
    } else if (function == "sum") {
        return sum();
    } else if (function == "max") {
        return max();
    } else if (function == "min") {
        return min();
    } else if (function == "count") {
        return count();
    } else {
        throw std::invalid_argument("Unknown aggregation function: " + function);
    }
}

DataFrame GroupByResult::agg(const std::unordered_map<std::string, std::string>& agg_dict) const {
    // For simplicity, we'll just implement a basic version
    // A full implementation would compute all results in a single pass
    std::unordered_map<std::string, DataFrame> results;
    
    for (const auto& [col, func] : agg_dict) {
        results[func] = agg(func);
    }
    
    // Combine results
    std::unordered_map<std::string, Column> combined_data;
    
    // First, add groupby keys
    for (const auto& key : keys_) {
        // We'd get unique values for each key, for simplicity just use 
        // first result's keys
        if (!results.empty()) {
            const auto& first_result = results.begin()->second;
            // Find the key in the columns
            for (const auto& col_name : first_result.columns()) {
                if (col_name == key) {
                    combined_data[key] = first_result.data_.at(key);
                    break;
                }
            }
        }
    }
    
    // Add aggregated columns with appropriate naming
    for (const auto& [col, func] : agg_dict) {
        const auto& result_df = results[func];
        // Skip if column doesn't exist in results
        bool found = false;
        for (const auto& col_name : result_df.columns()) {
            if (col_name == col) {
                found = true;
                // Format as "column_function"
                std::string new_col_name = col + "_" + func;
                combined_data[new_col_name] = result_df.data_.at(col);
                break;
            }
        }
    }
    
    return DataFrame(std::move(combined_data));
}

DataFrame GroupByResult::perform_aggregation(std::function<DataValue(const Column&)> agg_func) const {
    // Get the values for the groupby keys
    std::vector<Column> key_columns;
    key_columns.reserve(keys_.size());
    for (const auto& key : keys_) {
        key_columns.push_back(source_.data_.at(key));
    }
    
    // Identify unique groups with a hash-based approach for faster lookup
    using GroupKeyHash = std::vector<DataValue>;
    using GroupIndices = std::vector<size_t>;
    
    // Custom hasher for GroupKeyHash
    struct VectorDataValueHash {
        std::size_t operator()(const GroupKeyHash& key) const {
            std::size_t seed = key.size();
            for (const auto& val : key) {
                if (std::holds_alternative<double>(val)) {
                    // Use a simple hash for doubles
                    double d = std::get<double>(val);
                    seed ^= std::hash<double>{}(d) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                }
                else if (std::holds_alternative<int64_t>(val)) {
                    int64_t i = std::get<int64_t>(val);
                    seed ^= std::hash<int64_t>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                }
                else if (std::holds_alternative<std::string>(val)) {
                    const std::string& s = std::get<std::string>(val);
                    seed ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                }
                else if (std::holds_alternative<bool>(val)) {
                    bool b = std::get<bool>(val);
                    seed ^= std::hash<bool>{}(b) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                }
            }
            return seed;
        }
    };
    
    // Custom equality operator for GroupKeyHash
    struct VectorDataValueEqual {
        bool operator()(const GroupKeyHash& a, const GroupKeyHash& b) const {
            if (a.size() != b.size()) return false;
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i].index() != b[i].index()) return false;
                
                if (std::holds_alternative<double>(a[i])) {
                    if (std::get<double>(a[i]) != std::get<double>(b[i])) return false;
                }
                else if (std::holds_alternative<int64_t>(a[i])) {
                    if (std::get<int64_t>(a[i]) != std::get<int64_t>(b[i])) return false;
                }
                else if (std::holds_alternative<std::string>(a[i])) {
                    if (std::get<std::string>(a[i]) != std::get<std::string>(b[i])) return false;
                }
                else if (std::holds_alternative<bool>(a[i])) {
                    if (std::get<bool>(a[i]) != std::get<bool>(b[i])) return false;
                }
            }
            return true;
        }
    };
    
    // Use an unordered_map with custom hash and equality for better performance with large datasets
    std::unordered_map<GroupKeyHash, GroupIndices, VectorDataValueHash, VectorDataValueEqual> groups;
    const size_t row_count = source_.rows();
    
    // Reserve space to avoid rehashing
    groups.reserve(std::min(row_count / 10, size_t(1000)));
    
    // Determine if we should use parallel processing and the most appropriate algorithm
    bool use_parallel = source_.execution_policy_ == DataFrame::ExecutionPolicy::Parallel && 
                       row_count > 10000;
                       
    // For very large datasets with few unique values, use the radix hash algorithm
    // which is much faster for these cases
    bool use_radix_hash = row_count > 100000 && key_columns.size() == 1 && 
                         (source_.execution_policy_ == DataFrame::ExecutionPolicy::Parallel || 
                          source_.execution_policy_ == DataFrame::ExecutionPolicy::GPU);
    
    if (use_radix_hash) {
        // Specialized algorithm for single-key groupby with large datasets
        // This is a radix-bucket approach that's very fast for categorical data
        
        // Check if we're grouping by a string column (most common case for categories)
        const bool string_keys = std::holds_alternative<std::string>(key_columns[0][0]);
        
        if (string_keys) {
            // For string keys, use a string-optimized implementation
            std::unordered_map<std::string, GroupIndices> string_groups;
            
            // Estimate cardinality
            const size_t sample_size = std::min(row_count, size_t(10000));
            const size_t step = row_count / sample_size;
            
            std::unordered_set<std::string> unique_strings;
            for (size_t i = 0; i < row_count; i += step) {
                if (std::holds_alternative<std::string>(key_columns[0][i])) {
                    unique_strings.insert(std::get<std::string>(key_columns[0][i]));
                }
            }
            
            // Reserve space based on the estimated number of unique values
            size_t estimated_keys = (unique_strings.size() * row_count) / sample_size;
            estimated_keys = std::min(std::max(estimated_keys, size_t(100)), size_t(10000));
            string_groups.reserve(estimated_keys);
            
            // Use parallel processing with string keys
            const int num_threads = source_.num_threads_;
            const size_t chunk_size = (row_count + num_threads - 1) / num_threads;
            
            std::vector<std::unordered_map<std::string, GroupIndices>> thread_string_groups(num_threads);
            
            // Reserve thread-local maps
            for (auto& tg : thread_string_groups) {
                tg.reserve(estimated_keys / num_threads + 10);
            }
            
            // Process in parallel
            std::vector<std::thread> threads;
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    size_t start = t * chunk_size;
                    size_t end = std::min(start + chunk_size, row_count);
                    
                    auto& local_string_groups = thread_string_groups[t];
                    
                    for (size_t i = start; i < end; ++i) {
                        if (std::holds_alternative<std::string>(key_columns[0][i])) {
                            const std::string& key = std::get<std::string>(key_columns[0][i]);
                            local_string_groups[key].push_back(i);
                        }
                    }
                });
            }
            
            for (auto& thread : threads) {
                thread.join();
            }
            
            // Merge results
            for (const auto& local_groups : thread_string_groups) {
                for (const auto& [key, indices] : local_groups) {
                    auto& global_indices = string_groups[key];
                    global_indices.insert(global_indices.end(), indices.begin(), indices.end());
                }
            }
            
            // Convert string-based groups to our standard format
            for (const auto& [key, indices] : string_groups) {
                GroupKeyHash key_vec;
                key_vec.push_back(key);
                groups[key_vec] = indices;
            }
        }
        else {
            // For non-string keys, fall back to the standard approach
            // We can add specialized implementations for numeric keys if needed
            
            // Parallel groupby for large datasets
            const int num_threads = source_.num_threads_;
            const size_t chunk_size = (row_count + num_threads - 1) / num_threads;
            
            // Process as usual
            std::vector<std::unordered_map<GroupKeyHash, GroupIndices, VectorDataValueHash, VectorDataValueEqual>> 
                thread_groups(num_threads);
                
            // Reserve space using our improved estimates
            size_t estimated_groups = std::min(row_count / 10, size_t(1000));
            size_t groups_per_thread = (estimated_groups + num_threads - 1) / num_threads;
            
            for (auto& tg : thread_groups) {
                tg.reserve(groups_per_thread + 10);
            }
            
            // Process each chunk in parallel
            std::vector<std::thread> threads;
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    size_t start = t * chunk_size;
                    size_t end = std::min(start + chunk_size, row_count);
                    
                    auto& local_groups = thread_groups[t];
                    
                    for (size_t i = start; i < end; ++i) {
                        GroupKeyHash group_key;
                        group_key.reserve(key_columns.size());
                        
                        for (const auto& key_col : key_columns) {
                            group_key.push_back(key_col[i]);
                        }
                        
                        local_groups[group_key].push_back(i);
                    }
                });
            }
            
            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
            
            // Merge thread-local results into the main map
            for (const auto& local_groups : thread_groups) {
                for (const auto& [key, indices] : local_groups) {
                    auto& global_indices = groups[key];
                    global_indices.insert(global_indices.end(), indices.begin(), indices.end());
                }
            }
        }
    }
    else if (use_parallel) {
        // Standard parallel groupby for large datasets
        const int num_threads = source_.num_threads_;
        const size_t chunk_size = (row_count + num_threads - 1) / num_threads;
        
        // We need thread-local groups to avoid lock contention
        std::vector<std::unordered_map<GroupKeyHash, GroupIndices, VectorDataValueHash, VectorDataValueEqual>> 
            thread_groups(num_threads);
        
        // Estimate number of groups for better reservation
        size_t estimated_groups;
        
        // For larger datasets with many unique keys, we'll use a different strategy
        // that trades memory for performance with pre-sampling
        if (row_count > 100000 && key_columns.size() == 1) {
            // For single key column groupby with large datasets, sample the data to estimate cardinality
            const size_t sample_size = std::min(row_count, size_t(10000));
            const size_t step = row_count / sample_size;
            
            std::unordered_set<std::string> sample_values;
            
            for (size_t i = 0; i < row_count; i += step) {
                std::string value_str = value_to_string(key_columns[0][i]);
                sample_values.insert(value_str);
            }
            
            // Estimate total number of groups based on sample
            estimated_groups = (sample_values.size() * row_count) / sample_size;
            
            // Limit to reasonable values
            estimated_groups = std::min(std::max(estimated_groups, size_t(100)), size_t(10000));
        } else if (key_columns.size() <= 2) {
            // For 1 or 2 keys, estimate higher cardinality
            estimated_groups = std::min(row_count / 5, size_t(5000));
        } else {
            // For many keys, estimate lower cardinality (more likely to have sparse combinations)
            estimated_groups = std::min(row_count / 20, size_t(2000));
        }
        
        // Distribute estimated groups across threads
        size_t groups_per_thread = (estimated_groups + num_threads - 1) / num_threads;
        
        // Reserve space in each thread-local map with these improved estimates
        for (auto& tg : thread_groups) {
            tg.reserve(groups_per_thread + 10);  // Add some buffer
        }
        
        // Process each chunk in parallel
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, row_count);
                
                auto& local_groups = thread_groups[t];
                
                for (size_t i = start; i < end; ++i) {
                    GroupKeyHash group_key;
                    group_key.reserve(key_columns.size());
                    
                    for (const auto& key_col : key_columns) {
                        group_key.push_back(key_col[i]);
                    }
                    
                    local_groups[group_key].push_back(i);
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Merge thread-local results into the main map
        for (const auto& local_groups : thread_groups) {
            for (const auto& [key, indices] : local_groups) {
                auto& global_indices = groups[key];
                global_indices.insert(global_indices.end(), indices.begin(), indices.end());
            }
        }
    } 
    else {
        // Sequential grouping for smaller datasets
        for (size_t i = 0; i < row_count; ++i) {
            GroupKeyHash group_key;
            group_key.reserve(key_columns.size());
            
            for (const auto& key_col : key_columns) {
                group_key.push_back(key_col[i]);
            }
            
            groups[group_key].push_back(i);
        }
    }
    
    // Prepare result columns with pre-allocation
    std::unordered_map<std::string, Column> result_data;
    
    // Pre-allocate result columns
    const size_t group_count = groups.size();
    for (const auto& key : keys_) {
        result_data[key] = Column();
        result_data[key].reserve(group_count);
    }
    
    // Get all column names that aren't keys
    std::vector<std::string> value_columns;
    for (const auto& [col_name, _] : source_.data_) {
        if (std::find(keys_.begin(), keys_.end(), col_name) == keys_.end()) {
            value_columns.push_back(col_name);
            result_data[col_name] = Column();
            result_data[col_name].reserve(group_count);
        }
    }
    
    // Apply aggregations - potentially in parallel for large datasets
    if (use_parallel && value_columns.size() > 1) {
        // Prepare key result first (these are the same for all aggregations)
        std::vector<GroupKeyHash> unique_keys;
        unique_keys.reserve(group_count);
        
        for (const auto& [key, _] : groups) {
            unique_keys.push_back(key);
        }
        
        // Add keys to the result
        for (size_t k = 0; k < keys_.size(); ++k) {
            Column& key_result = result_data[keys_[k]];
            
            for (const auto& group_key : unique_keys) {
                key_result.push_back(group_key[k]);
            }
        }
        
        // Process value columns in parallel
        std::mutex result_mutex;
        std::vector<std::thread> threads;
        
        for (const auto& col_name : value_columns) {
            threads.emplace_back([&, col_name]() {
                const auto& col = source_.data_.at(col_name);
                Column thread_result;
                thread_result.reserve(group_count);
                
                for (const auto& group_key : unique_keys) {
                    const auto& indices = groups.at(group_key);
                    
                    // Extract values for this group
                    Column group_values;
                    group_values.reserve(indices.size());
                    
                    for (size_t idx : indices) {
                        group_values.push_back(col[idx]);
                    }
                    
                    // Apply aggregation function
                    thread_result.push_back(agg_func(group_values));
                }
                
                // Store results
                std::lock_guard<std::mutex> lock(result_mutex);
                result_data[col_name] = std::move(thread_result);
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    else {
        // Sequential processing for each group
        // Prepare key result columns
        for (size_t k = 0; k < keys_.size(); ++k) {
            Column& key_result = result_data[keys_[k]];
            
            for (const auto& [group_key, _] : groups) {
                key_result.push_back(group_key[k]);
            }
        }
        
        // Apply aggregation to each non-key column
        for (const auto& col_name : value_columns) {
            const auto& col = source_.data_.at(col_name);
            Column& agg_result = result_data[col_name];
            
            for (const auto& [_, indices] : groups) {
                // Extract values for this group with pre-allocation
                Column group_values;
                group_values.reserve(indices.size());
                
                for (size_t idx : indices) {
                    group_values.push_back(col[idx]);
                }
                
                // Apply aggregation function
                agg_result.push_back(agg_func(group_values));
            }
        }
    }
    
    return DataFrame(std::move(result_data));
}

} // namespace hpda