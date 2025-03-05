#include "hpda/dataframe.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <execution>
#include <iostream>
#include <thread>
#include <future>

namespace hpda {

 
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
        // Parallel execution using std::execution::par
        std::mutex result_mutex;
        std::for_each(std::execution::par, column_names.begin(), column_names.end(),
            [&](const std::string& col_name) {
                const auto& col = data_.at(col_name);
                DataValue val = operation(col);
                
                std::lock_guard<std::mutex> lock(result_mutex);
                result.push_back(val);
            });
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

DataFrame DataFrame::sort_values(const std::string& column, bool ascending) const {
    if (data_.find(column) == data_.end()) {
        throw std::out_of_range("Column '" + column + "' not found");
    }
    
    // Create indices and sort them
    const auto& sort_col = data_.at(column);
    std::vector<size_t> indices(sort_col.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    if (ascending) {
        std::sort(indices.begin(), indices.end(), [&sort_col](size_t i1, size_t i2) {
            return compare_values(sort_col[i1], sort_col[i2]);
        });
    } else {
        std::sort(indices.begin(), indices.end(), [&sort_col](size_t i1, size_t i2) {
            return compare_values(sort_col[i2], sort_col[i1]);
        });
    }
    
    // Create new dataframe with sorted data
    std::unordered_map<std::string, Column> sorted_data;
    for (const auto& [name, col] : data_) {
        Column sorted_col(col.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            sorted_col[i] = col[indices[i]];
        }
        sorted_data[name] = std::move(sorted_col);
    }
    
    return DataFrame(std::move(sorted_data));
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

DataFrame DataFrame::merge(const DataFrame& other, const std::string& on, 
                        const std::string& how) const {
    // Check if the 'on' column exists in both dataframes
    if (data_.find(on) == data_.end() || other.data_.find(on) == other.data_.end()) {
        throw std::invalid_argument("Join column not found in one or both DataFrames");
    }
    
    const auto& left_key = data_.at(on);
    const auto& right_key = other.data_.at(on);
    
    // Build hash map for the right dataframe's keys
    std::unordered_multimap<size_t, size_t> right_indices;
    for (size_t i = 0; i < right_key.size(); ++i) {
        // Using simple hash for demonstration
        size_t hash = std::hash<std::string>{}(value_to_string(right_key[i]));
        right_indices.emplace(hash, i);
    }
    
    // Prepare result dataframe
    std::unordered_map<std::string, Column> result_data;
    
    // Add columns from left dataframe
    for (const auto& [name, col] : data_) {
        result_data[name] = {};
    }
    
    // Add columns from right dataframe (excluding the join column to avoid duplication)
    for (const auto& [name, col] : other.data_) {
        if (name != on && data_.find(name) == data_.end()) {
            result_data[name] = {};
        }
    }
    
    // Perform the join operation
    for (size_t i = 0; i < left_key.size(); ++i) {
        size_t left_hash = std::hash<std::string>{}(value_to_string(left_key[i]));
        
        auto range = right_indices.equal_range(left_hash);
        bool match_found = range.first != range.second;
        
        // For LEFT and INNER joins, we need matches on the right
        if (match_found || how == "left" || how == "outer") {
            if (match_found) {
                // For each matching row in the right dataframe
                for (auto it = range.first; it != range.second; ++it) {
                    size_t right_idx = it->second;
                    
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
                        // Add a default null value
                        if (col.empty()) {
                            result_data[name].emplace_back(0.0);  // Default to double 0
                        } else {
                            // Use same type as the column
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
    
    // For RIGHT and OUTER joins, include unmatched right rows
    if (how == "right" || how == "outer") {
        // TODO: Implement right join logic
        // This would need to track which right rows were already matched
        // and include the unmatched ones with null values for left columns
    }
    
    return DataFrame(std::move(result_data));
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
        if (source.columns().find(key) == source.columns().end()) {
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
            combined_data[key] = first_result.columns().at(key);
        }
    }
    
    // Add aggregated columns with appropriate naming
    for (const auto& [col, func] : agg_dict) {
        const auto& result_df = results[func];
        // Skip if column doesn't exist in results
        if (result_df.columns().find(col) == result_df.columns().end()) {
            continue;
        }
        
        // Format as "column_function"
        std::string new_col_name = col + "_" + func;
        combined_data[new_col_name] = result_df.columns().at(col);
    }
    
    return DataFrame(std::move(combined_data));
}

DataFrame GroupByResult::perform_aggregation(std::function<DataValue(const Column&)> agg_func) const {
    // This is a simplified implementation of groupby
    // A real implementation would use a hash-based approach for better performance
    
    // Get the values for the groupby keys
    std::vector<Column> key_columns;
    for (const auto& key : keys_) {
        key_columns.push_back(source_.data_.at(key));
    }
    
    // Identify unique groups
    std::map<std::vector<DataValue>, std::vector<size_t>> groups;
    for (size_t i = 0; i < source_.rows(); ++i) {
        std::vector<DataValue> group_key;
        for (const auto& key_col : key_columns) {
            group_key.push_back(key_col[i]);
        }
        groups[group_key].push_back(i);
    }
    
    // Prepare result columns
    std::unordered_map<std::string, Column> result_data;
    
    // Add keys to the result
    for (size_t k = 0; k < keys_.size(); ++k) {
        Column key_result;
        
        for (const auto& [group_key, _] : groups) {
            key_result.push_back(group_key[k]);
        }
        
        result_data[keys_[k]] = std::move(key_result);
    }
    
    // Apply aggregation to each non-key column
    for (const auto& [col_name, col] : source_.data_) {
        // Skip key columns
        if (std::find(keys_.begin(), keys_.end(), col_name) != keys_.end()) {
            continue;
        }
        
        Column agg_result;
        
        for (const auto& [_, indices] : groups) {
            // Extract values for this group
            Column group_values;
            for (size_t idx : indices) {
                group_values.push_back(col[idx]);
            }
            
            // Apply aggregation function
            agg_result.push_back(agg_func(group_values));
        }
        
        result_data[col_name] = std::move(agg_result);
    }
    
    return DataFrame(std::move(result_data));
}

} // namespace hpda