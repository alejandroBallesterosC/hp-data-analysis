#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <functional>
#include <variant>
#include <mutex>
#include <optional>
#include <execution>
#include <future>

namespace hpda {

// Forward declarations
class Series;
class GroupByResult;

// Supported data types in our DataFrame
using DataValue = std::variant<int64_t, double, std::string, bool>;
using Column = std::vector<DataValue>;

class DataFrame {
public:
    // Constructors
    DataFrame() = default;
    DataFrame(const std::unordered_map<std::string, Column>& data);
    DataFrame(std::unordered_map<std::string, Column>&& data);
    
    // Basic properties
    size_t rows() const;
    size_t cols() const;
    std::vector<std::string> columns() const;
    
    // Column access
    Series operator[](const std::string& column_name) const;
    
    // Data access and modification
    void add_column(const std::string& name, const Column& data);
    void add_column(const std::string& name, Column&& data);
    void drop_column(const std::string& name);
    
    // Common operations
    DataFrame head(size_t n = 5) const;
    DataFrame tail(size_t n = 5) const;
    
    // Statistical functions
    Series mean() const;
    Series median() const;
    Series std() const;
    Series max() const;
    Series min() const;
    Series sum() const;
    
    // Data manipulation
    DataFrame sort_values(const std::string& column, bool ascending = true) const;
    
    // Joins and merges
    DataFrame merge(const DataFrame& other, const std::string& on, 
                   const std::string& how = "inner") const;
    
    // Groupby operations
    GroupByResult groupby(const std::string& column) const;
    GroupByResult groupby(const std::vector<std::string>& columns) const;
    
    // Resample for time series
    DataFrame resample(const std::string& rule, 
                      const std::string& time_column = "") const;
    
    // Parallelization options
    enum class ExecutionPolicy {
        Sequential,
        Parallel,
        GPU,
        Distributed
    };
    
    void set_execution_policy(ExecutionPolicy policy);
    ExecutionPolicy get_execution_policy() const;
    
    // Distributed execution options
    void set_num_threads(int threads);
    void set_cluster_nodes(const std::vector<std::string>& node_addresses);
    
private:
    std::unordered_map<std::string, Column> data_;
    ExecutionPolicy execution_policy_ = ExecutionPolicy::Sequential;
    int num_threads_ = 1;
    std::vector<std::string> cluster_nodes_;
    
    // Helper functions
    template<typename Op>
    Series apply_reduction(Op operation) const;
    
    // Helper for distributed execution
    void distribute_workload(std::function<void(size_t, size_t)> task) const;
};

// Series class represents a single column with operations
class Series {
public:
    Series() = default;
    explicit Series(const Column& data, const std::string& name = "");
    explicit Series(Column&& data, const std::string& name = "");
    
    // Basic properties
    size_t size() const;
    const std::string& name() const;
    
    // Element access
    const DataValue& operator[](size_t index) const;
    DataValue& operator[](size_t index);
    
    // Statistical functions
    double mean() const;
    double median() const;
    double std() const;
    DataValue max() const;
    DataValue min() const;
    DataValue sum() const;
    
    // Type conversion
    std::vector<double> to_double() const;
    
private:
    Column data_;
    std::string name_;
};

// GroupByResult class for aggregation operations
class GroupByResult {
public:
    GroupByResult(const DataFrame& source, const std::vector<std::string>& keys);
    
    // Aggregation functions
    DataFrame mean() const;
    DataFrame median() const;
    DataFrame std() const;
    DataFrame sum() const;
    DataFrame max() const;
    DataFrame min() const;
    DataFrame count() const;
    
    // Generic aggregation
    DataFrame agg(const std::string& function) const;
    DataFrame agg(const std::unordered_map<std::string, std::string>& agg_dict) const;
    
private:
    const DataFrame& source_;
    std::vector<std::string> keys_;
    
    DataFrame perform_aggregation(std::function<DataValue(const Column&)> agg_func) const;
};

} // namespace hpda