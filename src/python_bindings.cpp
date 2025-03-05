#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "hpda/dataframe.h"

namespace py = pybind11;

// Helper functions for Python conversions
namespace {

// Convert Python object to DataValue
hpda::DataValue py_to_data_value(const py::object& obj) {
    if (py::isinstance<py::int_>(obj)) {
        return static_cast<int64_t>(py::cast<int64_t>(obj));
    }
    else if (py::isinstance<py::float_>(obj)) {
        return py::cast<double>(obj);
    }
    else if (py::isinstance<py::str>(obj)) {
        return py::cast<std::string>(obj);
    }
    else if (py::isinstance<py::bool_>(obj)) {
        return py::cast<bool>(obj);
    }
    throw py::type_error("Unsupported Python type for DataFrame");
}

// Convert DataValue to Python object
py::object data_value_to_py(const hpda::DataValue& val) {
    if (std::holds_alternative<int64_t>(val)) {
        return py::cast(std::get<int64_t>(val));
    }
    else if (std::holds_alternative<double>(val)) {
        return py::cast(std::get<double>(val));
    }
    else if (std::holds_alternative<std::string>(val)) {
        return py::cast(std::get<std::string>(val));
    }
    else if (std::holds_alternative<bool>(val)) {
        return py::cast(std::get<bool>(val));
    }
    return py::none();
}

// Convert Python list/numpy array to Column
hpda::Column py_list_to_column(const py::list& list) {
    hpda::Column col;
    col.reserve(list.size());
    
    for (size_t i = 0; i < list.size(); ++i) {
        col.push_back(py_to_data_value(list[i]));
    }
    
    return col;
}

// Convert a Python dict of lists to a DataFrame
hpda::DataFrame py_dict_to_dataframe(const py::dict& dict) {
    std::unordered_map<std::string, hpda::Column> data;
    
    for (auto item : dict) {
        std::string key = py::cast<std::string>(item.first);
        if (py::isinstance<py::list>(item.second)) {
            data[key] = py_list_to_column(py::cast<py::list>(item.second));
        }
        else if (py::isinstance<py::array>(item.second)) {
            // Handle NumPy arrays (simplified)
            py::array arr = py::cast<py::array>(item.second);
            py::list converted_list;
            for (size_t i = 0; i < arr.size(); ++i) {
                converted_list.append(arr[i]);
            }
            data[key] = py_list_to_column(converted_list);
        }
        else {
            throw py::type_error("DataFrame columns must be lists or arrays");
        }
    }
    
    return hpda::DataFrame(std::move(data));
}

// Convert a DataFrame to a Python dict
py::dict dataframe_to_py_dict(const hpda::DataFrame& df) {
    py::dict result;
    
    for (const auto& col_name : df.columns()) {
        hpda::Series series = df[col_name];
        py::list col_data;
        
        for (size_t i = 0; i < series.size(); ++i) {
            col_data.append(data_value_to_py(series[i]));
        }
        
        result[py::cast(col_name)] = col_data;
    }
    
    return result;
}

}  // anonymous namespace

PYBIND11_MODULE(hpda, m) {
    m.doc() = "High Performance Data Analysis library with pandas-like API";
    
    // Enum for execution policy
    py::enum_<hpda::DataFrame::ExecutionPolicy>(m, "ExecutionPolicy")
        .value("Sequential", hpda::DataFrame::ExecutionPolicy::Sequential)
        .value("Parallel", hpda::DataFrame::ExecutionPolicy::Parallel)
        .value("GPU", hpda::DataFrame::ExecutionPolicy::GPU)
        .value("Distributed", hpda::DataFrame::ExecutionPolicy::Distributed)
        .export_values();
    
    // Series class
    py::class_<hpda::Series>(m, "Series")
        .def(py::init<>())
        .def("__len__", &hpda::Series::size)
        .def("__getitem__", [](const hpda::Series& s, size_t i) {
            if (i >= s.size()) {
                throw py::index_error("Series index out of range");
            }
            return data_value_to_py(s[i]);
        })
        .def("mean", &hpda::Series::mean)
        .def("median", &hpda::Series::median)
        .def("std", &hpda::Series::std)
        .def("min", [](const hpda::Series& s) {
            return data_value_to_py(s.min());
        })
        .def("max", [](const hpda::Series& s) {
            return data_value_to_py(s.max());
        })
        .def("sum", [](const hpda::Series& s) {
            return data_value_to_py(s.sum());
        })
        .def("name", &hpda::Series::name);
    
    // GroupByResult class
    py::class_<hpda::GroupByResult>(m, "GroupByResult")
        .def("mean", &hpda::GroupByResult::mean)
        .def("median", &hpda::GroupByResult::median)
        .def("std", &hpda::GroupByResult::std)
        .def("min", &hpda::GroupByResult::min)
        .def("max", &hpda::GroupByResult::max)
        .def("sum", &hpda::GroupByResult::sum)
        .def("count", &hpda::GroupByResult::count)
        .def("agg", py::overload_cast<const std::string&>(&hpda::GroupByResult::agg, py::const_))
        .def("agg", py::overload_cast<const std::unordered_map<std::string, std::string>&>(&hpda::GroupByResult::agg, py::const_));
    
    // DataFrame class
    py::class_<hpda::DataFrame>(m, "DataFrame")
        .def(py::init<>())
        .def(py::init([](const py::dict& data) {
            return py_dict_to_dataframe(data);
        }))
        .def("__len__", &hpda::DataFrame::rows)
        .def("__getitem__", [](const hpda::DataFrame& df, const std::string& col) {
            return df[col];
        })
        .def("columns", &hpda::DataFrame::columns)
        .def("shape", [](const hpda::DataFrame& df) {
            return py::make_tuple(df.rows(), df.cols());
        })
        .def("head", &hpda::DataFrame::head, py::arg("n") = 5)
        .def("tail", &hpda::DataFrame::tail, py::arg("n") = 5)
        .def("add_column", py::overload_cast<const std::string&, const hpda::Column&>(&hpda::DataFrame::add_column))
        .def("drop_column", &hpda::DataFrame::drop_column)
        .def("mean", &hpda::DataFrame::mean)
        .def("median", &hpda::DataFrame::median)
        .def("std", &hpda::DataFrame::std)
        .def("min", &hpda::DataFrame::min)
        .def("max", &hpda::DataFrame::max)
        .def("sum", &hpda::DataFrame::sum)
        .def("sort_values", &hpda::DataFrame::sort_values, py::arg("column"), py::arg("ascending") = true)
        .def("merge", &hpda::DataFrame::merge, py::arg("other"), py::arg("on"), py::arg("how") = "inner")
        .def("groupby", py::overload_cast<const std::string&>(&hpda::DataFrame::groupby, py::const_))
        .def("groupby", py::overload_cast<const std::vector<std::string>&>(&hpda::DataFrame::groupby, py::const_))
        .def("resample", &hpda::DataFrame::resample, py::arg("rule"), py::arg("time_column") = "")
        .def("set_execution_policy", &hpda::DataFrame::set_execution_policy)
        .def("get_execution_policy", &hpda::DataFrame::get_execution_policy)
        .def("set_num_threads", &hpda::DataFrame::set_num_threads)
        .def("set_cluster_nodes", &hpda::DataFrame::set_cluster_nodes)
        .def("to_dict", [](const hpda::DataFrame& df) {
            return dataframe_to_py_dict(df);
        });
    
    // Utility functions
    m.def("read_csv", [](const std::string& file_path, bool header = true, char delimiter = ',') {
        // Simplified implementation for demo purposes
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + file_path);
        }
        
        std::string line;
        std::vector<std::string> column_names;
        std::unordered_map<std::string, hpda::Column> data;
        
        // Read header if needed
        if (header && std::getline(file, line)) {
            std::istringstream ss(line);
            std::string token;
            while (std::getline(ss, token, delimiter)) {
                column_names.push_back(token);
                data[token] = {};
            }
        }
        
        // Read data
        size_t row_idx = 0;
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string token;
            size_t col_idx = 0;
            
            while (std::getline(ss, token, delimiter)) {
                // If no header, generate column names
                if (column_names.empty() || col_idx >= column_names.size()) {
                    std::string col_name = "col" + std::to_string(col_idx);
                    column_names.push_back(col_name);
                    data[col_name] = {};
                }
                
                // Try to infer type and add to the appropriate column
                try {
                    // Try to parse as integer
                    int64_t int_val = std::stoll(token);
                    data[column_names[col_idx]].push_back(int_val);
                } catch (...) {
                    try {
                        // Try to parse as double
                        double double_val = std::stod(token);
                        data[column_names[col_idx]].push_back(double_val);
                    } catch (...) {
                        // Default to string
                        data[column_names[col_idx]].push_back(token);
                    }
                }
                
                col_idx++;
            }
            
            row_idx++;
        }
        
        return hpda::DataFrame(std::move(data));
    }, py::arg("file_path"), py::arg("header") = true, py::arg("delimiter") = ',');
}