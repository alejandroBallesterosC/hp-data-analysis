"""
High Performance Data Analysis (HPDA) module.

This module provides a high-performance pandas-like API for data analysis 
with a C++ implementation for critical operations.
"""

# Import all public symbols from the compiled C++ extension
try:
    from ._hpda import (
        DataFrame,
        Series,
        GroupByResult,
        ExecutionPolicy,
        read_csv,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import C++ extension module. "
        "Make sure the library is properly installed with 'python setup.py install'. "
        f"Original error: {e}"
    )

__version__ = "0.1.0"