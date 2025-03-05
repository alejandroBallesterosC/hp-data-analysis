"""
Unit tests for the HPDA DataFrame.
"""
import unittest
import numpy as np

# Try to import the HPDA module
try:
    import hpda
except ImportError:
    # If not installed, try the local C++ extension
    try:
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import hpda
    except ImportError:
        print("WARNING: HPDA module not found. Tests will be skipped.")
        hpda = None


@unittest.skipIf(hpda is None, "HPDA module not available")
class TestDataFrame(unittest.TestCase):
    """Test cases for the DataFrame class."""

    def setUp(self):
        """Create a sample DataFrame for testing."""
        self.data = {
            "A": [1, 2, 3, 4, 5],
            "B": [10.1, 20.2, 30.3, 40.4, 50.5],
            "C": ["a", "b", "c", "d", "e"]
        }
        self.df = hpda.DataFrame(self.data)

    def test_create_dataframe(self):
        """Test creating a DataFrame."""
        self.assertEqual(len(self.df), 5)
        self.assertEqual(len(self.df.columns()), 3)
        self.assertIn("A", self.df.columns())
        self.assertIn("B", self.df.columns())
        self.assertIn("C", self.df.columns())

    def test_mean(self):
        """Test computing the mean of a column."""
        # Mean of column A should be (1+2+3+4+5)/5 = 3.0
        self.assertAlmostEqual(self.df["A"].mean(), 3.0)
        # Mean of column B should be (10.1+20.2+30.3+40.4+50.5)/5 = 30.3
        self.assertAlmostEqual(self.df["B"].mean(), 30.3)

    def test_median(self):
        """Test computing the median of a column."""
        # Median of column A should be 3.0
        self.assertAlmostEqual(self.df["A"].median(), 3.0)
        # Median of column B should be 30.3
        self.assertAlmostEqual(self.df["B"].median(), 30.3)

    def test_std(self):
        """Test computing the standard deviation of a column."""
        # Standard deviation of A
        a_values = [1, 2, 3, 4, 5]
        a_mean = sum(a_values) / len(a_values)
        a_std_expected = np.sqrt(sum((x - a_mean) ** 2 for x in a_values) / len(a_values))
        self.assertAlmostEqual(self.df["A"].std(), a_std_expected)

    def test_sort_values(self):
        """Test sorting the DataFrame by a column."""
        # Sort by column A in descending order
        sorted_df = self.df.sort_values("A", False)
        a_values = [sorted_df["A"][i] for i in range(len(sorted_df))]
        self.assertEqual(a_values, [5, 4, 3, 2, 1])

    def test_head_tail(self):
        """Test the head and tail methods."""
        head_df = self.df.head(2)
        self.assertEqual(len(head_df), 2)
        self.assertEqual(head_df["A"][0], 1)
        self.assertEqual(head_df["A"][1], 2)

        tail_df = self.df.tail(2)
        self.assertEqual(len(tail_df), 2)
        self.assertEqual(tail_df["A"][0], 4)
        self.assertEqual(tail_df["A"][1], 5)

    def test_groupby(self):
        """Test groupby functionality."""
        # Create a DataFrame with groups
        group_data = {
            "category": ["A", "B", "A", "B"],
            "value": [10, 20, 30, 40]
        }
        group_df = hpda.DataFrame(group_data)
        
        # Group by 'category' and compute mean
        result = group_df.groupby("category").mean()
        
        # Result should have categories A and B, with means 20 and 30
        result_dict = result.to_dict()
        self.assertEqual(len(result_dict["category"]), 2)
        
        # Find index of category A and B
        a_idx = result_dict["category"].index("A")
        b_idx = result_dict["category"].index("B")
        
        # Check mean values
        self.assertAlmostEqual(result_dict["value"][a_idx], 20.0)
        self.assertAlmostEqual(result_dict["value"][b_idx], 30.0)


if __name__ == "__main__":
    unittest.main()