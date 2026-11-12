"""Small synthetic checks for report counts; these are not dataset observations."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.quality import identify_label, quality_summary, write_report


class QualityTests(unittest.TestCase):
    def test_counts_preserve_raw_values(self):
        frame = pd.DataFrame({" rate ": [np.inf, np.nan, -1, -1],
                              "text": ["bad", "2", "3", "3"],
                              " Label": ["A", "B", "A", "A"]})
        original = frame.copy(deep=True)
        result = quality_summary(frame)
        self.assertEqual(result["records"], 4)
        self.assertEqual(result["features"], 2)
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(result["missing"][" rate "], 1)
        self.assertEqual(result["infinite"][" rate "], 1)
        self.assertEqual(result["nonnumeric_feature_values"]["text"], 1)
        self.assertEqual(result["negative_numeric_values"][" rate "], 2)
        self.assertEqual(result["classes"], {"A": 3, "B": 1})
        pd.testing.assert_frame_equal(frame, original)
        with tempfile.TemporaryDirectory() as directory:
            path = write_report(frame, {"dataset": "synthetic", "revision": "test"},
                                path=Path(directory) / "report.md")
            self.assertIn("examined sample", path.read_text(encoding="utf-8"))

    def test_ambiguous_label_requires_inspection(self):
        with self.assertRaises(ValueError):
            identify_label(pd.DataFrame({"Label": [], "class": []}))


if __name__ == "__main__":
    unittest.main()
