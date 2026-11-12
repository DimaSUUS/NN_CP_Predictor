import tempfile
import unittest
from pathlib import Path

import pandas as pd

from main import _read_indexed_csv, _unavailable_series


class MultiSeriesCsvTests(unittest.TestCase):
    def _write(self, frame):
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp.close()
        path = Path(tmp.name)
        frame.to_csv(path, index=False)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_same_timestamp_across_different_series_is_valid(self):
        path = self._write(pd.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "series_id": ["USD/RUB", "EUR/RUB", "USD/RUB"],
            "value": [90.0, 100.0, 91.0],
        }))
        frame = _read_indexed_csv(path)
        self.assertEqual(len(frame), 3)
        usd = frame[frame["series_id"] == "USD/RUB"]
        self.assertFalse(usd.index.has_duplicates)

    def test_duplicate_timestamp_within_same_series_is_rejected(self):
        path = self._write(pd.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-01"],
            "series_id": ["USD/RUB", "USD/RUB"], "value": [90.0, 91.0],
        }))
        with self.assertRaisesRegex(ValueError, r"series_id\+timestamp"):
            _read_indexed_csv(path)

    def test_missing_timestamp_column_has_clear_error(self):
        path = self._write(pd.DataFrame({"value": [1.0]}))
        with self.assertRaisesRegex(ValueError, "timestamp"):
            _read_indexed_csv(path)

    def test_failed_sources_mark_their_derived_series_unavailable(self):
        result = _unavailable_series({
            "cbr_currency.csv": {"status": "partial", "failed_currencies": ["EUR"]},
            "yfinance_data.csv": {"status": "failed", "stale_file_retained": True},
        })
        self.assertEqual(result, {"EUR_RUB", "SPY", "AAPL", "BTC-USD"})


if __name__ == "__main__":
    unittest.main()
