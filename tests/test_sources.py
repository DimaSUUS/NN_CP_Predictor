import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import config
import data_sources


class YFinanceColumnNormalizationTests(unittest.TestCase):
    def test_ticker_first_multiindex(self):
        columns = pd.MultiIndex.from_product(
            [["SPY", "AAPL"], ["Open", "Close"]], names=["Ticker", "Price"])
        raw = pd.DataFrame([[1, 10, 2, 20], [3, 11, 4, 21]],
                           index=pd.date_range("2024-01-01", periods=2), columns=columns)
        spy = data_sources._extract_ticker_frame(raw, "SPY", multi_ticker=True)
        self.assertEqual(spy.columns.tolist(), ["value"])
        self.assertEqual(spy["value"].tolist(), [10, 11])

    def test_price_first_multiindex_single_ticker(self):
        columns = pd.MultiIndex.from_product(
            [["Close", "Open"], ["SPY"]], names=["Price", "Ticker"])
        raw = pd.DataFrame([[10, 9], [11, 10]],
                           index=pd.date_range("2024-01-01", periods=2), columns=columns)
        spy = data_sources._extract_ticker_frame(raw, "SPY", multi_ticker=False)
        self.assertEqual(spy.columns.tolist(), ["value"])
        self.assertEqual(spy["value"].tolist(), [10, 11])


class CollectionManifestTests(unittest.TestCase):
    def test_failed_external_sources_do_not_overwrite_old_files_and_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            reports = root / "reports"
            processed = root / "processed"
            events = root / "events"
            plots = reports / "plots"
            gt_dir = reports / "ground_truth"
            detections = reports / "detections"
            for folder in (raw, reports, processed, events, plots, gt_dir, detections):
                folder.mkdir(parents=True, exist_ok=True)
            cbr_path = raw / "cbr_currency.csv"
            market_path = raw / "yfinance_data.csv"
            cbr_path.write_text("old cbr data", encoding="utf-8")
            market_path.write_text("old market data", encoding="utf-8")
            with patch.multiple(config,
                RAW_DIR=raw, REPORTS_DIR=reports, PROCESSED_DIR=processed,
                EVENTS_DIR=events, PLOTS_DIR=plots, GROUND_TRUTH_DIR=gt_dir,
                DETECTIONS_DIR=detections, GROUND_TRUTH_CSV=raw / "synthetic_ground_truth.csv"), \
                patch.object(data_sources, "fetch_cbr_currency", side_effect=RuntimeError("offline")), \
                patch.object(data_sources, "fetch_yfinance", side_effect=RuntimeError("database is locked")):
                result = data_sources.collect_all()
            self.assertEqual(cbr_path.read_text(encoding="utf-8"), "old cbr data")
            self.assertEqual(market_path.read_text(encoding="utf-8"), "old market data")
            self.assertEqual(result["sources"]["cbr_currency.csv"]["status"], "failed")
            self.assertTrue(result["sources"]["cbr_currency.csv"]["stale_file_retained"])
            self.assertEqual(result["sources"]["yfinance_data.csv"]["status"], "failed")
            manifest = json.loads((reports / "collection_status.json").read_text(encoding="utf-8"))
            self.assertIn("synth_random_walk.csv", manifest["sources"])
            self.assertTrue((raw / "synth_random_walk.csv").exists())


if __name__ == "__main__":
    unittest.main()
