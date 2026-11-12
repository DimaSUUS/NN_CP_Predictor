import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from main import cmd_process


class ProcessIntegrationTests(unittest.TestCase):
    def test_multi_series_sources_aliases_and_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, processed, events = root / "raw", root / "processed", root / "events"
            reports = root / "reports"
            plots, gt_dir, detections = reports / "plots", reports / "ground_truth", reports / "detections"
            for folder in (raw, processed, events, reports, plots, gt_dir, detections):
                folder.mkdir(parents=True, exist_ok=True)
            idx = pd.date_range("2024-01-01", periods=50, freq="D")
            cbr_rows = []
            for sid, base in (("USD/RUB", 90), ("EUR/RUB", 100), ("CNY/RUB", 12)):
                for i, ts in enumerate(idx):
                    cbr_rows.append({"timestamp": ts, "series_id": sid, "value": base + i * 0.01})
            pd.DataFrame(cbr_rows).to_csv(raw / "cbr_currency.csv", index=False)
            values = np.r_[np.zeros(25), np.ones(25) * 5]
            pd.DataFrame({"timestamp": idx, "series_id": "synth_mu_shift", "value": values}).to_csv(
                raw / "synth_mu_shift.csv", index=False)
            pd.DataFrame({"timestamp": idx, "value": values}).to_csv(
                raw / "synthetic_mu_shift.csv", index=False)
            pd.DataFrame({"timestamp": idx, "series_id": "AAPL", "value": np.arange(50) + 100}).to_csv(
                raw / "yfinance_data.csv", index=False)
            (raw / "synthetic_ground_truth.csv").write_text(
                "series_id,change_points,type\nsynth_mu_shift,25,mean_shift\n", encoding="utf-8")
            (reports / "collection_status.json").write_text(json.dumps({"sources": {
                "cbr_currency.csv": {"status": "success", "rows": 150},
                "yfinance_data.csv": {"status": "failed", "error": "database locked",
                                       "stale_file_retained": True}}}), encoding="utf-8")
            # Old derived files must not survive a failed current download.
            pd.DataFrame({"timestamp": idx, "value": np.arange(50)}).to_csv(
                processed / "AAPL.csv", index=False)
            pd.DataFrame({"timestamp": [idx[10]], "cp_index": [10]}).to_csv(
                detections / "AAPL.csv", index=False)
            patches = {
                "RAW_DIR": raw, "PROCESSED_DIR": processed, "EVENTS_DIR": events,
                "REPORTS_DIR": reports, "PLOTS_DIR": plots, "GROUND_TRUTH_DIR": gt_dir,
                "DETECTIONS_DIR": detections, "GROUND_TRUTH_CSV": raw / "synthetic_ground_truth.csv",
                "MANUAL_EVENTS": events / "events.csv", "CALENDAR_EVENTS": events / "calendar_events.csv",
            }
            with patch.multiple(config, **patches):
                cmd_process(argparse.Namespace(allow_stale=False))
            self.assertTrue((processed / "USD_RUB.csv").exists())
            self.assertTrue((processed / "EUR_RUB.csv").exists())
            self.assertTrue((processed / "CNY_RUB.csv").exists())
            self.assertTrue((processed / "synth_mu_shift.csv").exists())
            self.assertFalse((processed / "synthetic_mu_shift.csv").exists())
            self.assertFalse((processed / "AAPL.csv").exists())
            self.assertFalse((detections / "AAPL.csv").exists())
            # Same timestamp is valid across different series, but not duplicated within a series.
            for sid in ("USD_RUB", "EUR_RUB", "CNY_RUB", "synth_mu_shift"):
                df = pd.read_csv(processed / f"{sid}.csv", parse_dates=["timestamp"])
                self.assertEqual(len(df), 50)
                self.assertFalse(df["timestamp"].duplicated().any())


if __name__ == "__main__":
    unittest.main()
