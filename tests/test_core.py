import unittest
import numpy as np
import pandas as pd

import config
from core import (_fallback_detection, add_calendar_features, build_pipeline,
                  cluster_outliers, detect_change_points, evaluate_cp, mark_outliers)
from data_sources import generate_synthetic_series, generate_random_walk
from evaluation import summarize_outliers


class EvaluationTests(unittest.TestCase):
    def test_both_empty(self):
        self.assertEqual(evaluate_cp([], [] )["f1"], 1.0)

    def test_empty_predictions_do_not_get_perfect_precision(self):
        result = evaluate_cp([10], [])
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)

    def test_zero_tolerance_is_respected(self):
        self.assertEqual(evaluate_cp([10], [11], tolerance=0)["tp"], 0)

    def test_one_to_one_matching(self):
        result = evaluate_cp([10, 12], [11, 20], tolerance=2)
        self.assertEqual(result["tp"], 1)

    def test_matching_maximizes_number_of_matches_not_first_greedy_choice(self):
        result = evaluate_cp([10, 14], [12, 8], tolerance=4)
        self.assertEqual(result["tp"], 2)


class DataTests(unittest.TestCase):
    def test_synthetic_reproducibility_without_global_rng_side_effect(self):
        np.random.seed(123)
        before = np.random.get_state()
        a, cps_a = generate_synthetic_series(n_points=100, change_points=[40], seed=5)
        after = np.random.get_state()
        b, cps_b = generate_synthetic_series(n_points=100, change_points=[40], seed=5)
        self.assertTrue(np.array_equal(a["value"].to_numpy(), b["value"].to_numpy()))
        self.assertEqual(cps_a, cps_b)
        self.assertTrue(np.array_equal(before[1], after[1]))

    def test_invalid_change_points(self):
        with self.assertRaises(ValueError):
            generate_synthetic_series(n_points=100, change_points=[80, 40], seed=1)

    def test_random_walk(self):
        frame, cps = generate_random_walk(n_points=100, change_points=[50], seed=3)
        self.assertEqual(cps, [50])
        self.assertTrue((frame["value"] > 0).all())


class PipelineTests(unittest.TestCase):
    def test_pipeline_rejects_duplicate_timestamps(self):
        idx = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])
        frame = pd.DataFrame({"value": [1., 2., 3.]}, index=idx)
        with self.assertRaises(ValueError):
            build_pipeline(frame, {"stl_period": 0})

    def test_constant_series_no_outliers(self):
        frame = pd.DataFrame({"resid": np.zeros(100)})
        result = mark_outliers(frame)
        self.assertEqual(int(result["is_outlier"].sum()), 0)

    def test_cluster_candidates_preserve_raw_outlier_flags(self):
        frame = pd.DataFrame({"is_outlier": [0, 1, 1, 1, 1, 1, 0]})
        result, cps = cluster_outliers(frame, min_cp_len=5)
        self.assertEqual(cps, [3])
        self.assertEqual(int(result["cp_candidate"].sum()), 1)
        self.assertEqual(int(result["is_outlier"].sum()), 5)
        self.assertEqual(int(result["is_isolated_outlier"].sum()), 0)
        self.assertEqual(summarize_outliers(result), {
            "outliers": 5, "isolated_outliers": 0, "cp_candidates": 1, "points": 7
        })

    def test_outlier_metrics_use_same_definition_for_report_and_log(self):
        frame = pd.DataFrame({"is_outlier": [0, 1, 0, 1],
                              "is_isolated_outlier": [0, 1, 0, 1],
                              "cp_candidate": [0, 0, 0, 0]})
        counts = summarize_outliers(frame)
        self.assertEqual(counts["outliers"], 2)
        self.assertEqual(counts["isolated_outliers"], 2)

    def test_dp_fallback_mean_shift(self):
        rng = np.random.default_rng(12)
        x = np.r_[rng.normal(0, 1, 100), rng.normal(4, 1, 100)]
        cps = _fallback_detection(x, min_size=20, model="l2", penalty=10)
        self.assertTrue(any(abs(cp - 100) <= 5 for cp in cps), cps)

    def test_dp_fallback_variance_shift(self):
        rng = np.random.default_rng(13)
        x = np.r_[rng.normal(0, 1, 150), rng.normal(0, 3, 150)]
        x = (x - x.mean()) / x.std()
        cps = _fallback_detection(x, min_size=20, model="normal", penalty=30)
        self.assertTrue(any(abs(cp - 150) <= 15 for cp in cps), cps)

    def test_explicit_fallback_backend_is_available(self):
        rng = np.random.default_rng(21)
        x = np.r_[rng.normal(0, 1, 100), rng.normal(4, 1, 100)]
        cps = detect_change_points(x, penalty=10, model="l2", backend="fallback")
        self.assertTrue(any(abs(cp - 100) <= 5 for cp in cps), cps)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            detect_change_points([0.0, 1.0, 0.0, 1.0], penalty=1, backend="bad")


if __name__ == "__main__":
    unittest.main()
