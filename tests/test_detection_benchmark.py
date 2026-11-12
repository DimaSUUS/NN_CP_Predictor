import unittest

from core import build_pipeline, detect_for_series, evaluate_cp
from data_sources import generate_random_walk, generate_synthetic_series
import config


class SyntheticDetectionBenchmarkTests(unittest.TestCase):
    def _run(self, series_id, frame, truth):
        cfg = config.get_series_config(series_id)
        processed = build_pipeline(frame[["value"]], cfg)
        predicted = detect_for_series(processed, cfg)
        metrics = evaluate_cp(truth, predicted, tolerance=5)
        self.assertGreaterEqual(metrics["f1"], 0.99, (series_id, predicted, metrics))
        return predicted

    def test_mean_shift(self):
        frame, truth = generate_synthetic_series(
            n_points=1000, change_points=[300, 700], mu_values=[0.0, 5.0, 0.0],
            sigma_values=[1.0, 1.0, 1.0], seed=42, series_id="synth_mu_shift")
        predicted = self._run("synth_mu_shift", frame, truth)
        self.assertEqual(len(predicted), 2)

    def test_variance_shift(self):
        frame, truth = generate_synthetic_series(
            n_points=1000, change_points=[500], mu_before=0.0, mu_after=0.0,
            sigma_before=1.0, sigma_after=3.0, seed=43, series_id="synth_sigma_shift")
        predicted = self._run("synth_sigma_shift", frame, truth)
        self.assertEqual(len(predicted), 1)

    def test_random_walk_is_detected_in_log_return_space_without_oversegmentation(self):
        frame, truth = generate_random_walk(
            n_points=1000, change_points=[400], mu_before=0.003, mu_after=-0.005,
            sigma=0.01, start_price=100.0, seed=44, series_id="synth_random_walk")
        predicted = self._run("synth_random_walk", frame, truth)
        self.assertEqual(len(predicted), 1)

    def test_stationary_random_walk_returns_do_not_create_spurious_breaks(self):
        frame, truth = generate_random_walk(
            n_points=1000, change_points=[], mu_before=0.003, mu_after=0.003,
            sigma=0.01, start_price=100.0, seed=44, series_id="synth_random_walk")
        cfg = config.get_series_config("synth_random_walk")
        processed = build_pipeline(frame[["value"]], cfg)
        predicted = detect_for_series(processed, cfg)
        self.assertEqual(truth, [])
        self.assertLessEqual(len(predicted), 1, predicted)


if __name__ == "__main__":
    unittest.main()
