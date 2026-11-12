"""Reproducible, network-free synthetic baseline for change-point detection.

Usage:
  python baseline.py                        # default: fallback
  python baseline.py --backend fallback
  python baseline.py --backend rustures
  python baseline.py --backend auto
"""
from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from core import build_pipeline, detect_for_series
from data_sources import generate_random_walk, generate_synthetic_series
from evaluation import evaluate_cp


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_benchmarks(backend: str) -> dict:
    cases = []
    specifications = []
    frame, truth = generate_synthetic_series(
        n_points=1000, change_points=[300, 700], mu_values=[0.0, 5.0, 0.0],
        sigma_values=[1.0, 1.0, 1.0], seed=42, series_id="synth_mu_shift")
    specifications.append(("synth_mu_shift", frame, truth))
    frame, truth = generate_synthetic_series(
        n_points=1000, change_points=[500], mu_before=0.0, mu_after=0.0,
        sigma_before=1.0, sigma_after=3.0, seed=43, series_id="synth_sigma_shift")
    specifications.append(("synth_sigma_shift", frame, truth))
    frame, truth = generate_random_walk(
        n_points=1000, change_points=[400], mu_before=0.003, mu_after=-0.005,
        sigma=0.01, start_price=100.0, seed=44, series_id="synth_random_walk")
    specifications.append(("synth_random_walk", frame, truth))
    frame, truth = generate_random_walk(
        n_points=1000, change_points=[], mu_before=0.003, mu_after=0.003,
        sigma=0.01, start_price=100.0, seed=44, series_id="synth_random_walk")
    specifications.append(("synth_random_walk_no_change", frame, truth))

    for sid, frame, truth in specifications:
        base_sid = sid.replace("_no_change", "")
        cfg = config.get_series_config(base_sid)
        processed = build_pipeline(frame[["value"]], cfg)
        predicted = detect_for_series(processed, cfg, backend=backend)
        metrics = evaluate_cp(truth, predicted, tolerance=config.CP_EVAL_TOLERANCE)
        cases.append({"series": sid, "n_points": len(frame),
                      "ground_truth": truth, "predicted": predicted,
                      "metrics": metrics,
                      "false_positive_count": max(0, len(predicted) - metrics["tp"])})

    source_hashes = {}
    for filename in ("config.py", "core.py", "evaluation.py", "data_sources.py", "baseline.py"):
        source_path = Path(__file__).with_name(filename)
        source_hashes[filename] = hashlib.sha256(source_path.read_bytes()).hexdigest()

    return {
        "schema_version": 1,
        "source_sha256": source_hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend_requested": backend,
        "backend_available": {
            "rustures": package_version("rustures"),
            "ruptures_rs": package_version("ruptures-rs"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "statsmodels": package_version("statsmodels"),
        },
        "python_version": platform.python_version(),
        "random_seeds": {"mean_shift": 42, "variance_shift": 43, "random_walk": 44},
        "parameters": {
            "CP_PENALTY_MEAN": config.CP_PENALTY_MEAN,
            "CP_PENALTY_TREND": config.CP_PENALTY_TREND,
            "CP_PENALTY_VAR": config.CP_PENALTY_VAR,
            "CP_MIN_SIZE": config.CP_MIN_SIZE,
            "CP_DEDUP_TOL": config.CP_DEDUP_TOL,
            "CP_EVAL_TOLERANCE": config.CP_EVAL_TOLERANCE,
            "OUTLIER_WINDOW": config.OUTLIER_WINDOW,
            "OUTLIER_Z": config.OUTLIER_Z,
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="fallback",
                        choices=("fallback", "auto", "rustures", "ruptures_rs"))
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON path; default data/reports/baseline_<backend>.json")
    args = parser.parse_args()
    output = args.output or (config.REPORTS_DIR / f"baseline_{args.backend}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_benchmarks(args.backend)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Baseline saved: {output}")
    for case in result["cases"]:
        print(f"{case['series']}: truth={case['ground_truth']} predicted={case['predicted']} "
              f"F1={case['metrics']['f1']} FP={case['false_positive_count']}")


if __name__ == "__main__":
    main()