"""Grid-search CP penalties on synthetic benchmarks and report stable ranges.

Usage:
    python calibrate.py
    python calibrate.py --backend fallback
    python calibrate.py --output data/reports/calibration.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config
from core import build_pipeline, detect_for_series
from data_sources import generate_random_walk, generate_synthetic_series
from evaluation import evaluate_cp


def _build_cases():
    cases = []
    frame, truth = generate_synthetic_series(
        n_points=1000, change_points=[300, 700], mu_values=[0.0, 5.0, 0.0],
        sigma_values=[1.0, 1.0, 1.0], seed=42, series_id="synth_mu_shift")
    cases.append(("synth_mu_shift", frame, truth, "mean"))
    frame, truth = generate_synthetic_series(
        n_points=1000, change_points=[500], mu_before=0.0, mu_after=0.0,
        sigma_before=1.0, sigma_after=3.0, seed=43, series_id="synth_sigma_shift")
    cases.append(("synth_sigma_shift", frame, truth, "variance"))
    frame, truth = generate_random_walk(
        n_points=1000, change_points=[400], mu_before=0.003, mu_after=-0.005,
        sigma=0.01, start_price=100.0, seed=44, series_id="synth_random_walk")
    cases.append(("synth_random_walk", frame, truth, "mean"))
    frame, truth = generate_random_walk(
        n_points=1000, change_points=[], mu_before=0.003, mu_after=0.003,
        sigma=0.01, start_price=100.0, seed=44, series_id="synth_random_walk_no_change")
    cases.append(("synth_random_walk_no_change", frame, truth, "mean"))
    return cases


def _evaluate_grid(cases, penalty_key, grid, backend, tolerance):
    rows = []
    for sid, frame, truth, kind in cases:
        cfg = config.get_series_config(sid.replace("_no_change", ""))
        processed = build_pipeline(frame[["value"]], cfg)
        for p in grid:
            predicted = detect_for_series(processed, cfg, backend=backend,
                                          penalties={penalty_key: p})
            metrics = evaluate_cp(truth, predicted, tolerance=tolerance)
            fp = max(0, len(predicted) - metrics["tp"])
            rows.append({"series": sid, "kind": kind, "penalty": p,
                         "n_pred": len(predicted), "fp": fp,
                         "f1": metrics["f1"], "precision": metrics["precision"],
                         "recall": metrics["recall"]})
    return pd.DataFrame(rows)


def _stable_range(df, kind, target_f1=0.9, max_fp=1):
    subset = df[df["kind"] == kind]
    if subset.empty:
        return None
    grouped = subset.groupby("penalty").agg(
        min_f1=("f1", "min"),
        total_fp=("fp", "sum"),
    )
    good = grouped[(grouped["min_f1"] >= target_f1) & (grouped["total_fp"] <= max_fp)]
    if good.empty:
        return None
    return {"min": float(good.index.min()), "max": float(good.index.max()),
            "n_points": int(len(good))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="fallback",
                        choices=("fallback", "auto", "rustures", "ruptures_rs"))
    parser.add_argument("--tolerance", type=int, default=config.CP_EVAL_TOLERANCE)
    parser.add_argument("--output", type=Path,
                        default=config.REPORTS_DIR / "calibration.json")
    args = parser.parse_args()

    config.ensure_dirs()
    cases = _build_cases()

    mean_grid = list(range(2, 51, 2))
    var_grid = list(range(5, 101, 5))

    mean_df = _evaluate_grid(cases, "mean", mean_grid, args.backend, args.tolerance)
    var_df = _evaluate_grid(cases, "variance", var_grid, args.backend, args.tolerance)

    mean_range = _stable_range(mean_df, "mean")
    var_range = _stable_range(var_df, "variance")

    print("=== MEAN penalty grid ===")
    print(mean_df.groupby("penalty").agg({"f1": "mean", "fp": "sum"}).to_string())
    print("\n=== VARIANCE penalty grid ===")
    print(var_df.groupby("penalty").agg({"f1": "mean", "fp": "sum"}).to_string())
    print("\n=== Suggested stable ranges ===")
    print("mean:", mean_range)
    print("variance:", var_range)

    output = {
        "backend": args.backend,
        "tolerance": args.tolerance,
        "mean_grid": mean_grid,
        "var_grid": var_grid,
        "mean_results": mean_df.to_dict(orient="records"),
        "var_results": var_df.to_dict(orient="records"),
        "suggested_mean_range": mean_range,
        "suggested_variance_range": var_range,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()