"""CLI для сбора данных, обработки, детекции, оценки и отчётов."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging

import pandas as pd

import config
import data_sources
import report
from core import (StreamSimulator, build_pipeline, detect_for_series,
                  generate_calendar_events, load_events, load_ground_truth)
from evaluation import evaluate_cp

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("main")

BACKEND_CHOICES = ("fallback", "auto", "rustures", "ruptures_rs")


def _read_indexed_csv(path) -> pd.DataFrame:
    """Read a time-series CSV, allowing equal timestamps across different series."""
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError(f"В {path} отсутствует столбец timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        bad = int(frame["timestamp"].isna().sum())
        raise ValueError(f"В {path} есть {bad} некорректных timestamp")
    if "series_id" in frame.columns:
        if frame["series_id"].isna().any():
            raise ValueError(f"В {path} есть строки без series_id")
        duplicated = frame.duplicated(subset=["series_id", "timestamp"], keep=False)
        if duplicated.any():
            sample = frame.loc[duplicated, ["series_id", "timestamp"]].head(3).to_dict("records")
            raise ValueError(f"В {path} есть дубликаты пары series_id+timestamp: {sample}")
    elif frame["timestamp"].duplicated().any():
        raise ValueError(f"В {path} есть повторяющиеся timestamp")
    return frame.set_index("timestamp").sort_index()


def _unavailable_series(collection_status: dict) -> set[str]:
    """Series whose last external collection failed and therefore must not be reused."""
    unavailable: set[str] = set()
    cbr = collection_status.get("cbr_currency.csv", {})
    if cbr.get("status") in {"failed", "empty"}:
        unavailable.update({"USD_RUB", "EUR_RUB", "CNY_RUB"})
    elif cbr.get("status") == "partial":
        for currency in cbr.get("failed_currencies", []):
            unavailable.add(f"{currency}_RUB")
    market = collection_status.get("yfinance_data.csv", {})
    if market.get("status") in {"failed", "empty"}:
        unavailable.update({"SPY", "AAPL", "BTC-USD"})
    elif market.get("status") == "success":
        available = set(market.get("series", []))
        unavailable.update({"SPY", "AAPL", "BTC-USD"} - available)
    return unavailable


def cmd_collect(args):
    logger.info("=== Collect data ===")
    config.ensure_dirs()
    data_sources.collect_all()


def cmd_process(args):
    logger.info("=== Process data ===")
    config.ensure_dirs()
    processed_ids: set[str] = set()
    status_path = config.REPORTS_DIR / "collection_status.json"
    collection_status = {}
    if status_path.exists():
        try:
            collection_status = json.loads(status_path.read_text(encoding="utf-8")).get("sources", {})
        except Exception as exc:
            logger.warning("Не удалось прочитать manifest сбора %s: %s", status_path, exc)
    unavailable = _unavailable_series(collection_status) if not getattr(args, "allow_stale", False) else set()
    for path in sorted(config.RAW_DIR.glob("*.csv")):
        if path.name == config.GROUND_TRUTH_CSV.name:
            continue
        source_status = collection_status.get(path.name, {})
        if (not getattr(args, "allow_stale", False) and
                source_status.get("status") in {"failed", "empty"} and
                source_status.get("stale_file_retained")):
            logger.warning("Пропускаю %s: последняя загрузка не удалась, файл старый; "
                           "для осознанной обработки используйте --allow-stale", path.name)
            continue
        try:
            frame = _read_indexed_csv(path)
        except Exception as exc:
            logger.warning("Пропускаю %s: %s", path.name, exc)
            if path.name == "cbr_currency.csv":
                unavailable.update({"USD_RUB", "EUR_RUB", "CNY_RUB"})
            elif path.name == "yfinance_data.csv":
                unavailable.update({"SPY", "AAPL", "BTC-USD"})
            else:
                alias = {"synthetic_mu_shift": "synth_mu_shift",
                         "synthetic_sigma_shift": "synth_sigma_shift",
                         "synthetic_random_walk": "synth_random_walk"}.get(path.stem, path.stem)
                unavailable.add(alias)
            continue
        if "value" not in frame:
            logger.warning("Пропускаю %s: нет столбца value", path.name)
            if path.name == "cbr_currency.csv":
                unavailable.update({"USD_RUB", "EUR_RUB", "CNY_RUB"})
            elif path.name == "yfinance_data.csv":
                unavailable.update({"SPY", "AAPL", "BTC-USD"})
            else:
                unavailable.add({"synthetic_mu_shift": "synth_mu_shift",
                                 "synthetic_sigma_shift": "synth_sigma_shift",
                                 "synthetic_random_walk": "synth_random_walk"}.get(path.stem, path.stem))
            continue
        if "series_id" in frame:
            series_ids = frame["series_id"].dropna().astype(str).unique()
            for sid in series_ids:
                sid = str(sid)
                safe_sid = sid.replace("/", "_").replace(" ", "_")
                if safe_sid in unavailable:
                    logger.warning("Пропускаю %s: источник не обновлён в последнем сборе", safe_sid)
                    continue
                if sid in processed_ids:
                    logger.info("Пропускаю дубликат источника для %s из %s; ряд уже обработан", sid, path.name)
                    continue
                sub = frame.loc[frame["series_id"].astype(str) == sid, ["value"]].copy()
                if len(sub) < 30:
                    logger.warning("Skip %s: only %s points", sid, len(sub))
                    unavailable.add(safe_sid)
                    continue
                cfg = config.get_series_config(safe_sid)
                try:
                    out = build_pipeline(sub, cfg)
                except Exception as exc:
                    logger.warning("Ошибка обработки %s: %s", sid, exc)
                    unavailable.add(safe_sid)
                    continue
                out.to_csv(config.PROCESSED_DIR / f"{safe_sid}.csv", index_label="timestamp")
                processed_ids.add(sid)
                logger.info("%s.csv: %s points, stl_period=%s", safe_sid, len(out), cfg["stl_period"])
        else:
            sid = {"synthetic_mu_shift": "synth_mu_shift",
                   "synthetic_sigma_shift": "synth_sigma_shift",
                   "synthetic_random_walk": "synth_random_walk"}.get(path.stem, path.stem)
            if sid in unavailable:
                logger.warning("Пропускаю %s: источник не обновлён в последнем сборе", sid)
                continue
            if sid in processed_ids:
                logger.info("Пропускаю устаревший дубликат %s для %s", path.name, sid)
                continue
            if len(frame) < 30:
                logger.warning("Skip %s: only %s points", sid, len(frame))
                unavailable.add(sid)
                continue
            cfg = config.get_series_config(sid)
            try:
                out = build_pipeline(frame[["value"]].copy(), cfg)
            except Exception as exc:
                logger.warning("Ошибка обработки %s: %s", sid, exc)
                unavailable.add(sid)
                continue
            out.to_csv(config.PROCESSED_DIR / f"{sid}.csv", index_label="timestamp")
            processed_ids.add(sid)
            logger.info("%s.csv: %s points, stl_period=%s", sid, len(out), cfg["stl_period"])
    for sid in sorted(unavailable - processed_ids):
        for folder, suffix in ((config.PROCESSED_DIR, ".csv"),
                               (config.DETECTIONS_DIR, ".csv"),
                               (config.PLOTS_DIR, ".png"),
                               (config.GROUND_TRUTH_DIR, "_gt.png")):
            artifact = folder / f"{sid}{suffix}"
            if artifact.exists():
                artifact.unlink()
                logger.info("Удалён устаревший производный артефакт %s", artifact)
    # Stage 2: календарные события по умолчанию не генерируются.
    n = generate_calendar_events()
    logger.info("Calendar events regenerated: %s", n)


def cmd_detect(args):
    """Сохраняет детекцию отдельно от processed data, не перезаписывая входы."""
    logger.info("=== Detect change points ===")
    config.ensure_dirs()
    rows = []
    gt_map = load_ground_truth()
    for path in sorted(config.PROCESSED_DIR.glob("*.csv")):
        sid = path.stem
        try:
            df = _read_indexed_csv(path)
            cps = detect_for_series(df, config.get_series_config(sid), debug=args.debug,
                                    backend=getattr(args, "backend", "fallback"))
        except Exception as exc:
            logger.exception("Ошибка детекции %s: %s", sid, exc)
            continue
        result = pd.DataFrame({"timestamp": [df.index[cp] for cp in cps],
                               "cp_index": cps, "series_id": sid})
        result.to_csv(config.DETECTIONS_DIR / f"{sid}.csv", index=False)
        gt = gt_map.get(sid, [])
        rows.append({"series": sid, "n_predicted": len(cps), "n_ground_truth": len(gt),
                     "predicted_idx": ";".join(map(str, cps))})
        logger.info("%s: %s CPs -> %s", sid, len(cps), cps)
    summary = pd.DataFrame(rows, columns=["series", "n_predicted", "n_ground_truth", "predicted_idx"])
    summary.to_csv(config.REPORTS_DIR / "detection_summary.csv", index=False)
    print("\n" + (summary.to_string(index=False) if not summary.empty else "Нет обработанных рядов") + "\n")


def cmd_evaluate(args):
    logger.info("=== Evaluate vs ground truth ===")
    gt_map = load_ground_truth()
    if not gt_map:
        logger.error("Нет ground truth — сначала выполните collect.")
        return
    rows = []
    for sid, true_cps in gt_map.items():
        path = config.PROCESSED_DIR / f"{sid}.csv"
        if not path.exists():
            logger.warning("Нет обработанного ряда для %s", sid)
            continue
        df = _read_indexed_csv(path)
        pred = detect_for_series(df, config.get_series_config(sid), debug=args.debug,
                                 backend=getattr(args, "backend", "fallback"))
        metrics = evaluate_cp(true_cps, pred, tolerance=args.tolerance)
        metrics.update({"series": sid, "true_idx": ";".join(map(str, true_cps)),
                        "pred_idx": ";".join(map(str, pred))})
        rows.append(metrics)
    columns = ["series", "n_true", "n_pred", "tp", "precision", "recall", "f1",
               "mean_offset", "true_idx", "pred_idx"]
    result = pd.DataFrame(rows, columns=columns)
    result.to_csv(config.REPORTS_DIR / "evaluation.csv", index=False)
    print("\n" + "=" * 100 + "\nCHANGE POINT DETECTION — EVALUATION\n" + "=" * 100)
    print(result.to_string(index=False) if not result.empty else "Нет рядов для оценки")
    print("=" * 100 + "\n")


def cmd_report(args):
    logger.info("=== Report ===")
    config.ensure_dirs()
    gt_map = load_ground_truth()
    rows = []
    for path in sorted(config.PROCESSED_DIR.glob("*.csv")):
        sid = path.stem
        df = _read_indexed_csv(path)
        detection_path = config.DETECTIONS_DIR / f"{sid}.csv"
        df["predicted_cp"] = 0
        if detection_path.exists():
            det = pd.read_csv(detection_path)
            if "timestamp" in det:
                timestamps = pd.to_datetime(det["timestamp"], errors="coerce").dropna()
                valid = df.index.get_indexer(timestamps)
                for pos in valid[valid >= 0]:
                    df.iloc[pos, df.columns.get_loc("predicted_cp")] = 1
        gt = gt_map.get(sid, [])
        # Stage 2: только ручные события; автокалендарь отключён через config.
        events = load_events(sid)
        if not events.empty:
            events = events[events["date"].dt.normalize().isin(df.index.normalize())]
        report.plot_series(df, sid, config.PLOTS_DIR / f"{sid}.png",
                           ground_truth_cps=gt, events=events)
        if not events.empty:
            report.plot_ground_truth(df, events, sid,
                                     config.GROUND_TRUTH_DIR / f"{sid}_gt.png")
        rows.append(report.summarize(df, sid, gt))
    summary = pd.DataFrame(rows)
    summary.to_csv(config.REPORTS_DIR / "summary.csv", index=False)
    cols = ["series", "points", "outliers", "isolated_outliers", "cp_candidates", "predicted_cps",
            "holidays", "weekends", "n_gt_cps"]
    print("\n" + "=" * 100 + "\nSUMMARY\n" + "=" * 100)
    if summary.empty:
        print("Нет обработанных рядов")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(summary[cols].to_string(index=False))
    print("=" * 100 + "\n")


def cmd_inspect(args):
    path = config.PROCESSED_DIR / f"{args.series}.csv"
    if not path.exists():
        logger.error("Не найден файл: %s", path)
        return
    df = _read_indexed_csv(path)
    print(f"\nSeries: {args.series}\nColumns: {df.columns.tolist()}\nRows: {len(df)}")
    print(df.head(15).to_string() + "\n")


def cmd_check(args):
    print("\n=== RAW ===")
    for path in sorted(config.RAW_DIR.glob("*.csv")):
        if path.name == config.GROUND_TRUTH_CSV.name:
            print(f"  {path.name}: ground-truth metadata (not a time series)")
            continue
        try:
            df = _read_indexed_csv(path)
        except Exception as exc:
            print(f"  {path.name}: ERROR: {exc}")
            continue
        sid_values = df["series_id"].dropna().astype(str).unique() if "series_id" in df else [path.stem]
        for sid in sid_values:
            sub = df if "series_id" not in df else df[df["series_id"].astype(str) == sid]
            weekends = int((sub.index.dayofweek >= 5).sum())
            missing = int(pd.to_numeric(sub["value"], errors="coerce").isna().sum()) if "value" in sub else -1
            print(f"  {path.name} | {sid}: rows={len(sub)}, weekends={weekends}, missing_value={missing}")
    print("\n=== PROCESSED ===")
    for path in sorted(config.PROCESSED_DIR.glob("*.csv")):
        try:
            df = _read_indexed_csv(path)
            weekends = int((df.index.dayofweek >= 5).sum())
            missing = int(pd.to_numeric(df.get("value", pd.Series(index=df.index, dtype=float)), errors="coerce").isna().sum())
            print(f"  {path.stem}: rows={len(df)}, weekends={weekends}, missing_value={missing}")
        except Exception as exc:
            print(f"  {path.stem}: ERROR: {exc}")
    print()


def cmd_stream(args):
    path = config.PROCESSED_DIR / f"{args.series}.csv"
    if not path.exists():
        logger.error("Не найден файл: %s", path)
        return
    df = _read_indexed_csv(path)
    simulator = StreamSimulator(df, delay_seconds=args.delay)
    print(f"Streaming simulator {args.series} ({len(df)} points, delay={args.delay}s); детекция не выполняется")
    count = 0
    for ts, value in simulator.stream():
        count += 1
        if count <= 3 or count % 100 == 0:
            print(f"  {ts.date()} value={float(value):.4f}")
        if args.limit and count >= args.limit:
            break
    print(f"Done: {count} points\n")


def main():
    parser = argparse.ArgumentParser(description="Time series change-point project")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("collect").set_defaults(func=cmd_collect)
    p_process = sub.add_parser("process")
    p_process.add_argument("--allow-stale", action="store_true",
                           help="обработать старые внешние CSV, даже если последняя загрузка не удалась")
    p_process.set_defaults(func=cmd_process)
    p_detect = sub.add_parser("detect")
    p_detect.add_argument("--debug", action="store_true")
    p_detect.add_argument("--backend", choices=BACKEND_CHOICES, default="fallback",
                          help="алгоритм сегментации (default: fallback, без внешних зависимостей)")
    p_detect.set_defaults(func=cmd_detect)
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--debug", action="store_true")
    p_eval.add_argument("--tolerance", type=int, default=None)
    p_eval.add_argument("--backend", choices=BACKEND_CHOICES, default="fallback",
                        help="алгоритм сегментации (default: fallback, без внешних зависимостей)")
    p_eval.set_defaults(func=cmd_evaluate)
    sub.add_parser("report").set_defaults(func=cmd_report)
    sub.add_parser("check").set_defaults(func=cmd_check)
    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("--series", required=True)
    p_inspect.set_defaults(func=cmd_inspect)
    p_stream = sub.add_parser("stream")
    p_stream.add_argument("--series", required=True)
    p_stream.add_argument("--delay", type=float, default=0.0)
    p_stream.add_argument("--limit", type=int, default=0)
    p_stream.set_defaults(func=cmd_stream)
    args = parser.parse_args()

    # Явный backend: проверяем доступность до запуска, чтобы не падать в середине.
    backend = getattr(args, "backend", None)
    if backend == "ruptures_rs" and importlib.util.find_spec("ruptures_rs") is None:
        parser.error("для --backend ruptures_rs установите пакет ruptures-rs")
    if backend == "rustures" and importlib.util.find_spec("rustures") is None:
        parser.error("для --backend rustures установите пакет rustures")

    config.ensure_dirs()
    args.func(args)


if __name__ == "__main__":
    main()