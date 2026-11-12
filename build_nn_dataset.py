"""Экспорт датасета для обучения NN предсказанию времени до следующей CP.

Вход:  data/processed/*.csv              (реальные ряды)
       data/reports/detections/*.csv     (метки CP для реальных)
       data/raw/synthetic_batch/*.csv    (синтетика, опционально)
       data/raw/synthetic_batch/ground_truth.csv

Выход: data/reports/nn_dataset/{train,val,test}/
         X.npy          # shape (N, window, n_features), float32, UNCOMPRESSED
         y_bin.npy, y_tau.npy, mask.npy
         series.npy, timestamp.npy, source.npy
       metadata.json
       preview.csv
       bin_distribution.txt

Важно: X сохраняется как .npy (не .npz) и пишется через open_memmap —
пик RAM = один ряд (~20 MB), не зависит от числа рядов.
При обучении: X = np.load('X.npy', mmap_mode='r') — lazy loading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

WINDOW = 200
HORIZON_DAYS = 180
BIN_SIZE_DAYS = 5
N_BINS = HORIZON_DAYS // BIN_SIZE_DAYS  # 36
CENSORED_BIN = N_BINS                   # 36

FEATURES = [
    "value_z", "log_return",
    "roll_std_30", "roll_std_60", "roll_mad_120",
    "label_cp", "is_outlier", "days_since_last_cp",
    "day_of_week", "month_sin", "month_cos",
]


# --------------------------------------------------------------------------- #
# Признаки
# --------------------------------------------------------------------------- #
def _compute_features(value: pd.Series) -> pd.DataFrame:
    value = pd.to_numeric(value, errors="coerce").astype(float)
    is_price_like = bool((value > 0).all())

    df = pd.DataFrame(index=value.index)
    roll_mean_120 = value.rolling(120, min_periods=30).mean()
    roll_std_120 = value.rolling(120, min_periods=30).std()
    df["value_z"] = ((value - roll_mean_120) /
                     roll_std_120.replace(0, np.nan)).fillna(0.0)

    if is_price_like:
        df["log_return"] = np.log(value).diff().fillna(0.0)
    else:
        df["log_return"] = value.diff().fillna(0.0)

    df["roll_std_30"] = df["log_return"].rolling(30, min_periods=10).std().fillna(0.0)
    df["roll_std_60"] = df["log_return"].rolling(60, min_periods=20).std().fillna(0.0)
    df["roll_mad_120"] = (df["log_return"]
                          .rolling(120, min_periods=30)
                          .apply(lambda x: np.median(np.abs(x - np.median(x))))
                          .fillna(0.0))

    idx = value.index
    df["day_of_week"] = idx.dayofweek / 6.0
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    return df


def _prepare_series(value: pd.Series, cp_positions: list[int] | None,
                    is_outlier: pd.Series | None) -> pd.DataFrame:
    value = pd.to_numeric(value, errors="coerce").astype(float)
    df = _compute_features(value)

    df["label_cp"] = 0
    if cp_positions:
        for cp in cp_positions:
            if 0 <= cp < len(df):
                df.iloc[cp, df.columns.get_loc("label_cp")] = 1

    if is_outlier is not None:
        df["is_outlier"] = (pd.to_numeric(is_outlier, errors="coerce")
                            .fillna(0).astype(int).to_numpy())
    else:
        df["is_outlier"] = 0

    cp_flags = df["label_cp"].to_numpy()
    cp_pos = np.where(cp_flags == 1)[0]
    days_since = np.full(len(df), -1, dtype=float)
    for i in range(len(df)):
        past = cp_pos[cp_pos <= i]
        if len(past) > 0:
            days_since[i] = i - past[-1]
    df["days_since_last_cp"] = days_since

    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0.0)
    return df


def _build_windows_inplace(df: pd.DataFrame, window: int, horizon: int):
    """Возвращает (X, y_tau, y_bin, mask, timestamps) для одного ряда.

    X shape: (n_windows, window, n_features).
    """
    values = df[FEATURES].to_numpy(dtype=np.float32)
    cp_flags = df["label_cp"].to_numpy(dtype=int)
    timestamps = df.index.to_numpy()
    n = len(df)
    if n < window + 2:
        return None

    cp_pos = np.where(cp_flags == 1)[0]
    n_win = n - window + 1

    X = np.empty((n_win, window, len(FEATURES)), dtype=np.float32)
    y_tau = np.empty(n_win, dtype=np.int32)
    y_bin = np.empty(n_win, dtype=np.int32)
    mask = np.empty(n_win, dtype=np.int8)
    ts = np.empty(n_win, dtype=object)

    pos = 0
    for t in range(window - 1, n):
        future = cp_pos[cp_pos > t]
        if len(future) == 0:
            tau = horizon + 1
            b = CENSORED_BIN
            m = 0
        else:
            tau = int(future[0] - t)
            if tau > horizon:
                b = CENSORED_BIN
                m = 0
            else:
                b = min((tau - 1) // BIN_SIZE_DAYS, CENSORED_BIN - 1)
                m = 1
        X[pos] = values[t - window + 1: t + 1]
        y_tau[pos] = tau
        y_bin[pos] = b
        mask[pos] = m
        ts[pos] = timestamps[t]
        pos += 1

    return X, y_tau, y_bin, mask, ts


# --------------------------------------------------------------------------- #
# Фаза 1: сканирование метаданных (без X!)
# --------------------------------------------------------------------------- #
def _scan_real_series(processed_dir: Path, detections_dir: Path, window: int):
    metas = []
    for path in sorted(processed_dir.glob("*.csv")):
        try:
            n = sum(1 for _ in open(path, encoding="utf-8")) - 1
        except Exception:
            continue
        if n < window + 2:
            continue
        metas.append({
            "series_id": path.stem, "kind": "real",
            "path": path, "n_windows": n - window + 1,
            "detection_path": detections_dir / f"{path.stem}.csv",
        })
    return metas


def _scan_synthetic_series(synth_dir: Path, window: int, max_series: int | None):
    gt_path = synth_dir / "ground_truth.csv"
    if not gt_path.exists():
        print(f"SKIP synthetic: не найден {gt_path}")
        return []
    gt = pd.read_csv(gt_path)
    metas = []
    for _, row in gt.iterrows():
        sid = str(row["series_id"])
        csv_path = synth_dir / f"{sid}.csv"
        if not csv_path.exists():
            continue
        try:
            n = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        except Exception:
            continue
        if n < window + 2:
            continue
        raw_cps = row.get("change_points")
        cps = ([] if pd.isna(raw_cps)
               else [int(p.strip()) for p in str(raw_cps).split(";") if p.strip()])
        metas.append({
            "series_id": sid, "kind": "synthetic",
            "path": csv_path, "n_windows": n - window + 1,
            "cp_positions": cps,
        })
    if max_series and len(metas) > max_series:
        rng = np.random.default_rng(0)
        idx = sorted(rng.choice(len(metas), size=max_series, replace=False))
        metas = [metas[i] for i in idx]
        print(f"  synthetic: подвыборка {max_series} из {len(gt)}")
    return metas


# --------------------------------------------------------------------------- #
# Сплиты: все реальные → train, синтетика 80/10/10
# --------------------------------------------------------------------------- #
def _assign_splits(real_metas, synth_metas, seed: int = 42):
    rng = np.random.default_rng(seed)
    synth = list(synth_metas)
    rng.shuffle(synth)
    n = len(synth)
    if n < 10:
        return {"train": real_metas + synth, "val": [], "test": []}
    n_test = max(1, int(round(n * 0.1)))
    n_val = max(1, int(round(n * 0.1)))
    return {
        "train": real_metas + synth[n_test + n_val:],
        "val": synth[n_test:n_test + n_val],
        "test": synth[:n_test],
    }


# --------------------------------------------------------------------------- #
# Фаза 2: запись split'а через memmap
# --------------------------------------------------------------------------- #
def _write_split(split_dir: Path, metas: list[dict], window: int, horizon: int):
    split_dir.mkdir(parents=True, exist_ok=True)
    n_total = sum(m["n_windows"] for m in metas)
    if n_total == 0:
        print(f"  {split_dir.name}: пустой split, пропуск")
        return None

    from numpy.lib.format import open_memmap

    X = open_memmap(split_dir / "X.npy", mode="w+", dtype="float32",
                    shape=(n_total, window, len(FEATURES)))
    y_bin_arr = np.empty(n_total, dtype=np.int32)
    y_tau_arr = np.empty(n_total, dtype=np.int32)
    mask_arr = np.empty(n_total, dtype=np.int8)
    series_arr = np.empty(n_total, dtype=object)
    ts_arr = np.empty(n_total, dtype=object)
    source_arr = np.empty(n_total, dtype=object)

    offset = 0
    for i, m in enumerate(metas):
        try:
            df = pd.read_csv(m["path"], parse_dates=["timestamp"]).set_index("timestamp")
        except Exception as exc:
            print(f"    SKIP {m['series_id']}: {exc}")
            continue

        if m["kind"] == "real":
            cp_positions = []
            det_path = m["detection_path"]
            if det_path.exists():
                det = pd.read_csv(det_path)
                cp_ts = set(pd.to_datetime(det["timestamp"], errors="coerce")
                            .dropna().dt.normalize())
                cp_positions = [int(j) for j, ts in enumerate(df.index.normalize())
                                if ts in cp_ts]
            is_outlier = df["is_outlier"] if "is_outlier" in df else None
        else:
            cp_positions = m["cp_positions"]
            is_outlier = None

        prepared = _prepare_series(df["value"], cp_positions, is_outlier)
        built = _build_windows_inplace(prepared, window, horizon)
        if built is None:
            continue
        X_b, y_tau_b, y_bin_b, mask_b, ts_b = built
        n_actual = len(X_b)

        X[offset:offset + n_actual] = X_b
        y_tau_arr[offset:offset + n_actual] = y_tau_b
        y_bin_arr[offset:offset + n_actual] = y_bin_b
        mask_arr[offset:offset + n_actual] = mask_b
        series_arr[offset:offset + n_actual] = m["series_id"]
        ts_arr[offset:offset + n_actual] = ts_b
        source_arr[offset:offset + n_actual] = m["kind"]

        offset += n_actual
        del X_b, prepared, df

        if (i + 1) % 50 == 0:
            print(f"    {split_dir.name}: {i + 1}/{len(metas)} series, {offset} windows")

    X.flush()
    del X  # закрыть memmap

    # Если offset < n_total (были skip), обрезаем
    np.save(split_dir / "y_bin.npy", y_bin_arr[:offset])
    np.save(split_dir / "y_tau.npy", y_tau_arr[:offset])
    np.save(split_dir / "mask.npy", mask_arr[:offset])
    np.save(split_dir / "series.npy", series_arr[:offset], allow_pickle=True)
    np.save(split_dir / "timestamp.npy", ts_arr[:offset], allow_pickle=True)
    np.save(split_dir / "source.npy", source_arr[:offset], allow_pickle=True)

    # Переписать X.npy, если offset < n_total
    if offset < n_total:
        from numpy.lib.format import open_memmap
        final_X = open_memmap(split_dir / "X.npy", mode="r+")
        # Обрезать нельзя memmap in-place — сделаем трюк: оставим как есть,
        # но в metadata запишем offset. Или пересоздадим.
        # Проще: пометить в metadata "actual_n", тренировочный код читает [:actual_n].
        print(f"    WARN: {n_total - offset} окон потеряно, X.npy обрезан логически")

    return {
        "n_series": len(metas),
        "n_windows_allocated": n_total,
        "n_windows_actual": offset,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=config.REPORTS_DIR / "nn_dataset")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--horizon", type=int, default=HORIZON_DAYS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-synthetic-batch", action="store_true")
    parser.add_argument("--synthetic-dir", type=Path,
                        default=config.RAW_DIR / "synthetic_batch")
    parser.add_argument("--max-synthetic", type=int, default=None,
                        help="ограничить число синтетических рядов (для быстрой проверки)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # --- Фаза 1: скан ---
    print("=== Scan real series ===")
    real_metas = _scan_real_series(config.PROCESSED_DIR,
                                   config.DETECTIONS_DIR, args.window)
    print(f"  найдено {len(real_metas)} рядов")

    synth_metas = []
    if args.include_synthetic_batch:
        print("\n=== Scan synthetic batch ===")
        synth_metas = _scan_synthetic_series(args.synthetic_dir,
                                             args.window, args.max_synthetic)
        print(f"  найдено {len(synth_metas)} рядов")
    else:
        print("\n(синтетика не подключена — флаг --include-synthetic-batch)")

    if not real_metas and not synth_metas:
        print("Нет данных для экспорта.")
        return

    # --- Сплиты ---
    splits = _assign_splits(real_metas, synth_metas, seed=args.seed)
    print("\n=== Splits ===")
    for name, metas in splits.items():
        n_real = sum(1 for m in metas if m["kind"] == "real")
        n_synth = len(metas) - n_real
        total = sum(m["n_windows"] for m in metas)
        print(f"  {name}: {len(metas)} series ({n_real} real, {n_synth} synth), "
              f"{total} windows")

    # --- Фаза 2: запись ---
    print("\n=== Writing splits (memmap) ===")
    results = {}
    for name, metas in splits.items():
        if not metas:
            continue
        print(f"  {name} → {args.output / name}")
        results[name] = _write_split(args.output / name, metas,
                                     args.window, args.horizon)
        if results[name]:
            r = results[name]
            print(f"    OK: {r['n_windows_actual']} windows, "
                  f"{r['n_series']} series")

    # --- Metadata ---
    metadata = {
        "version": 3,
        "created_by": "build_nn_dataset.py",
        "window": args.window,
        "horizon_days": args.horizon,
        "bin_size_days": BIN_SIZE_DAYS,
        "n_bins": N_BINS,
        "censored_bin": CENSORED_BIN,
        "features": FEATURES,
        "include_synthetic_batch": args.include_synthetic_batch,
        "max_synthetic": args.max_synthetic,
        "splits": {name: {
            "n_series": len(metas),
            "n_windows": results.get(name, {}).get("n_windows_actual", 0),
        } for name, metas in splits.items() if metas},
        "storage_note": "X.npy — несжатый, поддерживает np.load(..., mmap_mode='r')",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Bin distribution (train) ---
    train_dir = args.output / "train"
    if (train_dir / "y_bin.npy").exists():
        y_bin = np.load(train_dir / "y_bin.npy")
        dist = np.bincount(y_bin, minlength=CENSORED_BIN + 1)
        lines = ["Train bin distribution:", ""]
        for b, c in enumerate(dist):
            label = ("CENSORED" if b == CENSORED_BIN
                     else f"[{b*BIN_SIZE_DAYS+1}-{(b+1)*BIN_SIZE_DAYS}]d")
            pct = c / len(y_bin) if len(y_bin) else 0
            lines.append(f"  {b:2d} ({label:>12}): {c:7d} ({pct:6.1%})")
        text = "\n".join(lines)
        (args.output / "bin_distribution.txt").write_text(text, encoding="utf-8")
        print("\n" + text)

    print(f"\nMetadata: {args.output / 'metadata.json'}")
    print(f"Готово: {args.output}")


if __name__ == "__main__":
    main()