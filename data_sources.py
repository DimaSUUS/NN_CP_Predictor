"""Источники данных: ЦБ РФ, yfinance и синтетические ряды."""
from __future__ import annotations

import logging
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)
CBR_BASE = "https://www.cbr.ru/scripts"
CBR_CODES = {"USD": "R01235", "EUR": "R01239", "CNY": "R01375"}


def _empty_source_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name="timestamp")
    return pd.DataFrame({"value": pd.Series(dtype=float),
                         "series_id": pd.Series(dtype=str),
                         "source": pd.Series(dtype=str)}, index=idx)


# --------------------------------------------------------------------------- #
# CBR
# --------------------------------------------------------------------------- #
def fetch_cbr_currency(currency: str, date_from: str = "2020-01-01",
                       date_to: str | None = None) -> pd.DataFrame:
    if currency not in CBR_CODES:
        raise ValueError(f"Unsupported currency: {currency}")
    start = pd.to_datetime(date_from)
    end = pd.Timestamp.now().normalize() if date_to is None else pd.to_datetime(date_to)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("Некорректный диапазон дат")
    response = requests.get(
        f"{CBR_BASE}/XML_dynamic.asp",
        params={"date_req1": start.strftime("%d/%m/%Y"),
                "date_req2": end.strftime("%d/%m/%Y"),
                "VAL_NM_RQ": CBR_CODES[currency]},
        headers={"User-Agent": "Mozilla/5.0 (compatible; time-series-cp/1.0)"},
        timeout=30,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    records = []
    for record in root.findall("Record"):
        value_node, nominal_node = record.find("Value"), record.find("Nominal")
        date_text = record.get("Date")
        if value_node is None or not value_node.text or not date_text:
            continue
        raw_value = float(value_node.text.replace(" ", "").replace(",", "."))
        nominal = float(nominal_node.text) if nominal_node is not None and nominal_node.text else 1.0
        if not np.isfinite(raw_value) or not np.isfinite(nominal) or nominal <= 0:
            continue
        records.append({"timestamp": pd.to_datetime(date_text, format="%d.%m.%Y"),
                        "value": raw_value / nominal,
                        "series_id": f"{currency}/RUB", "source": "cbr"})
    if not records:
        logger.warning("CBR %s: нет записей за указанный период", currency)
        return _empty_source_frame()
    df = pd.DataFrame(records).drop_duplicates("timestamp").set_index("timestamp").sort_index()
    logger.info("CBR %s: %s rows", currency, len(df))
    return df


# --------------------------------------------------------------------------- #
# yfinance (опционален)
# --------------------------------------------------------------------------- #
def _extract_ticker_frame(raw: pd.DataFrame, ticker: str, multi_ticker: bool) -> pd.DataFrame:
    """Нормализовать варианты колонок yfinance, включая MultiIndex для одного тикера."""
    frame = raw
    if isinstance(raw.columns, pd.MultiIndex):
        levels = [list(raw.columns.get_level_values(i).unique()) for i in range(raw.columns.nlevels)]
        ticker_level = next((i for i, vals in enumerate(levels) if ticker in vals), None)
        if ticker_level is not None:
            try:
                frame = raw.xs(ticker, axis=1, level=ticker_level, drop_level=True)
            except KeyError:
                frame = raw
        if isinstance(frame.columns, pd.MultiIndex):
            close_cols = [col for col in frame.columns if any(str(part).lower() == "close" for part in col)]
            if close_cols:
                frame = frame.loc[:, close_cols[0]]
        elif "Close" in frame.columns:
            frame = frame["Close"]
    elif multi_ticker and ticker in raw.columns:
        frame = raw[ticker]
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name="value")
    if "value" not in frame.columns:
        if "Close" in frame.columns:
            frame = frame.rename(columns={"Close": "value"})
        elif len(frame.columns) == 1:
            frame = frame.rename(columns={frame.columns[0]: "value"})
        else:
            raise ValueError(f"Не удалось определить Close для {ticker}: {frame.columns.tolist()}")
    return frame[["value"]]


def fetch_yfinance(tickers: list[str], start: str = "2020-01-01",
                   end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    if not tickers:
        return _empty_source_frame()
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, interval=interval,
                      group_by="ticker", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        logger.warning("yfinance: нет данных для %s", tickers)
        return _empty_source_frame()
    raw = raw.copy()
    if not isinstance(raw.index, pd.DatetimeIndex):
        raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw.index.name = "timestamp"
    frames = []
    for ticker in tickers:
        frame = _extract_ticker_frame(raw, ticker, multi_ticker=len(tickers) > 1)
        frame = frame.copy()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.loc[frame["value"].notna() & np.isfinite(frame["value"].to_numpy(dtype=float))]
        frame["series_id"] = ticker
        frame["source"] = "yfinance"
        frames.append(frame)
    if not frames:
        return _empty_source_frame()
    result = pd.concat(frames, axis=0).sort_index()
    result.index.name = "timestamp"
    return result


# --------------------------------------------------------------------------- #
# Синтетические ряды (общие утилиты)
# --------------------------------------------------------------------------- #
def _validate_change_points(n_points: int, change_points: list[int] | None) -> list[int]:
    cps = [n_points // 2] if change_points is None else [int(cp) for cp in change_points]
    if n_points < 1:
        raise ValueError("n_points должен быть положительным")
    if cps != sorted(cps) or len(set(cps)) != len(cps):
        raise ValueError("change_points должны быть отсортированы и уникальны")
    if any(cp <= 0 or cp >= n_points for cp in cps):
        raise ValueError("Каждая точка разладки должна удовлетворять 0 < cp < n_points")
    return cps


def _segments(change_points: list[int], n_points: int):
    bounds = [0, *change_points, n_points]
    for i in range(len(bounds) - 1):
        yield i, bounds[i], bounds[i + 1]


def _build_timestamps(n_points: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n_points, freq="D", name="timestamp")


def generate_synthetic_series(n_points: int = 1000, change_points: list[int] | None = None,
                              mu_before: float = 0.0, mu_after: float = 5.0,
                              sigma_before: float = 1.0, sigma_after: float = 1.0,
                              seed: int | None = None, series_id: str = "synthetic",
                              mu_values: list[float] | None = None,
                              sigma_values: list[float] | None = None):
    """Сгенерировать гауссовы сегменты; change point — индекс начала нового сегмента."""
    cps = _validate_change_points(int(n_points), change_points)
    segment_count = len(cps) + 1
    if mu_values is not None and len(mu_values) != segment_count:
        raise ValueError(f"mu_values должен иметь длину {segment_count}")
    if sigma_values is not None and len(sigma_values) != segment_count:
        raise ValueError(f"sigma_values должен иметь длину {segment_count}")
    mus = list(mu_values) if mu_values is not None else [mu_before] + [mu_after] * len(cps)
    sigmas = list(sigma_values) if sigma_values is not None else [sigma_before] + [sigma_after] * len(cps)
    if any(not np.isfinite(float(s)) or float(s) <= 0 for s in sigmas):
        raise ValueError("Все sigma должны быть конечными и > 0")
    rng = np.random.default_rng(seed)
    values = np.empty(int(n_points), dtype=float)
    for seg_idx, start, end in _segments(cps, int(n_points)):
        values[start:end] = rng.normal(float(mus[seg_idx]), float(sigmas[seg_idx]), end - start)
    df = pd.DataFrame({"value": values, "series_id": series_id, "source": "synthetic"},
                      index=_build_timestamps(int(n_points)))
    logger.info("Synthetic '%s': %s pts, CPs=%s", series_id, n_points, cps)
    return df, cps


def generate_random_walk(n_points: int = 1000, change_points: list[int] | None = None,
                         mu_before: float = 0.0005, mu_after: float = -0.002,
                         sigma: float = 0.01, start_price: float = 100.0,
                         seed: int | None = None, series_id: str = "synth_rw"):
    cps = _validate_change_points(int(n_points), change_points)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma должен быть конечным и > 0")
    if not np.isfinite(start_price) or start_price <= 0:
        raise ValueError("start_price должен быть конечным и > 0")
    rng = np.random.default_rng(seed)
    returns = np.empty(int(n_points), dtype=float)
    for seg_idx, start, end in _segments(cps, int(n_points)):
        mu = mu_before if seg_idx == 0 else mu_after
        returns[start:end] = rng.normal(mu, sigma, end - start)
    prices = start_price * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"value": prices, "series_id": series_id, "source": "synthetic"},
                      index=_build_timestamps(int(n_points)))
    logger.info("Synthetic random walk '%s': %s pts, CPs=%s", series_id, n_points, cps)
    return df, cps

def generate_random_walk_multisegment(
    n_points: int,
    change_points: list[int],
    mus: list[float],
    sigma: float,
    start_price: float = 100.0,
    seed: int | None = None,
    series_id: str = "synth_rw_multi",
):
    """Random walk с несколькими drift-сегментами (для trend-сценариев NN)."""
    cps = _validate_change_points(int(n_points), change_points)
    if len(mus) != len(cps) + 1:
        raise ValueError(f"mus должен иметь длину {len(cps) + 1}")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma должен быть конечным и > 0")
    if not np.isfinite(start_price) or start_price <= 0:
        raise ValueError("start_price должен быть конечным и > 0")
    rng = np.random.default_rng(seed)
    returns = np.empty(int(n_points), dtype=float)
    for seg_idx, start, end in _segments(cps, int(n_points)):
        returns[start:end] = rng.normal(float(mus[seg_idx]), sigma, end - start)
    prices = start_price * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"value": prices, "series_id": series_id, "source": "synthetic"},
                      index=_build_timestamps(int(n_points)))
    logger.info("Synthetic RW multi '%s': %s pts, CPs=%s", series_id, n_points, cps)
    return df, cps

def _sample_cps_via_gaps(n_points: int, n_cps: int, min_gap: int,
                         rng: np.random.Generator) -> list[int] | None:
    """Сгенерировать n_cps позиций с зазором >= min_gap между ними и краями.

    Работает через семплирование длин (n_cps + 1) сегментов:
    сначала каждому даём min_gap, остаток «свободного места» разбиваем
    экспоненциально (даёт вариативность зазоров), затем накапливаем.

    Гарантированно валидно, если n_points >= (n_cps + 1) * min_gap.
    Никакого rejection —成功率 100%.
    """
    n_segments = n_cps + 1
    min_total = n_segments * min_gap
    if n_points < min_total:
        return None
    free = n_points - min_total
    if free > 0:
        # Экспоненциальные веса → разная длина сегментов, а не одинаковые
        raw = rng.exponential(1.0, n_segments)
        props = raw / raw.sum()
        extras = (free * props).astype(int)
        # Остаток от округления — в первый сегмент
        diff = free - int(extras.sum())
        extras[0] += diff
    else:
        extras = np.zeros(n_segments, dtype=int)

    lengths = [min_gap + int(e) for e in extras]
    cps = []
    pos = 0
    for i in range(n_cps):
        pos += lengths[i]
        cps.append(int(pos))
    return cps

# --------------------------------------------------------------------------- #
# Массовая генерация для pre-training NN
# --------------------------------------------------------------------------- #
def generate_synthetic_batch(
    n_series: int = 1000,
    n_points_range: tuple[int, int] = (800, 1500),      # ← было (1000, 3000)
    n_cps_range: tuple[int, int] = (3, 8),              # ← было (2, 6)
    min_gap: int = 60,                                   # ← было 120
    output_dir: Path | None = None,
    seed: int = 42,
) -> Path:
    """Сгенерировать пачку синтетических рядов для pre-training NN.

    Плотные CP (min_gap=60, 3-8 на ряд, длина 800-1500) дают ~30% censored,
    а не 85% как при редких CP. Это критично для баланса классов.

    Каждый ряд — случайный сценарий (mean / variance / trend), случайная
    длина и число CP. Trend теперь с несколькими reversal (n_cps>=3), чтобы
    не упираться в 1 CP на ряд и не давать 90% censored.

    Сохраняет:
      - {output_dir}/{series_id}.csv
      - {output_dir}/ground_truth.csv
    """
    output_dir = Path(output_dir) if output_dir else (config.RAW_DIR / "synthetic_batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    gt_records = []
    n_done = 0

    for i in range(n_series):
        n_points = int(rng.integers(n_points_range[0], n_points_range[1] + 1))
        scenario = str(rng.choice(["mean", "variance", "trend"]))

        # Все сценарии, включая trend, получают n_cps из общего диапазона.
        # Для trend это несколько drift reversal — реалистично для финансов.
        n_cps = int(rng.integers(n_cps_range[0], n_cps_range[1] + 1))

        cps = _sample_cps_via_gaps(n_points, n_cps, min_gap, rng)
        if cps is None:
            continue   # только если ряд физически слишком короткий для n_cps

        sid = f"synth_batch_{i:04d}_{scenario}"
        rw_seed = int(rng.integers(0, 2**31))

        if scenario == "mean":
            mu_values = [float(rng.normal(0, 3)) for _ in range(n_cps + 1)]
            for j in range(1, len(mu_values)):
                if abs(mu_values[j] - mu_values[j - 1]) < 1.5:
                    mu_values[j] = mu_values[j - 1] + float(rng.choice([-1, 1])) * 3.0
            sigma = float(rng.uniform(0.5, 2.0))
            frame, truth = generate_synthetic_series(
                n_points=n_points, change_points=cps,
                mu_values=mu_values,
                sigma_values=[sigma] * (n_cps + 1),
                seed=rw_seed, series_id=sid)

        elif scenario == "variance":
            mu = float(rng.normal(0, 1))
            sigma_values = [float(rng.uniform(0.3, 2.5)) for _ in range(n_cps + 1)]
            for j in range(1, len(sigma_values)):
                if abs(sigma_values[j] - sigma_values[j - 1]) < 0.5:
                    sigma_values[j] = sigma_values[j - 1] * 2.0
            frame, truth = generate_synthetic_series(
                n_points=n_points, change_points=cps,
                mu_values=[mu] * (n_cps + 1),
                sigma_values=sigma_values,
                seed=rw_seed, series_id=sid)

        else:  # trend: несколько drift reversal
            # Чередуем знак drift, чтобы каждая CP была видимым reversal.
            mus = []
            current = float(rng.normal(0, 0.002))
            for _ in range(n_cps + 1):
                mus.append(current)
                current = current + float(rng.choice([-1, 1])) * float(rng.uniform(0.005, 0.010))
            sigma = float(rng.uniform(0.005, 0.02))
            start_price = float(rng.uniform(50, 200))
            frame, truth = generate_random_walk_multisegment(
                n_points=n_points, change_points=cps,
                mus=mus, sigma=sigma, start_price=start_price,
                seed=rw_seed, series_id=sid)

        frame[["value"]].to_csv(output_dir / f"{sid}.csv", index_label="timestamp")
        gt_records.append({
            "series_id": sid,
            "change_points": ";".join(map(str, truth)),
            "scenario": scenario,
            "n_points": n_points,
            "n_cps": len(truth),
        })
        n_done += 1

    if gt_records:
        pd.DataFrame(gt_records).to_csv(output_dir / "ground_truth.csv", index=False)
    logger.info("Synthetic batch: %s series → %s", n_done, output_dir)
    print(f"Synthetic batch: {n_done} series, ground truth → {output_dir / 'ground_truth.csv'}")
    return output_dir


# --------------------------------------------------------------------------- #
# Сборка всех источников
# --------------------------------------------------------------------------- #
def collect_all() -> dict[str, dict]:
    """Собрать источники независимо, сохранить явный статус успех/сбой.

    Провал загрузки НЕ перезаписывает существующий CSV. Манифест статусов
    позволяет process отличить свежий файл от старого после сбоя.
    yfinance помечается как "skipped" (а не "failed"), если пакет не установлен.
    """
    import json
    from datetime import datetime, timezone

    config.ensure_dirs()
    run_time = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, dict] = {"run_at_utc": run_time, "sources": {}}

    # --- CBR ---
    frames = []
    cbr_failures = []
    for currency in ("USD", "EUR", "CNY"):
        try:
            frame = fetch_cbr_currency(currency)
            if not frame.empty:
                frames.append(frame)
                manifest["sources"][f"cbr_{currency}"] = {"status": "success", "rows": len(frame)}
            else:
                cbr_failures.append(currency)
                manifest["sources"][f"cbr_{currency}"] = {"status": "empty", "rows": 0}
        except Exception as exc:
            cbr_failures.append(currency)
            manifest["sources"][f"cbr_{currency}"] = {"status": "failed", "error": str(exc)}
            logger.warning("CBR %s: %s", currency, exc)
    cbr_path = config.RAW_DIR / "cbr_currency.csv"
    if frames:
        cbr = pd.concat(frames).sort_index()
        cbr.to_csv(cbr_path, index_label="timestamp")
        manifest["sources"]["cbr_currency.csv"] = {
            "status": "partial" if cbr_failures else "success", "rows": len(cbr),
            "failed_currencies": cbr_failures, "file": cbr_path.name}
    else:
        manifest["sources"]["cbr_currency.csv"] = {
            "status": "failed", "rows": 0, "failed_currencies": cbr_failures,
            "file": cbr_path.name, "stale_file_retained": cbr_path.exists()}
        logger.warning("CBR: нет свежих данных; существующий файл не перезаписан")

    # --- yfinance (опционален) ---
    market_path = config.RAW_DIR / "yfinance_data.csv"
    try:
        import yfinance  # noqa: F401
        yf_available = True
    except ImportError:
        yf_available = False

    if not yf_available:
        manifest["sources"]["yfinance_data.csv"] = {
            "status": "skipped",
            "reason": "yfinance not installed; install with `pip install yfinance`",
            "file": market_path.name,
            "stale_file_retained": market_path.exists(),
        }
        logger.info("yfinance не установлен — рыночные данные пропущены "
                    "(это не ошибка, установите пакет при необходимости)")
    else:
        try:
            market = fetch_yfinance(["SPY", "AAPL", "BTC-USD"])
            if market.empty:
                manifest["sources"]["yfinance_data.csv"] = {
                    "status": "empty", "rows": 0, "file": market_path.name,
                    "stale_file_retained": market_path.exists()}
                logger.warning("yfinance: пустой ответ; существующий файл не перезаписан")
            else:
                market.to_csv(market_path, index_label="timestamp")
                manifest["sources"]["yfinance_data.csv"] = {
                    "status": "success", "rows": len(market), "file": market_path.name,
                    "series": sorted(market["series_id"].unique().tolist())}
        except Exception as exc:
            manifest["sources"]["yfinance_data.csv"] = {
                "status": "failed", "error": str(exc), "file": market_path.name,
                "stale_file_retained": market_path.exists()}
            logger.warning("yfinance: %s", exc)

    # --- Базовые синтетические сценарии (для baseline + регрессии) ---
    gt_records = []
    scenarios = [
        ("synth_mu_shift", generate_synthetic_series,
         dict(n_points=1000, change_points=[300, 700],
              mu_values=[0.0, 5.0, 0.0], sigma_values=[1.0, 1.0, 1.0], seed=42)),
        ("synth_sigma_shift", generate_synthetic_series,
         dict(n_points=1000, change_points=[500],
              mu_before=0.0, mu_after=0.0, sigma_before=1.0, sigma_after=3.0, seed=43)),
        ("synth_random_walk", generate_random_walk,
         dict(n_points=1000, change_points=[400],
              mu_before=0.003, mu_after=-0.005, sigma=0.01,
              start_price=100.0, seed=44)),
    ]
    for sid, generator, kwargs in scenarios:
        frame, cps = generator(series_id=sid, **kwargs)
        frame.to_csv(config.RAW_DIR / f"{sid}.csv", index_label="timestamp")
        gt_records.append({"series_id": sid,
                           "change_points": ";".join(map(str, cps)),
                           "type": {"synth_mu_shift": "mean_shift",
                                    "synth_sigma_shift": "variance_shift",
                                    "synth_random_walk": "trend_reversal"}[sid]})
        manifest["sources"][f"{sid}.csv"] = {"status": "success", "rows": len(frame),
                                             "synthetic": True}
    pd.DataFrame(gt_records).to_csv(config.GROUND_TRUTH_CSV, index=False)

    # --- Манифест ---
    status_path = config.REPORTS_DIR / "collection_status.json"
    status_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    failed = [name for name, info in manifest["sources"].items()
              if info.get("status") in {"failed", "empty"}]
    if failed:
        logger.warning("Collection finished with failed/empty sources: %s; see %s",
                       failed, status_path)
    else:
        logger.info("Data collection completed successfully; see %s", status_path)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Источники данных и синтетика.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("collect", help="Собрать CBR + yfinance + базовую синтетику")

    p_batch = sub.add_parser("synthetic-batch",
                             help="Сгенерировать пачку синтетики для pre-training NN")
    p_batch.add_argument("--n-series", type=int, default=1000)
    p_batch.add_argument("--min-points", type=int, default=1000)
    p_batch.add_argument("--max-points", type=int, default=3000)
    p_batch.add_argument("--min-cps", type=int, default=2)
    p_batch.add_argument("--max-cps", type=int, default=6)
    p_batch.add_argument("--min-gap", type=int, default=120)
    p_batch.add_argument("--output", type=Path,
                         default=config.RAW_DIR / "synthetic_batch")
    p_batch.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    config.ensure_dirs()

    if args.command == "collect":
        collect_all()
    elif args.command == "synthetic-batch":
        generate_synthetic_batch(
            n_series=args.n_series,
            n_points_range=(args.min_points, args.max_points),
            n_cps_range=(args.min_cps, args.max_cps),
            min_gap=args.min_gap,
            output_dir=args.output,
            seed=args.seed,
        )