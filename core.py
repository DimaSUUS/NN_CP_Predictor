"""Базовые операции над временными рядами и офлайн-детекция change points.

Важно: build_pipeline() и mark_outliers() в режиме по умолчанию являются
ретроспективными (offline). Их признаки нельзя напрямую подавать в онлайн-
прогноз, если они рассчитаны по всему временному ряду.
OnlineDetector — причинный (causal) детектор, использующий только прошлое.
"""
from __future__ import annotations

import importlib.util
import logging
import time
from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

import config
from evaluation import evaluate_cp, summarize_outliers

try:
    import holidays  # type: ignore
except ImportError:  # календарные признаки не должны ломать остальной pipeline
    holidays = None

logger = logging.getLogger(__name__)


def _has_module(name: str) -> bool:
    """Проверка доступности пакета без его импорта (безопасно для opz. зависимостей)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Базовая валидация и offline-разложение
# --------------------------------------------------------------------------- #
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="raise")
    if out.index.hasnans:
        raise ValueError("DatetimeIndex содержит NaT")
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError("Индекс содержит повторяющиеся временные метки")
    return out


def stl_decompose(df: pd.DataFrame, value_col: str = "value", period: int = 7,
                  seasonal: int | None = None, robust: bool = True) -> pd.DataFrame:
    """Добавить STL trend/seasonal/resid. Для period < 2 STL отключён."""
    df = _ensure_datetime_index(df)
    if value_col not in df:
        raise KeyError(f"Нет столбца {value_col!r}")
    values = pd.to_numeric(df[value_col], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{value_col!r} содержит пропуски или нечисловые значения")
    period = int(period or 0)
    if period < 2 or len(df) < 2 * period:
        df["trend"] = values.astype(float)
        df["seasonal"] = 0.0
        df["resid"] = 0.0
        logger.info("STL skipped (period=%s, n=%s)", period, len(df))
        return df
    if seasonal is None:
        seasonal = min(7, period if period % 2 else period + 1)
    seasonal = int(seasonal)
    if seasonal < 3 or seasonal % 2 == 0:
        raise ValueError("Параметр STL seasonal должен быть нечётным целым >= 3")
    result = STL(values.astype(float), period=period, seasonal=seasonal,
                 robust=robust).fit()
    df["trend"] = result.trend
    df["seasonal"] = result.seasonal
    df["resid"] = result.resid
    return df


# --------------------------------------------------------------------------- #
# Offline выбросы и кластеризация
# --------------------------------------------------------------------------- #
def mark_outliers(df: pd.DataFrame, resid_col: str = "resid", window: int | None = None,
                  z_threshold: float | None = None, causal: bool = False) -> pd.DataFrame:
    """Пометить выбросы по robust z-score.

    causal=False — offline (центрированные окна и глобальная fallback-оценка).
    causal=True  — только прошлые наблюдения, без сигнала до накопления истории.
    """
    df = df.copy()
    window = config.OUTLIER_WINDOW if window is None else int(window)
    z_threshold = config.OUTLIER_Z if z_threshold is None else float(z_threshold)
    if window < 3:
        raise ValueError("window должен быть >= 3")
    if z_threshold <= 0:
        raise ValueError("z_threshold должен быть > 0")
    if resid_col not in df:
        df["is_outlier"] = 0
        return df
    r = pd.to_numeric(df[resid_col], errors="coerce").astype(float)
    if r.empty or r.isna().all():
        df["is_outlier"] = 0
        return df
    if causal:
        history = r.shift(1)
        min_periods = window
        med = history.rolling(window, min_periods=min_periods).median()
        abs_dev = (history - med).abs()
        sigma = 1.4826 * abs_dev.rolling(window, min_periods=min_periods).median()
        valid = sigma.where(sigma > np.finfo(float).eps)
        z = (r - med).abs() / valid
        flags = (z > z_threshold) & med.notna() & valid.notna()
    else:
        global_median = float(r.median())
        global_mad = float((r - global_median).abs().median())
        global_sigma = 1.4826 * global_mad
        if not np.isfinite(global_sigma) or global_sigma <= np.finfo(float).eps:
            global_sigma = float(r.std(ddof=1))
        if not np.isfinite(global_sigma) or global_sigma <= np.finfo(float).eps:
            df["is_outlier"] = 0
            return df
        local_med = r.rolling(window=window, center=True, min_periods=window).median()
        local_mad = (r - local_med).abs().rolling(window=window, center=True,
                                                  min_periods=window).median()
        local_sigma = 1.4826 * local_mad
        med = local_med.fillna(global_median)
        sigma = local_sigma.where(local_sigma > np.finfo(float).eps, global_sigma)
        z = (r - med).abs() / sigma
        flags = z > z_threshold
    df["is_outlier"] = flags.fillna(False).astype(int)
    counts = summarize_outliers(df)
    logger.info("Outliers flagged: %s / %s", counts["outliers"], counts["points"])
    return df


def cluster_outliers(df: pd.DataFrame, min_cp_len: int = 5):
    """Короткие серии выбросов остаются outlier; длинные — кандидат CP.

    Это эвристика, а не полноценный CP-детектор. В упрощённом пайплайне
    отключена по умолчанию; если нужна — включите cfg["cluster_outliers"]=True.
    """
    if min_cp_len < 1:
        raise ValueError("min_cp_len должен быть >= 1")
    df = df.copy()
    labels = pd.to_numeric(df.get("is_outlier", pd.Series(0, index=df.index)),
                           errors="coerce").fillna(0).to_numpy(dtype=int)
    labels = (labels == 1).astype(int)
    new_outlier = np.zeros(len(labels), dtype=int)
    candidates: list[int] = []
    i = 0
    while i < len(labels):
        if labels[i] == 0:
            i += 1
            continue
        j = i + 1
        while j < len(labels) and labels[j] == 1:
            j += 1
        if j - i < min_cp_len:
            new_outlier[i:j] = 1
        else:
            candidates.append((i + j) // 2)
        i = j
    df["is_isolated_outlier"] = new_outlier
    df["cp_candidate"] = 0
    for cp in candidates:
        if 0 <= cp < len(df):
            df.iloc[cp, df.columns.get_loc("cp_candidate")] = 1
    return df, candidates


# --------------------------------------------------------------------------- #
# Календарные признаки
# --------------------------------------------------------------------------- #
def add_calendar_features(df: pd.DataFrame, country: str | None = None,
                          weekends: bool = False, paydays: bool = False,
                          tax_days: bool = False) -> pd.DataFrame:
    df = _ensure_datetime_index(df)
    idx = df.index
    if len(idx) == 0:
        for col in ("is_holiday", "is_weekend", "is_payday", "is_tax_day",
                    "is_month_start", "is_month_end", "is_quarter_end", "day_of_week",
                    "is_special_day"):
            df[col] = pd.Series(index=idx, dtype="int64")
        return df
    df["is_holiday"] = 0
    if country and holidays is not None:
        try:
            years = range(int(idx.year.min()), int(idx.year.max()) + 1)
            holiday_map = (holidays.NYSE(years=years) if country == "NYSE"
                           else holidays.country_holidays(country, years=years))
            df["is_holiday"] = np.fromiter((d in holiday_map for d in idx.date),
                                           dtype=np.int8, count=len(idx))
        except Exception as exc:
            logger.warning("Не удалось вычислить праздники %s: %s", country, exc)
    elif country and holidays is None:
        logger.warning("Пакет holidays не установлен; is_holiday будет равен 0")
    df["is_weekend"] = idx.dayofweek.isin([5, 6]).astype(int) if weekends else 0
    df["is_payday"] = idx.day.isin([10, 11, 12, 24, 25, 26]).astype(int) if paydays else 0
    df["is_tax_day"] = idx.day.isin([25, 28]).astype(int) if tax_days else 0
    df["is_month_start"] = idx.is_month_start.astype(int)
    df["is_month_end"] = idx.is_month_end.astype(int)
    df["is_quarter_end"] = idx.is_quarter_end.astype(int)
    df["day_of_week"] = idx.dayofweek
    df["is_special_day"] = ((df["is_holiday"] == 1) | (df["is_weekend"] == 1) |
                            (df["is_payday"] == 1) | (df["is_tax_day"] == 1)).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_pipeline(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Offline pipeline; проверяет вход и добавляет признаки."""
    df = _ensure_datetime_index(df)
    if "value" not in df:
        raise KeyError("Входной DataFrame должен содержать столбец 'value'")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df.empty:
        raise ValueError("Нельзя обработать пустой временной ряд")
    if df["value"].isna().any() or not np.isfinite(df["value"].to_numpy(dtype=float)).all():
        raise ValueError("Столбец value содержит пропуски или нечисловые значения")
    stl_period = cfg.get("stl_period", 7) or 0
    df = stl_decompose(df, period=stl_period)
    # Если STL отключён (period<2), resid=0; флаги строим по наблюдаемому value.
    outlier_col = "value" if stl_period < 2 or len(df) < 2 * stl_period else "resid"
    df = mark_outliers(df, resid_col=outlier_col, causal=False)
    # Кластерная эвристика — опциональна, финальный список CP даёт detect_for_series.
    if cfg.get("cluster_outliers", False):
        df, _ = cluster_outliers(df, min_cp_len=config.CP_CLUSTER_MIN)
    else:
        df["is_isolated_outlier"] = 0
        df["cp_candidate"] = 0
    df = add_calendar_features(df, country=cfg.get("country"),
                               weekends=cfg.get("weekends", False),
                               paydays=cfg.get("paydays", False),
                               tax_days=cfg.get("tax_days", False))
    df["detrended"] = df["value"] - df["seasonal"]
    df["diff"] = df["detrended"].diff()
    return df


# --------------------------------------------------------------------------- #
# Детекция change points: три backend-а
# --------------------------------------------------------------------------- #
def _fallback_detection(x: np.ndarray, min_size: int, model: str = "l2",
                        penalty: float = 10.0) -> list[int]:
    """Точный DP fallback для l2/normal (O(n²)); используется по умолчанию."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    if n < 2 * min_size or np.ptp(x) <= np.finfo(float).eps:
        return []
    if model not in {"l2", "normal"}:
        raise ValueError(f"Fallback не поддерживает model={model!r}")
    cs = np.r_[0.0, np.cumsum(x)]
    cs2 = np.r_[0.0, np.cumsum(x * x)]
    eps = max(float(np.var(x)) * 1e-12, np.finfo(float).eps)
    best = np.full(n + 1, np.inf)
    prev = np.full(n + 1, -1, dtype=int)
    best[0] = 0.0
    for end in range(min_size, n + 1):
        starts = [0, *range(min_size, end - min_size + 1)] if end >= 2 * min_size else [0]
        for start in starts:
            if start != 0 and prev[start] < 0:
                continue
            length = end - start
            total = cs[end] - cs[start]
            ss = cs2[end] - cs2[start]
            var = max((ss - total * total / length) / length, eps)
            cost = (ss - total * total / length) if model == "l2" else length * np.log(var)
            candidate = best[start] + cost + (penalty if start > 0 else 0.0)
            if candidate < best[end]:
                best[end], prev[end] = candidate, start
    if prev[n] < 0:
        return []
    cps = []
    end = n
    while end > 0:
        start = int(prev[end])
        if start < 0:
            return []
        if start > 0:
            cps.append(start)
        end = start
    return sorted(cps)


def _rustures_detection(x: np.ndarray, min_size: int, model: str,
                        penalty: float) -> list[int]:
    """Опциональный адаптер для Rust-библиотеки `rustures`.

    ВАЖНО: rustures — не drop-in замена ruptures. API похож (fit/predict),
    но между версиями может отличаться. Адаптер намеренно защитный.
    Установка: pip install rustures
    """
    try:
        import rustures  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Backend rustures запрошен, но пакет не установлен. "
                           "Установите: pip install rustures") from exc
    try:
        algo = rustures.Pelt(model=model, min_size=min_size)
        fitted = algo.fit(x.reshape(-1, 1))
        result = fitted.predict(pen=penalty)
    except Exception as exc:
        raise RuntimeError(f"Ошибка вызова rustures.Pelt (model={model!r}): {exc}") from exc
    return [int(cp) for cp in result if 0 < int(cp) < len(x)]


def _ruptures_rs_detection(x: np.ndarray, min_size: int, model: str,
                           penalty: float) -> list[int]:
    """Legacy-адаптер для ruptures-rs (не входит в обязательные зависимости)."""
    try:
        import ruptures_rs as rpt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Backend ruptures_rs запрошен, но пакет не установлен. "
                           "Установите: pip install ruptures-rs") from exc
    try:
        algo = rpt.Pelt(model=model, min_size=min_size).fit(x.reshape(-1, 1))
        result = algo.predict(pen=penalty)
    except Exception as exc:
        raise RuntimeError(f"Ошибка вызова ruptures_rs.Pelt (model={model!r}): {exc}") from exc
    if not result or result[-1] != len(x):
        raise RuntimeError(f"ruptures_rs вернул некорректные границы: {result!r}")
    return [int(cp) for cp in result[:-1] if 0 < int(cp) < len(x)]


def detect_change_points(series: Iterable[float], penalty: float,
                         min_size: int | None = None, model: str = "l2",
                         backend: str = "fallback") -> list[int]:
    """Найти границы сегментов; возвращаем индексы начала нового сегмента.

    backend:
      - "fallback" (default) — встроенный DP, без внешних зависимостей;
      - "rustures" — Rust-библиотека, если установлена;
      - "ruptures_rs" — legacy, если установлена;
      - "auto" — rustures → ruptures_rs → fallback (с предупреждением).
    """
    min_size = config.CP_MIN_SIZE if min_size is None else int(min_size)
    penalty = float(penalty)
    if min_size < 2:
        raise ValueError("min_size должен быть >= 2")
    if penalty < 0:
        raise ValueError("penalty должен быть >= 0")
    if backend not in {"auto", "fallback", "rustures", "ruptures_rs"}:
        raise ValueError("backend должен быть auto, fallback, rustures или ruptures_rs")
    x = np.asarray(series, dtype=float).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("В series есть NaN или бесконечные значения")
    n = len(x)
    if n < 2 * min_size or np.ptp(x) <= np.finfo(float).eps:
        return []
    std = float(x.std())
    x = (x - x.mean()) / std if std > 0 else x - x.mean()

    if backend == "fallback":
        return _fallback_detection(x, min_size, model=model, penalty=penalty)
    if backend == "rustures":
        return _rustures_detection(x, min_size, model=model, penalty=penalty)
    if backend == "ruptures_rs":
        return _ruptures_rs_detection(x, min_size, model=model, penalty=penalty)

    # auto: предпочитаем ускоренные backend-ы, если они реально установлены
    if _has_module("rustures"):
        return _rustures_detection(x, min_size, model=model, penalty=penalty)
    if _has_module("ruptures_rs"):
        return _ruptures_rs_detection(x, min_size, model=model, penalty=penalty)
    logger.warning("Ни rustures, ни ruptures_rs не установлены; использую DP fallback")
    return _fallback_detection(x, min_size, model=model, penalty=penalty)


def _deduplicate_cps(cps: Iterable[int], tolerance: int) -> list[int]:
    values = sorted(set(int(x) for x in cps))
    if not values:
        return []
    groups: list[list[int]] = [[values[0]]]
    for cp in values[1:]:
        if cp - groups[-1][-1] <= tolerance:
            groups[-1].append(cp)
        else:
            groups.append([cp])
    return [int(round(float(np.median(group)))) for group in groups]


def detect_for_series(df: pd.DataFrame, cfg: dict | None = None, debug: bool = False,
                      backend: str = "fallback", penalties: dict | None = None) -> list[int]:
    """Offline CP detection по сигналу, соответствующему семантике ряда.

    - log_returns: и mean-, и variance-проходы на лог-доходностях;
    - иначе: mean на detrended levels + variance на первых разностях.
    Параметр `penalties` позволяет переопределить config.* из калибратора.
    """
    cfg = cfg or {}
    penalties = penalties or {}
    p_mean = float(penalties.get("mean", config.CP_PENALTY_MEAN))
    p_var = float(penalties.get("variance", config.CP_PENALTY_VAR))
    if df.empty or "value" not in df:
        return []

    raw = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("value содержит NaN или бесконечные значения")

    passes: dict[str, list[int]] = {}
    if cfg.get("detection_space") == "log_returns":
        if np.any(raw <= 0):
            raise ValueError("log_returns требует строго положительных значений")
        signal = np.diff(np.log(raw))
        offset = 1
        if len(signal) >= 2 * config.CP_MIN_SIZE:
            passes["mean"] = [cp + offset for cp in detect_change_points(
                signal, p_mean, model="l2", backend=backend)]
            passes["variance"] = [cp + offset for cp in detect_change_points(
                signal, p_var, model="normal", backend=backend)]
    else:
        if "detrended" in df:
            values = pd.to_numeric(df["detrended"], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).all():
                passes["mean"] = detect_change_points(values, p_mean, model="l2", backend=backend)
        increments = np.diff(raw)
        if len(increments) >= 2 * config.CP_MIN_SIZE:
            passes["variance"] = [cp + 1 for cp in detect_change_points(
                increments, p_var, model="normal", backend=backend)]

    if debug:
        logger.info("%s: detector passes=%s", cfg.get("kind", "unknown"), passes)
    return _deduplicate_cps((cp for found in passes.values() for cp in found),
                            config.CP_DEDUP_TOL)


# --------------------------------------------------------------------------- #
# Online-детектор (причинный, только прошлые наблюдения)
# --------------------------------------------------------------------------- #
class OnlineDetector:
    """Causal per-step детектор выбросов/сдвигов.

    Использует rolling median/MAD по ПРЕДЫДУЩИМ значениям; текущее значение
    в оценку не входит. Готов к потоковой подаче без пересчёта всего ряда.
    """

    def __init__(self, window: int | None = None, z_threshold: float | None = None,
                 min_history: int | None = None):
        self.window = int(window if window is not None else config.OUTLIER_WINDOW)
        self.z_threshold = float(z_threshold if z_threshold is not None else config.OUTLIER_Z)
        self.min_history = int(min_history if min_history is not None
                               else max(10, self.window // 2))
        if self.window < 3:
            raise ValueError("window должен быть >= 3")
        if self.z_threshold <= 0:
            raise ValueError("z_threshold должен быть > 0")
        if self.min_history < 2:
            raise ValueError("min_history должен быть >= 2")
        self._history: deque = deque(maxlen=self.window)

    def update(self, value) -> dict:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return {"alarm": False, "z": None, "reason": "not_numeric"}
        if not np.isfinite(v):
            return {"alarm": False, "z": None, "reason": "not_finite"}
        self._history.append(v)
        if len(self._history) < self.min_history:
            return {"alarm": False, "z": None, "reason": "warmup"}
        prior = np.asarray(self._history)[:-1]  # строго прошлое
        med = float(np.median(prior))
        mad = float(np.median(np.abs(prior - med)))
        sigma = 1.4826 * mad
        if sigma <= np.finfo(float).eps:
            return {"alarm": False, "z": 0.0, "reason": "no_variance"}
        z = abs(v - med) / sigma
        return {"alarm": bool(z > self.z_threshold), "z": float(z), "reason": "ok"}

    def reset(self) -> None:
        self._history.clear()


# --------------------------------------------------------------------------- #
# Ground truth, события, календарь, поток
# --------------------------------------------------------------------------- #
def load_events(series_id: str | None = None, include_calendar: bool | None = None) -> pd.DataFrame:
    include_calendar = config.USE_CALENDAR_EVENTS if include_calendar is None else include_calendar
    sources = [(config.MANUAL_EVENTS, 1)]
    if include_calendar:
        sources.append((config.CALENDAR_EVENTS, 2))
    frames = []
    for path, priority in sources:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = {"date", "series_id", "type"}
        if not required.issubset(frame.columns):
            logger.warning("Пропускаю файл событий %s: нужны столбцы %s", path, sorted(required))
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date", "series_id", "type"])
        frame["_priority"] = priority
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "series_id", "type"])
    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values("_priority").drop_duplicates(
        subset=["date", "series_id", "type"], keep="first")
    events = events.drop(columns="_priority").sort_values(["series_id", "date"])
    if series_id is not None:
        events = events[events["series_id"].astype(str) == str(series_id)]
    return events.reset_index(drop=True)


def load_ground_truth() -> dict[str, list[int]]:
    if not config.GROUND_TRUTH_CSV.exists():
        return {}
    gt = pd.read_csv(config.GROUND_TRUTH_CSV)
    if not {"series_id", "change_points"}.issubset(gt.columns):
        raise ValueError(f"В {config.GROUND_TRUTH_CSV} нужны series_id и change_points")
    result: dict[str, list[int]] = {}
    for _, row in gt.iterrows():
        raw = row["change_points"]
        if pd.isna(raw) or str(raw).strip() == "":
            points = []
        else:
            points = [int(part.strip()) for part in str(raw).split(";") if part.strip()]
        result[str(row["series_id"])] = points
    return result


def generate_calendar_events() -> int:
    """Автогенерация календарных событий (за флагом в config)."""
    if not config.GENERATE_CALENDAR_EVENTS:
        logger.info("generate_calendar_events отключён (config.GENERATE_CALENDAR_EVENTS=False)")
        return 0
    all_events = []
    for series_id, cfg in config.SERIES_CONFIG.items():
        path = config.PROCESSED_DIR / f"{series_id}.csv"
        if not path.exists() or not any((cfg.get("country"), cfg.get("weekends"),
                                         cfg.get("paydays"), cfg.get("tax_days"))):
            continue
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        if frame.empty or "timestamp" not in frame:
            continue
        idx = pd.DatetimeIndex(frame["timestamp"].dropna().sort_values().unique())
        if len(idx) == 0:
            continue
        holiday_map = {}
        if cfg.get("country") and holidays is not None:
            try:
                years = range(int(idx.year.min()), int(idx.year.max()) + 1)
                h = (holidays.NYSE(years=years) if cfg["country"] == "NYSE"
                     else holidays.country_holidays(cfg["country"], years=years))
                holiday_map = dict(h)
            except Exception as exc:
                logger.warning("Праздники %s: %s", series_id, exc)
        for ts in idx:
            flags = []
            if ts.date() in holiday_map:
                flags.append(f"holiday:{holiday_map[ts.date()]}")
            if cfg.get("weekends") and ts.dayofweek in (5, 6): flags.append("weekend")
            if cfg.get("paydays") and ts.day in (10, 11, 12, 24, 25, 26): flags.append("payday")
            if cfg.get("tax_days") and ts.day in (25, 28): flags.append("tax_day")
            if flags:
                all_events.append({"date": ts.normalize(), "series_id": series_id,
                                   "type": "seasonal_event", "direction": "", "magnitude": "",
                                   "source": "auto:holidays+calendar", "description": "+".join(flags)})
    config.ensure_dirs()
    if not all_events:
        pd.DataFrame(columns=["date", "series_id", "type", "direction", "magnitude", "source", "description"]).to_csv(config.CALENDAR_EVENTS, index=False)
        logger.info("Calendar events: none")
        return 0
    result = pd.DataFrame(all_events).drop_duplicates(subset=["date", "series_id", "type"])
    result.to_csv(config.CALENDAR_EVENTS, index=False)
    logger.info("Calendar events: %s rows -> %s", len(result), config.CALENDAR_EVENTS)
    return len(result)


class StreamSimulator:
    """Симулятор поступления точек; сам по себе не выполняет детекцию."""
    def __init__(self, df: pd.DataFrame, value_col: str = "value",
                 delay_seconds: float = 0.0, start_from: int = 0):
        if delay_seconds < 0:
            raise ValueError("delay_seconds должен быть >= 0")
        if start_from < 0:
            raise ValueError("start_from должен быть >= 0")
        if value_col not in df:
            raise KeyError(f"Нет столбца {value_col!r}")
        self.df = _ensure_datetime_index(df).iloc[start_from:].copy()
        self.value_col = value_col
        self.delay_seconds = float(delay_seconds)

    def stream(self):
        for ts, row in self.df.iterrows():
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            yield ts, row[self.value_col]