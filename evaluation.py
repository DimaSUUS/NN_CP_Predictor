"""Единые метрики качества временных рядов.

Модуль содержит только вычисление метрик: без чтения файлов, логирования в CLI
или визуализации. Его используют core, CLI и отчёты, чтобы определения совпадали.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

import config


def summarize_outliers(df: pd.DataFrame) -> dict[str, int]:
    """Возвращает согласованные счётчики флагов выбросов и CP-кандидатов.

    is_outlier — исходные точки, помеченные robust detector до кластерной эвристики.
    is_isolated_outlier — короткие группы выбросов, оставленные после кластеризации.
    cp_candidate — маркеры центров длинных групп выбросов; это не подтверждённые CP.
    """
    def count_flag(column: str) -> int:
        if column not in df:
            return 0
        values = pd.to_numeric(df[column], errors="coerce").fillna(0)
        return int((values == 1).sum())

    return {
        "outliers": count_flag("is_outlier"),
        "isolated_outliers": count_flag("is_isolated_outlier"),
        "cp_candidates": count_flag("cp_candidate"),
        "points": int(len(df)),
    }


def evaluate_cp(true_cps: Iterable[int], pred_cps: Iterable[int],
                tolerance: int | None = None) -> dict:
    """One-to-one оценка change points с максимальным числом совпадений.

    Пустые truth и predictions дают F1=1 (нет событий и нет ложных сигналов).
    Если одна из сторон непуста, а совпадений нет, соответствующие метрики равны 0.
    """
    tolerance = config.CP_EVAL_TOLERANCE if tolerance is None else int(tolerance)
    if tolerance < 0:
        raise ValueError("tolerance должен быть >= 0")
    true = sorted(set(int(x) for x in true_cps))
    pred = sorted(set(int(x) for x in pred_cps))
    n_true, n_pred = len(true), len(pred)
    if not n_true and not n_pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "mean_offset": None, "n_true": 0, "n_pred": 0, "tp": 0}

    # DP: максимум one-to-one совпадений, затем минимум суммарного смещения.
    dp: list[list[tuple[int, int, list[tuple[int, int]]]]] = [
        [(0, 0, []) for _ in range(n_pred + 1)] for _ in range(n_true + 1)]
    for i in range(1, n_true + 1):
        for j in range(1, n_pred + 1):
            options = [dp[i - 1][j], dp[i][j - 1]]
            distance = abs(true[i - 1] - pred[j - 1])
            if distance <= tolerance:
                prev = dp[i - 1][j - 1]
                options.append((prev[0] + 1, prev[1] + distance,
                                prev[2] + [(i - 1, j - 1)]))
            dp[i][j] = max(options, key=lambda item: (item[0], -item[1]))
    matches = dp[n_true][n_pred][2]
    offsets = [abs(true[i] - pred[j]) for i, j in matches]
    tp = len(matches)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3),
            "mean_offset": round(float(np.mean(offsets)), 2) if offsets else None,
            "n_true": n_true, "n_pred": n_pred, "tp": tp}
