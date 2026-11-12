"""Графики и сводки по обработанным временным рядам."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import summarize_outliers
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)
COLORS = {"change_point": "#d62728", "outlier": "#ff7f0e", "seasonal_event": "#2ca02c"}


def _safe_y(df: pd.DataFrame, date, col: str = "value"):
    if date not in df.index or col not in df:
        return np.nan
    y = df.loc[date, col]
    return y.iloc[0] if isinstance(y, pd.Series) else y


def _ensure_plot_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="raise")
    required = {"value", "trend", "seasonal", "resid", "detrended"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Для графика не хватает колонок: {sorted(missing)}")
    return out.sort_index()


def plot_series(df: pd.DataFrame, series_name: str, save_path,
                ground_truth_cps=None, events=None):
    df = _ensure_plot_columns(df)
    if df.empty:
        logger.warning("Пропускаю пустой график %s", series_name)
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_cps = ground_truth_cps or []
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    idx = df.index
    ax = axes[0]
    ax.plot(idx, df["value"], lw=0.9, color="steelblue", label="value")
    if "is_outlier" in df and df["is_outlier"].sum() > 0:
        out = df[df["is_outlier"] == 1]
        ax.scatter(out.index, out["value"], color="red", s=20, zorder=5,
                   label=f"outliers ({len(out)})")
    if events is not None and not events.empty:
        for _, event in events.iterrows():
            if event.get("type") == "outlier":
                date = pd.Timestamp(event["date"]).normalize()
                if date in idx:
                    ax.scatter(date, _safe_y(df, date), color=COLORS["outlier"], s=50,
                               zorder=6, edgecolor="black", lw=0.6)
    if "is_holiday" in df and df["is_holiday"].sum() > 0:
        hol = df[df["is_holiday"] == 1]
        ax.scatter(hol.index, hol["value"], color="gold", s=8, alpha=0.6, zorder=4)
    ax.set_title(f"{series_name} — value")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    for axis, col, color in [(axes[1], "trend", "orange"),
                             (axes[2], "seasonal", "green"),
                             (axes[3], "resid", "purple")]:
        axis.plot(idx, df[col], color=color, lw=1.0 if col == "trend" else 0.7)
        if col == "resid":
            axis.axhline(0, color="black", lw=0.5, alpha=0.5)
        axis.set_title(col)
        axis.grid(alpha=0.3)
    axes[4].plot(idx, df["detrended"], color="crimson", lw=0.9)
    axes[4].set_title("detrended = value − seasonal")
    axes[4].grid(alpha=0.3)
    for cp in ground_truth_cps:
        if 0 <= int(cp) < len(df):
            for axis in axes:
                axis.axvline(idx[int(cp)], color="black", ls="--", lw=1.0, alpha=0.6)
    if ground_truth_cps:
        axes[0].text(0.99, 1.02, f"true CPs: {ground_truth_cps}", transform=axes[0].transAxes,
                     ha="right", va="bottom", fontsize=9, color="black")
    if "predicted_cp" in df:
        for ts in df.index[df["predicted_cp"].fillna(0).astype(int) == 1]:
            for axis in axes:
                axis.axvline(ts, color="blue", ls=":", lw=0.9, alpha=0.6)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_ground_truth(df: pd.DataFrame, events: pd.DataFrame, series_name: str, save_path):
    df = _ensure_plot_columns(df)
    if df.empty:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    events = events if events is not None else pd.DataFrame(columns=["date", "type"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    idx = df.index
    ax = axes[0]
    ax.plot(idx, df["value"], lw=0.9, color="steelblue", label="value")
    n_cp = n_out = n_seas = 0
    for _, event in events.iterrows():
        date = pd.Timestamp(event["date"]).normalize()
        if date < idx.min().normalize() or date > idx.max().normalize():
            continue
        kind = event["type"]
        if kind == "change_point":
            ax.axvline(date, color=COLORS["change_point"], ls="--", lw=1.4, alpha=0.8)
            n_cp += 1
        elif kind == "outlier" and date in idx:
            ax.scatter(date, _safe_y(df, date), color=COLORS["outlier"], s=60,
                       zorder=5, edgecolor="black", lw=0.6)
            n_out += 1
        elif kind == "seasonal_event" and date in idx:
            ax.scatter(date, _safe_y(df, date), color=COLORS["seasonal_event"], s=30,
                       marker="^", zorder=4, alpha=0.6)
            n_seas += 1
    legend = [
        Line2D([0], [0], color=COLORS["change_point"], ls="--", lw=1.5, label=f"change_point ({n_cp})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["outlier"], markersize=8,
               markeredgecolor="black", label=f"outlier ({n_out})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=COLORS["seasonal_event"],
               markersize=8, label=f"seasonal_event ({n_seas})"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    ax.set_title(f"{series_name} — ground truth ({len(events)} events)")
    ax.grid(alpha=0.3)
    if "resid" in df:
        axes[1].plot(idx, df["resid"], lw=0.6, color="purple", alpha=0.8)
        axes[1].axhline(0, color="black", lw=0.4, alpha=0.5)
        for _, event in events[events["type"] == "change_point"].iterrows():
            axes[1].axvline(pd.Timestamp(event["date"]), color=COLORS["change_point"],
                            ls="--", lw=1.0, alpha=0.6)
        axes[1].set_title("resid (STL) with change points")
        axes[1].grid(alpha=0.3)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def summarize(df: pd.DataFrame, series_name: str, gt_cps):
    dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    def stat(col, fn, default=0.0):
        return round(float(fn(pd.to_numeric(df[col], errors="coerce"))), 4) if col in df and len(df) else default
    outlier_metrics = summarize_outliers(df)
    return {
        "series": series_name, "points": len(df),
        "date_from": dates.min().strftime("%Y-%m-%d") if len(dates) else None,
        "date_to": dates.max().strftime("%Y-%m-%d") if len(dates) else None,
        "value_mean": stat("value", lambda x: x.mean()),
        "value_std": stat("value", lambda x: x.std()),
        "value_min": stat("value", lambda x: x.min()),
        "value_max": stat("value", lambda x: x.max()),
        "outliers": outlier_metrics["outliers"],
        "isolated_outliers": outlier_metrics["isolated_outliers"],
        "cp_candidates": outlier_metrics["cp_candidates"],
        "predicted_cps": int(df["predicted_cp"].sum()) if "predicted_cp" in df else 0,
        "holidays": int(df["is_holiday"].sum()) if "is_holiday" in df else 0,
        "weekends": int(df["is_weekend"].sum()) if "is_weekend" in df else 0,
        "paydays": int(df["is_payday"].sum()) if "is_payday" in df else 0,
        "n_gt_cps": len(gt_cps),
    }
