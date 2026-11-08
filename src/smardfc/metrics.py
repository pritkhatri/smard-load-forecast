"""Forecast error metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mape(y, p) -> float:
    """Mean absolute percentage error in percent (load is always > 0)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p) / np.abs(y)) * 100)


def skill(mae_model: float, mae_reference: float) -> float:
    """1 - MAE_model / MAE_reference. 0 = no better than the reference, 1 = perfect."""
    return float(1.0 - mae_model / mae_reference)


def summary(y, p) -> dict:
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p), "n": int(len(y))}


def by_group(df: pd.DataFrame, pred_col: str, group: str) -> pd.DataFrame:
    """MAE / RMSE / MAPE per value of `group` (e.g. local hour, weekday)."""
    rows = []
    for key, g in df.groupby(group):
        rows.append({group: key, **summary(g["y"], g[pred_col])})
    return pd.DataFrame(rows).set_index(group)
