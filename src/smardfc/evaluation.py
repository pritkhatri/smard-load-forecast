"""Evaluation protocol: expanding-window cross-validation plus a final test period.

* the last `test_days` days are never used for fitting, model choice or tuning
* the earlier "development" days are cut into `n_folds` consecutive validation windows
  of `val_days` days; fold i is trained on everything before its window (expanding)
* all models are scored on the same rows (target and both baselines available)
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics
from .features import FEATURES
from .models import LGBMModel, NaiveColumn

BASELINES = ("naive_day", "naive_week", "naive_mean4w")
REFERENCE = "naive_week"


@dataclass
class Split:
    dev_days: list
    test_days: list
    folds: list  # list of (train_days, val_days)
    train_days: list | None = None  # days used to fit the model scored on the test period


def make_split(
    days: list, test_days: int = 90, n_folds: int = 5, val_days: int = 60, purge_days: int = 0
) -> Split:
    """Chronological split. `purge_days` removes that many days between the end of every
    training set and the start of the window it is scored on. With purge_days=0 the
    afternoon targets of the last training day lie after the first scored day's cut-off
    (their labels were not yet known when that forecast was issued); purge_days=1 removes it."""
    days = list(days)
    dev, test = days[:-test_days], days[-test_days:]
    folds = []
    for i in range(n_folds):
        v_end = len(dev) - (n_folds - 1 - i) * val_days
        v_start = v_end - val_days
        folds.append((dev[: v_start - purge_days], dev[v_start:v_end]))
    return Split(dev, test, folds, train_days=dev[: len(dev) - purge_days])


def model_zoo(include_nn: bool = True) -> dict[str, Callable]:
    zoo: dict[str, Callable] = {
        "naive_day": lambda: NaiveColumn("naive_day"),
        "naive_week": lambda: NaiveColumn("naive_week"),
        "naive_mean4w": lambda: NaiveColumn("same_hour_mean_4w"),
        "lgbm": lambda: LGBMModel(residual=False),
        "lgbm_resid": lambda: LGBMModel(residual=True),
    }
    if include_nn and importlib.util.find_spec("torch") is not None:
        from .nn import MLPModel

        zoo["mlp"] = lambda: MLPModel()
    return zoo


def _rows(feats: pd.DataFrame, days) -> pd.DataFrame:
    return feats[feats["day"].isin(set(days))]


def _scorable(df: pd.DataFrame) -> pd.Series:
    return df["y"].notna() & df["naive_week"].notna() & df["naive_day"].notna()


def fit_predict(factory, train: pd.DataFrame, test: pd.DataFrame):
    model = factory()
    tr = train[train["y"].notna()]
    model.fit(tr[FEATURES + ["naive_day", "naive_week"]], tr["y"].to_numpy())
    return model, model.predict(test[FEATURES + ["naive_day", "naive_week"]])


def predict_frame(feats: pd.DataFrame, train_days, test_days, zoo) -> tuple[pd.DataFrame, dict]:
    train, test = _rows(feats, train_days), _rows(feats, test_days)
    out = test[["day", "lhour", "y"]].copy()
    fitted = {}
    for name, factory in zoo.items():
        fitted[name], out[name] = fit_predict(factory, train, test)
    return out[_scorable(test)], fitted


def score(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Overall MAE/RMSE/MAPE per model, with skill against each seasonal-naive baseline."""
    rows = {m: metrics.summary(preds["y"], preds[m]) for m in models}
    df = pd.DataFrame(rows).T
    for b in (b for b in BASELINES if b in df.index):
        df[f"skill_vs_{b}"] = [metrics.skill(df.loc[m, "MAE"], df.loc[b, "MAE"]) for m in df.index]
    return df


def run_cv(feats: pd.DataFrame, split: Split, zoo) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-fold scores, pooled out-of-fold predictions)."""
    fold_rows, pooled = [], []
    for i, (train_days, val_days) in enumerate(split.folds, 1):
        preds, _ = predict_frame(feats, train_days, val_days, zoo)
        sc = score(preds, list(zoo)).reset_index(names="model")
        sc.insert(0, "fold", i)
        sc.insert(1, "val_start", str(val_days[0].date()))
        sc.insert(2, "val_end", str(val_days[-1].date()))
        fold_rows.append(sc)
        preds = preds.assign(fold=i)
        pooled.append(preds)
    return pd.concat(fold_rows, ignore_index=True), pd.concat(pooled)


def cv_summary(folds: pd.DataFrame) -> pd.DataFrame:
    cols = ["MAE", "RMSE", "MAPE", f"skill_vs_{REFERENCE}"]
    g = folds.groupby("model")[cols]
    out = pd.concat({"mean": g.mean(), "std": g.std(ddof=1)}, axis=1)
    out.columns = [f"{c}_{s}" for s, c in out.columns]
    order = list(dict.fromkeys(folds["model"]))
    return out.loc[order]


def run_test(feats: pd.DataFrame, split: Split, zoo):
    """Fit on all development days, score once on the held-out test days."""
    train_days = split.train_days if split.train_days is not None else split.dev_days
    preds, fitted = predict_frame(feats, train_days, split.test_days, zoo)
    return preds, fitted


def error_by(preds: pd.DataFrame, model: str, key: str) -> pd.DataFrame:
    df = preds.copy()
    if key == "weekday":
        df["weekday"] = pd.to_datetime(df["day"]).dt.dayofweek
    return metrics.by_group(df, model, "lhour" if key == "hour" else key)


def best_and_worst_week(preds: pd.DataFrame, model: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start dates (Mondays) of the full test week with the lowest / highest MAE."""
    df = preds.copy()
    df["day"] = pd.to_datetime(df["day"])
    df["week"] = df["day"].dt.to_period("W-SUN").dt.start_time
    g = df.groupby("week")
    counts = g["y"].count()
    full = counts[counts >= 7 * 23].index
    mae = g.apply(lambda x: metrics.mae(x["y"], x[model]), include_groups=False).loc[full]
    return mae.idxmin(), mae.idxmax()


def stats_dict(df: pd.DataFrame) -> dict:
    return {k: {c: (None if np.isnan(v) else float(v)) for c, v in r.items()}
            for k, r in df.to_dict("index").items()}  # fmt: skip
