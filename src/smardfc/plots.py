"""Result figures (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {
    "naive_day": "#999999", "naive_week": "#555555", "naive_mean4w": "#8c564b", "lgbm": "#1f77b4",
    "lgbm_resid": "#d62728", "mlp": "#2ca02c",
}  # fmt: skip
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_error_by_hour(tables: dict[str, pd.DataFrame], path, title: str):
    """MAE per target hour of day; horizon = hour + 13 h after the last observed hour."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, t in tables.items():
        ax.plot(t.index, t["MAE"], marker="o", ms=3, label=m, color=COLORS.get(m))
    ax.set_xlabel(
        "target hour of day (local); horizon = hour + 13 h after the last observed hour (11:00)"
    )
    ax.set_ylabel("MAE [MW]")
    ax.set_title(title)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path)


def plot_error_by_weekday(tables: dict[str, pd.DataFrame], path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    n = len(tables)
    width = 0.8 / n
    for i, (m, t) in enumerate(tables.items()):
        ax.bar(t.index + (i - (n - 1) / 2) * width, t["MAE"], width, label=m, color=COLORS.get(m))
    ax.set_xticks(range(7), WD)
    ax.set_ylabel("MAE [MW]")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path)


def plot_week(preds: pd.DataFrame, week_start, model: str, path, title: str):
    from .data import TZ

    start = pd.Timestamp(week_start)
    df = preds.copy()
    df["day"] = pd.to_datetime(df["day"])
    df = df[(df["day"] >= start) & (df["day"] < start + pd.Timedelta(days=7))]
    x = df.index.tz_convert(TZ).tz_localize(None)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(x, df["y"], color="black", lw=1.4, label="actual")
    ax.plot(x, df[model], color=COLORS.get(model), lw=1.2, label=model)
    ax.plot(x, df["naive_week"], color="#888888", lw=0.9, ls="--", label="naive_week")
    ax.set_ylabel("load [MW]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    _save(fig, path)


def plot_importance(imp: pd.Series, path, title: str, top: int = 15):
    imp = (imp / imp.sum() * 100).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(imp.index, imp.values, color=COLORS["lgbm"])
    ax.set_xlabel("share of total gain [%]")
    ax.set_title(title)
    _save(fig, path)


def plot_cv_folds(folds: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, g in folds.groupby("model", sort=False):
        ax.plot(g["fold"], g["MAE"], marker="o", label=m, color=COLORS.get(m))
    ax.set_xlabel("CV fold (later folds = more recent validation window)")
    ax.set_ylabel("MAE [MW]")
    ax.set_xticks(sorted(folds["fold"].unique()))
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("MAE per cross-validation fold")
    _save(fig, path)


def ensure(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
