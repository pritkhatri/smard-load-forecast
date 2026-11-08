"""End-to-end commands: data, evaluate, train. Used by the CLI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from . import __version__, plots
from . import evaluation as ev
from .data import clean, fetch_smard, load_csv, load_processed, save_processed
from .features import FEATURES, build_features, feature_days
from .models import LGBM_PARAMS, LGBMModel

MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT = "smard-load-forecast"
# Served model: best CV MAE among the LightGBM models (the API image has no PyTorch, so the MLP
# is not served). The test results had already been seen when this was decided; see README.
HEADLINE_MODEL = "lgbm"


def prepare_data(
    region="Amprion", start="2022-01-01", end=None, csv=None, refresh=False, out="data/processed"
):
    s = load_csv(csv) if csv else fetch_smard(region=region, start=start, end=end, refresh=refresh)
    s, report = clean(s)
    save_processed(s, report, out)
    return s, report


def _mlflow_setup():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)


DAY_KINDS = ["normal weekday", "weekend", "holiday", "bridge day", "day next to holiday"]


def day_kinds(feats: pd.DataFrame) -> pd.Series:
    """Label every target day: NRW holiday, bridge day, weekend, day next to a holiday, other."""
    f = feats.groupby("day")[["is_holiday", "is_bridge", "holiday_prev", "holiday_next", "dow"]]
    d = f.first()
    kind = np.select(
        [d.is_holiday == 1, d.is_bridge == 1, d.dow >= 5, (d.holiday_prev + d.holiday_next) > 0],
        ["holiday", "bridge day", "weekend", "day next to holiday"],
        default="normal weekday",
    )
    return pd.Series(kind, index=d.index, name="kind")


def error_by_daytype(preds: pd.DataFrame, kinds: pd.Series, names: list[str], scope: str):
    df = preds.assign(kind=pd.to_datetime(preds["day"]).map(kinds))
    rows = []
    for kind, g in df.groupby("kind"):
        for m in names:
            row = {"scope": scope, "kind": kind, "model": m}
            row |= {"MAE": ev.metrics.mae(g["y"], g[m]), "n_hours": len(g)}
            rows.append(row | {"n_days": g["day"].nunique()})
    return pd.DataFrame(rows)


def best_naive(cv: pd.DataFrame) -> str:
    naive = [m for m in cv.index if m.startswith("naive")]
    return min(naive, key=lambda m: cv.loc[m, "MAE_mean"])


def write_metrics_json(out, meta, cv, folds, test, by_hour_test, by_hour_cv, best):
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, float)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    ref = ev.REFERENCE
    payload = {
        "meta": meta,
        "best_naive_baseline_by_cv_mae": best,
        "skill_reference": ref,
        "cv_summary": cv.to_dict("index"),
        "cv_folds": folds.to_dict("records"),
        "test": test.to_dict("index"),
        "test_by_hour": {m: t.to_dict("index") for m, t in by_hour_test.items()},
        "cv_by_hour": {m: t.to_dict("index") for m, t in by_hour_cv.items()},
    }
    (out / "metrics.json").write_text(json.dumps(clean(payload), indent=2))


def evaluate(
    data_path="data/processed/load.parquet",
    results="results",
    use_nn=True,
    test_days=90,
    n_folds=5,
    val_days=60,
    until: str | None = None,
    purge_days: int = 0,
) -> dict:
    load = load_processed(data_path)
    if until:  # keep only data before this local date, e.g. to backtest an earlier window
        load = load[load.index < pd.Timestamp(until).tz_localize("Europe/Berlin")]
    out = plots.ensure(results)
    days = feature_days(load)
    feats = build_features(load, days)
    split = ev.make_split(
        days, test_days=test_days, n_folds=n_folds, val_days=val_days, purge_days=purge_days
    )
    zoo = ev.model_zoo(include_nn=use_nn)
    names = list(zoo)

    folds, pooled = ev.run_cv(feats, split, zoo)
    cv = ev.cv_summary(folds)
    test_preds, fitted = ev.run_test(feats, split, zoo)
    test = ev.score(test_preds, names)

    folds.to_csv(out / "cv_folds.csv", index=False)
    cv.to_csv(out / "cv_summary.csv")
    test.to_csv(out / "test_summary.csv")
    test_preds.assign(timestamp_utc=test_preds.index.astype(str)).to_csv(
        out / "test_predictions.csv", index=False
    )

    by_hour_test = {m: ev.error_by(test_preds, m, "hour") for m in names}
    by_wd_test = {m: ev.error_by(test_preds, m, "weekday") for m in names}
    by_hour_cv = {m: ev.error_by(pooled, m, "hour") for m in names}
    pd.concat(by_hour_test, names=["model"]).to_csv(out / "test_by_hour.csv")
    pd.concat(by_wd_test, names=["model"]).to_csv(out / "test_by_weekday.csv")
    pd.concat(by_hour_cv, names=["model"]).to_csv(out / "cv_by_hour.csv")

    tp = test_preds.assign(month=pd.to_datetime(test_preds["day"]).dt.strftime("%Y-%m"))
    pd.concat({m: ev.metrics.by_group(tp, m, "month") for m in names}, names=["model"]).to_csv(
        out / "test_by_month.csv"
    )

    imp = fitted[HEADLINE_MODEL].importance()
    imp.rename("gain").to_csv(out / "feature_importance.csv")

    shown = [m for m in names if m != "naive_day"]
    plots.plot_error_by_hour(
        {m: by_hour_test[m] for m in shown}, out / "error_by_hour_test.png",
        f"Test period: MAE by target hour ({split.test_days[0].date()} to "
        f"{split.test_days[-1].date()})")  # fmt: skip
    plots.plot_error_by_hour(
        {m: by_hour_cv[m] for m in shown}, out / "error_by_hour_cv.png",
        "Cross-validation (all folds pooled): MAE by target hour")  # fmt: skip
    plots.plot_error_by_weekday(
        {m: by_wd_test[m] for m in shown}, out / "error_by_weekday_test.png",
        "Test period: MAE by weekday")  # fmt: skip
    plots.plot_cv_folds(folds, out / "cv_folds.png")
    plots.plot_importance(
        imp, out / "feature_importance.png", f"{HEADLINE_MODEL}: feature importance (gain)"
    )
    best, worst = ev.best_and_worst_week(test_preds, HEADLINE_MODEL)
    plots.plot_week(test_preds, best, HEADLINE_MODEL, out / "week_best.png",
                    f"Best test week (starting {best.date()}): {HEADLINE_MODEL}")  # fmt: skip
    plots.plot_week(test_preds, worst, HEADLINE_MODEL, out / "week_worst.png",
                    f"Worst test week (starting {worst.date()}): {HEADLINE_MODEL}")  # fmt: skip

    meta = {
        "version": __version__,
        "data_first": str(load.index[0]),
        "data_last": str(load.index[-1]),
        "n_feature_days": len(days),
        "test_start": str(split.test_days[0].date()),
        "test_end": str(split.test_days[-1].date()),
        "cv_folds": [
            {"fold": i, "train_days": len(t), "val_start": str(v[0].date()),
             "val_end": str(v[-1].date())}
            for i, (t, v) in enumerate(split.folds, 1)
        ],
        "worst_week": str(worst.date()),
        "best_week": str(best.date()),
        "models": names,
        "until": until,
        "purge_days": purge_days,
    }  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    report = Path(data_path).parent / "validation_report.json"
    if report.exists():
        shutil.copy(report, out / "data_validation.json")
    (out / "results.md").write_text(_markdown(cv, test, meta))
    best = best_naive(cv)
    if best != ev.REFERENCE:
        print(f"warning: best naive baseline by CV MAE is {best}, but skill uses {ev.REFERENCE}")
    kinds = day_kinds(feats)
    pd.concat(
        [
            error_by_daytype(pooled, kinds, names, "cv"),
            error_by_daytype(test_preds, kinds, names, "test"),
        ]
    ).to_csv(out / "error_by_daytype.csv", index=False)
    write_metrics_json(out, meta, cv, folds, test, by_hour_test, by_hour_cv, best)
    _log_mlflow(names, folds, cv, test, split, out)
    return {"cv": cv, "test": test, "meta": meta}


def _fmt(x, nd=0):
    return f"{x:,.{nd}f}"


def _markdown(cv: pd.DataFrame, test: pd.DataFrame, meta: dict) -> str:
    lines = [
        f"Test period {meta['test_start']} to {meta['test_end']} "
        "(held out from fitting and tuning).",
        "MAE and RMSE in MW, MAPE in %. Skill = 1 - MAE / MAE of the reference baseline.",
        "",
        f"| model | CV MAE (mean +- std over {len(meta['cv_folds'])} folds) "
        "| CV skill vs naive_week "
        "| test MAE | test RMSE | test MAPE | test skill vs naive_week |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in cv.index:
        c, t = cv.loc[m], test.loc[m]
        lines.append(
            f"| {m} | {_fmt(c['MAE_mean'])} +- {_fmt(c['MAE_std'])} "
            f"| {c['skill_vs_naive_week_mean']:+.3f} +- {c['skill_vs_naive_week_std']:.3f} "
            f"| {_fmt(t['MAE'])} | {_fmt(t['RMSE'])} | {t['MAPE']:.2f} "
            f"| {t['skill_vs_naive_week']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def _log_mlflow(names, folds, cv, test, split, out: Path):
    _mlflow_setup()
    for m in names:
        with mlflow.start_run(run_name=f"{m}"):
            mlflow.set_tags({"model": m, "version": __version__, "results_dir": str(out)})
            mlflow.log_params({"model": m, "n_features": len(FEATURES), **_params(m)})
            for _, r in folds[folds["model"] == m].iterrows():
                mlflow.log_metric("cv_fold_MAE", r["MAE"], step=int(r["fold"]))
            for k in ("MAE", "RMSE", "MAPE"):
                mlflow.log_metric(f"cv_{k}_mean", float(cv.loc[m, f"{k}_mean"]))
            mlflow.log_metric("cv_MAE_std", float(cv.loc[m, "MAE_std"]))
            for k in ("MAE", "RMSE", "MAPE", "skill_vs_naive_week", "skill_vs_naive_day"):
                mlflow.log_metric(f"test_{k}", float(test.loc[m, k]))
            for f in ("cv_summary.csv", "test_summary.csv", "meta.json"):
                mlflow.log_artifact(str(out / f))


def _params(m: str) -> dict:
    if m.startswith("lgbm"):
        return {k: v for k, v in LGBM_PARAMS.items() if k != "verbose"} | {
            "residual": m == "lgbm_resid"
        }
    return {}


def train(data_path="data/processed/load.parquet", model_path="models/model.joblib") -> Path:
    """Fit the headline model on ALL available days and save it for the API.

    The metrics in results/ come from `evaluate`, whose model never sees the test
    period. This final model additionally uses the test period, so there is no
    held-out score for it.
    """
    load = load_processed(data_path)
    feats = build_features(load, feature_days(load))
    feats = feats[feats["y"].notna()]
    model = LGBMModel(residual=HEADLINE_MODEL == "lgbm_resid").fit(feats, feats["y"].to_numpy())
    bundle = {
        "model": model,
        "name": HEADLINE_MODEL,
        "trained_until": str(load.index[-1]),
        "version": __version__,
    }
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    _mlflow_setup()
    with mlflow.start_run(run_name="train_final"):
        mlflow.log_params({"model": HEADLINE_MODEL, "trained_until": bundle["trained_until"]})
        mlflow.log_artifact(str(path))
    return path
