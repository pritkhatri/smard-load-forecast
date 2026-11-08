"""Integration test: the whole evaluate / train / API path on small synthetic data."""

import json

import pandas as pd
from fastapi.testclient import TestClient

from smardfc import pipeline
from smardfc.api import create_app
from smardfc.data import DataReport, save_processed


def test_evaluate_and_train_end_to_end(tmp_path, monkeypatch, synthetic_load):
    monkeypatch.chdir(tmp_path)  # mlflow.db and results/ go to the temp dir
    save_processed(synthetic_load, DataReport(), "data/processed")
    res = pipeline.evaluate(use_nn=False, test_days=14, n_folds=2, val_days=20, results="results")
    for name in ("results.md", "cv_folds.csv", "test_summary.csv", "week_worst.png",
                 "feature_importance.png", "error_by_hour_test.png", "meta.json"):  # fmt: skip
        assert (tmp_path / "results" / name).exists(), name
    meta = json.loads((tmp_path / "results" / "meta.json").read_text())
    assert meta["test_start"] > meta["cv_folds"][-1]["val_end"]  # test comes after all CV data
    assert set(res["test"].index) == {
        "naive_day",
        "naive_week",
        "naive_mean4w",
        "lgbm",
        "lgbm_resid",
    }
    assert (tmp_path / "mlflow.db").exists()
    mj = json.loads((tmp_path / "results" / "metrics.json").read_text())
    assert mj["best_naive_baseline_by_cv_mae"].startswith("naive")
    assert mj["test"]["lgbm"]["MAE"] == float(res["test"].loc["lgbm", "MAE"])
    assert set(mj["test_by_hour"]["lgbm"]) == {str(h) for h in range(24)}
    dt = pd.read_csv(tmp_path / "results" / "error_by_daytype.csv")
    assert {"cv", "test"} == set(dt["scope"]) and "normal weekday" in set(dt["kind"])

    path = pipeline.train()
    assert path.exists()
    client = TestClient(create_app(str(path), "data/processed/load.parquet"))
    r = client.post("/predict", json={"date": "2022-06-20"})
    assert r.status_code == 200 and len(r.json()["forecasts"]) == 24
    assert pd.notna([f["load_mw"] for f in r.json()["forecasts"]]).all()


def test_model_zoo_skips_mlp_without_torch(monkeypatch):
    import importlib.util

    from smardfc.evaluation import model_zoo

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert "mlp" not in model_zoo(include_nn=True)


def test_day_kinds_labels_holiday_bridge_weekend():
    from smardfc.features import build_features
    from smardfc.pipeline import day_kinds

    idx = pd.date_range("2022-04-01", "2022-06-10", freq="h", tz="UTC")
    f = build_features(
        pd.Series(1.0, index=idx), ["2022-05-26", "2022-05-27", "2022-05-28", "2022-05-31"]
    )
    k = day_kinds(f)
    assert k[pd.Timestamp("2022-05-26")] == "holiday"  # Ascension Day
    assert k[pd.Timestamp("2022-05-27")] == "bridge day"
    assert k[pd.Timestamp("2022-05-28")] == "weekend"
    assert k[pd.Timestamp("2022-05-31")] == "normal weekday"


def test_served_model_uses_exactly_the_feature_columns(tmp_path, monkeypatch, synthetic_load):
    """The saved model must not see the target or any other column outside FEATURES."""
    import joblib

    from smardfc.features import FEATURES

    monkeypatch.chdir(tmp_path)
    save_processed(synthetic_load, DataReport(), "data/processed")
    path = pipeline.train()
    names = list(joblib.load(path)["model"].model.booster_.feature_name())
    assert names == list(FEATURES)
