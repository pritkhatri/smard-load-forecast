import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from smardfc.api import create_app
from smardfc.data import save_processed
from smardfc.evaluation import fit_predict
from smardfc.features import build_features, feature_days
from smardfc.models import LGBMModel


@pytest.fixture(scope="module")
def feats(synthetic_load):
    return build_features(synthetic_load, feature_days(synthetic_load))


def test_lgbm_learns_the_synthetic_structure(feats):
    """The synthetic series is a deterministic cycle plus N(0, 150 MW) noise, so the best
    possible MAE is about 120 MW while the series std is about 3000 MW. This only checks
    that the pipeline learns the cycle; it says nothing about real data."""
    cut = pd.Timestamp("2022-05-15")
    train, test = feats[feats["day"] < cut], feats[feats["day"] >= cut]
    _, p = fit_predict(lambda: LGBMModel(residual=True, n_estimators=100), train, test)
    ok = test["y"].notna().to_numpy()
    assert abs(test["y"].to_numpy()[ok] - p[ok]).mean() < 300


def test_residual_model_falls_back_when_week_ago_missing(feats):
    train = feats[feats["day"] < pd.Timestamp("2022-05-15")]
    m = LGBMModel(residual=True, n_estimators=20).fit(train, train["y"].to_numpy())
    row = feats.iloc[[100]].copy()
    row["lag_168"] = float("nan")
    assert pd.notna(m.predict(row)).all()


@pytest.fixture()
def client(tmp_path, synthetic_load, feats):
    train = feats[feats["day"] < pd.Timestamp("2022-06-01")]
    model = LGBMModel(n_estimators=30).fit(train, train["y"].to_numpy())
    mp, dp = tmp_path / "m.joblib", tmp_path / "load.parquet"
    joblib.dump({"model": model, "name": "lgbm", "trained_until": "x", "version": "t"}, mp)
    save_processed(synthetic_load, type("R", (), {"to_dict": lambda self: {}})(), tmp_path)
    return TestClient(create_app(str(mp), str(dp)))


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_24_hours(client):
    r = client.post("/predict", json={"date": "2022-06-10"})
    assert r.status_code == 200
    body = r.json()
    assert [f["hour"] for f in body["forecasts"]] == list(range(24))
    assert all(10000 < f["load_mw"] < 40000 for f in body["forecasts"])
    assert body["cutoff"].startswith("2022-06-09 12:00:00")


def test_predict_handles_dst_day(client):
    assert len(client.post("/predict", json={"date": "2022-03-27"}).json()["forecasts"]) == 23


def test_predict_rejects_dates_without_history(client):
    assert client.post("/predict", json={"date": "2023-01-01"}).status_code == 422  # future
    assert client.post("/predict", json={"date": "2021-12-05"}).status_code == 422  # too early
    assert client.post("/predict", json={"date": "not-a-date"}).status_code == 422


def test_health_and_predict_when_model_missing(tmp_path):
    c = TestClient(create_app(str(tmp_path / "no.joblib"), str(tmp_path / "no.parquet")))
    assert c.get("/health").json()["status"].startswith("unavailable")
    assert c.post("/predict", json={"date": "2022-06-10"}).status_code == 503
