"""FastAPI service: day-ahead forecast for a requested date.

Configuration through environment variables:
    SMARDFC_MODEL  path to the joblib bundle written by `smardfc train`
    SMARDFC_DATA   path to the processed load parquet written by `smardfc data`
The service forecasts from the load history it finds in SMARDFC_DATA, applying the same
12:00 cut-off as in training. It does not fetch new data.
"""

from __future__ import annotations

import datetime as dt
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .data import TZ, load_processed
from .features import MIN_HISTORY_DAYS, build_features, last_observation_time


class PredictRequest(BaseModel):
    date: dt.date = Field(description="Target day (local time), e.g. 2026-10-20")


class HourlyForecast(BaseModel):
    timestamp: str = Field(description="Start of the hour, ISO 8601 with local UTC offset")
    hour: int
    load_mw: float


class PredictResponse(BaseModel):
    date: dt.date
    cutoff: str
    model: str
    trained_until: str
    forecasts: list[HourlyForecast]


class Health(BaseModel):
    status: str
    model: str | None = None
    data_last: str | None = None


def create_app(model_path: str | None = None, data_path: str | None = None) -> FastAPI:
    model_path = model_path or os.environ.get("SMARDFC_MODEL", "models/model.joblib")
    data_path = data_path or os.environ.get("SMARDFC_DATA", "data/processed/load.parquet")
    app = FastAPI(title="smardfc", version="0.1.0")
    app.state.bundle = None
    app.state.load = None
    app.state.error = None
    try:
        app.state.bundle = joblib.load(model_path)
        app.state.load = load_processed(data_path)
    except Exception as exc:  # reported by /health, /predict answers 503
        app.state.error = f"{type(exc).__name__}: {exc}"

    @app.get("/health", response_model=Health)
    def health(request: Request):
        st = request.app.state
        if st.bundle is None or st.load is None:
            return Health(status=f"unavailable: {st.error}")
        return Health(status="ok", model=st.bundle["name"], data_last=str(st.load.index[-1]))

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, request: Request):
        st = request.app.state
        if st.bundle is None or st.load is None:
            raise HTTPException(503, f"model or data not loaded: {st.error}")
        load = st.load
        day = pd.Timestamp(req.date)
        if last_observation_time(day) > load.index[-1]:
            raise HTTPException(
                422, f"history ends {load.index[-1]}; the cut-off for {req.date} is later"
            )
        if last_observation_time(day) - pd.Timedelta(days=MIN_HISTORY_DAYS) < load.index[0]:
            raise HTTPException(422, f"not enough history before {req.date} (need 29 days)")
        feats = build_features(load, [day])
        pred = st.bundle["model"].predict(feats)
        local = feats.index.tz_convert(TZ)
        return PredictResponse(
            date=req.date,
            cutoff=str(pd.Timestamp(last_observation_time(day) + pd.Timedelta(hours=1))
                       .tz_convert(TZ)),
            model=st.bundle["name"],
            trained_until=st.bundle["trained_until"],
            forecasts=[
                HourlyForecast(timestamp=t.isoformat(), hour=int(t.hour), load_mw=float(v))
                for t, v in zip(local, pred, strict=True)
            ],
        )  # fmt: skip

    return app


def app_factory() -> FastAPI:  # for `uvicorn --factory`
    return create_app()
