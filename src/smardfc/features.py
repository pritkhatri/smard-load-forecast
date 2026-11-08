"""Feature engineering for day-ahead forecasting.

Forecast task: for a target day D (local time, 00:00 to 24:00) the forecast is issued
at the cut-off, D-1 12:00 Europe/Berlin. The last load value assumed known is the
hour starting 11:00 on D-1 (so the hour ending at the cut-off). Publication delays of
real SMARD data are ignored.

Leakage is prevented by construction: every load-derived feature is computed from
`avail = load.loc[:last_obs]`, a slice that ends at the cut-off, and values are looked
up by timestamp in that slice only. The target `y` is the only column that reads
beyond the cut-off. tests/test_leakage.py checks this by truncating the series.
"""

from __future__ import annotations

from collections.abc import Iterable

import holidays
import numpy as np
import pandas as pd

from .data import TZ

CUTOFF_HOUR = 12
LAGS_H = (24, 48, 168, 336, 504, 672)  # hours before the target hour
MIN_HISTORY_DAYS = 29  # 28 days of lags plus the day of the first lag

_HOL = holidays.Germany(subdiv="NW", years=range(2019, 2032))


def cutoff_for(day) -> pd.Timestamp:
    """Forecast cut-off for target day `day`: D-1 12:00 local time (tz-aware)."""
    d = pd.Timestamp(day).normalize()
    return (d - pd.Timedelta(days=1) + pd.Timedelta(hours=CUTOFF_HOUR)).tz_localize(TZ)


def last_observation_time(day) -> pd.Timestamp:
    """Start of the last hour that is complete at the cut-off, in UTC."""
    return (cutoff_for(day) - pd.Timedelta(hours=1)).tz_convert("UTC")


def target_index(day) -> pd.DatetimeIndex:
    """Hourly UTC timestamps of local day `day` (23, 24 or 25 of them)."""
    d = pd.Timestamp(day).normalize()
    lo = d.tz_localize(TZ)
    hi = (d + pd.Timedelta(days=1)).tz_localize(TZ)
    return pd.date_range(lo, hi, freq="h", inclusive="left").tz_convert("UTC").as_unit("ns")


def is_holiday(day) -> bool:
    return pd.Timestamp(day).date() in _HOL


def _is_special_eve(day) -> bool:
    t = pd.Timestamp(day)
    return (t.month, t.day) in {(12, 24), (12, 31)}


def _calendar_flags(date: pd.Timestamp) -> dict:
    prev_d, next_d = date - pd.Timedelta(days=1), date + pd.Timedelta(days=1)
    weekend = date.day_of_week >= 5
    nxt_free = is_holiday(next_d) or next_d.day_of_week >= 5
    prv_free = is_holiday(prev_d) or prev_d.day_of_week >= 5
    hol = is_holiday(date)
    return {
        "is_holiday": int(hol),
        "is_special_eve": int(_is_special_eve(date)),
        "holiday_prev": int(is_holiday(prev_d)),
        "holiday_next": int(is_holiday(next_d)),
        # a working day squeezed between a day off and a weekend/holiday
        "is_bridge": int((not weekend) and (not hol) and nxt_free and prv_free),
    }


def build_features(load: pd.Series, days: Iterable) -> pd.DataFrame:
    """Feature frame for the given target days, indexed by target timestamp (UTC).

    Columns: features, `naive_week`, `naive_day`, `y` (actual, NaN if unknown),
    `day` (local target date) and `lhour` (local hour of day).
    """
    frames = []
    hours = lambda k: pd.Timedelta(hours=k)  # noqa: E731
    for day in days:
        day = pd.Timestamp(day).normalize()
        last = last_observation_time(day)
        avail = load.loc[:last]  # nothing after the cut-off is visible below
        tgt = target_index(day)
        local = tgt.tz_convert(TZ)
        f: dict[str, np.ndarray | float | int] = {}

        for k in LAGS_H:
            f[f"lag_{k}"] = avail.reindex(tgt - hours(k)).to_numpy(float)
        same_hour = np.column_stack([f[f"lag_{k}"] for k in (168, 336, 504, 672)])
        f["same_hour_mean_4w"] = np.nanmean(same_hour, axis=1) if len(tgt) else np.array([])
        f["same_hour_std_4w"] = np.nanstd(same_hour, axis=1) if len(tgt) else np.array([])

        w24 = avail.loc[last - hours(23) : last]
        w168 = avail.loc[last - hours(167) : last]
        f["last_obs"] = float(avail.reindex([last]).iloc[0])
        f["mean_last_24h"] = float(w24.mean())
        f["mean_last_168h"] = float(w168.mean())
        f["std_last_24h"] = float(w24.std())
        f["hours_ahead"] = ((tgt - last) / hours(1)).to_numpy(float)

        hour = local.hour.to_numpy()
        doy = local.day_of_year.to_numpy()
        f["hour"] = hour
        f["dow"] = local.day_of_week.to_numpy()
        f["month"] = local.month.to_numpy()
        f["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
        f["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        for key, val in _calendar_flags(day).items():
            f[key] = val
        # the week-ago reference day may be a holiday, which makes it a poor template
        f["lag168_is_holiday"] = np.array(
            [int(is_holiday((t - hours(168)).tz_convert(TZ).date())) for t in tgt]
        )

        df = pd.DataFrame(f, index=tgt)
        df["naive_week"] = df["lag_168"]
        # latest same-hour value available at the cut-off: yesterday if its hour is
        # already complete (target hours 00:00-11:00), else the day before yesterday
        df["naive_day"] = df["lag_24"].where(df["lag_24"].notna(), df["lag_48"])
        df["y"] = load.reindex(tgt).to_numpy(float)
        df["day"] = day
        df["lhour"] = hour
        frames.append(df)
    return pd.concat(frames) if frames else pd.DataFrame()


def feature_days(load: pd.Series) -> list[pd.Timestamp]:
    """All local days with enough history for the lags and complete data."""
    local = load.index.tz_convert(TZ)
    first = local[0].normalize().tz_localize(None) + pd.Timedelta(days=MIN_HISTORY_DAYS)
    # first midnight after the end of the data; a trailing partial day is dropped
    end_midnight = (local[-1] + pd.Timedelta(hours=1)).normalize().tz_localize(None)
    last_day = end_midnight - pd.Timedelta(days=1)
    return list(pd.date_range(first, last_day, freq="D"))


FEATURES = [
    "lag_24", "lag_48", "lag_168", "lag_336", "lag_504", "lag_672",
    "same_hour_mean_4w", "same_hour_std_4w",
    "last_obs", "mean_last_24h", "mean_last_168h", "std_last_24h", "hours_ahead",
    "hour", "dow", "month", "doy_sin", "doy_cos", "hour_sin", "hour_cos",
    "is_holiday", "is_special_eve", "holiday_prev", "holiday_next", "is_bridge",
    "lag168_is_holiday",
]  # fmt: skip
