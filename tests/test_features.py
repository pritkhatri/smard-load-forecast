import numpy as np
import pandas as pd

from smardfc.features import (
    FEATURES,
    build_features,
    feature_days,
    is_holiday,
    target_index,
)  # fmt: skip


def test_dst_day_lengths():
    assert len(target_index("2022-03-27")) == 23  # spring forward
    assert len(target_index("2022-10-30")) == 25  # fall back
    assert len(target_index("2022-05-02")) == 24


def test_nrw_holidays():
    assert is_holiday("2022-05-26")  # Ascension Day
    assert is_holiday("2022-11-01")  # All Saints, NRW only
    assert not is_holiday("2022-01-06")  # Epiphany is not a holiday in NRW
    assert not is_holiday("2022-05-02")


def test_feature_values_match_definitions(synthetic_load):
    day = pd.Timestamp("2022-03-10")
    f = build_features(synthetic_load, [day])
    t = f.index[5]
    for k in (24, 168, 336):
        expected = synthetic_load.get(t - pd.Timedelta(hours=k), np.nan)
        got = f[f"lag_{k}"].iloc[5]
        if k == 24 and f["lhour"].iloc[5] >= 12:
            assert np.isnan(got)
        else:
            assert got == expected
    assert f["y"].iloc[5] == synthetic_load.loc[t]
    assert set(FEATURES) <= set(f.columns)
    assert len(f) == 24


def test_bridge_day_flag():
    s = pd.Series(1.0, index=pd.date_range("2022-01-01", periods=24 * 40, freq="h", tz="UTC"))
    # Friday 27 May 2022 follows Ascension Day (Thursday 26 May) and precedes a weekend
    f = build_features(
        pd.Series(1.0, index=pd.date_range("2022-04-01", "2022-06-10", freq="h", tz="UTC")),
        ["2022-05-27", "2022-05-30"],
    )
    assert f[f["day"] == "2022-05-27"]["is_bridge"].eq(1).all()
    assert f[f["day"] == "2022-05-30"]["is_bridge"].eq(0).all()
    del s


def test_feature_days_skip_history_and_partial_day(synthetic_load):
    partial = synthetic_load.iloc[:-5]  # last local day incomplete
    days = feature_days(partial)
    assert days[0] == pd.Timestamp("2021-12-30")  # 29 days after the first (local) day
    assert days[-1] < pd.Timestamp("2022-11-15")
