"""No feature may use data after the forecast cut-off (12:00 on the day before)."""

import numpy as np
import pandas as pd
import pytest

from smardfc.features import FEATURES, build_features, cutoff_for, last_observation_time

COLS = FEATURES + ["naive_week", "naive_day"]


@pytest.mark.parametrize(
    "day", ["2022-02-15", "2022-03-27", "2022-04-18", "2022-06-01", "2022-10-30", "2022-10-31"]
)
def test_features_identical_when_future_is_removed(synthetic_load, day):
    """Cut the series at the cut-off. If any feature read later data, it would change.

    Includes the spring (27 Mar, 23 h) and autumn (30 Oct, 25 h) daylight-saving days.
    """
    full = build_features(synthetic_load, [day])
    truncated = synthetic_load.loc[: last_observation_time(day)]
    cut = build_features(truncated, [day])
    pd.testing.assert_frame_equal(full[COLS], cut[COLS], check_exact=True)
    assert cut["y"].isna().all()  # the target is not in the truncated series


def test_features_change_when_pre_cutoff_data_changes(synthetic_load):
    """Sanity check on the test above: features do react to data before the cut-off."""
    day = "2022-03-10"
    base = build_features(synthetic_load, [day])
    altered = synthetic_load.copy()
    altered.loc[last_observation_time(day)] += 5000
    new = build_features(altered, [day])
    assert not np.allclose(base["last_obs"], new["last_obs"])


def test_poisoning_future_values_does_not_change_features(synthetic_load):
    day = "2022-03-10"
    base = build_features(synthetic_load, [day])
    poisoned = synthetic_load.copy()
    poisoned.loc[poisoned.index > last_observation_time(day)] = 1e9
    out = build_features(poisoned, [day])
    pd.testing.assert_frame_equal(base[COLS], out[COLS], check_exact=True)


def test_lag_24_is_missing_for_hours_after_the_cutoff(synthetic_load):
    f = build_features(synthetic_load, ["2022-03-10"])
    assert f.loc[f["lhour"] <= 11, "lag_24"].notna().all()
    assert f.loc[f["lhour"] >= 12, "lag_24"].isna().all()


def test_cutoff_is_noon_local_time_the_day_before():
    c = cutoff_for("2022-03-28")  # day after the spring DST change (UTC+2 from 27 March)
    assert str(c) == "2022-03-27 12:00:00+02:00"
    assert str(last_observation_time("2022-03-28")) == "2022-03-27 09:00:00+00:00"
