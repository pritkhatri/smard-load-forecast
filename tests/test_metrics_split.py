import numpy as np
import pandas as pd
import pytest

from smardfc import metrics
from smardfc.evaluation import cv_summary, make_split, score


def test_metrics_known_values():
    y, p = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert metrics.mae(y, p) == 15.0
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert metrics.mape(y, p) == pytest.approx((10 / 100 + 20 / 200) / 2 * 100)


def test_skill_score():
    assert metrics.skill(80, 100) == pytest.approx(0.2)
    assert metrics.skill(100, 100) == 0.0
    assert metrics.skill(120, 100) < 0


def test_by_group_splits_by_key():
    df = pd.DataFrame({"y": [10.0, 10.0, 20.0], "p": [11.0, 9.0, 25.0], "h": [0, 0, 1]})
    t = metrics.by_group(df, "p", "h")
    assert t.loc[0, "MAE"] == 1.0 and t.loc[1, "MAE"] == 5.0


def test_split_is_chronological_expanding_and_leaves_test_untouched():
    days = list(pd.date_range("2022-01-30", periods=1000, freq="D"))
    sp = make_split(days, test_days=90, n_folds=5, val_days=60)
    assert sp.test_days == days[-90:]
    assert sp.dev_days == days[:-90]
    seen_val = []
    prev_train = 0
    for train, val in sp.folds:
        assert max(train) < min(val)  # no look-ahead
        assert len(val) == 60
        assert len(train) > prev_train  # expanding window
        prev_train = len(train)
        seen_val += val
    assert not set(seen_val) & set(sp.test_days)
    assert sp.folds[-1][1][-1] == sp.dev_days[-1]  # last fold ends where development ends
    assert len(set(seen_val)) == len(seen_val)  # validation windows do not overlap


def test_score_and_cv_summary():
    df = pd.DataFrame(
        {
            "y": [100.0, 100.0],
            "naive_day": [90.0, 110.0],
            "naive_week": [80.0, 120.0],
            "m": [95.0, 105.0],
        }
    )
    sc = score(df, ["naive_day", "naive_week", "m"])
    assert sc.loc["m", "MAE"] == 5.0
    assert sc.loc["m", "skill_vs_naive_week"] == pytest.approx(1 - 5 / 20)
    folds = pd.concat([sc.reset_index(names="model").assign(fold=i) for i in (1, 2)])
    cs = cv_summary(folds)
    assert cs.loc["m", "MAE_std"] == 0.0


def test_purge_days_leave_a_gap_between_training_and_scored_days():
    days = list(pd.date_range("2022-01-30", periods=1000, freq="D"))
    sp = make_split(days, test_days=90, n_folds=5, val_days=60, purge_days=1)
    for train, val in sp.folds:
        assert (min(val) - max(train)).days == 2  # one full day is skipped
    assert max(sp.train_days) == sp.dev_days[-2]
    base = make_split(days, test_days=90, n_folds=5, val_days=60)
    assert max(base.train_days) == base.dev_days[-1]
