import pandas as pd
import pytest

from smardfc.features import build_features, feature_days

torch = pytest.importorskip("torch")


def test_mlp_fits_and_predicts_finite(synthetic_load):
    from smardfc.nn import MLPModel

    f = build_features(synthetic_load, feature_days(synthetic_load))
    train, test = (
        f[f["day"] < pd.Timestamp("2022-05-15")],
        f[f["day"] >= pd.Timestamp("2022-05-15")],
    )
    m = MLPModel(hidden=(16,), epochs=3).fit(train, train["y"].to_numpy())
    p = m.predict(test)
    assert p.shape == (len(test),) and pd.notna(p).all()
