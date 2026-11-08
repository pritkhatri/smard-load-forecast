"""Models behind one fit/predict interface on the feature frame."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES


class NaiveColumn:
    """Seasonal-naive baseline: predicts the value in an existing column."""

    def __init__(self, column: str):
        self.column = column

    def fit(self, X: pd.DataFrame, y) -> NaiveColumn:  # noqa: N803
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return X[self.column].to_numpy(float)


LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=0,
    n_jobs=4,
    verbose=-1,
)


class LGBMModel:
    """LightGBM on the target itself (residual=False) or on y - lag_168 (residual=True).

    The residual variant predicts the deviation from the same hour one week earlier,
    which makes it easier for a tree model to follow level shifts and trends. Where the
    week-ago value is missing it falls back to the 4-week same-hour mean.
    """

    def __init__(self, residual: bool = False, **params):
        self.residual = residual
        self.params = {**LGBM_PARAMS, **params}
        self.model: lgb.LGBMRegressor | None = None
        self.features = list(FEATURES)

    def _base(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return X["lag_168"].fillna(X["same_hour_mean_4w"]).to_numpy(float)

    def fit(self, X: pd.DataFrame, y) -> LGBMModel:  # noqa: N803
        y = np.asarray(y, float)
        target = y - self._base(X) if self.residual else y
        ok = np.isfinite(target)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X.loc[ok, self.features], target[ok])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        out = self.model.predict(X[self.features])
        return out + self._base(X) if self.residual else out

    def importance(self) -> pd.Series:
        gain = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain, index=self.features).sort_values(ascending=False)
