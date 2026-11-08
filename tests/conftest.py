import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_load() -> pd.Series:
    """Deterministic hourly UTC load with daily/weekly/yearly cycles, 2021-12-01..2022-11-15.

    Test data only; no metric in the README comes from it.
    """
    idx = pd.date_range("2021-12-01", "2022-11-15 23:00", freq="h", tz="UTC").as_unit("ns")
    local = idx.tz_convert("Europe/Berlin")
    rng = np.random.default_rng(0)
    y = (
        20000
        + 3000 * np.sin(2 * np.pi * (local.hour - 6) / 24)
        - 2500 * (local.day_of_week >= 5)
        - 1500 * np.cos(2 * np.pi * local.day_of_year / 365)
        + rng.normal(0, 150, len(idx))
    )
    return pd.Series(y, index=idx, name="load_mw")
