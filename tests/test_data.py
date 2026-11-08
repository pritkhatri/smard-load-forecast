import io

import httpx
import numpy as np
import pandas as pd

from smardfc.data import clean, fetch_smard, load_csv

DE_CSV = (
    "Datum von;Datum bis;Netzlast [MWh] Originalauflösungen\n"
    "01.01.2022 00:00;01.01.2022 01:00;21.345,50\n"
    "01.01.2022 01:00;01.01.2022 02:00;20.100,25\n"
    "01.01.2022 02:00;01.01.2022 03:00;-\n"
)
EN_CSV = (
    "Start date;End date;Total grid load [MWh] Calculated resolutions\n"
    "Jan 1, 2022 12:00 AM;Jan 1, 2022 1:00 AM;21345.5\n"
    "Jan 1, 2022 1:00 AM;Jan 1, 2022 2:00 AM;20100.25\n"
)


def test_german_csv_decimal_comma_and_missing_marker():
    s = load_csv(io.StringIO(DE_CSV))
    assert s.iloc[0] == 21345.5 and s.iloc[1] == 20100.25 and np.isnan(s.iloc[2])
    assert str(s.index[0]) == "2021-12-31 23:00:00+00:00"  # local midnight = 23:00 UTC


def test_english_csv_layout():
    s = load_csv(io.StringIO(EN_CSV))
    assert list(s.values) == [21345.5, 20100.25]


def test_csv_handles_autumn_dst_repeated_hour():
    rows = ["Datum von;Datum bis;Netzlast [MWh]"]
    for h in ("00:00", "01:00", "02:00", "02:00", "03:00"):  # 30 Oct 2022 has 02:00 twice
        rows.append(f"30.10.2022 {h};30.10.2022 {h};1.000,0")
    s = load_csv(io.StringIO("\n".join(rows)))
    assert s.index.is_unique and s.index.is_monotonic_increasing
    assert (s.index[1:] - s.index[:-1] == pd.Timedelta(hours=1)).all()


def test_clean_duplicates_gaps_and_nonpositive():
    idx = pd.date_range("2022-01-01", periods=48, freq="h", tz="UTC").as_unit("ns")
    s = pd.Series(np.linspace(1000, 2000, 48), index=idx)
    s = pd.concat([s, s.iloc[[10]] + 1])  # duplicate timestamp, last value wins
    s = s.drop(idx[[20, 21]])  # short gap -> interpolated
    s = s.drop(idx[30:40])  # long gap -> stays missing
    s.loc[idx[5]] = -3.0  # impossible value -> treated as missing, then interpolated
    out, rep = clean(s)
    assert rep.n_duplicates == 1
    assert rep.n_nonpositive == 1
    assert rep.n_gaps == 3 and rep.longest_gap_hours == 10
    assert rep.n_interpolated == 3
    assert rep.n_left_missing == 10
    assert out.loc[idx[20]] == np.interp(20, [19, 22], [out.loc[idx[19]], out.loc[idx[22]]])
    assert out.index.is_monotonic_increasing and len(out) == 48


def test_clean_counts_dst_days():
    idx = pd.date_range("2022-03-25", "2022-11-02", freq="h", tz="UTC").as_unit("ns")
    out, rep = clean(pd.Series(1000.0, index=idx))
    assert rep.n_short_days_23h == 1 and rep.n_long_days_25h == 1
    assert rep.n_missing_hours == 0


def test_fetch_smard_parses_chunks_and_caches(tmp_path):
    """The SMARD API is replaced by a mock transport; chunks are cached on disk."""
    t0 = int(pd.Timestamp("2022-01-01", tz="Europe/Berlin").timestamp() * 1000)
    hour = 3_600_000
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("index_hour.json"):
            return httpx.Response(
                200, json={"timestamps": [t0 - 7 * 24 * hour, t0 + 7 * 24 * hour]}
            )
        series = [[t0 + i * hour, 1000.0 + i] for i in range(48)]
        series[3][1] = None  # SMARD marks missing values as null
        return httpx.Response(200, json={"series": series})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    kw = dict(start="2022-01-01", end="2022-01-03", cache_dir=tmp_path, client=client)
    s = fetch_smard(**kw)
    assert len(s) == 48 and np.isnan(s.iloc[3]) and s.iloc[0] == 1000.0
    assert str(s.index[0]) == "2021-12-31 23:00:00+00:00"
    n_after_first = len(calls)
    fetch_smard(**kw)  # second call: only the index is requested again, chunks come from disk
    assert len(calls) == n_after_first + 1
