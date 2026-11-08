"""Load, cache and validate hourly grid-load data from SMARD (Bundesnetzagentur).

Two sources are supported:
  * the SMARD chart-data API (fetch_smard), cached as raw JSON in data/raw/
  * a CSV from the SMARD download centre (load_csv), German or English layout
"""

from __future__ import annotations

import io
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

TZ = "Europe/Berlin"
API = "https://www.smard.de/app/chart_data"
# 410 = "Realisierter Stromverbrauch: Gesamt (Netzlast)"
FILTER_CONSUMPTION = 410


def _utc_hourly_index(idx) -> pd.DatetimeIndex:
    """UTC DatetimeIndex with a fixed resolution so that pandas 2 and 3 agree."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        raise ValueError("expected a timezone-aware index")
    return idx.tz_convert("UTC").as_unit("ns")


# --------------------------------------------------------------------------
# SMARD API
# --------------------------------------------------------------------------
def _get_json(client: httpx.Client, url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = client.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_smard(
    region: str = "Amprion",
    start: str = "2022-01-01",
    end: str | None = None,
    cache_dir: str | Path = "data/raw",
    filter_id: int = FILTER_CONSUMPTION,
    refresh: bool = False,
    client: httpx.Client | None = None,
) -> pd.Series:
    """Hourly realised consumption in MW for one control zone, UTC index.

    `start` is a local (Europe/Berlin) date, `end` an exclusive local date. With
    end=None the data is cut at the start of the current month, i.e. only complete
    months are returned. Weekly chunks are cached in cache_dir; the two newest
    chunks are re-downloaded when refresh=True.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client()
    try:
        index = _get_json(client, f"{API}/{filter_id}/{region}/index_hour.json")["timestamps"]
        start_ts = pd.Timestamp(start).tz_localize(TZ)
        if end is None:
            now = pd.Timestamp.now(tz=TZ)
            end_ts = pd.Timestamp(year=now.year, month=now.month, day=1).tz_localize(TZ)
        else:
            end_ts = pd.Timestamp(end).tz_localize(TZ)
        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)
        before = [t for t in index if t <= start_ms]
        chunks = ([max(before)] if before else []) + [t for t in index if start_ms < t < end_ms]
        newest = set(index[-2:])
        rows: list[list] = []
        for t in chunks:
            path = cache / f"{filter_id}_{region}_hour_{t}.json"
            if path.exists() and not (refresh and t in newest):
                payload = json.loads(path.read_text())
            else:
                url = f"{API}/{filter_id}/{region}/{filter_id}_{region}_hour_{t}.json"
                payload = _get_json(client, url)
                path.write_text(json.dumps(payload))
            rows.extend(payload.get("series", []))
    finally:
        if own:
            client.close()

    if not rows:
        raise RuntimeError("SMARD returned no data for the requested range")
    ts = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
    vals = pd.to_numeric(pd.Series([r[1] for r in rows]), errors="coerce").to_numpy(float)
    s = pd.Series(vals, index=_utc_hourly_index(ts), name="load_mw")
    return s[(s.index >= start_ts.tz_convert("UTC")) & (s.index < end_ts.tz_convert("UTC"))]


# --------------------------------------------------------------------------
# CSV from the download centre
# --------------------------------------------------------------------------
_VALUE_HINT = re.compile(r"netzlast|verbrauch|load|consumption", re.I)


def _to_number(col: pd.Series, decimal_comma: bool) -> pd.Series:
    s = col.astype(str).str.strip().replace({"-": np.nan, "": np.nan, "nan": np.nan})
    if decimal_comma:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_csv(path_or_buf) -> pd.Series:
    """Read a SMARD download-centre CSV ("Realisierter Stromverbrauch", hourly).

    Handles the German layout ("Datum von;Datum bis;Netzlast [MWh]...", decimal
    comma, dd.mm.yyyy) and the English layout ("Start date;End date;Total grid
    load [MWh]...", decimal point, "Jan 1, 2022 12:00 AM"). Timestamps are local
    time; the autumn DST hour is resolved from the order of the rows.
    """
    text = (
        Path(path_or_buf).read_text(encoding="utf-8-sig")
        if isinstance(path_or_buf, (str, Path))
        else path_or_buf.read()
    )
    first = text.splitlines()[0]
    sep = ";" if first.count(";") >= first.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    if df.shape[1] < 3:
        raise ValueError(f"expected at least 3 columns, got {list(df.columns)}")

    value_cols = [c for c in df.columns[2:] if _VALUE_HINT.search(c)]
    col = value_cols[0] if value_cols else df.columns[2]
    german = bool(re.search(r"datum|netzlast|verbrauch", " ".join(df.columns), re.I))

    raw_ts = df.iloc[:, 0].str.strip()
    if german:
        ts = pd.to_datetime(raw_ts, format="%d.%m.%Y %H:%M")
    else:
        try:
            ts = pd.to_datetime(raw_ts, format="%b %d, %Y %I:%M %p")
        except ValueError:
            ts = pd.to_datetime(raw_ts, format="mixed")
    vals = _to_number(df[col], decimal_comma=german).to_numpy(float)

    try:
        loc = pd.DatetimeIndex(ts).tz_localize(TZ, ambiguous="infer")
    except Exception:
        # fall back: treat the second occurrence of a repeated local hour as DST end
        dup = pd.Series(ts).duplicated(keep="first").to_numpy()
        loc = pd.DatetimeIndex(ts).tz_localize(TZ, ambiguous=dup)
    return pd.Series(vals, index=_utc_hourly_index(loc), name="load_mw")


# --------------------------------------------------------------------------
# validation and cleaning
# --------------------------------------------------------------------------
@dataclass
class DataReport:
    n_raw: int = 0
    first: str = ""
    last: str = ""
    n_duplicates: int = 0
    n_nonpositive: int = 0
    n_missing_hours: int = 0
    n_gaps: int = 0
    longest_gap_hours: int = 0
    n_interpolated: int = 0
    n_left_missing: int = 0
    n_flagged_outliers: int = 0
    n_short_days_23h: int = 0
    n_long_days_25h: int = 0
    gaps: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def clean(s: pd.Series, max_interp_gap: int = 3) -> tuple[pd.Series, DataReport]:
    """Validate an hourly UTC series and return (clean series, report).

    * duplicates (same timestamp): last value kept
    * non-positive values: treated as missing (grid load is never <= 0)
    * gaps: the series is reindexed to a complete hourly grid; gaps of at most
      `max_interp_gap` hours are filled by linear interpolation, longer gaps stay NaN
    * DST: the index is UTC, so 23/25-hour local days need no special handling; they
      are counted in the report for transparency
    * outliers: flagged (not modified) when the change versus the same hour one week
      earlier is more than 8 robust standard deviations from the median change
    """
    rep = DataReport(n_raw=len(s))
    s = s.sort_index()
    dup = s.index.duplicated(keep="last")
    rep.n_duplicates = int(dup.sum())
    s = s[~dup]

    nonpos = (s <= 0).fillna(False)
    rep.n_nonpositive = int(nonpos.sum())
    s = s.mask(nonpos)

    full = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC").as_unit("ns")
    s = s.reindex(full)
    rep.first, rep.last = str(full[0]), str(full[-1])

    miss = s.isna().to_numpy()
    rep.n_missing_hours = int(miss.sum())
    values = s.to_numpy(float).copy()
    i = 0
    while i < len(values):
        if miss[i]:
            j = i
            while j < len(values) and miss[j]:
                j += 1
            n = j - i
            rep.n_gaps += 1
            rep.longest_gap_hours = max(rep.longest_gap_hours, n)
            rep.gaps.append({"start": str(full[i]), "hours": n})
            if n <= max_interp_gap and i > 0 and j < len(values):
                values[i:j] = np.interp(np.arange(i, j), [i - 1, j], [values[i - 1], values[j]])
                rep.n_interpolated += n
            i = j
        else:
            i += 1
    out = pd.Series(values, index=full, name="load_mw")
    rep.n_left_missing = int(out.isna().sum())

    delta = out - out.shift(168)
    med = delta.median()
    mad = (delta - med).abs().median()
    if mad > 0:
        rep.n_flagged_outliers = int((((delta - med).abs() / (1.4826 * mad)) > 8).sum())

    local = out.index.tz_convert(TZ)
    per_day = pd.Series(1, index=local.date).groupby(level=0).sum()
    # a partial first/last day would look like a 23-hour day; leave it out of the count
    if local[0].hour != 0:
        per_day = per_day.iloc[1:]
    if local[-1].hour != 23 and len(per_day):
        per_day = per_day.iloc[:-1]
    rep.n_short_days_23h = int((per_day == 23).sum())
    rep.n_long_days_25h = int((per_day == 25).sum())
    return out, rep


def save_processed(s: pd.Series, rep: DataReport, out_dir: str | Path = "data/processed") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s.rename("load_mw").to_frame().to_parquet(out / "load.parquet")
    (out / "validation_report.json").write_text(json.dumps(rep.to_dict(), indent=2))


def load_processed(path: str | Path = "data/processed/load.parquet") -> pd.Series:
    df = pd.read_parquet(path)
    s = df["load_mw"]
    s.index = _utc_hourly_index(s.index)
    return s
