"""Diagnostics behind the README caveats: region vs holidays, the August 2026 drop, and
the evening-hour errors. Writes results/investigation.md and results/investigation_august.png.

Needs network on first run (it downloads the other SMARD control zones and the German
total into data/raw_other/ for comparison) and results/test_predictions.csv from `evaluate`.
Report only; nothing here feeds back into the model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import holidays
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smardfc.data import TZ, clean, fetch_smard, load_processed

OUT = Path("results")
ZONES = ["Amprion", "50Hertz", "TenneT", "TransnetBW", "DE"]
lines: list[str] = []


def h(title):
    lines.extend(["", f"## {title}", ""])


def p(text=""):
    lines.append(text)


def table(df, fmt="{:,.0f}"):
    p("| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |")
    p("|" + "---|" * (len(df.columns) + 1))
    for idx, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (int, float, np.floating)) else str(v) for v in row]
        p("| " + " | ".join([str(idx)] + cells) + " |")
    p()


# ---------------------------------------------------------------- data
load = load_processed()
zones = {"Amprion": load}
for z in ZONES[1:]:
    raw = fetch_smard(
        region=z, start="2022-01-01", end="2026-11-01", cache_dir=f"data/raw_other/{z}"
    )
    zones[z], _ = clean(raw)
loc = {z: s.tz_convert(TZ) for z, s in zones.items()}
daily = {z: s.resample("D").mean() for z, s in loc.items()}

# ------------------------------------------------------------ 1. provenance
h("1. Data provenance")
meta = json.loads(sorted(Path("data/raw").glob("410_Amprion_hour_*.json"))[0].read_text())[
    "meta_data"
]
p(f"Raw chunk meta_data (first file): `{json.dumps(meta)}`")
p()
p(f"Amprion series: {load.index[0]} to {load.index[-1]} (UTC), {len(load)} hourly values.")
ratio = (loc["Amprion"]["2026-10"].mean() / loc["DE"]["2026-10"].mean()) * 100
p(f"Mean load October 2026: Amprion {loc['Amprion']['2026-10'].mean():,.0f} MW, "
  f"Germany total {loc['DE']['2026-10'].mean():,.0f} MW, so Amprion is {ratio:.0f}% of the total.")  # fmt: skip

# ------------------------------------------------------------ 2. holidays vs region
h("2. Holiday calendar vs region")
test_days = pd.date_range("2026-08-03", "2026-10-31")
all_days = pd.date_range("2022-01-01", "2026-10-31")
nw = holidays.Germany(subdiv="NW", years=range(2022, 2027))
states = ["RP", "SL", "HE", "BW", "BY", "NI"]
rows = []
for st in states:
    cal = holidays.Germany(subdiv=st, years=range(2022, 2027))
    for d in all_days:
        if d.date() in cal and d.date() not in nw:
            rows.append({"state": st, "date": d, "name": cal.get(d.date()),
                         "weekday": d.day_name()[:3], "in_test": d in test_days})  # fmt: skip
miss = pd.DataFrame(rows)
p(
    "Amprion is a control area, not a state: it spans NRW plus parts of other states "
    "(the exact split is not in this repo; the states below are an assumption)."
)
p()
p(
    f"Holidays in the test period (2026-08-03 to 2026-10-31) in the NW calendar: "
    f"{[str(d.date()) + ' ' + d.day_name()[:3] for d in test_days if d.date() in nw]}"
)
tm = miss[miss.in_test].drop_duplicates(["date", "name"])
p("Holidays in other candidate states that the NW calendar does not mark, test period:")
for _, r in tm.groupby(["date", "name"]).agg(states=("state", ", ".join)).reset_index().iterrows():
    p(f"- {r['date'].date()} {r['date'].day_name()[:3]}: {r['name']} ({r['states']})")
p()
p("Such holidays over the whole data set (2022-01-01 to 2026-10-31), by state and name:")
p()
summ = miss.groupby(["name"]).agg(
    days=("date", "nunique"), states=("state", lambda x: ", ".join(sorted(set(x))))
)
summ.index.name = "holiday"
table(summ, "{}")
p(f"Distinct dates marked by at least one other state but not NW: {miss.date.nunique()}; "
  f"of those on a Mon-Fri: {miss.drop_duplicates('date').weekday.isin(['Mon','Tue','Wed','Thu','Fri']).sum()}.")  # fmt: skip

# ------------------------------------------------------------ 3. August
h("3. August 2026 investigation")
a = load.loc["2026-07-01":"2026-09-30"]
full = pd.date_range(a.index[0], a.index[-1], freq="h", tz="UTC")
p(f"Jul-Sep 2026 Amprion: {len(a)} values, {len(full) - len(a)} missing hours, "
  f"{int(a.index.duplicated().sum())} duplicate timestamps, min {a.min():,.0f} MW, max {a.max():,.0f} MW.")  # fmt: skip
p(f"Consecutive identical hourly values (flat stretches): {int((a.diff() == 0).sum())}.")
d_loc = a.tz_convert(TZ)
day_prof = d_loc.groupby(d_loc.index.date).apply(lambda x: tuple(np.round(x.values, 1)))
dup_days = int(day_prof.duplicated().sum())
p(f"Local days whose 24-hour profile exactly equals another day's: {dup_days}.")
delta = load - load.shift(168)
mad = (delta - delta.median()).abs().median()
z = (delta - delta.median()).abs() / (1.4826 * mad)
aug_out = z["2026-08-01":"2026-08-31"]
p(f"Hours in Aug 2026 with a week-over-week change above 8 robust SD: {int((aug_out > 8).sum())} "
  f"(whole data set: {int((z > 8).sum())} of {len(z)}).")  # fmt: skip


# weekly level vs same weekday-aligned week of 2025 (364 days earlier), non-holiday Mon-Fri
def weekday_mean(s_daily, hol):
    d = s_daily[(s_daily.index.dayofweek < 5) & ~s_daily.index.map(lambda t: t.date() in hol)]
    return d


yoy = {}
for zname in ZONES:
    d = weekday_mean(daily[zname], nw)
    wk = d.groupby(d.index.to_period("W-SUN").start_time.tz_localize(None)).mean()
    prev = wk.copy()
    prev.index = prev.index + pd.Timedelta(days=364)
    yoy[zname] = (wk / prev.reindex(wk.index) - 1) * 100
yy = pd.DataFrame(yoy)["2026-06-01":"2026-10-26"]
yy.index = [str(i.date()) for i in yy.index]
yy.index.name = "week starting"
p()
p(
    "Year-over-year change in mean Mon-Fri load (NW public holidays excluded), by week, in %. "
    "Compared with the week 364 days earlier."
)
p()
table(yy, "{:+.1f}")

# seasonal check: is an August dip normal? Aug weekday mean vs mean of Jul and Sep
p(
    "Mean non-holiday Mon-Fri load in August relative to the mean of July and September of "
    "the same year (Amprion), in %:"
)
p()
rel = {}
wd = weekday_mean(daily["Amprion"], nw)
for yr in range(2022, 2027):
    g = {m: wd[(wd.index.year == yr) & (wd.index.month == m)].mean() for m in (7, 8, 9)}
    rel[yr] = (g[8] / ((g[7] + g[9]) / 2) - 1) * 100
table(pd.DataFrame({"Aug vs avg(Jul,Sep)": rel}).rename_axis("year"), "{:+.1f}")

# MW-level view: do the zone changes add up to the national change?
p()
p(
    "Change in mean Mon-Fri load in MW, 2026-08-10 to 2026-08-28 vs the same weekdays 364 days "
    "earlier. If the four control zones add up to Germany, a shift between zones nets to ~0:"
)
p()
rows = {}
for zname in ZONES:
    c = loc[zname]["2026-08-10":"2026-08-28 23:00"]
    c = c[c.index.dayofweek < 5]
    pv = pd.Series(loc[zname].reindex(c.index - pd.Timedelta(days=364)).values, index=c.index)
    rows[zname] = {"2026 mean": c.mean(), "2025 mean": pv.mean(), "change MW": c.mean() - pv.mean()}
mw = pd.DataFrame(rows).T
mw.loc["sum of 4 zones"] = mw.loc[["Amprion", "50Hertz", "TenneT", "TransnetBW"]].sum()
mw.index.name = "zone"
table(mw)

# hour-of-day shape of the drop, Mon-Fri 2026-08-10..2026-08-28 vs the same days 364 d earlier
cur = loc["Amprion"]["2026-08-10":"2026-08-28 23:00"]
cur = cur[cur.index.dayofweek < 5]
prev_idx = cur.index - pd.Timedelta(days=364)
prv = pd.Series(loc["Amprion"].reindex(prev_idx).values, index=cur.index)
ratio_h = (cur.groupby(cur.index.hour).mean() / prv.groupby(prv.index.hour).mean() - 1) * 100
p(
    "Change by local hour, Mon-Fri 2026-08-10 to 2026-08-28 vs the same weekdays 364 days "
    "earlier (Amprion), in %:"
)
p()
p(", ".join(f"{int(hh):02d}h {v:+.1f}" for hh, v in ratio_h.items()))

# re-download check: were the Jul-Sep 2026 values revised after the first download?
p()
p(
    "Re-download check: Amprion 2026-06-01 to 2026-10-01 fetched again into a fresh cache and "
    "compared with the series used for the results."
)
p()
fresh = fetch_smard(
    region="Amprion", start="2026-06-01", end="2026-10-01", cache_dir="data/raw_recheck"
)
old = load.reindex(fresh.index)
diff = (fresh - old).abs()
p(
    f"Hours compared: {len(fresh)}; hours that differ: {int((diff > 1e-6).sum())}; max abs difference: {diff.max():.3f} MW."
)

# MW offset by hour (constant offset vs proportional)
mwh = cur.groupby(cur.index.hour).mean() - prv.groupby(prv.index.hour).mean()
p()
p("Same comparison as above in MW by local hour (2026 minus 2025):")
p()
p(", ".join(f"{int(hh):02d}h {v:+,.0f}" for hh, v in mwh.items()))

# daily onset and recovery: Amprion minus the weekday-aligned 2025 value, and the other three zones
dd = {}
for zname in ["Amprion", "50Hertz", "TenneT", "TransnetBW"]:
    d26 = daily[zname]["2026-07-01":"2026-09-30"]
    d25 = daily[zname].reindex(d26.index - pd.Timedelta(days=364))
    dd[zname] = pd.Series(d26.values - d25.values, index=d26.index)
dd = pd.DataFrame(dd)
dd["other 3 zones"] = dd[["50Hertz", "TenneT", "TransnetBW"]].sum(axis=1)
dd = dd[["Amprion", "other 3 zones"]]
p()
p(
    "Daily mean load, 2026 minus the same weekday 364 days earlier, MW (Amprion and the sum of "
    "the other three zones), around the start and the end of the break:"
)
p()
sel = dd.loc["2026-07-28":"2026-08-10"].round(0).copy()
sel.index = [f"{i.date()} {i.day_name()[:3]}" for i in sel.index]
sel.index.name = "day"
table(sel)
sel = dd.loc["2026-08-28":"2026-09-10"].round(0).copy()
sel.index = [f"{i.date()} {i.day_name()[:3]}" for i in sel.index]
sel.index.name = "day"
table(sel)

# test skill without August
pr0 = pd.read_csv(OUT / "test_predictions.csv")
pr0["day"] = pd.to_datetime(pr0["day"])
no_aug = pr0[pr0.day >= "2026-09-01"]
p("Test skill vs naive_week excluding August (days from 2026-09-01), and for the full test period:")
p()
rows = {}
for m in ["lgbm", "lgbm_resid", "mlp"]:
    a_ = (no_aug[m] - no_aug.y).abs().mean()
    b_ = (no_aug.naive_week - no_aug.y).abs().mean()
    a2 = (pr0[m] - pr0.y).abs().mean()
    b2 = (pr0.naive_week - pr0.y).abs().mean()
    rows[m] = {"skill Sep-Oct": 1 - a_ / b_, "skill full test": 1 - a2 / b2}
table(pd.DataFrame(rows).T.rename_axis("model"), "{:+.3f}")

# figure
import matplotlib.dates as mdates  # noqa: E402

s26 = loc["Amprion"]["2026-08-03":"2026-09-06 23:00"]
nv26 = loc["Amprion"].shift(168, freq="h").reindex(s26.index)
s25i = s26.index - pd.Timedelta(days=364)
s25 = loc["Amprion"].reindex(s25i)
nv25 = loc["Amprion"].shift(168, freq="h").reindex(s25i)
x = s26.index.tz_localize(None)
fig, ax = plt.subplots(4, 1, figsize=(10, 12))
ylim = (9000, 24000)
ax[0].plot(x, s26.values, "k", lw=1, label="actual 2026")
ax[0].plot(x, nv26.values, color="#888", ls="--", lw=0.9, label="naive_week 2026")
ax[0].set_title("Amprion, 2026-08-03 to 2026-09-06")
ax[1].plot(x, s25.values, "k", lw=1, label="actual 2025 (same weekdays)")
ax[1].plot(x, nv25.values, color="#888", ls="--", lw=0.9, label="naive_week 2025")
ax[1].set_title("Same weeks of 2025 (shifted by 364 days to align weekdays); same y-range as above")
for a_ in ax[:2]:
    a_.set_ylim(*ylim)
    a_.set_ylabel("MW")
    a_.legend(fontsize=8, loc="lower right")  # noqa: E702
    a_.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    a_.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax[2].bar(dd.index.tz_localize(None), dd["Amprion"], color="#1f77b4", label="Amprion", width=0.8)
ax[2].plot(dd.index.tz_localize(None), dd["other 3 zones"], color="#d62728", lw=1.4,
           label="50Hertz + TenneT + TransnetBW")  # fmt: skip
ax[2].axhline(0, color="k", lw=0.6)
ax[2].set_title("Daily mean load, 2026 minus the same weekday of 2025 [MW], 1 Jul to 30 Sep")
ax[2].set_ylabel("MW")
ax[2].legend(fontsize=8)  # noqa: E702
ax[2].xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
for zname in ZONES:
    ax[3].plot(pd.to_datetime(yy.index), yy[zname], marker="o", ms=3, label=zname,
               lw=2.2 if zname in ("Amprion", "DE") else 1)  # fmt: skip
ax[3].axhline(0, color="k", lw=0.6)
ax[3].set_title("Year-over-year change of weekly mean Mon-Fri load [%]")
ax[3].legend(ncol=5, fontsize=8)
for a_ in ax:
    a_.tick_params(axis="x", labelrotation=30)
fig.tight_layout()
fig.savefig(OUT / "investigation_august.png", dpi=130)  # noqa: E702

# ------------------------------------------------------------ 4. evening hours
h("4. Evening hours in the test period")
pr = pd.read_csv(OUT / "test_predictions.csv")
pr["day"] = pd.to_datetime(pr["day"])
pr["err_lgbm"] = (pr.lgbm - pr.y).abs()
pr["err_nw"] = (pr.naive_week - pr.y).abs()
ev = pr[pr.lhour.isin([19, 20, 21])]
p("MAE by local hour (MW) in the test period: lgbm vs naive_week")
p()
hh = pr.groupby("lhour")[["err_lgbm", "err_nw"]].mean()
hh.columns = ["lgbm", "naive_week"]
hh["lgbm - naive_week"] = hh.lgbm - hh.naive_week
table(hh.loc[16:23].rename_axis("hour"))
p("Hours 19-21 only, by month (MAE, MW) and mean signed error lgbm-actual:")
p()
bym = (
    ev.assign(m=ev.day.dt.strftime("%Y-%m"))
    .groupby("m")
    .agg(lgbm=("err_lgbm", "mean"), naive_week=("err_nw", "mean"), n=("y", "size"))
)
bym["lgbm bias"] = ev.assign(m=ev.day.dt.strftime("%Y-%m"), e=ev.lgbm - ev.y).groupby("m").e.mean()
table(bym)
dst = (ev.day >= "2026-10-24") & (ev.day <= "2026-10-26")
p(f"Around the DST change (days 2026-10-24 to 2026-10-26, hours 19-21): lgbm MAE "
  f"{ev[dst].err_lgbm.mean():,.0f}, naive_week {ev[dst].err_nw.mean():,.0f} (n={int(dst.sum())}). "
  f"Without those days: lgbm {ev[~dst].err_lgbm.mean():,.0f}, naive_week {ev[~dst].err_nw.mean():,.0f}.")  # fmt: skip
p()
p("Share of the test days where lgbm has the larger absolute error than naive_week, hours 19-21: "
  f"{(ev.err_lgbm > ev.err_nw).mean() * 100:.0f}% (all hours: {(pr.err_lgbm > pr.err_nw).mean() * 100:.0f}%).")  # fmt: skip


def sunset_local(day, lat=51.5, lon=7.0):
    """Approximate sunset (local clock, Europe/Berlin) with the NOAA sunrise equation."""
    n = day.dayofyear
    g = 2 * math.pi / 365 * (n - 1)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))  # fmt: skip
    dec = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
           + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))  # fmt: skip
    ha = math.degrees(math.acos(math.cos(math.radians(90.833)) / (math.cos(math.radians(lat))
        * math.cos(dec)) - math.tan(math.radians(lat)) * math.tan(dec)))  # fmt: skip
    utc_min = 720 - 4 * lon + 4 * ha - eqt
    t = pd.Timestamp(day).normalize().tz_localize("UTC") + pd.Timedelta(minutes=utc_min)
    return t.tz_convert(TZ)


p()
p("Approximate sunset, Essen area (51.5 N, 7.0 E, NOAA formula, about +-3 min), local clock time:")
p()
for d in [
    "2026-08-03",
    "2026-08-31",
    "2026-09-15",
    "2026-10-01",
    "2026-10-24",
    "2026-10-26",
    "2026-10-31",
]:
    p(f"- {d}: {sunset_local(pd.Timestamp(d)).strftime('%H:%M')}")

(OUT / "investigation.md").write_text(
    "# Investigation (generated by scripts/investigate.py)\n" + "\n".join(lines) + "\n"
)
print("\n".join(lines))
