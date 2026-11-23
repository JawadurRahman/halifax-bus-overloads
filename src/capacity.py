"""Round 4: scheduled-capacity features from Halifax Transit GTFS schedules.

Source: Transitland feed archive, feed f-dxfy-halifaxtransit~hrm (27 versions, Aug 2021 - Nov 2026),
downloaded to data/external/gtfs/<sha1>.zip with the versions listed in needed_versions.json
[sha1, fetched_at, earliest_service_date, latest_service_date].
Licence: Contains information licenced under the Open Government Licence - Halifax.

No look-ahead: for each service date we use the LATEST version fetched BEFORE that date whose
calendar covers it (a published schedule is known in advance). If no version fetched before the
date covers it, the earliest covering version is used and `sched_version_late` = 1.

Features per route group x service date x period (trips counted by their first departure time):
    sched_trips            scheduled trips starting in the period, both directions
    sched_trips_peakdir    trips in the direction with more trips
    sched_trips_per_hour   sched_trips / period length (h)
    load_per_trip          boardings_mean4w / sched_trips          (recent demand per scheduled bus)
    peakdir_load_per_trip  peakdir_mean4w / sched_trips_peakdir
    sched_change_4w        sched_trips / sched_trips 4 same-weekdays ago (clipped 0.25-4)

Output: data/processed/panel_capacity.parquet (panel_features rows + these columns)
Run from the project root:  python src/capacity.py
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd

from load import PERIODS, PERIOD_ORDER, PROCESSED, ROOT, route_group

GTFS_DIR = ROOT / "data" / "external" / "gtfs"
CAP = ["sched_trips", "sched_trips_peakdir", "sched_trips_per_hour",
       "load_per_trip", "peakdir_load_per_trip", "sched_change_4w"]
PERIOD_HOURS = {p: hi - lo for p, (lo, hi) in PERIODS.items()}


def _hours(t: pd.Series) -> pd.Series:
    hms = t.str.split(":", expand=True).astype(float)
    return hms[0] + hms[1] / 60 + hms[2] / 3600


def load_version(sha: str, routes: list[str]):
    """Per service_id: scheduled trip counts by route group x period x direction."""
    with zipfile.ZipFile(GTFS_DIR / f"{sha}.zip") as z:
        r = pd.read_csv(z.open("routes.txt"), dtype=str, usecols=["route_id", "route_short_name"])
        t = pd.read_csv(z.open("trips.txt"), dtype=str, usecols=["route_id", "service_id", "trip_id", "direction_id"])
        st = pd.read_csv(z.open("stop_times.txt"), dtype=str, usecols=["trip_id", "stop_sequence", "departure_time"])
        cal = pd.read_csv(z.open("calendar.txt"), dtype=str) if "calendar.txt" in z.namelist() else pd.DataFrame()
        cd = pd.read_csv(z.open("calendar_dates.txt"), dtype=str) if "calendar_dates.txt" in z.namelist() else pd.DataFrame()
    st["stop_sequence"] = st.stop_sequence.astype(int)
    first = st.sort_values(["trip_id", "stop_sequence"]).drop_duplicates("trip_id")
    first = first[first.departure_time.notna()]
    first["start_h"] = _hours(first.departure_time)
    t = t.merge(r, on="route_id").merge(first[["trip_id", "start_h"]], on="trip_id")
    t["route"] = route_group(t.route_short_name.str.strip())
    t = t[t.route.isin(routes)]
    t["period"] = pd.NA
    for p, (lo, hi) in PERIODS.items():
        t.loc[(t.start_h >= lo) & (t.start_h < hi), "period"] = p
    t = t[t.period.notna()]
    counts = t.groupby(["service_id", "route", "period", "direction_id"]).size().rename("n").reset_index()
    return counts, cal, cd


def active_service_ids(cal: pd.DataFrame, cd: pd.DataFrame, d: pd.Timestamp) -> set[str]:
    ds, wd = d.strftime("%Y%m%d"), ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][d.dayofweek]
    on = set()
    if len(cal):
        m = (cal.start_date <= ds) & (cal.end_date >= ds) & (cal[wd] == "1")
        on = set(cal.loc[m, "service_id"])
    if len(cd):
        x = cd[cd.date == ds]
        on |= set(x.loc[x.exception_type == "1", "service_id"])
        on -= set(x.loc[x.exception_type == "2", "service_id"])
    return on


def scheduled_trips(dates: pd.DatetimeIndex, routes: list[str]) -> pd.DataFrame:
    versions = pd.DataFrame(json.load(open(GTFS_DIR / "needed_versions.json")),
                            columns=["sha", "fetched", "start", "end"])
    for c in ["fetched", "start", "end"]:
        versions[c] = pd.to_datetime(versions[c])
    cache, rows, late = {}, [], 0
    for d in dates:
        cover = versions[(versions.start <= d) & (versions.end >= d)]
        known = cover[cover.fetched < d]
        if len(known):
            sha, is_late = known.sort_values("fetched").sha.iloc[-1], 0
        elif len(cover):
            sha, is_late = cover.sort_values("fetched").sha.iloc[0], 1
            late += 1
        else:
            continue
        if sha not in cache:
            cache[sha] = load_version(sha, routes)
        counts, cal, cd = cache[sha]
        act = counts[counts.service_id.isin(active_service_ids(cal, cd, d))]
        by_dir = act.groupby(["route", "period", "direction_id"]).n.sum()
        tot = by_dir.groupby(level=["route", "period"]).sum().rename("sched_trips")
        pk = by_dir.groupby(level=["route", "period"]).max().rename("sched_trips_peakdir")
        out = pd.concat([tot, pk], axis=1).reset_index().assign(date=d, sched_version=sha[:8], sched_version_late=is_late)
        rows.append(out)
    print(f"schedule versions used: {len(cache)}; dates using a version fetched on/after the date: {late}")
    return pd.concat(rows, ignore_index=True)


def download() -> None:
    """Download the versions in needed_versions.json from the Transitland v2 API.

    Needs an API key with feed-version download access in the TRANSITLAND_API_KEY environment variable.
    """
    import os
    import urllib.request
    key = os.environ.get("TRANSITLAND_API_KEY")
    assert key, "set TRANSITLAND_API_KEY (Transitland API key with feed-version download access)"
    for sha, *_ in json.load(open(GTFS_DIR / "needed_versions.json")):
        out = GTFS_DIR / f"{sha}.zip"
        if out.exists():
            continue
        req = urllib.request.Request(f"https://transit.land/api/v2/rest/feed_versions/{sha}/download",
                                     headers={"apikey": key})
        tmp = out.with_suffix(".part")
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            f.write(r.read())
        zipfile.ZipFile(tmp).testzip()
        tmp.rename(out)
        print("downloaded", sha)


def main() -> None:
    import sys
    if "--download" in sys.argv:
        download()
    from features import lag_same_weekday   # same-weekday look-back helper (skips holidays)
    full = pd.read_parquet(PROCESSED / "panel.parquet")
    pf = pd.read_parquet(PROCESSED / "panel_features.parquet")
    routes = list(pf.route.cat.categories)
    for d in (full, pf):
        for c in ["route", "period"]:
            d[c] = d[c].astype(str)
    grid_dates = pd.DatetimeIndex(sorted(full.date[full.date >= "2021-11-22"].unique()))
    sched = scheduled_trips(grid_dates, routes)

    g = full[full.date >= "2021-11-22"][["route", "period", "date", "dow"]].merge(
        sched, on=["route", "period", "date"], how="left")
    # a route-period with an active schedule but no trips starting in it = 0 scheduled trips
    has_sched = g.date.isin(sched.date.unique())
    for c in ["sched_trips", "sched_trips_peakdir"]:
        g.loc[has_sched & g[c].isna(), c] = 0
    g = lag_same_weekday(g, "sched_trips").rename(columns={"sched_trips_lag1w": "_l1", "sched_trips_mean4w": "_m4"})
    # sched_trips 4 same-weekdays ago: shift(4) within route x period x weekday
    g = g.sort_values(["route", "period", "dow", "date"])
    g["_t4"] = g.groupby(["route", "period", "dow"]).sched_trips.shift(4)
    g["sched_change_4w"] = (g.sched_trips / g._t4).replace([np.inf, -np.inf], np.nan).clip(0.25, 4)

    out = pf.merge(g[["route", "period", "date", "sched_trips", "sched_trips_peakdir", "sched_change_4w",
                      "sched_version", "sched_version_late"]], on=["route", "period", "date"], how="left",
                   validate="one_to_one")
    out["sched_trips_per_hour"] = out.sched_trips / out.period.map(PERIOD_HOURS)
    trips = out.sched_trips.where(out.sched_trips > 0)
    out["load_per_trip"] = out.boardings_mean4w / trips
    out["peakdir_load_per_trip"] = out.peakdir_mean4w / out.sched_trips_peakdir.where(out.sched_trips_peakdir > 0)
    out["route"] = pd.Categorical(out.route, categories=routes)
    out["period"] = pd.Categorical(out.period, categories=PERIOD_ORDER)
    assert len(out) == len(pf)
    out.to_parquet(PROCESSED / "panel_capacity.parquet", index=False)

    print(f"-> wrote panel_capacity.parquet {out.shape}")
    print("missing %:\n", (100 * out[CAP].isna().mean()).round(2).to_string())
    print("rows using a late-fetched version:", int(out.sched_version_late.fillna(0).sum()))
    print("\n", out[CAP].describe().T[["mean", "min", "50%", "max"]].round(2).to_string())
    t = out[out.year <= 2024]
    print("\noverload rate by load_per_trip decile (train 2022-24):",
          t.groupby(pd.qcut(t.load_per_trip, 10, labels=False), observed=True).y.mean().round(3).to_dict())
    print("scheduled PM trips, route 3, by year:",
          out[(out.route == "3") & (out.period == "PM")].groupby("year").sched_trips.median().to_dict())


if __name__ == "__main__":
    main()
