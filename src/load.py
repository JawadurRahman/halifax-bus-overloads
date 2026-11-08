"""Load and clean the raw Halifax Transit datasets.

Outputs (data/processed/):
    overloads.parquet  - one row per bus overload incident, Halifax local time,
                         mapped to an APC service date + half-hour slot and a period
    apc.parquet        - all APC years (2017-2026) in one half-hourly table

Run from the project root:  python src/load.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

TZ = "America/Halifax"
APC_YEARS = range(2017, 2027)

# APC service day runs 04:00 -> 27:59 (hours after midnight are written as 24.x-27.x
# and belong to the PREVIOUS service day). Clock times before this hour therefore
# belong to the previous service day.
SERVICE_DAY_START_HOUR = 4

# Periods on the service-hour clock: [start, end)
PERIODS = {"AM": (6, 10), "MID": (10, 14), "PM": (14, 19), "EVE": (19, 24)}
PERIOD_ORDER = list(PERIODS)

# Route families: route 10 (Dalhousie) was split into 10A/10B/10C on 2024-11-18.
# We treat them as one route "10" in the modelling panel (decision 2026-11-08).
ROUTE_GROUPS = {"10A": "10", "10B": "10", "10C": "10"}

# APC counter-glitch rule (decision 2026-11-08): a half-hour value is an error if it is
# > OUTLIER_ABS boardings AND > OUTLIER_RATIO x the median for that
# year/route/direction/half-hour slot. Such values are set to NaN (= "no data").
OUTLIER_ABS = 500
OUTLIER_RATIO = 5

# Overload points outside the Halifax area (2 points near lat 45.9, lon -64.6)
MAX_LATITUDE = 44.9


def route_group(route: pd.Series) -> pd.Series:
    return route.replace(ROUTE_GROUPS)


def assign_period(service_hour: pd.Series) -> pd.Series:
    """Map a service hour (float, e.g. 7.5 or 24.5) to AM/MID/PM/EVE, else NaN."""
    out = pd.Series(pd.NA, index=service_hour.index, dtype="object")
    for name, (lo, hi) in PERIODS.items():
        out[(service_hour >= lo) & (service_hour < hi)] = name
    return pd.Categorical(out, categories=PERIOD_ORDER, ordered=True)


# ---------------------------------------------------------------------------
# Overload incidents
# ---------------------------------------------------------------------------
def load_overloads(verbose: bool = True) -> pd.DataFrame:
    path = RAW / "overloads" / "overload_incidents.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"ROUTE_NUMBER": str})
    n0 = len(df)

    assert (df["INCIDENT_OVERLOAD"] == "Passenger Overload Start").all()

    # 1) UTC -> Halifax local time, before anything else
    t_utc = pd.to_datetime(df["DATETIME_OCURRED"], format="%m/%d/%Y %I:%M:%S %p")
    df["datetime_utc"] = t_utc.dt.tz_localize("UTC")
    df["datetime_local"] = df["datetime_utc"].dt.tz_convert(TZ)

    # 2) drop ferry rows and out-of-area points
    is_ferry = df["ROUTE_NUMBER"].str.startswith("Fer")
    bad_loc = df["LATITUDE"] > MAX_LATITUDE
    df = df[~is_ferry & ~bad_loc].copy()

    # 3) map to APC service date + half-hour slot
    local = df["datetime_local"]
    clock_hour = local.dt.hour + local.dt.minute / 60 + local.dt.second / 3600
    after_midnight = local.dt.hour < SERVICE_DAY_START_HOUR
    cal_date = local.dt.tz_localize(None).dt.normalize()
    df["service_date"] = cal_date - pd.to_timedelta(after_midnight.astype(int), unit="D")
    service_hour = clock_hour + 24 * after_midnight
    df["route_hour"] = np.floor(service_hour * 2) / 2  # half-hour slot, APC convention
    df["clock_hour"] = local.dt.hour
    df["period"] = assign_period(df["route_hour"])

    df = df.rename(
        columns={
            "INCIDENT_REPORT_ID": "incident_id",
            "ROUTE_NUMBER": "route",
            "LATITUDE": "lat",
            "LONGITUDE": "lon",
        }
    )
    df["route_group"] = route_group(df["route"])
    df["year"] = df["service_date"].dt.year
    df["weekday"] = df["service_date"].dt.dayofweek  # 0 = Monday

    cols = [
        "incident_id", "datetime_utc", "datetime_local", "service_date", "route_hour",
        "clock_hour", "period", "route", "route_group", "lat", "lon", "year", "weekday",
    ]
    df = df[cols].sort_values("datetime_local").reset_index(drop=True)

    if verbose:
        print(f"[overloads] raw rows: {n0:,}")
        print(f"[overloads] dropped ferry: {is_ferry.sum()}  out-of-area (lat>{MAX_LATITUDE}): {bad_loc.sum()}")
        print(f"[overloads] clean rows: {len(df):,}  "
              f"({df.datetime_local.min():%Y-%m-%d} -> {df.datetime_local.max():%Y-%m-%d}, local)")
        print(f"[overloads] after-midnight incidents (moved to previous service day): {int((df.route_hour >= 24).sum())}")
        print(f"[overloads] outside AM/MID/PM/EVE periods: {df.period.isna().sum()}")
    return df


# ---------------------------------------------------------------------------
# APC boardings
# ---------------------------------------------------------------------------
def _read_apc_year(year: int) -> pd.DataFrame:
    path = RAW / "apc" / f"apc_{year}.csv"
    df = pd.read_csv(
        path,
        dtype={"Route_Number": str, "Route_Name": str, "Route_Direction": str,
               "Ridership_Total": "Int32", "Route_Hour": float},
        usecols=["Route_Number", "Route_Name", "Route_Direction", "Ridership_Total",
                 "Route_Hour", "Route_Date"],
    )
    # Date formats differ: archive "2017.10.04 10:00:00", current "10/22/2026 10:00:00 AM".
    # The time part is a dummy; keep only the date.
    date_str = df["Route_Date"].str.split(" ").str[0]
    fmt = "%Y.%m.%d" if "." in date_str.iloc[0] else "%m/%d/%Y"
    df["service_date"] = pd.to_datetime(date_str, format=fmt)
    assert df["service_date"].notna().all(), year
    df["file_year"] = year
    return df.drop(columns="Route_Date")


def load_apc(years=APC_YEARS, verbose: bool = True) -> pd.DataFrame:
    parts = []
    for y in years:
        d = _read_apc_year(y)
        parts.append(d)
        if verbose:
            print(f"[apc] {y}: {len(d):>9,} rows  {d.service_date.min():%Y-%m-%d} -> "
                  f"{d.service_date.max():%Y-%m-%d}  ({d.service_date.nunique()} dates, "
                  f"{d.Route_Number.nunique()} routes)")
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(
        columns={
            "Route_Number": "route",
            "Route_Name": "route_name",
            "Route_Direction": "direction",
            "Ridership_Total": "ridership_raw",
            "Route_Hour": "route_hour",
        }
    )
    df["route_name"] = df["route_name"].str.strip()

    # sanity: no duplicate keys
    key = ["route", "direction", "service_date", "route_hour"]
    n_dup = df.duplicated(key).sum()
    assert n_dup == 0, f"{n_dup} duplicate APC keys"

    # counter-glitch outliers -> NaN
    med = df.groupby(["file_year", "route", "direction", "route_hour"])["ridership_raw"].transform("median")
    df["is_outlier"] = (df["ridership_raw"] > OUTLIER_ABS) & (df["ridership_raw"] > OUTLIER_RATIO * med)
    df["ridership"] = df["ridership_raw"].astype("float32").where(~df["is_outlier"])

    df["route_group"] = route_group(df["route"])
    df["period"] = assign_period(df["route_hour"])
    df["weekday"] = df["service_date"].dt.dayofweek
    df["year"] = df["service_date"].dt.year.astype("int16")
    df["route_hour"] = df["route_hour"].astype("float32")
    df["ridership_raw"] = df["ridership_raw"].astype("int32")
    for c in ["route", "route_group", "route_name", "direction"]:
        df[c] = df[c].astype("category")

    df = df[["service_date", "year", "weekday", "route", "route_group", "route_name",
             "direction", "route_hour", "period", "ridership_raw", "ridership", "is_outlier"]]
    df = df.sort_values(["service_date", "route", "direction", "route_hour"]).reset_index(drop=True)

    if verbose:
        n_out = int(df.is_outlier.sum())
        print(f"[apc] total rows: {len(df):,}")
        print(f"[apc] outliers set to NaN: {n_out} rows "
              f"(removed {df.loc[df.is_outlier, 'ridership_raw'].sum():,} of "
              f"{df.ridership_raw.sum():,} boardings)")
        print("[apc] outliers by year:", df[df.is_outlier].groupby('year').size().to_dict())
        print(f"[apc] max ridership after cleaning: {df.ridership.max():.0f}")
        all_days = pd.date_range(df.service_date.min(), df.service_date.max(), freq="D")
        missing = all_days.difference(pd.DatetimeIndex(df.service_date.unique()))
        print(f"[apc] calendar days with NO APC data at all: {len(missing)}")
        if len(missing):
            print("       ", ", ".join(d.strftime("%Y-%m-%d") for d in missing[:40]),
                  "..." if len(missing) > 40 else "")
    return df


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    ovl = load_overloads()
    ovl.to_parquet(PROCESSED / "overloads.parquet", index=False)
    print(f"-> wrote {PROCESSED / 'overloads.parquet'}\n")

    apc = load_apc()
    apc.to_parquet(PROCESSED / "apc.parquet", index=False)
    print(f"-> wrote {PROCESSED / 'apc.parquet'}")


if __name__ == "__main__":
    main()
