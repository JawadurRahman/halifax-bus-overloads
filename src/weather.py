"""Environment Canada weather for Halifax Stanfield -> features per date x period.

Station: HALIFAX STANFIELD INT'L A, Climate ID 8202251, Station ID 50620 (hourly + daily).
Raw files: data/external/weather/hourly/stanfield_hourly_YYYY_MM.csv
           data/external/weather/daily/stanfield_daily_YYYY.csv

Hourly data has temperature, wind, visibility and a "Weather" text field but NO precipitation
amounts at this station (100% missing), so precipitation / snowfall / snow-on-ground come from
the daily file and are the same for all four periods of a day.

Outputs (data/processed/):
    weather_features.parquet   one row per date x period
    panel_weather.parquet      panel.parquet (2022+) with weather features joined

Assumption: weather for the SAME day/period is used as a feature. In practice a forecast would be
available in advance, so this measures the value of (perfect) weather forecasts.

Run from the project root:  python src/weather.py            (build features)
                            python src/weather.py --download (re-download raw files first)
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from load import PERIOD_ORDER, PERIODS, PROCESSED, ROOT, TZ

WEATHER_DIR = ROOT / "data" / "external" / "weather"
STATION_ID = 50620
START, END = "2021-12-01", "2026-10-31"
BULK_URL = ("https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv"
            "&stationID={sid}&Year={y}&Month={m}&Day=1&timeframe={tf}&submit=Download+Data")

# "bad weather" thresholds (documented in notebooks/02_models.ipynb)
BAD_SNOW_CM = 2.0       # daily snowfall, AND snow reported during the period
BAD_PRECIP_MM = 15.0    # daily total precipitation, AND rain reported during the period
BAD_WIND_KMH = 50.0     # max hourly wind speed in the period
# Visibility is NOT used in the flag: Stanfield is an inland airport ~30 km from downtown with
# frequent morning fog that does not reflect conditions on city streets (vis_min is still a feature).


def download() -> None:
    (WEATHER_DIR / "hourly").mkdir(parents=True, exist_ok=True)
    (WEATHER_DIR / "daily").mkdir(parents=True, exist_ok=True)
    for p in pd.period_range(START[:7], END[:7], freq="M"):
        out = WEATHER_DIR / "hourly" / f"stanfield_hourly_{p.year}_{p.month:02d}.csv"
        urllib.request.urlretrieve(BULK_URL.format(sid=STATION_ID, y=p.year, m=p.month, tf=1), out)
    for y in range(int(START[:4]), int(END[:4]) + 1):
        out = WEATHER_DIR / "daily" / f"stanfield_daily_{y}.csv"
        urllib.request.urlretrieve(BULK_URL.format(sid=STATION_ID, y=y, m=1, tf=2), out)
    print("downloaded to", WEATHER_DIR)


def _read(pattern: str) -> pd.DataFrame:
    files = sorted((WEATHER_DIR).glob(pattern))
    assert files, f"no files for {pattern}"
    return pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)


def load_hourly() -> pd.DataFrame:
    h = _read("hourly/*.csv")
    h.columns = [c.replace("°", "") for c in h.columns]
    # Timestamps are Local STANDARD Time (AST = UTC-4) all year, no daylight saving.
    lst = pd.to_datetime(h["Date/Time (LST)"])
    local = lst.dt.tz_localize("Etc/GMT+4").dt.tz_convert(TZ).dt.tz_localize(None)  # Etc/GMT+4 = UTC-4
    wx = h["Weather"].where(h["Weather"].ne("NA")).astype("string")
    out = pd.DataFrame({
        "local": local,
        "temp": pd.to_numeric(h["Temp (C)"], errors="coerce"),
        "wind": pd.to_numeric(h["Wind Spd (km/h)"], errors="coerce"),
        "vis": pd.to_numeric(h["Visibility (km)"], errors="coerce"),
        "wx": wx,
    })
    out["date"] = out["local"].dt.normalize()
    hour = out["local"].dt.hour
    out["period"] = pd.NA
    for name, (lo, hi) in PERIODS.items():
        out.loc[(hour >= lo) & (hour < hi), "period"] = name
    out["wx_reported"] = out["wx"].notna().astype(bool)
    w = out["wx"].fillna("").astype(str)
    out["wx_freezing"] = w.str.contains("Freezing Rain|Freezing Drizzle|Ice Pellets", regex=True)
    out["wx_snow"] = w.str.contains("Snow", regex=False) & ~w.str.fullmatch(r"(Blowing|Drifting) Snow.*")
    out["wx_rain"] = w.str.contains("Rain|Drizzle", regex=True) & ~out["wx_freezing"]
    return out


def load_daily() -> pd.DataFrame:
    d = _read("daily/*.csv")
    d.columns = [c.replace("°", "") for c in d.columns]
    out = pd.DataFrame({
        "date": pd.to_datetime(d["Date/Time"]),
        "precip_mm": pd.to_numeric(d["Total Precip (mm)"], errors="coerce"),
        "rain_mm": pd.to_numeric(d["Total Rain (mm)"], errors="coerce"),
        "snow_cm": pd.to_numeric(d["Total Snow (cm)"], errors="coerce"),
        "snow_ground_cm": pd.to_numeric(d["Snow on Grnd (cm)"], errors="coerce"),
        "gust_max_kmh": pd.to_numeric(d["Spd of Max Gust (km/h)"], errors="coerce"),
    }).sort_values("date").reset_index(drop=True)
    # snow on ground: not reported May-Oct (= no snow); short winter gaps forward-filled (<=3 days)
    summer = out["date"].dt.month.between(5, 10)
    out.loc[summer & out.snow_ground_cm.isna() & (out.snow_cm.fillna(0) == 0), "snow_ground_cm"] = 0.0
    out["snow_ground_cm"] = out["snow_ground_cm"].ffill(limit=3)
    # shoulder months (Apr, Nov, ...): still missing and no snowfall in the past 7 days -> bare ground
    no_recent_snow = out["snow_cm"].fillna(0).rolling(7, min_periods=1).sum() == 0
    out.loc[out.snow_ground_cm.isna() & no_recent_snow, "snow_ground_cm"] = 0.0
    # gusts are only reported above a threshold (~31 km/h) -> missing means "no notable gust"
    out["gust_max_kmh"] = out["gust_max_kmh"].fillna(0.0)
    out["snow_cm_prev1d"] = out["snow_cm"].shift(1)       # yesterday's snowfall (clean-up days)
    return out


def build_features() -> pd.DataFrame:
    h = load_hourly()
    h = h[h.period.notna()]
    g = h.groupby(["date", "period"])
    per = g.agg(
        temp_mean=("temp", "mean"), temp_min=("temp", "min"),
        wind_mean=("wind", "mean"), wind_max=("wind", "max"),
        vis_min=("vis", "min"),
        n_wx_reported=("wx_reported", "sum"),
    )
    for k in ["wx_snow", "wx_rain", "wx_freezing"]:
        # share of hours WITH a weather report that mention snow / rain / freezing precip
        per[k] = (g[k].sum().astype(float) / per["n_wx_reported"].astype(float).replace(0, np.nan))
    per = per.reset_index()

    d = load_daily()
    f = per.merge(d, on="date", how="left")
    f["bad_weather"] = (
        ((f.snow_cm >= BAD_SNOW_CM) & (f.wx_snow > 0))
        | ((f.precip_mm >= BAD_PRECIP_MM) & (f.wx_rain > 0))
        | (f.wx_freezing > 0)
        | (f.wind_max >= BAD_WIND_KMH)
    ).astype("int8")
    f["period"] = pd.Categorical(f["period"], categories=PERIOD_ORDER, ordered=True)
    f = f[(f.date >= START) & (f.date <= END)].sort_values(["date", "period"]).reset_index(drop=True)
    return f.drop(columns="n_wx_reported")


WEATHER_COLS = ["temp_mean", "temp_min", "wind_mean", "wind_max", "vis_min",
                "wx_snow", "wx_rain", "wx_freezing", "precip_mm", "rain_mm", "snow_cm",
                "snow_ground_cm", "gust_max_kmh", "snow_cm_prev1d", "bad_weather"]


def main() -> None:
    if "--download" in sys.argv:
        download()
    f = build_features()
    f.to_parquet(PROCESSED / "weather_features.parquet", index=False)
    print(f"weather features: {f.shape}, {f.date.min():%Y-%m-%d} -> {f.date.max():%Y-%m-%d}")
    print("missing %:\n", (100 * f[WEATHER_COLS].isna().mean()).round(2).to_string())
    print("\nbad_weather share by period:", f.groupby("period", observed=True).bad_weather.mean().round(3).to_dict())

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel = panel[panel.date >= "2022-01-01"]
    pw = panel.merge(f, on=["date", "period"], how="left", validate="many_to_one")
    pw["period"] = pd.Categorical(pw["period"], categories=PERIOD_ORDER, ordered=True)
    assert len(pw) == len(panel)
    pw.to_parquet(PROCESSED / "panel_weather.parquet", index=False)
    print(f"\n-> wrote panel_weather.parquet {pw.shape}")
    print("panel rows with any missing weather feature: "
          f"{pw[WEATHER_COLS].isna().any(axis=1).mean():.2%}")
    print("bad_weather share in panel:", f"{pw.bad_weather.mean():.3f}",
          "| overload rate bad vs not:", pw.groupby("bad_weather").y.mean().round(4).to_dict())
    chk = f[f.date.isin(pd.to_datetime(["2022-02-04", "2024-01-29", "2026-01-26"]))]
    print("\nlow-ridership days from EDA:\n",
          chk[["date", "period", "temp_mean", "snow_cm", "precip_mm", "wx_snow", "wx_freezing", "wind_max", "vis_min", "bad_weather"]].to_string(index=False))


if __name__ == "__main__":
    main()
