"""Round-3 feature groups: overload history, university calendar, better ridership.

All features use ONLY information from before the predicted day. Look-backs run over the
panel's own date grid, which already excludes weekends and holidays, so a holiday never
counts as "last week" or "yesterday".

Input : data/processed/panel.parquet (2020+, for look-back history)
        data/processed/panel_weather.parquet (2022+, the modelling rows)
        data/processed/apc.parquet
        data/external/academic_calendar.csv
Output: data/processed/panel_features.parquet  (same rows as panel_weather + new columns)

Run from the project root:  python src/features.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from load import PERIOD_ORDER, PROCESSED, ROOT

CALENDAR_CSV = ROOT / "data" / "external" / "academic_calendar.csv"
DAYS_SINCE_CAP = 60
RATIO_CLIP = (0.25, 4.0)

HIST = ["ovl_days_prev5", "ovl_weeks_prev4_sameday", "days_since_ovl"]
CAL = ["in_term", "week_of_term", "is_first3w_fall", "is_break", "is_exam"]
RIDE = ["boardings_lag1w", "boardings_mean4w", "peak_hh_lag1w", "peak_hh_mean4w",
        "peakdir_lag1w", "peakdir_mean4w", "yoy_ratio", "lag_ratio"]


# ---------------------------------------------------------------------------
# 1. overload history
# ---------------------------------------------------------------------------
def overload_history(p: pd.DataFrame) -> pd.DataFrame:
    """p: full panel (route, period, date, dow, y), weekdays without holidays."""
    p = p.sort_values(["route", "period", "date"]).copy()
    prev = p.groupby(["route", "period"]).y.shift(1)
    # weekdays with an overload among the previous 5 (non-holiday) weekdays: 0-5
    p["ovl_days_prev5"] = prev.groupby([p.route, p.period]).transform(
        lambda s: s.rolling(5, min_periods=1).sum())
    # overload weeks among the previous 4 same-weekday dates: 0-4
    p = p.sort_values(["route", "period", "dow", "date"])
    prev_wd = p.groupby(["route", "period", "dow"]).y.shift(1)
    p["ovl_weeks_prev4_sameday"] = prev_wd.groupby([p.route, p.period, p.dow]).transform(
        lambda s: s.rolling(4, min_periods=1).sum())
    # calendar days since the last overload on this route + period (before today), capped
    p = p.sort_values(["route", "period", "date"])
    last = p["date"].where(p.y == 1)
    last = last.groupby([p.route, p.period]).shift(1)
    last = last.groupby([p.route, p.period]).ffill()
    p["days_since_ovl"] = (p["date"] - last).dt.days.fillna(DAYS_SINCE_CAP).clip(upper=DAYS_SINCE_CAP)
    return p[["route", "period", "date"] + HIST]


# ---------------------------------------------------------------------------
# 2. university calendar
# ---------------------------------------------------------------------------
def calendar_features(dates: pd.Series) -> pd.DataFrame:
    cal = pd.read_csv(CALENDAR_CSV, parse_dates=["start_date", "end_date"])
    d = pd.DataFrame({"date": pd.to_datetime(dates.unique())}).sort_values("date")
    for c in CAL:
        d[c] = 0
    covered = pd.Series(False, index=d.index)
    for _, r in cal.iterrows():
        inside = d.date.between(r.start_date, r.end_date)
        if r.event in ("fall_classes", "winter_classes"):
            wk = ((d.date - r.start_date).dt.days // 7 + 1).where(inside, 0)
            d["week_of_term"] = np.where(inside, wk, d["week_of_term"])
            d.loc[inside, "in_term"] = 1
            if r.event == "fall_classes":
                d.loc[inside & (wk <= 3), "is_first3w_fall"] = 1
        elif r.event.endswith("_break"):
            d.loc[inside, "is_break"] = 1
        elif r.event.endswith("_exams"):
            d.loc[inside, "is_exam"] = 1
    # a study break is not class time (week_of_term keeps counting through it)
    d.loc[d.is_break == 1, "in_term"] = 0
    first_year = int(cal.academic_year.str[:4].astype(int).min())
    assert d.date.min() >= pd.Timestamp(f"{first_year}-09-01"), "calendar does not cover the panel"
    return d


# ---------------------------------------------------------------------------
# 3. better ridership (lagged)
# ---------------------------------------------------------------------------
def ridership_daily(routes: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Same-day peak half-hour and peak-direction boardings per route x date x period (NOT features
    by themselves - only their lags are used)."""
    a = pd.read_parquet(PROCESSED / "apc.parquet",
                        columns=["service_date", "route_group", "direction", "route_hour", "period", "ridership"],
                        filters=[("service_date", ">=", start), ("service_date", "<=", end)])
    a = a[a.route_group.isin(routes) & a.period.notna()].copy()
    for c in ["route_group", "period", "direction"]:
        a[c] = a[c].astype(str)
    keys = ["route_group", "service_date", "period"]
    slot = a.groupby(keys + ["route_hour"]).ridership.sum(min_count=1)
    peak_hh = slot.groupby(level=keys).max().rename("peak_hh")
    dirn = a.groupby(keys + ["direction"]).ridership.sum(min_count=1)
    peak_dir = dirn.groupby(level=keys).max().rename("peakdir")
    out = pd.concat([peak_hh, peak_dir], axis=1).reset_index()
    return out.rename(columns={"route_group": "route", "service_date": "date"})


def lag_same_weekday(p: pd.DataFrame, col: str) -> pd.DataFrame:
    """1-week lag and mean of the previous 4 same-weekday (non-holiday) values of `col`."""
    p = p.sort_values(["route", "period", "dow", "date"])
    prev = p.groupby(["route", "period", "dow"])[col].shift(1)
    p[f"{col}_lag1w"] = prev
    p[f"{col}_mean4w"] = prev.groupby([p.route, p.period, p.dow]).transform(
        lambda s: s.rolling(4, min_periods=1).mean())
    return p


def ridership_features(full: pd.DataFrame) -> pd.DataFrame:
    routes = sorted(full.route.unique())
    rd = ridership_daily(routes, full.date.min() - pd.Timedelta(days=60), full.date.max())
    p = full[["route", "period", "date", "dow", "boardings_lag1w", "boardings_mean4w"]].merge(
        rd, on=["route", "period", "date"], how="left")
    p = lag_same_weekday(p, "peak_hh")
    p = lag_same_weekday(p, "peakdir")

    # year-over-year: this 4-week mean vs the same weeks a year earlier (52 weeks back, same weekday,
    # nearest available non-holiday date within +-7 days)
    p = p.sort_values("date")
    prior = p[["route", "period", "dow", "date", "boardings_mean4w"]].rename(
        columns={"boardings_mean4w": "mean4w_lastyear", "date": "date_ly"})
    look = p[["route", "period", "dow", "date"]].assign(date_ly=p.date - pd.Timedelta(days=364))
    look = look.sort_values("date_ly")
    prior = prior.sort_values("date_ly")
    ly = pd.merge_asof(look, prior, on="date_ly", by=["route", "period", "dow"],
                       direction="nearest", tolerance=pd.Timedelta(days=7))
    p = p.merge(ly[["route", "period", "date", "mean4w_lastyear"]], on=["route", "period", "date"], how="left")
    p["yoy_ratio"] = (p.boardings_mean4w / p.mean4w_lastyear).clip(*RATIO_CLIP)
    p["lag_ratio"] = (p.boardings_lag1w / p.boardings_mean4w).clip(*RATIO_CLIP)
    p = p.rename(columns={"peak_hh_lag1w": "peak_hh_lag1w", "peakdir_lag1w": "peakdir_lag1w"})
    return p[["route", "period", "date"] + [c for c in RIDE if c not in ("boardings_lag1w", "boardings_mean4w")]]


def main() -> None:
    full = pd.read_parquet(PROCESSED / "panel.parquet")
    for c in ["route", "period"]:
        full[c] = full[c].astype(str)
    pw = pd.read_parquet(PROCESSED / "panel_weather.parquet")
    routes = list(pw.route.cat.categories)
    for c in ["route", "period"]:
        pw[c] = pw[c].astype(str)

    hist = overload_history(full)
    ride = ridership_features(full)
    cal = calendar_features(pw.date)

    out = (pw.merge(hist, on=["route", "period", "date"], how="left", validate="one_to_one")
             .merge(ride, on=["route", "period", "date"], how="left", validate="one_to_one")
             .merge(cal, on="date", how="left", validate="many_to_one"))
    assert len(out) == len(pw)
    out["route"] = pd.Categorical(out.route, categories=routes)
    out["period"] = pd.Categorical(out.period, categories=PERIOD_ORDER)
    out.to_parquet(PROCESSED / "panel_features.parquet", index=False)

    print(f"-> wrote panel_features.parquet {out.shape} (same rows as panel_weather: {len(pw):,})")
    new = HIST + CAL + RIDE[2:]
    print("\nmissing % of new features:\n", (100 * out[new].isna().mean()).round(2).to_string())
    print("\n", out[new].describe().T[["mean", "min", "50%", "max"]].round(2).to_string())
    # quick signal check (2022-2024 = training years only)
    t = out[out.year <= 2024]
    print("\noverload rate by ovl_days_prev5 (train):", t.groupby("ovl_days_prev5").y.mean().round(3).to_dict())
    print("overload rate by is_first3w_fall (train):", t.groupby("is_first3w_fall").y.mean().round(3).to_dict())
    print("overload rate in_term vs not (train):", t.groupby("in_term").y.mean().round(3).to_dict())
    print("overload rate exam vs not (train):", t.groupby("is_exam").y.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
