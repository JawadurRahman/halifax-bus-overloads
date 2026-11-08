"""Build the modelling panel: route x weekday date x period.

Rows   : top-N routes (by overload count since 2023) x non-holiday weekdays (NS public+optional+government holidays) x {AM, MID, PM, EVE}
Label  : y = 1 if at least one overload incident in that cell
Features (past information only, to avoid leakage - passed-up riders board the next bus,
so same-period boardings are partly caused by the outcome):
    boardings_lag1w   : boardings same route+period, same weekday, previous (non-holiday) week
    boardings_mean4w  : mean of the previous 4 (non-holiday) same-weekday values
    dow, period, month, year, is_covid (2020)
Weather / university calendar: placeholders (see add_weather_features / add_calendar_features).

Run from the project root after src/load.py:  python src/build_panel.py
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from load import PERIOD_ORDER, PROCESSED

TOP_N_ROUTES = 10
TOP_ROUTES_SINCE = "2023-01-01"
PANEL_START = "2020-01-01"          # overload data starts 2020-01-06
LAG_WEEKS = 4
LAG_BUFFER_DAYS = 7 * (LAG_WEEKS + 3)  # load APC this far before PANEL_START for lags


# Holiday rule (decision 2026-11-08): NS statutory holidays PLUS the "optional" and
# "government" categories (Boxing Day, Thanksgiving, Victoria Day, Natal Day, Truth &
# Reconciliation Day, Easter Monday, ...). APC ridership on those days is 0.3-0.6x a
# normal weekday, i.e. they behave like holidays for transit.
HOLIDAY_CATEGORIES = ("public", "optional", "government")


def ns_holidays(years) -> set[pd.Timestamp]:
    return {pd.Timestamp(d) for d in
            holidays.Canada(prov="NS", years=years, categories=HOLIDAY_CATEGORIES)}


def top_routes(ovl: pd.DataFrame, n: int = TOP_N_ROUTES) -> list[str]:
    since = ovl[ovl.service_date >= TOP_ROUTES_SINCE]
    counts = since.groupby("route_group").size().sort_values(ascending=False)
    print(f"Top {n} route groups by overload count since {TOP_ROUTES_SINCE}:")
    print(counts.head(n + 3).to_string(), "\n")
    return counts.head(n).index.tolist()


def period_boardings(apc: pd.DataFrame, routes: list[str]) -> pd.DataFrame:
    """Total boardings (both directions) per route_group x service_date x period.

    NaN if the route has no APC rows at all in that period that day (missing != 0).
    """
    a = apc[apc.route_group.isin(routes) & apc.period.notna()]
    g = a.groupby(["route_group", "service_date", "period"], observed=True)
    out = g.agg(boardings=("ridership", "sum"), n_slots=("ridership", "count")).reset_index()
    out.loc[out.n_slots == 0, "boardings"] = np.nan
    out["route_group"] = out["route_group"].astype(str)
    out["period"] = out["period"].astype(str)
    return out[["route_group", "service_date", "period", "boardings"]]


def add_lag_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Lags over the previous same-weekday dates, skipping holidays.

    `grid` must already exclude holidays and weekends, so shift(1) within
    (route, period, weekday) is the most recent non-holiday same weekday.
    """
    grid = grid.sort_values(["route", "period", "dow", "date"]).copy()
    g = grid.groupby(["route", "period", "dow"], sort=False)
    prev = g["boardings"].shift(1)
    grid["boardings_lag1w"] = prev
    grid["lag1w_days_ago"] = (grid["date"] - g["date"].shift(1)).dt.days
    grid["boardings_mean4w"] = (
        prev.groupby([grid.route, grid.period, grid.dow], sort=False)
        .transform(lambda s: s.rolling(LAG_WEEKS, min_periods=1).mean())
    )
    grid["n_weeks_mean4w"] = (
        prev.notna().groupby([grid.route, grid.period, grid.dow], sort=False)
        .transform(lambda s: s.rolling(LAG_WEEKS, min_periods=1).sum())
    ).astype("int8")
    return grid


def add_weather_features(panel: pd.DataFrame) -> pd.DataFrame:
    """PLACEHOLDER - Environment Canada (Halifax Stanfield) weather, to be added.

    Planned: daily/period precipitation, snowfall, temperature, etc. merged on `date`
    (and period, if hourly data is used). Use the forecast-available / same-day values
    only if they would be known at prediction time.
    """
    return panel


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """PLACEHOLDER - university academic calendar (term start, exams, breaks)."""
    return panel


def build_panel() -> pd.DataFrame:
    ovl = pd.read_parquet(PROCESSED / "overloads.parquet")
    routes = top_routes(ovl)

    panel_end = ovl.service_date.max()
    lag_start = pd.Timestamp(PANEL_START) - pd.DateOffset(days=LAG_BUFFER_DAYS)

    apc = pd.read_parquet(
        PROCESSED / "apc.parquet",
        columns=["service_date", "route_group", "route_name", "period", "ridership"],
        filters=[("service_date", ">=", lag_start), ("service_date", "<=", panel_end)],
    )
    bpp = period_boardings(apc, routes)

    # dominant route name per route_group x date (route 5 changes meaning in 2021)
    names = (
        apc[apc.route_group.isin(routes)]
        .assign(route_group=lambda d: d.route_group.astype(str), route_name=lambda d: d.route_name.astype(str))
        .groupby(["route_group", "service_date"]).route_name.agg(lambda s: s.mode().iat[0])
        .rename("route_name").reset_index()
    )

    # date grid: weekdays, minus NS statutory holidays
    hol = ns_holidays(range(lag_start.year, panel_end.year + 1))
    dates = pd.date_range(lag_start, panel_end, freq="D")
    is_hol = dates.isin(list(hol))
    weekdays = dates[(dates.dayofweek < 5) & ~is_hol]
    hol_wd = sorted({n for d, n in
                     holidays.Canada(prov="NS", years=range(2020, panel_end.year + 1),
                                     categories=HOLIDAY_CATEGORIES).items()})
    print("Holiday names used:", "; ".join(hol_wd))
    print(f"Weekday holidays dropped ({PANEL_START} -> {panel_end:%Y-%m-%d}): "
          f"{int(((dates.dayofweek < 5) & is_hol & (dates >= PANEL_START)).sum())}")

    grid = pd.MultiIndex.from_product(
        [routes, weekdays, PERIOD_ORDER], names=["route", "date", "period"]
    ).to_frame(index=False)
    grid["dow"] = grid["date"].dt.dayofweek
    grid = grid.merge(
        bpp.rename(columns={"route_group": "route", "service_date": "date"}),
        on=["route", "date", "period"], how="left",
    )
    grid = add_lag_features(grid)

    # labels
    o = ovl[ovl.route_group.isin(routes) & ovl.period.notna()].copy()
    o["period"] = o["period"].astype(str)
    cnt = (o.groupby(["route_group", "service_date", "period"]).size()
           .rename("n_overloads").reset_index()
           .rename(columns={"route_group": "route", "service_date": "date"}))
    panel = grid[grid.date >= PANEL_START].merge(cnt, on=["route", "date", "period"], how="left")
    panel["n_overloads"] = panel["n_overloads"].fillna(0).astype("int16")
    panel["y"] = (panel["n_overloads"] > 0).astype("int8")

    # how many in-scope incidents did not land in the panel (weekends, holidays, night)?
    in_scope = ovl[ovl.route_group.isin(routes) & (ovl.service_date >= PANEL_START)]
    print(f"Incidents on top routes since {PANEL_START}: {len(in_scope):,}; "
          f"counted in panel: {int(panel.n_overloads.sum()):,} "
          f"(rest are weekends/holidays/outside 06-24)")

    panel = panel.merge(names.rename(columns={"route_group": "route", "service_date": "date"}),
                        on=["route", "date"], how="left")
    panel["month"] = panel["date"].dt.month.astype("int8")
    panel["year"] = panel["date"].dt.year.astype("int16")
    panel["is_covid"] = (panel["year"] == 2020).astype("int8")
    panel["dow"] = panel["dow"].astype("int8")
    panel["route"] = pd.Categorical(panel["route"], categories=routes)
    panel["period"] = pd.Categorical(panel["period"], categories=PERIOD_ORDER, ordered=True)

    panel = add_weather_features(panel)
    panel = add_calendar_features(panel)

    cols = ["route", "route_name", "date", "period", "dow", "month", "year", "is_covid",
            "boardings_lag1w", "lag1w_days_ago", "boardings_mean4w", "n_weeks_mean4w",
            "n_overloads", "y"]
    # NB: same-period `boardings` is intentionally NOT kept (leakage).
    panel = panel[cols].sort_values(["date", "route", "period"]).reset_index(drop=True)
    return panel


def main() -> None:
    panel = build_panel()
    out = PROCESSED / "panel.parquet"
    panel.to_parquet(out, index=False)
    print(f"\n-> wrote {out}  shape={panel.shape}")
    print(f"date range: {panel.date.min():%Y-%m-%d} -> {panel.date.max():%Y-%m-%d}")
    print(f"positive rate overall: {panel.y.mean():.3%}   excl. 2020: {panel[panel.is_covid == 0].y.mean():.3%}")
    print("\npositive rate by year:\n", panel.groupby("year").y.agg(["mean", "sum", "size"]).round(4).to_string())
    print("\nmissing lag features (%):\n",
          (100 * panel[["boardings_lag1w", "boardings_mean4w"]].isna().mean()).round(2).to_string())
    print("\nlag1w_days_ago distribution:\n", panel.lag1w_days_ago.value_counts().sort_index().to_string())
    print("\n", panel.head().to_string())


if __name__ == "__main__":
    main()
