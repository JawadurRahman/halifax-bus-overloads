"""Diagnostics for joining overload incidents to APC boardings.

For each year: % of (non-ferry) overload incidents that match an APC row on
  (a) route + service date + half-hour slot   (exact slot, any direction)
  (b) route + service date + slot +/- 30 min
  (c) route + service date                    (route had any APC data that day)
Also lists unmatched routes and checks route numbering across years.

Run from the project root after src/load.py:  python src/join_check.py
"""
from __future__ import annotations

import pandas as pd

from load import PROCESSED


def main() -> None:
    ovl = pd.read_parquet(PROCESSED / "overloads.parquet")
    apc = pd.read_parquet(PROCESSED / "apc.parquet", columns=["service_date", "route", "route_hour"])
    apc["route"] = apc["route"].astype(str)

    slots = apc[["route", "service_date", "route_hour"]].drop_duplicates()
    slot_set = set(zip(slots.route, slots.service_date, slots.route_hour))
    day_set = set(zip(slots.route, slots.service_date))

    o = ovl.copy()
    o["m_slot"] = [(r, d, h) in slot_set for r, d, h in zip(o.route, o.service_date, o.route_hour)]
    o["m_slot_pm30"] = [
        any((r, d, h + k) in slot_set for k in (-0.5, 0.0, 0.5))
        for r, d, h in zip(o.route, o.service_date, o.route_hour)
    ]
    o["m_day"] = [(r, d) in day_set for r, d in zip(o.route, o.service_date)]

    print("=== Match rate by year (overloads, ferry excluded) ===")
    tab = o.groupby("year").agg(
        n=("route", "size"),
        exact_slot_pct=("m_slot", "mean"),
        slot_pm30_pct=("m_slot_pm30", "mean"),
        route_day_pct=("m_day", "mean"),
    )
    tab.loc["all"] = [len(o), o.m_slot.mean(), o.m_slot_pm30.mean(), o.m_day.mean()]
    for c in ["exact_slot_pct", "slot_pm30_pct", "route_day_pct"]:
        tab[c] = (100 * tab[c]).round(1)
    tab["n"] = tab["n"].astype(int)
    print(tab.to_string(), "\n")

    print("=== Unmatched incidents (exact slot) by route ===")
    um = o[~o.m_slot]
    um_tab = (
        um.groupby("route").agg(n_unmatched=("route", "size"), no_apc_that_day=("m_day", lambda s: int((~s).sum())))
        .join(o.groupby("route").size().rename("n_total"))
        .sort_values("n_unmatched", ascending=False)
    )
    print(um_tab.head(25).to_string(), "\n")

    routes_in_apc = set(apc.route.unique())
    never = sorted(set(o.route) - routes_in_apc)
    print("Overload routes that NEVER appear in APC:", never or "none", "\n")

    print("=== Route numbering across years ===")
    apc_years = apc.assign(y=apc.service_date.dt.year).groupby("route").y.agg(["min", "max"])
    ovl_years = o.groupby("route").year.agg(["min", "max", "size"])
    rt = ovl_years.join(apc_years, lsuffix="_ovl", rsuffix="_apc").sort_values("size", ascending=False)
    rt.columns = ["ovl_first", "ovl_last", "n_ovl", "apc_first", "apc_last"]
    print(rt.head(30).to_string())
    print("\nOverload routes used outside the APC years for that route:")
    bad = rt[(rt.ovl_first < rt.apc_first) | (rt.ovl_last > rt.apc_last)]
    print(bad.to_string() if len(bad) else "none")


if __name__ == "__main__":
    main()
