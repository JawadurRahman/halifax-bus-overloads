# Predicting When Halifax Buses Leave People Behind

2026-11-08 · Jawadur Rahman · CSCI 6505 project update

**In one sentence:** we can predict which bus routes and times are likely to leave riders behind, clearly better than simple rules, and the most useful clue is whether it has been happening there recently.

## The problem

When a Halifax Transit bus is full, the driver has to drive past people waiting at the stop. Halifax Transit calls this a **passenger overload** and logs each one. In September 2026, the city published these records for the first time.

Being left at the stop is a real problem for riders: you're late for class or work, and in winter you're standing in the cold waiting for the next bus. If Halifax Transit knew *in advance* where and when this is likely, it could add a bus or adjust the schedule before it happens.

## Our question

1. **Can we predict, ahead of time, which route and time of day will have a bus leaving people behind?**
2. **What actually helps the prediction?** Is it ridership, weather, the university calendar, or something else?

## How we set it up

We split every weekday into four periods: **morning** (6–10 am), **midday** (10 am–2 pm), **afternoon** (2–7 pm) and **evening** (7 pm–midnight).

We looked at the **10 routes** with the most overloads (3, 8, 9A, 9B, 1, 4, 90, 10, 2 and 5). They account for 81% of all overloads since 2023.

That gives 40 route-periods per day (10 routes × 4 periods). For each one, the question is simply:

> **Will a bus on this route leave someone behind during this period? Yes or no.**

This is a **classification** problem. About 7% of route-periods have an overload, so it's a fairly rare event.

## The data

| Data | What it tells us | Source |
| --- | --- | --- |
| **Overload reports** | When and where a full bus passed riders (9,354 reports, 2020–2026). It does *not* say how many people were left behind. | [HRM Open Data](https://data-hrm.hub.arcgis.com/datasets/HRM::transit-passenger-overload-incidents/about) |
| **Passenger counts (APC)** | How many people boarded each route, every half hour (2017–2026). Counted automatically by sensors on the bus doors. | [HRM Open Data](https://data-hrm.hub.arcgis.com/datasets/HRM::transit-automated-passenger-counts/about) |
| **Weather** | Temperature, rain, snow and wind each day | [Environment Canada](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html), Halifax airport |
| **University calendar** | When Dal terms, breaks and exams happen | Dalhousie academic calendars |

## What we found in the data first

- Overloads happen mostly in the **afternoon rush, 3–5 pm**.
- They are more common **Tuesday to Thursday** than Monday or Friday.
- They **jump every September and October**, when classes start, even though total ridership only rises a little.
- A handful of routes (3, 8, 9A, 9B, 1) have most of them.

## How we tested the predictions

To be fair, we trained on the past and tested on the future, just like it would be used in real life:

- **Learn from:** 2022–2024
- **Fine-tune on:** 2025
- **Final test:** January–October 2026, which the models never saw while being built

We compared our models against simple rules anyone could use without machine learning:

| Method | How it works, in plain words |
| --- | --- |
| **Random guessing** | Pick route-periods at random. |
| **Same as last week** | "There was an overload here last Tuesday afternoon, so expect one this Tuesday too." Knows what's happening *now*, but nothing about the usual pattern. |
| **Historical rate** | "In past Octobers, Route 1 afternoons had overloads 30% of the time, so the risk is 30%." Knows the *usual* pattern, but nothing about what's happening now. |
| **Basic model** | A machine-learning model (LightGBM) that learns from route, time of day, weekday and month. |
| **Final model** | The same model, plus **recent overload history**: how often this route-period had overloads in the last few weeks. Knows both the usual pattern *and* what's happening now. |

## Results

**How we score it:** imagine Halifax Transit can only check the **2 riskiest route-periods each day** (out of 40) and send an extra bus there. In 2026, 656 route-periods actually had an overload. How many of those would each method have caught?

| Method | Overloads caught (out of 656) | Overall score (0 to 1, higher is better) |
| --- | --- | --- |
| Random guessing | about 33 (5%) | 0.08 |
| Same as last week | 129 (20%) | 0.14 |
| Historical rate | 135 (21%) | 0.25 |
| Basic model | 159 (24%) | 0.29 |
| **Final model** | **195 (30%)** | **0.36** |

The **overall score** (called PR-AUC) checks how well each method puts real overloads near the top of its risk list, across every cutoff, not just the top 2.

**What this means:**

- The final model catches about **60 more overloads** than the best simple rule (195 vs 135).
- When it flags a route-period as risky, it's **right about half the time** (47%).
- We checked that this improvement is real and not luck: we resampled the test data many times, and the final model beat the historical rate every time.

## What helps and what doesn't

We added each kind of information one at a time to see whether it improved the prediction.

| Added information | Did it help? |
| --- | --- |
| **Recent overload history** (overloads on this route-period in the past few weeks) | **Yes, clearly** |
| Ridership counts (how many people boarded recently) | No |
| University calendar (week of term, breaks, exams) | No |
| Weather (even with perfect knowledge of the actual weather) | No |
| Bus schedules (how many buses were scheduled on each route and time period) | No (only slightly without overload history) |

**Why didn't ridership, the calendar or weather help?** They're not useless; they just don't add anything new:

- **Ridership** mostly tells the model "this route is busy at this time", which it already knows from the route, time and month.
- **The calendar** shows a real pattern (overloads are 3–4 times more common in the first weeks of fall term), but the month already captures it.
- **Weather** has a small effect in a sensible direction: in heavy rain or snow, overloads are slightly *less* likely, probably because fewer people ride. It's too small to improve the predictions.

## Adding bus schedules

An overload happens when there are more riders than buses, and until now the model only knew about riders. So we added Halifax Transit's published schedules (GTFS, 27 archived versions from 2021–2026, via [Transitland](https://www.transit.land/feeds/f-dxfy-halifaxtransit~hrm)). For each route, day and time period, we counted the scheduled trips and computed **expected riders per bus**. For each day we only used a schedule published before that day.

- **The pattern is real:** when expected riders per bus is in the top tenth, overloads happen 25% of the time, vs 0.5% in the bottom tenth.
- **But it doesn't improve the predictions** once recent overload history is in the model. The final model scored 0.36 with schedules vs 0.37 without, a difference within noise.
- **The reason:** Halifax barely changed scheduled service on these 10 routes from 2022 to 2026 (only routes 90 and 10 changed). So "how many buses" is almost the same as "which route and time", which the model already knows.
- **It does help explain the trend:** riders per scheduled afternoon bus rose from 38 in 2022 to about 50 in 2023–2025 while service stayed flat. That's a likely reason overloads became more common.

Because we pick the final model on 2025 data before testing, the final model is now the version with schedules (score 0.36, 195 overloads caught). It is statistically the same as the version without them.

## The main takeaway

> **Buses leaving people behind is a structural problem, not a random one.** It depends on *which route*, *what time* and *which month*, and above all on **whether it has been happening there recently**. A route-period that keeps overloading probably just doesn't have enough buses at that time.

In practice, this means Halifax Transit's own overload reports are an early-warning signal. If a route-period has overloaded several times in the past few weeks, it will probably keep doing so, and that's where extra capacity is needed.

## Limitations

- **We predict *whether* riders were left behind, not *how many*.** The data doesn't record how many people were passed up.
- **Reports depend on drivers.** Some drivers or routes may report more consistently than others, and overloads per rider almost tripled from 2022 to 2023, which may partly be a change in reporting.
- **The overload rate changes from year to year** (about 3% in 2022, about 9% in 2023–2025, 8% in 2026), which makes each new year a bit harder to predict.
- **One weather station**, at the airport about 30 km from downtown.
- **The fall 2025 Dal faculty lockout** delayed most classes to September 23. Our calendar used the official start date, so the calendar features were off for that fall. Worth checking: did the September 2025 overload spike also start late?
- **Weekends, holidays and late night (midnight–6 am) were left out.**
- **Schedules show planned buses, not actual ones.** Cancelled or late trips aren't visible, and GTFS doesn't say which buses are the bigger articulated ones.
- **We've looked at the 2026 test results several times** across rounds. Each new idea was scored on it only once, but a fresh test period (Nov 2026 onward) would make the final numbers more convincing.

## What's next

1. Check the 2025 lockout: did the September overload spike shift later that year?
2. Prepare the presentation (Nov 24 / 26 / Dec 1).
3. Write the 5-page paper (due Dec 11).

The code, notebooks and figures are in this repository: `src/`, `notebooks/` (01 = exploring the data, 02 = first models, 03 = new features, 04 = full walkthrough, 05 = bus schedules) and `reports/figures/`.
