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
| **Final model** | A tree-based model (XGBoost) that also uses **recent overload history**: how often this route-period had overloads in the last few weeks. Knows both the usual pattern *and* what's happening now. It also gets ridership, the calendar, weather and bus schedules, which turned out not to matter much. |

## Results

**How we score it:** the model doesn't say yes or no; it gives each route-period a **risk score** from 0 to 1. To compare methods fairly we use one **overall score (PR-AUC)**, from 0 to 1, which measures how well a method puts the real overloads ahead of the quiet route-periods. Random guessing gets 0.08 (the share of "yes" answers); perfect is 1.

| Method | Overall score (0 to 1, higher is better) |
| --- | --- |
| Random guessing | 0.08 |
| Same as last week | 0.14 |
| Historical rate | 0.25 |
| Basic model | 0.29 |
| **Final model** | **0.36** |

**As yes/no answers.** To get a yes/no answer we draw a line on the risk score, the **cut-off**: above it the model says "yes, overload". We chose the cut-off on 2025 (0.79, the best balance between catching overloads and avoiding false alarms) and kept it for 2026. In 2026, 656 route-periods actually had an overload:

| | Really an overload | Really no overload |
| --- | --- | --- |
| **Model said "yes"** | **299 caught** | 507 false alarms |
| **Model said "no"** | 357 missed | 7,157 correctly quiet |

**What this means:**

- The final model **catches 46% of all overloads** (299 of 656). That's called **recall**.
- When it says "yes", it's **right 37% of the time** (299 of 806). That's called **precision**.
- To catch the same 299 overloads, the best simple rule (historical rate) would have to say "yes" about 1,050 times, so the model raises about **240 fewer false alarms**.
- If you pick one route-period that had an overload and one that didn't, the model gives the real overload the higher risk score **86% of the time**.
- A lower cut-off catches more (at 0.5: 79% of overloads), but then only 22% of the "yes" answers are right. Where to draw the line is a choice; the overall score covers every choice at once.
- We checked that the improvement is real and not luck: we re-scored the model on 1,000 reshuffled versions of 2026, and it beat the historical rate every time.

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

On 2025 the LightGBM version with schedules scored best so far (0.36 on 2026), statistically the same as the version without them. The next section tries other kinds of model.

## Trying other models

Our first two models were logistic regression (a straight-line model) and LightGBM (boosted decision trees). Maybe a different kind of model would do better, so we tried three more, each tuned on 2025 and tested on 2026 once, with the same clues:

- **Random Forest:** hundreds of independent decision trees that vote.
- **Neural network:** a small multi-layer network.
- **XGBoost:** a second boosted-trees library.

| Model | Score (PR-AUC 2026) with overload history | with every clue + schedules |
| --- | --- | --- |
| LightGBM | 0.36 | 0.36 |
| Logistic regression | 0.37 | 0.37 |
| Random Forest | 0.36 | 0.35 |
| Neural network | 0.36 | 0.35 |
| XGBoost | 0.36 | 0.36 |

**No model does better.** All five land within about 0.01 of each other, and none beats LightGBM by more than noise. For every model, recent overload history gives the big jump and the other clues add little. So the limit is the information in the data, not the choice of model.

**The final model is now XGBoost.** We pick the final model by its score on 2025, before looking at 2026, and XGBoost with every clue came first on 2025 by a hair (0.4006 vs 0.3999 for LightGBM). On 2026 it scores 0.356, versus 0.363 for LightGBM: the same within noise. We kept the rule rather than switching back after seeing the 2026 results, since choosing on the test year would make the test unfair.

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
- **We've looked at the 2026 test results several times** across rounds (five so far). Each new idea was scored on it only once, but a fresh test period (Nov 2026 onward) would make the final numbers more convincing.

## What's next

1. Check the 2025 lockout: did the September overload spike shift later that year?
2. Prepare the presentation (Nov 24 / 26 / Dec 1).
3. Write the 5-page paper (due Dec 11).

The code, notebooks and figures are in this repository: `src/`, `notebooks/` (01 = exploring the data, 02 = first models, 03 = new features, 04 = full walkthrough, 05 = bus schedules; `src/round5.py` = other models) and `reports/figures/`.
