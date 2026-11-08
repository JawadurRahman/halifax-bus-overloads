# Predicting Halifax Bus Overloads — Project Update

2026-11-08 · Jawadur Rahman

We can predict which Halifax Transit routes and time periods are likely to pass up waiting riders, clearly better than a simple historical lookup. Adding weather did not make the predictions better.

## The question

When a bus is full, it drives past people waiting at the stop. Halifax Transit logs each one as a "passenger overload", and released these records publicly in September 2026.

We asked two things. Can we predict, ahead of time, which route and time period on a weekday will have an overload? And does weather improve that prediction beyond past ridership? If overloads can be predicted, extra buses could be sent before riders are left behind.

Each prediction is a yes/no for one route, one weekday and one period: morning (6–10), midday (10–14), afternoon (14–19) or evening (19–24).

## Data

We combined three public datasets, all for Halifax.

| Dataset | Source | Coverage | What it tells us |
| --- | --- | --- | --- |
| [Passenger overload incidents](https://data-hrm.hub.arcgis.com/datasets/HRM::transit-passenger-overload-incidents/about) | HRM Open Data | 9,354 incidents, Jan 2020 – Oct 2026 | When and where a full bus passed riders. It does not record how many people were left. |
| [Automated passenger counts](https://data-hrm.hub.arcgis.com/datasets/HRM::transit-automated-passenger-counts/about) | HRM Open Data (current + archived) | 9.1 million rows, Oct 2017 – Nov 2026 | How many people boarded each route, every half hour |
| [Hourly and daily weather](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html) | Environment Canada, Halifax Stanfield airport | Dec 2021 – Oct 2026 | Temperature, wind, rain, snowfall, snow on the ground |

We focused on the 10 busiest routes for overloads (3, 8, 9A, 1, 4, 90, 9B, 10, 2 and 5), which account for 81% of incidents since 2023.

## What we did

1. **Cleaned the data.** Converted overload times from UTC to Halifax time, removed ferry incidents and counter glitches, and matched each overload to the ridership counts (95% matched to the exact half-hour).
2. **Built one table to predict from.** One row per route, weekday and period: 68,080 rows from 2020 to 2026. Holidays were removed. About 6.6% of rows had at least one overload.
3. **Explored the patterns.** Overloads peak between 3 and 5 pm, are about twice as common Tuesday to Thursday as on Monday or Friday, and jump every September and October when classes start.
4. **Added weather.** Temperature, wind, rain, snowfall and snow on the ground for each day and period.
5. **Trained and tested models.** We trained on 2022–2024, tuned on 2025, and tested once on January–October 2026, which the models had never seen. 2020 was left out because COVID capacity limits make it unlike other years.

We compared two simple rules against two machine-learning models:

- **"Same as last week"**: predict an overload if there was one at the same time last week.
- **"Historical rate"**: how often that route and period had overloads in that month in past years.
- **Logistic regression**: a standard statistical model.
- **LightGBM**: a decision-tree model that can pick up combinations, such as one route being busy only in September afternoons.

## Results

On 2026 data, LightGBM was the best method: it ranks risky route-periods 16% better than the historical-rate rule, and the difference is statistically significant.

| Method | Prediction score on 2026 (0–1, higher is better) | Overloads caught if we flag the riskiest 5% (of 656) |
| --- | --- | --- |
| LightGBM | **0.289** | **154** (23%) |
| Logistic regression | 0.276 | 144 (22%) |
| Historical rate | 0.249 | 135 (21%) |
| Same as last week | 0.141 | 129 (20%) |
| Random guessing | 0.079 | about 33 (5%) |

The score is PR-AUC: how well a method puts the real overloads at the top of its risk list. Flagging the top 5% means about 2 of the 40 route-periods each day.

- LightGBM's flags were right 37% of the time, and caught 229 of the 823 incidents reported in 2026 (28%).
- Its gain over the historical rate is +0.040, with a 95% confidence interval of +0.021 to +0.063.
- Most of the predictive power comes from **which route, which period, which month and which weekday**. The models mainly learn where and when overloads happen.

## Does weather help?

No. Adding weather did not reliably improve the predictions, even though we gave the models the actual weather for each day, as if the forecast were perfect.

| Features given to LightGBM | Score on 2025 (tuning) | Score on 2026 (test) |
| --- | --- | --- |
| A. Route, period, weekday, month | 0.372 | 0.292 |
| B. A + past ridership | 0.370 | 0.287 |
| C. B + weather | 0.375 | 0.289 |

Weather's small gain on 2025 (+0.006) did not hold up on 2026: +0.002, with a 95% confidence interval of −0.005 to +0.008. Past ridership did not help either, because route, period and month already capture it.

The weather effects that do show up are small and point in a sensible direction. Heavy rain, strong gusts and big snowfalls make overloads slightly *less* likely, probably because fewer people ride.

## Caveats and next steps

The models work, but the gain is modest and these results have limits.

- **Modest in practice.** Flagging the riskiest 5% catches about a quarter of overloads, and about 6 in 10 flags are false alarms.
- **Reported, not actual, overloads.** Overloads per million boardings almost tripled from 2022 to 2023. Some of that may be a change in how drivers report, not just more crowding.
- **The overload rate shifts by year:** 3% in 2022, about 9% in 2023–2025, 8% in 2026. This makes 2026 harder to score well on.
- **One weather station.** The airport is about 30 km from downtown and has no hourly rainfall, so rain and snow are daily totals.
- **Risk scores are rankings, not real probabilities.** They would need recalibrating before being read as percentages.

The university calendar and recent overload history were tested next; see the round 3 section below. The code, notebooks and figures are in this repository (`src/`, `notebooks/01_eda.ipynb`, `notebooks/02_models.ipynb`, `notebooks/03_features.ipynb`, `reports/figures/`).

## Round 3: new feature groups (2026-11-08)

We tested three new groups of features, each built only from information available before the predicted day. Only one helped: **recent overload history**.

| Added to route + period + weekday + month | Did it help? | Gain on 2026 (PR-AUC, 95% CI) |
| --- | --- | --- |
| Overload history (overloads on that route-period in the past 1–4 weeks, days since the last one) | **Yes** | **+0.069** (+0.044 to +0.094) |
| University calendar (term weeks, breaks, exams; Dal dates) | No | +0.008 (−0.006 to +0.025) |
| Better ridership (peak half-hour, busier direction, year-over-year change, "busier than usual") | No | −0.002 (−0.019 to +0.019) |
| Weather, on top of all three | No | +0.004 (−0.005 to +0.013) |

- The best model (LightGBM with all features, chosen on 2025) scores **0.370** on 2026. That compares with 0.292 for the earlier model and 0.249 for the historical-rate rule.
- Flagging its riskiest 5% of route-periods catches **194 of 656 overloads (30%)**, covering 294 of 823 incidents, with **47% of flags correct**. The earlier model caught 159, with 38% of flags correct.
- A model with overload history alone does almost as well (0.361), so the history features carry nearly all of the gain.
- The calendar shows a real pattern: overloads are 3–4 times as common in the first weeks of fall term. But that pattern is already captured by the month.

**What it means.** Our conclusion mostly holds: overload risk is structural (where and when) rather than driven by day-to-day demand or weather.
- The refinement: the best extra signal is *where overloads have just been happening*. A route and period that keeps overloading probably has a persistent capacity shortfall.
- That signal also follows each year's overall level. This helps with the year-to-year drift that a time-only model cannot follow.
- Ridership counts, the term calendar and weather still add nothing on top.

**A note on the test set.** We had already looked at the 2026 results once, in the previous round. This time every choice was made on 2025 data, and each new feature set was scored on 2026 only once.
