# Predicting Halifax Transit Passenger Overloads

CSCI 6505 (Machine Learning, Dalhousie) course project.

## Project question

Can we predict which Halifax Transit routes and time periods will have buses passing up
waiting riders (a "passenger overload"), and does weather improve that prediction beyond
past ridership alone?

This is a binary classification problem. The unit of prediction is
**route × date × time period**:

| Period | Hours (Halifax local time) |
|--------|----------------------------|
| AM     | 06:00–10:00 |
| MID    | 10:00–14:00 |
| PM     | 14:00–19:00 |
| EVE    | 19:00–24:00 |

The panel covers weekdays only (statutory holidays excluded) and the busiest routes.
`y = 1` if at least one overload incident was reported in that cell.

The original project description (`docs/Project Bus Overload + Bus Boardings.pdf`) also lists
the university academic calendar as a candidate feature. A placeholder is kept for it next to
the weather features.

## Data sources

| Dataset | Source | Local file(s) | Original file name |
|---|---|---|---|
| Passenger overload incidents (2020–2026) | [HRM Open Data](https://data-hrm.hub.arcgis.com/datasets/HRM::transit-passenger-overload-incidents/about) | `data/raw/overloads/overload_incidents.csv` | `Transit_Passenger_Overload_Incidents_-666457333674270898.csv` |
| Automated Passenger Counts, current year (2026) | [HRM Open Data](https://data-hrm.hub.arcgis.com/datasets/HRM::transit-automated-passenger-counts/about) | `data/raw/apc/apc_2026.csv` | `Transit_Automated_Passenger_Counts_8663944865722026735.csv` |
| Automated Passenger Counts, archived (2017–2025) | HRM Open Data catalogue: search "Transit Automated Passenger Counts Archived" | `data/raw/apc/apc_2017.csv` … `apc_2025.csv` | `transit-automated-passenger-counts-archived/transit-automated-passenger-counts-YYYY.csv` |
| Weather, Dec 2021 – Oct 2026 | [Environment Canada historical data](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html): HALIFAX STANFIELD INT'L A, Climate ID 8202251, Station ID 50620, hourly + daily bulk CSV | `data/external/weather/hourly/stanfield_hourly_YYYY_MM.csv`, `.../daily/stanfield_daily_YYYY.csv` | downloaded by `src/weather.py --download` |

Dataset metadata sheets are in `docs/`.

Data files are **not** tracked in git (see `.gitignore`). Download them from the links above
and place them at the paths shown.

## Data-processing decisions (round 1, 2026-11-08)

- **Overload times** are UTC; converted to `America/Halifax`. Incidents before 04:00 local
  are assigned to the previous APC service day (APC hours 24.0–27.5).
- **Dropped incidents:** 24 ferry rows (`FerD`) and 2 points with latitude > 44.9.
- **APC schema:** identical in all years (2017–2026), half-hourly with direction; only the
  date format differs. 2017 covers Oct–Dec only.
- **APC counter glitches:** a half-hour value > 500 **and** > 5× the year/route/direction/slot
  median is set to NaN (161 rows, 0.13% of boardings). The raw value is kept in `ridership_raw`.
  Missing APC rows mean "no data", never 0.
- **Route 10** (Dalhousie) was split into 10A/10B/10C on 2024-11-18; all four are merged into
  `route_group = "10"`.
- **Holidays:** `holidays.Canada(prov="NS", categories=("public","optional","government"))`.
  This adds Boxing Day, Thanksgiving, Victoria Day, Natal Day and Truth & Reconciliation Day,
  whose ridership is 0.3–0.6× a normal weekday. They are dropped from the panel and skipped
  in the lag features.
- **Panel:** top 10 route groups by overload count since 2023 (3, 8, 9A, 1, 4, 90, 9B, 10, 2, 5);
  periods AM 06–10, MID 10–14, PM 14–19, EVE 19–24; lag features use only previous weeks'
  same-weekday values. 2020 is flagged `is_covid`.

- **Weather (round 2):** Halifax Stanfield (8202251 / 50620). Hourly precipitation is not reported
  there, so precipitation/snow come from the daily file. Hourly LST timestamps are converted to
  Halifax clock time. Same-day weather is used as a feature (perfect-forecast assumption).
- **Modelling split (round 2):** 2022+ only; train 2022–2024, validation 2025, test 2026 (Jan–Oct).
  Test models are trained on 2022–2024; a refit on 2022–2025 is reported as a sensitivity check.

See the summaries at the end of `notebooks/01_eda.ipynb` and `notebooks/02_models.ipynb` for findings.

## Folder layout

```
ML project/
├── data/
│   ├── raw/
│   │   ├── overloads/overload_incidents.csv
│   │   └── apc/apc_2017.csv ... apc_2026.csv
│   ├── external/        weather downloads + academic_calendar.csv (tracked)
│   └── processed/       cleaned parquet files + modelling panel
├── docs/                project description + dataset metadata PDFs
├── notebooks/           01_eda, 02_models, 03_features, 04_final_walkthrough (start here)
├── src/
│   ├── load.py          load + clean raw data, write data/processed/*.parquet
│   ├── join_check.py    overload ↔ APC join diagnostics
│   ├── build_panel.py   build data/processed/panel.parquet
│   ├── weather.py       download EC weather, build features -> panel_weather.parquet
│   ├── train.py         baselines, LogReg, LightGBM, metrics, ablation (round 2)
│   ├── features.py      overload history, calendar, ridership -> panel_features.parquet
│   └── round3.py        round 3 ablation A-F (validation stage; test stage runs once)
├── reports/             results tables (val_*, test_*, r3_*), model selections
│   └── figures/         figures saved by the notebooks
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.12 (Windows commands shown).

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

Then run the pipeline from the project root:

```bash
.venv\Scripts\python src\load.py
```

```bash
.venv\Scripts\python src\join_check.py
```

```bash
.venv\Scripts\python src\build_panel.py
```

```bash
.venv\Scripts\python src\weather.py --download
```

```bash
.venv\Scripts\python src\train.py
```

```bash
.venv\Scripts\python src\train.py --test
```

```bash
.venv\Scripts\python src\features.py
```

```bash
.venv\Scripts\python src\round3.py
```

Open `notebooks/01_eda.ipynb`, `notebooks/02_models.ipynb` and `notebooks/03_features.ipynb` with the `.venv` kernel (`.venv\Scripts\jupyter lab`).

> **Note on pinned versions:** on the original dev machine, Windows Smart App Control blocked
> the compiled DLLs of the newest pandas, pyarrow and scikit-learn releases
> ("An Application Control policy has blocked this file"). Older releases are pinned in
> `requirements.txt` for that reason. LightGBM 4.7 also crashed intermittently with a native
> access violation, so `lightgbm==4.5.0` and `numpy==2.1.3` are pinned too.
