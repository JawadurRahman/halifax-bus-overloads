"""Baselines, models, metrics and the weather ablation.

Split (by time, 2022+ only):  train 2022-2024 | validation 2025 | test 2026 (Jan-Oct)

Stage 1 (validation):  python src/train.py           -> reports/val_results.csv, reports/model_selection.json
Stage 2 (test, once the models are chosen):
                       python src/train.py --test    -> reports/test_results.csv

Feature sets (ablation = research question):
    A  time      : route, period, day of week, month
    B  + lags    : A + boardings_lag1w, boardings_mean4w
    C  + weather : B + weather features (same-day/period weather = perfect-forecast assumption)
"""
from __future__ import annotations

import json
import sys
import warnings
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import PERIOD_ORDER, PROCESSED, ROOT  # noqa: E402
from weather import WEATHER_COLS  # noqa: E402

REPORTS = ROOT / "reports"
SEED = 0

TRAIN_YEARS, VAL_YEARS, TEST_YEARS = (2022, 2023, 2024), (2025,), (2026,)

TIME = ["route", "period", "dow", "month"]
LAGS = ["boardings_lag1w", "boardings_mean4w"]
WEATHER = WEATHER_COLS
FEATURE_SETS = {"A_time": TIME, "B_lags": TIME + LAGS, "C_weather": TIME + LAGS + WEATHER}

LGBM_GRID = {"num_leaves": [7, 15, 31], "min_child_samples": [20, 100]}
LOGREG_GRID = {"C": [0.01, 0.1, 1.0], "interact": [False, True]}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_data(panel_file: str = "panel_weather.parquet") -> pd.DataFrame:
    """Modelling panel (2022+) + y_lag1w (overload in the previous non-holiday same weekday).

    panel_file: panel_weather.parquet (round 2) or panel_features.parquet (round 3, same rows).
    """
    full = pd.read_parquet(PROCESSED / "panel.parquet")
    for c in ["route", "period"]:
        full[c] = full[c].astype(str)
    full = full.sort_values(["route", "period", "dow", "date"])
    full["y_lag1w"] = full.groupby(["route", "period", "dow"]).y.shift(1)

    df = pd.read_parquet(PROCESSED / panel_file)
    routes = list(df["route"].cat.categories) if hasattr(df["route"], "cat") else sorted(df.route.unique())
    for c in ["route", "period"]:
        df[c] = df[c].astype(str)
    df = df.merge(full[["route", "period", "date", "y_lag1w"]], on=["route", "period", "date"], how="left")
    df["route"] = pd.Categorical(df["route"], categories=routes)
    df["period"] = pd.Categorical(df["period"], categories=PERIOD_ORDER)
    df["route_period"] = df["route"].astype(str) + "_" + df["period"].astype(str)
    df["split"] = np.select(
        [df.year.isin(TRAIN_YEARS), df.year.isin(VAL_YEARS), df.year.isin(TEST_YEARS)],
        ["train", "val", "test"], "other")
    return df.sort_values(["date", "route", "period"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _rank_desc(score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices sorted by descending score, ties broken at random."""
    return np.lexsort((rng.random(len(score)), -score))


def evaluate(d: pd.DataFrame, score, name: str = "") -> dict:
    score = np.asarray(score, dtype=float)
    y = d["y"].to_numpy()
    n_ovl = d["n_overloads"].to_numpy()
    rng = np.random.default_rng(SEED)
    out = {"model": name, "n": len(y), "prevalence": y.mean(),
           "pr_auc": average_precision_score(y, score),
           "roc_auc": roc_auc_score(y, score),
           "brier": brier_score_loss(y, np.clip(score, 0, 1))}
    order = _rank_desc(score, rng)
    for k in (0.05, 0.10):
        top = order[: int(np.ceil(k * len(y)))]
        out[f"recall_top{int(k*100)}"] = y[top].sum() / y.sum()
        out[f"incidents_top{int(k*100)}"] = n_ovl[top].sum() / n_ovl.sum()
    # per-day budget: flag the top 2 / top 4 of each day's 40 route-periods
    tmp = pd.DataFrame({"date": d["date"].to_numpy(), "s": score + 1e-12 * rng.random(len(y)), "y": y})
    r = tmp.groupby("date")["s"].rank(ascending=False, method="first")
    for k, per_day in ((5, 2), (10, 4)):
        out[f"recall_day{k}"] = y[(r <= per_day).to_numpy()].sum() / y.sum()
    return out


def bootstrap_diff(d: pd.DataFrame, s1, s2, n_boot: int = 1000, metric=average_precision_score) -> dict:
    """Paired bootstrap over DATES of metric(s2) - metric(s1) (default: PR-AUC)."""
    rng = np.random.default_rng(SEED)
    dates = d["date"].to_numpy()
    uniq, inv = np.unique(dates, return_inverse=True)
    idx_by_day = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    y = d["y"].to_numpy(); s1 = np.asarray(s1); s2 = np.asarray(s2)
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_day[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if y[pick].sum() == 0:
            continue
        diffs.append(metric(y[pick], s2[pick]) - metric(y[pick], s1[pick]))
    diffs = np.array(diffs)
    return {"diff": metric(y, s2) - metric(y, s1),
            "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
            "p_le_0": float((diffs <= 0).mean())}


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def baseline_hist_rate(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Training-set overload rate for route x period x month (fallback route x period, global)."""
    keys = ["route", "period", "month"]
    t = train.assign(route=train.route.astype(str), period=train.period.astype(str))
    g = target.assign(route=target.route.astype(str), period=target.period.astype(str))
    r3 = t.groupby(keys).y.mean().rename("r3")
    r2 = t.groupby(["route", "period"]).y.mean().rename("r2")
    out = g.join(r3, on=keys).join(r2, on=["route", "period"])
    return out["r3"].fillna(out["r2"]).fillna(t.y.mean()).to_numpy()


def baseline_last_week(target: pd.DataFrame) -> np.ndarray:
    """y from the previous non-holiday same weekday (missing -> 0)."""
    return target["y_lag1w"].fillna(0).to_numpy()


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
# boardings-type counts -> log1p; ratios -> log; everything else numeric -> as is (all then scaled)
LOG1P_COLS = LAGS + ["peak_hh_lag1w", "peak_hh_mean4w", "peakdir_lag1w", "peakdir_mean4w"]
LOG_COLS = ["yoy_ratio", "lag_ratio"]


def make_logreg(features: list[str], C: float = 1.0, interact: bool = False) -> Pipeline:
    cat = [c for c in TIME if c in features] + (["route_period"] if interact else [])
    lag = [c for c in features if c in LOG1P_COLS]
    rat = [c for c in features if c in LOG_COLS]
    wx = [c for c in features if c not in TIME and c not in LOG1P_COLS and c not in LOG_COLS]
    tr = [("cat", OneHotEncoder(handle_unknown="ignore"), cat)]
    if lag:
        tr.append(("lag", Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                    ("sc", StandardScaler())]), lag))
    if rat:
        tr.append(("ratio", Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
                                      ("sc", StandardScaler())]), rat))
    if wx:   # "wx" kept as the name for round-2 compatibility; holds weather and other numeric features
        tr.append(("wx", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), wx))
    return Pipeline([("prep", ColumnTransformer(tr)),
                     ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000))])


def logreg_cols(features, interact):
    return features + (["route_period"] if interact else [])


def fit_lgbm(train: pd.DataFrame, features: list[str], num_leaves: int, min_child_samples: int,
             val: pd.DataFrame | None = None, n_estimators: int | None = None, seed: int = SEED):
    # int32 labels: LightGBM 4.7 + numpy 2.5 crashes (access violation) on int8 label arrays
    y = train["y"].astype("int32").to_numpy()
    params = dict(objective="binary", learning_rate=0.03, num_leaves=num_leaves,
                  min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                  metric="average_precision", random_state=seed, verbose=-1, n_jobs=4)
    if n_estimators is None:
        m = lgb.LGBMClassifier(n_estimators=3000, **params)
        m.fit(train[features], y, eval_set=[(val[features], val["y"].astype("int32").to_numpy())],
              callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
    else:
        m = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
        m.fit(train[features], y)
    return m


# ---------------------------------------------------------------------------
# stage 1: validation
# ---------------------------------------------------------------------------
def run_validation(df: pd.DataFrame, verbose: bool = True, feature_sets: dict | None = None):
    feature_sets = feature_sets or FEATURE_SETS
    tr, va = df[df.split == "train"], df[df.split == "val"]
    rows, scores, selection = [], {}, {}

    s = baseline_hist_rate(tr, va); scores["Baseline: hist rate"] = s
    rows.append(evaluate(va, s, "Baseline: hist rate"))
    s = baseline_last_week(va); scores["Baseline: last week"] = s
    rows.append(evaluate(va, s, "Baseline: last week"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fs_name, feats in feature_sets.items():
            # logistic regression
            best = None
            for C, inter in product(LOGREG_GRID["C"], LOGREG_GRID["interact"]):
                cols = logreg_cols(feats, inter)
                m = make_logreg(feats, C, inter).fit(tr[cols], tr.y)
                p = m.predict_proba(va[cols])[:, 1]
                ap = average_precision_score(va.y, p)
                if verbose:
                    print(f"  logreg {fs_name:10s} C={C:<5} interact={inter!s:5}  val PR-AUC={ap:.4f}")
                if best is None or ap > best[0]:
                    best = (ap, {"C": C, "interact": inter}, p)
            name = f"LogReg {fs_name}"
            selection[name] = best[1]; scores[name] = best[2]
            rows.append(evaluate(va, best[2], name) | {"params": json.dumps(best[1])})

            # LightGBM
            best = None
            for nl, mcs in product(LGBM_GRID["num_leaves"], LGBM_GRID["min_child_samples"]):
                m = fit_lgbm(tr, feats, nl, mcs, val=va)
                p = m.predict_proba(va[feats])[:, 1]
                ap = average_precision_score(va.y, p)
                if verbose:
                    print(f"  lgbm   {fs_name:10s} leaves={nl:<3} min_child={mcs:<4} "
                          f"trees={m.best_iteration_:<5} val PR-AUC={ap:.4f}")
                if best is None or ap > best[0]:
                    best = (ap, {"num_leaves": nl, "min_child_samples": mcs,
                                 "n_estimators": int(m.best_iteration_)}, p)
            name = f"LightGBM {fs_name}"
            selection[name] = best[1]; scores[name] = best[2]
            rows.append(evaluate(va, best[2], name) | {"params": json.dumps(best[1])})

    res = pd.DataFrame(rows)
    return res, scores, selection


# ---------------------------------------------------------------------------
# stage 2: test
# ---------------------------------------------------------------------------
def fit_selected(train: pd.DataFrame, name: str, params: dict, feature_sets: dict | None = None,
                 seed: int = SEED):
    fs = name.split(" ", 1)[1]
    feats = (feature_sets or FEATURE_SETS)[fs]
    if name.startswith("LogReg"):
        cols = logreg_cols(feats, params["interact"])
        m = make_logreg(feats, params["C"], params["interact"]).fit(train[cols], train.y)
        return m, cols
    m = fit_lgbm(train, feats, params["num_leaves"], params["min_child_samples"],
                 n_estimators=params["n_estimators"], seed=seed)
    return m, feats


def run_test(df: pd.DataFrame, selection: dict, train_splits=("train",), feature_sets: dict | None = None):
    tr, te = df[df.split.isin(train_splits)], df[df.split == "test"]
    rows, scores, models = [], {}, {}
    s = baseline_hist_rate(tr, te); scores["Baseline: hist rate"] = s
    rows.append(evaluate(te, s, "Baseline: hist rate"))
    s = baseline_last_week(te); scores["Baseline: last week"] = s
    rows.append(evaluate(te, s, "Baseline: last week"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, params in selection.items():
            m, cols = fit_selected(tr, name, params, feature_sets)
            p = m.predict_proba(te[cols])[:, 1]
            scores[name] = p; models[name] = (m, cols)
            rows.append(evaluate(te, p, name))
    return pd.DataFrame(rows), scores, models


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load_data()
    print(df.groupby("split").y.agg(["size", "mean"]).round(4).to_string(), "\n")
    if "--test" not in sys.argv:
        res, _, selection = run_validation(df)
        res.to_csv(REPORTS / "val_results.csv", index=False)
        (REPORTS / "model_selection.json").write_text(json.dumps(selection, indent=2))
        print("\n", res.drop(columns="params", errors="ignore").round(4).to_string(index=False))
    else:
        selection = json.loads((REPORTS / "model_selection.json").read_text())
        res, _, _ = run_test(df, selection)
        res.to_csv(REPORTS / "test_results.csv", index=False)
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
