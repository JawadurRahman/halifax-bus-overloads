"""Round 5: three more model families on the same rows, split, features and rules.

Question: is the result a property of the data, or of the two models we used (logistic regression, LightGBM)?

New models (each tuned on 2025 only; 2026 evaluated ONCE):
    RandomForest   bagged decision trees (sklearn), class_weight="balanced_subsample"
    MLP            small neural network (sklearn MLPClassifier) on the logistic-regression preprocessing
                   (one-hot route/period/weekday/month, log-transformed counts, scaled numbers)
    XGBoost        a second gradient-boosting library (hist trees, native categorical route/period),
                   scale_pos_weight + early stopping on 2025, like LightGBM

Feature sets (from rounds 2-4):
    A_time, B_history, F_all3_weather, I_all_capacity

    python src/round5.py           validation stage  -> reports/r5_val_results.csv, r5_selection.json
    python src/round5.py --test    test stage, ONCE  -> reports/r5_test_*
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import train as T
from round3 import gains_vs
from round4 import FEATURE_SETS_R4   # also registers the capacity columns for the log transforms

FEATURE_SETS_R5 = {k: FEATURE_SETS_R4[k] for k in ["A_time", "B_history", "F_all3_weather", "I_all_capacity"]}
FAMILIES_R5 = ["RandomForest", "MLP", "XGBoost"]
GRIDS = {
    "RandomForest": [{"max_features": mf, "min_samples_leaf": leaf} for mf, leaf in product(["sqrt", 0.3], [20, 100])],
    "MLP": [{"hidden": h, "alpha": a} for h, a in product([(64,), (64, 32)], [1e-4, 1e-2])],
    "XGBoost": [{"max_depth": d, "min_child_weight": w} for d, w in product([3, 6], [1, 10])],
}
OUT = T.REPORTS
TEST_SCORES = OUT / "r5_test_scores.parquet"


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def make_rf(feats, p, seed=T.SEED):
    cat = [c for c in ["route", "period"] if c in feats]
    num = [c for c in feats if c not in cat]
    prep = ColumnTransformer([("cat", OrdinalEncoder(), cat),
                              ("num", SimpleImputer(strategy="median"), num)])
    rf = RandomForestClassifier(n_estimators=500, max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
                                class_weight="balanced_subsample", n_jobs=4, random_state=seed)
    return Pipeline([("prep", prep), ("clf", rf)])


def make_mlp(feats, p, seed=T.SEED):
    pipe = T.make_logreg(feats)            # reuse the one-hot / log / scaling preprocessing
    pipe.steps[-1] = ("clf", MLPClassifier(hidden_layer_sizes=p["hidden"], alpha=p["alpha"], early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=10, max_iter=300,
                                           random_state=seed))
    return pipe


def fit_xgb(tr, feats, p, va=None, n_estimators=None, seed=T.SEED):
    y = tr.y.astype("int32").to_numpy()
    kw = dict(tree_method="hist", enable_categorical=True, learning_rate=0.03, max_depth=p["max_depth"],
              min_child_weight=p["min_child_weight"], subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              scale_pos_weight=(y == 0).sum() / (y == 1).sum(), eval_metric="aucpr", random_state=seed, n_jobs=4)
    if n_estimators is None:
        m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=150, **kw)
        m.fit(tr[feats], y, eval_set=[(va[feats], va.y.astype("int32").to_numpy())], verbose=False)
    else:
        m = xgb.XGBClassifier(n_estimators=n_estimators, **kw)
        m.fit(tr[feats], y, verbose=False)
    return m


def fit(family, tr, feats, p, va=None, seed=T.SEED):
    if family == "XGBoost":
        return fit_xgb(tr, feats, p, va=va, n_estimators=p.get("n_estimators"), seed=seed)
    m = (make_rf if family == "RandomForest" else make_mlp)(feats, p, seed)
    return m.fit(tr[feats], tr.y.astype(int))


def load():
    return T.load_data("panel_capacity.parquet")


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def run_val():
    df = load()
    tr, va = df[df.split == "train"], df[df.split == "val"]
    rows, sel, scores = [], {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for family, (fs, feats) in product(FAMILIES_R5, FEATURE_SETS_R5.items()):
            best, t0 = None, time.time()
            for p in GRIDS[family]:
                m = fit(family, tr, feats, p, va=va)
                s = m.predict_proba(va[feats])[:, 1]
                ap = average_precision_score(va.y, s)
                if family == "XGBoost":
                    p = p | {"n_estimators": int(m.best_iteration) + 1}
                if best is None or ap > best[0]:
                    best = (ap, p, s)
            name = f"{family} {fs}"
            sel[name], scores[name] = best[1], best[2]
            rows.append(T.evaluate(va, best[2], name) | {"params": json.dumps(best[1], default=list)})
            print(f"  {name:28s} val PR-AUC {best[0]:.4f}  {best[1]}  ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    r4 = pd.read_csv(OUT / "r4_val_results.csv")
    r4 = r4[r4.model.str.startswith(("LightGBM", "LogReg", "Baseline")) &
            r4.model.str.contains("|".join(["Baseline"] + list(FEATURE_SETS_R5)))]
    allres = pd.concat([r4, res], ignore_index=True)
    best = allres[~allres.model.str.startswith("Baseline")].sort_values("pr_auc", ascending=False).iloc[0].model
    sel["_best_on_validation"] = best
    res.to_csv(OUT / "r5_val_results.csv", index=False)
    (OUT / "r5_selection.json").write_text(json.dumps(sel, indent=2, default=list))
    return allres, best


def run_test_once(force=False):
    if TEST_SCORES.exists() and not force:
        raise SystemExit(f"{TEST_SCORES} exists: the round-5 test has already been run once. Not re-running.")
    df = load()
    sel = json.loads((OUT / "r5_selection.json").read_text())
    best = sel.pop("_best_on_validation")
    tr, te = df[df.split == "train"], df[df.split == "test"].reset_index(drop=True)
    # earlier models' 2026 scores (same rows), aligned by key
    r4 = pd.read_parquet(OUT / "r4_test_scores.parquet")
    r4["date"] = pd.to_datetime(r4.date)
    al = te[["date", "route", "period"]].astype({"route": str, "period": str}).merge(
        r4, on=["date", "route", "period"], how="left", validate="one_to_one")
    assert al.y.astype(int).tolist() == te.y.astype(int).tolist()
    scores = {c: al[c].to_numpy() for c in r4.columns if c.startswith(("LightGBM", "LogReg", "Baseline"))}
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, p in sel.items():
            family, fs = name.split(" ", 1)
            if "hidden" in p:
                p["hidden"] = tuple(p["hidden"])
            m = fit(family, tr, FEATURE_SETS_R5[fs], p)
            scores[name] = m.predict_proba(te[FEATURE_SETS_R5[fs]])[:, 1]
            rows.append(T.evaluate(te, scores[name], name))
            print(f"  {name:28s} test PR-AUC {rows[-1]['pr_auc']:.4f}", flush=True)
    res = pd.DataFrame(rows)
    # each new model vs LightGBM on the same feature set, and the best-on-validation model vs the hist-rate rule
    gains = pd.concat([gains_vs(te, scores, f"LightGBM {fs}", [f"{fam} {fs}" for fam in FAMILIES_R5])
                       for fs in FEATURE_SETS_R5])
    gains_base = gains_vs(te, scores, "Baseline: hist rate", [best])
    res.to_csv(OUT / "r5_test_results.csv", index=False)
    gains.to_csv(OUT / "r5_test_gains_vs_lightgbm.csv", index=False)
    gains_base.to_csv(OUT / "r5_test_gains_vs_baseline.csv", index=False)
    pd.DataFrame({k: v for k, v in scores.items() if k.split(" ")[0] in FAMILIES_R5}).assign(
        date=te.date.to_numpy(), route=te.route.astype(str).to_numpy(), period=te.period.astype(str).to_numpy(),
        y=te.y.to_numpy()).to_parquet(TEST_SCORES, index=False)
    (OUT / "r5_test_run.json").write_text(json.dumps(
        {"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "best_on_validation": best,
         "trained_on": "2022-2024", "note": "single evaluation of the round-5 model families on 2026"}, indent=2))
    return res, gains, gains_base, best


def main():
    pd.set_option("display.width", 220)
    cols = ["model", "pr_auc", "roc_auc", "recall_top5", "recall_day5"]
    if "--test" in sys.argv:
        res, gains, gb, best = run_test_once(force="--force" in sys.argv)
        print("\nTEST 2026 (single evaluation)\n", res[cols].round(4).to_string(index=False))
        g = gains[gains.metric == "PR-AUC"]
        print("\nvs LightGBM, same features (PR-AUC):\n", g[["model", "vs", "diff", "ci_low", "ci_high"]].round(4).to_string(index=False))
        print("\nbest on validation:", best, "\n", gb.round(4).to_string(index=False))
    else:
        allres, best = run_val()
        print("\nVALIDATION 2025\n", allres[cols].round(4).to_string(index=False))
        print("\nbest on validation:", best)


if __name__ == "__main__":
    main()
