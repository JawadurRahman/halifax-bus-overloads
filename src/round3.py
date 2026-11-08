"""Round 3: do overload history, the university calendar or better ridership features help?

Same rows, split and metrics as round 2 (src/train.py). Feature sets:
    A  route, period, weekday, month        (round-2 baseline)
    B  A + overload history
    C  A + university calendar
    D  A + better ridership
    E  A + all three
    F  E + weather
Settings are chosen on validation 2025 only.

    python src/round3.py           validation stage (re-runnable)  -> reports/r3_val_*.csv, r3_selection.json
    python src/round3.py --test    test stage, ONCE                  -> reports/r3_test_*.csv/parquet
The test stage refuses to run a second time (pass --force to override; don't).
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import train as T
from features import CAL, HIST, RIDE

FEATURE_SETS_R3 = {
    "A_time": T.TIME,
    "B_history": T.TIME + HIST,
    "C_calendar": T.TIME + CAL,
    "D_ridership": T.TIME + RIDE,
    "E_all3": T.TIME + HIST + CAL + RIDE,
    "F_all3_weather": T.TIME + HIST + CAL + RIDE + T.WEATHER,
}
FAMILIES = ["LightGBM", "LogReg"]
OUT = T.REPORTS
TEST_SCORES = OUT / "r3_test_scores.parquet"


def recall_top(k):
    def f(y, s):
        n = int(np.ceil(k * len(y)))
        return y[np.argsort(-s, kind="stable")[:n]].sum() / y.sum()
    f.__name__ = f"recall_top{int(k * 100)}"
    return f


METRICS = {"PR-AUC": average_precision_score, "recall top 5%": recall_top(0.05), "recall top 10%": recall_top(0.10)}


def gains_vs(d: pd.DataFrame, scores: dict, ref: str, names: list[str]) -> pd.DataFrame:
    """Paired day-bootstrap 95% CI of metric(model) - metric(ref) for each model and metric."""
    noise = 1e-9 * np.random.default_rng(0).random(len(d))   # fixed tie-break for top-k
    rows = []
    for n in names:
        if n == ref:
            continue
        for mname, f in METRICS.items():
            r = T.bootstrap_diff(d, np.asarray(scores[ref]) + noise, np.asarray(scores[n]) + noise, metric=f)
            rows.append({"model": n, "vs": ref, "metric": mname, **r})
    return pd.DataFrame(rows)


def load():
    return T.load_data("panel_features.parquet")


def run_val(verbose: bool = False):
    df = load()
    res, scores, sel = T.run_validation(df, verbose=verbose, feature_sets=FEATURE_SETS_R3)
    va = df[df.split == "val"]
    # gains over A within each model family
    gains = pd.concat([gains_vs(va, scores, f"{fam} A_time",
                                [f"{fam} {k}" for k in FEATURE_SETS_R3]) for fam in FAMILIES])
    # LightGBM seed stability on validation
    tr = df[df.split == "train"]
    seed_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fs, feats in FEATURE_SETS_R3.items():
            prm = sel[f"LightGBM {fs}"]
            aps = [average_precision_score(va.y, T.fit_lgbm(tr, feats, prm["num_leaves"], prm["min_child_samples"],
                                                            n_estimators=prm["n_estimators"], seed=s)
                                           .predict_proba(va[feats])[:, 1]) for s in range(5)]
            seed_rows.append({"feature_set": fs, "mean": np.mean(aps), "sd": np.std(aps)})
    seeds = pd.DataFrame(seed_rows)
    best = res[res.model.str.startswith(("LightGBM", "LogReg"))].sort_values("pr_auc", ascending=False).iloc[0].model
    sel["_best_on_validation"] = best

    res.to_csv(OUT / "r3_val_results.csv", index=False)
    gains.to_csv(OUT / "r3_val_gains_vs_A.csv", index=False)
    seeds.to_csv(OUT / "r3_val_seed_stability.csv", index=False)
    (OUT / "r3_selection.json").write_text(json.dumps(sel, indent=2))
    pd.DataFrame({k: np.asarray(v) for k, v in scores.items()}).assign(
        date=va.date.to_numpy(), route=va.route.astype(str).to_numpy(), period=va.period.astype(str).to_numpy(),
        y=va.y.to_numpy()).to_parquet(OUT / "r3_val_scores.parquet", index=False)
    return res, gains, seeds, sel


def run_test_once(force: bool = False):
    if TEST_SCORES.exists() and not force:
        raise SystemExit(f"{TEST_SCORES} exists: the 2026 test has already been run once. Not re-running.")
    df = load()
    sel = json.loads((OUT / "r3_selection.json").read_text())
    best = sel.pop("_best_on_validation")
    res, scores, _ = T.run_test(df, sel, train_splits=("train",), feature_sets=FEATURE_SETS_R3)
    te = df[df.split == "test"]
    gains = pd.concat([gains_vs(te, scores, f"{fam} A_time", [f"{fam} {k}" for k in FEATURE_SETS_R3])
                       for fam in FAMILIES])
    gains_base = gains_vs(te, scores, "Baseline: hist rate", [best, "LightGBM A_time"])
    res.to_csv(OUT / "r3_test_results.csv", index=False)
    gains.to_csv(OUT / "r3_test_gains_vs_A.csv", index=False)
    gains_base.to_csv(OUT / "r3_test_gains_vs_baseline.csv", index=False)
    pd.DataFrame({k: np.asarray(v) for k, v in scores.items()}).assign(
        date=te.date.to_numpy(), route=te.route.astype(str).to_numpy(), period=te.period.astype(str).to_numpy(),
        y=te.y.to_numpy(), n_overloads=te.n_overloads.to_numpy()).to_parquet(TEST_SCORES, index=False)
    (OUT / "r3_test_run.json").write_text(json.dumps(
        {"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "best_on_validation": best,
         "trained_on": "2022-2024", "note": "single evaluation of the round-3 feature sets on 2026"}, indent=2))
    return res, gains, gains_base, best


def main():
    pd.set_option("display.width", 220)
    cols = ["model", "pr_auc", "roc_auc", "recall_top5", "recall_top10", "recall_day5"]
    if "--test" in sys.argv:
        res, gains, gb, best = run_test_once(force="--force" in sys.argv)
        print("TEST 2026 (trained 2022-2024; single evaluation)\n", res[cols].round(4).to_string(index=False))
        print("\nbest model chosen on validation:", best)
        print("\ngains vs A (test):\n", gains.round(4).to_string(index=False))
        print("\ngains vs hist-rate baseline (test):\n", gb.round(4).to_string(index=False))
    else:
        res, gains, seeds, sel = run_val(verbose=False)
        print("VALIDATION 2025\n", res[cols].round(4).to_string(index=False))
        print("\nLightGBM seed stability (val PR-AUC, 5 seeds):\n", seeds.round(4).to_string(index=False))
        print("\ngains vs A (validation):\n", gains.round(4).to_string(index=False))
        print("\nbest on validation:", sel["_best_on_validation"])
        print("selected settings:", json.dumps({k: v for k, v in sel.items() if not k.startswith("_")}))


if __name__ == "__main__":
    main()
