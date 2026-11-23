"""Round 4: does scheduled capacity (GTFS) improve on the round-3 models?

Same rows, split, metrics and rules as rounds 2-3 (settings and model choice on 2025 only; 2026 evaluated
once). Feature sets:
    A_time               route, period, weekday, month                  (reference)
    B_history            A + overload history                           (reference)
    F_all3_weather       A + history + calendar + ridership + weather   (round-3 final, reference)
    G_capacity           A + capacity
    H_history_capacity   B + capacity
    I_all_capacity       F + capacity

    python src/round4.py           validation stage  -> reports/r4_val_*, r4_selection.json
    python src/round4.py --test    test stage, ONCE  -> reports/r4_test_*
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import train as T
from capacity import CAP
from features import CAL, HIST, RIDE
from round3 import FAMILIES, gains_vs

FEATURE_SETS_R4 = {
    "A_time": T.TIME,
    "B_history": T.TIME + HIST,
    "F_all3_weather": T.TIME + HIST + CAL + RIDE + T.WEATHER,
    "G_capacity": T.TIME + CAP,
    "H_history_capacity": T.TIME + HIST + CAP,
    "I_all_capacity": T.TIME + HIST + CAL + RIDE + T.WEATHER + CAP,
}
# pairs that answer "does capacity add something?" (reference -> with capacity)
PAIRS = [("A_time", "G_capacity"), ("B_history", "H_history_capacity"), ("F_all3_weather", "I_all_capacity")]
OUT = T.REPORTS
TEST_SCORES = OUT / "r4_test_scores.parquet"

# capacity counts/ratios: log1p for counts and per-trip loads in logistic regression
T.LOG1P_COLS += ["sched_trips", "sched_trips_peakdir", "sched_trips_per_hour", "load_per_trip", "peakdir_load_per_trip"]
T.LOG_COLS += ["sched_change_4w"]


def load():
    return T.load_data("panel_capacity.parquet")


def pair_gains(d, scores):
    rows = []
    for fam in FAMILIES:
        for a, b in PAIRS:
            rows.append(gains_vs(d, scores, f"{fam} {a}", [f"{fam} {b}"]))
    return pd.concat(rows)


def save_scores(d, scores, path, extra=()):
    df = pd.DataFrame({k: np.asarray(v) for k, v in scores.items()}).assign(
        date=d.date.to_numpy(), route=d.route.astype(str).to_numpy(), period=d.period.astype(str).to_numpy(),
        y=d.y.to_numpy())
    for c in extra:
        df[c] = d[c].to_numpy()
    df.to_parquet(path, index=False)


def run_val(verbose=False):
    df = load()
    res, scores, sel = T.run_validation(df, verbose=verbose, feature_sets=FEATURE_SETS_R4)
    va = df[df.split == "val"]
    gains = pair_gains(va, scores)
    tr = df[df.split == "train"]
    seed_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fs, feats in FEATURE_SETS_R4.items():
            prm = sel[f"LightGBM {fs}"]
            aps = [average_precision_score(va.y, T.fit_lgbm(tr, feats, prm["num_leaves"], prm["min_child_samples"],
                                                            n_estimators=prm["n_estimators"], seed=s)
                                           .predict_proba(va[feats])[:, 1]) for s in range(5)]
            seed_rows.append({"feature_set": fs, "mean": np.mean(aps), "sd": np.std(aps)})
    seeds = pd.DataFrame(seed_rows)
    best = res[res.model.str.startswith(("LightGBM", "LogReg"))].sort_values("pr_auc", ascending=False).iloc[0].model
    sel["_best_on_validation"] = best
    res.to_csv(OUT / "r4_val_results.csv", index=False)
    gains.to_csv(OUT / "r4_val_capacity_gains.csv", index=False)
    seeds.to_csv(OUT / "r4_val_seed_stability.csv", index=False)
    (OUT / "r4_selection.json").write_text(json.dumps(sel, indent=2))
    save_scores(va, scores, OUT / "r4_val_scores.parquet")
    return res, gains, seeds, sel


def run_test_once(force=False):
    if TEST_SCORES.exists() and not force:
        raise SystemExit(f"{TEST_SCORES} exists: the round-4 test has already been run once. Not re-running.")
    df = load()
    sel = json.loads((OUT / "r4_selection.json").read_text())
    best = sel.pop("_best_on_validation")
    res, scores, _ = T.run_test(df, sel, train_splits=("train",), feature_sets=FEATURE_SETS_R4)
    te = df[df.split == "test"]
    gains = pair_gains(te, scores)
    gains_base = gains_vs(te, scores, "Baseline: hist rate", [best])
    res.to_csv(OUT / "r4_test_results.csv", index=False)
    gains.to_csv(OUT / "r4_test_capacity_gains.csv", index=False)
    gains_base.to_csv(OUT / "r4_test_gains_vs_baseline.csv", index=False)
    save_scores(te, scores, TEST_SCORES, extra=("n_overloads",))
    (OUT / "r4_test_run.json").write_text(json.dumps(
        {"run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "best_on_validation": best,
         "trained_on": "2022-2024", "note": "single evaluation of the round-4 (capacity) feature sets on 2026"}, indent=2))
    return res, gains, gains_base, best


def main():
    pd.set_option("display.width", 220)
    cols = ["model", "pr_auc", "roc_auc", "recall_top5", "recall_top10"]
    if "--test" in sys.argv:
        res, gains, gb, best = run_test_once(force="--force" in sys.argv)
        print("TEST 2026 (single evaluation)\n", res[cols].round(4).to_string(index=False))
        print("\nbest on validation:", best)
        print("\ncapacity gains (test):\n", gains.round(4).to_string(index=False))
        print("\nbest vs hist-rate baseline (test):\n", gb.round(4).to_string(index=False))
    else:
        res, gains, seeds, sel = run_val()
        print("VALIDATION 2025\n", res[cols].round(4).to_string(index=False))
        print("\nLightGBM seed stability:\n", seeds.round(4).to_string(index=False))
        print("\ncapacity gains (validation):\n", gains.round(4).to_string(index=False))
        print("\nbest on validation:", sel["_best_on_validation"])


if __name__ == "__main__":
    main()
