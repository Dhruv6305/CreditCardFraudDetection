"""Phase 2: time-aware cross-validation of every model on the development set.
Run:  python -m src.cv                    (all models, flags off)
      python -m src.cv --ablate xgb_d3    (feature-flag ablation for one model)
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .baselines import baseline_fold_table
from .config import N_SPLITS, RESULTS_DIR, SELECTION_FOLDS
from .data import load_raw, prepare, split_dev_test, xy
from .evaluate import pr_auc, recall_at_precision
from .models import MODEL_NAMES, build_pipeline, fit_pipeline


def run_cv(dev, name, small_flag=False, overnight_flag=False):
    """Returns (per-fold table, out-of-fold scores, fold id per row; 0 = never validated)."""
    X, y = xy(dev)
    oof = np.full(len(dev), np.nan)
    fold_id = np.zeros(len(dev), dtype=int)
    rows = []
    for k, (tr, va) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(X), start=1):
        pipe = fit_pipeline(build_pipeline(name, small_flag, overnight_flag), X.iloc[tr], y[tr])
        s = pipe.predict_proba(X.iloc[va])[:, 1]
        oof[va], fold_id[va] = s, k
        rows.append({"model": name, "small_flag": small_flag, "overnight_flag": overnight_flag,
                     "fold": k, "train_fraud": int(y[tr].sum()), "val_rows": len(va),
                     "val_fraud": int(y[va].sum()), "pr_auc": pr_auc(y[va], s),
                     "recall_at_p50": recall_at_precision(y[va], s)})
    return pd.DataFrame(rows), oof, fold_id


def summarise(results: pd.DataFrame, group=("model",)) -> pd.DataFrame:
    sel = results[results["fold"].isin(SELECTION_FOLDS)]
    out = sel.groupby(list(group)).agg(
        mean_pr_auc=("pr_auc", "mean"), std_pr_auc=("pr_auc", "std"),
        mean_recall_p50=("recall_at_p50", "mean")).round(4)
    return out.sort_values("mean_pr_auc", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablate", help="model name for the feature-flag ablation")
    args = parser.parse_args()

    dev, _test = split_dev_test(prepare(load_raw()))   # test slice is never used here

    if args.ablate:
        frames = []
        for small, night in itertools.product([False, True], repeat=2):
            res, _, _ = run_cv(dev, args.ablate, small, night)
            frames.append(res)
        allres = pd.concat(frames)
        print(allres.round(4).to_string(index=False))
        print("\nSummary on folds", SELECTION_FOLDS)
        print(summarise(allres, group=("small_flag", "overnight_flag")))
        allres.to_csv(RESULTS_DIR / f"ablation_{args.ablate}.csv", index=False)
        return

    frames = [baseline_fold_table(dev)]
    for name in MODEL_NAMES:
        print(f"running {name} ...")
        res, _, _ = run_cv(dev, name)
        frames.append(res)
    allres = pd.concat(frames, ignore_index=True)
    print(allres.round(4).to_string(index=False))
    print("\nSummary on folds", SELECTION_FOLDS, "(fold 1 reported only)")
    print(summarise(allres))
    allres.to_csv(RESULTS_DIR / "cv_results.csv", index=False)


if __name__ == "__main__":
    main()