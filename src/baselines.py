"""Phase 1 check: prepare data, show the split, and score the trivial baselines per CV fold.
Run:  python -m src.baselines            (set FRAUD_CSV to use a local file)
"""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import N_SPLITS, RESULTS_DIR
from .data import load_raw, prepare, split_dev_test, xy
from .evaluate import pr_auc, recall_at_precision, rule_scores


def baseline_fold_table(dev: pd.DataFrame) -> pd.DataFrame:
    X, y = xy(dev)
    rows = []
    for k, (tr, va) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(X), start=1):
        s = rule_scores(X.iloc[va])
        rows.append({"model": "rule_amount_le_1", "fold": k, "train_fraud": int(y[tr].sum()),
                     "val_rows": len(va), "val_fraud": int(y[va].sum()),
                     "pr_auc": pr_auc(y[va], s), "recall_at_p50": recall_at_precision(y[va], s)})
    return pd.DataFrame(rows)


def main():
    df = prepare(load_raw())
    dev, test = split_dev_test(df)
    for name, part in (("dev", dev), ("test", test)):
        print(f"{name:4s} rows={len(part):7d} fraud={int(part['Class'].sum()):4d} "
              f"hours={part['Time'].min() / 3600:.1f}-{part['Time'].max() / 3600:.1f}")
    # NOTE: from here on, only `dev` is used. `test` is untouched until select_and_test.py.
    table = baseline_fold_table(dev)
    print(table.round(4).to_string(index=False))
    table.to_csv(RESULTS_DIR / "baseline_cv.csv", index=False)


if __name__ == "__main__":
    main()