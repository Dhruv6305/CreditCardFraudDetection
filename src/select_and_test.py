"""Phase 3: pick the model, choose the threshold, score the test slice ONCE, save artifacts.
Run:  python -m src.select_and_test [--model xgb_d3] [--small-flag] [--overnight-flag] [--force]
"""
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import (COST_FN, COST_FP, DATA_DIR, INPUT_COLS, MODELS_DIR, RANDOM_STATE,
                     RESULTS_DIR, SELECTION_FOLDS, SENSITIVITY_COST_FN)
from .cv import run_cv, summarise
from .data import load_raw, prepare, split_dev_test, xy
from .evaluate import (best_threshold_by_cost, metrics_at_threshold, pr_auc,
                       recall_at_precision, rule_scores)
from .models import build_pipeline, fit_pipeline

TEST_LOCK = RESULTS_DIR / "test_metrics.json"


def paired_report(results: pd.DataFrame, a: str, b: str):
    """Within-fold comparison of a vs b on the selection folds."""
    sel = results[results["fold"].isin(SELECTION_FOLDS)]
    pa = sel[sel["model"] == a].set_index("fold")["pr_auc"]
    pb = sel[sel["model"] == b].set_index("fold")["pr_auc"]
    diff = (pa - pb).dropna()
    wins = int((diff > 0).sum())
    print(f"{a} vs {b}: wins {wins}/{len(diff)} folds, mean diff {diff.mean():+.4f}, "
          f"std of diffs {diff.std():.4f}")
    clear = wins >= 3 and diff.mean() > diff.std()
    print("  -> clear win" if clear else "  -> NOT a clear win; prefer the simpler model if close")


def save_shap(pipe, dev, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        print("shap/matplotlib missing; skipping SHAP")
        return
    if "scale" in pipe.named_steps:
        print("SHAP skipped for the linear pipeline")
        return
    fraud = dev[dev["Class"] == 1]
    genuine = dev[dev["Class"] == 0].sample(min(3000, int((dev["Class"] == 0).sum())),
                                            random_state=RANDOM_STATE)
    sample = pd.concat([fraud, genuine])[INPUT_COLS]
    feats = pipe.named_steps["features"].transform(sample)
    sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(feats)
    if isinstance(sv, list):
        sv = sv[1]
    if getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    plt.figure()
    shap.summary_plot(sv, feats, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="override the automatically chosen model")
    ap.add_argument("--small-flag", action="store_true")
    ap.add_argument("--overnight-flag", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-score the test slice anyway")
    args = ap.parse_args()

    if TEST_LOCK.exists() and not args.force:
        sys.exit("The test slice was already scored (results/test_metrics.json exists). "
                 "Re-scoring it after more tuning defeats the purpose. Use --force only deliberately.")

    dev, test = split_dev_test(prepare(load_raw()))
    Xdev, ydev = xy(dev)

    # ---- 1. choose the model from the CV results (folds 2-5 only) ----
    cv_results = pd.read_csv(RESULTS_DIR / "cv_results.csv")
    summary = summarise(cv_results)
    print(summary, "\n")
    models_only = summary.drop(index="rule_amount_le_1", errors="ignore")
    ranked = list(models_only.index)
    name = args.model or ranked[0]
    print("chosen model:", name)
    if len(ranked) > 1 and name == ranked[0]:
        paired_report(cv_results, ranked[0], ranked[1])

    # ---- 2. threshold from pooled out-of-fold predictions (folds 2-5) ----
    _, oof, fold_id = run_cv(dev, name, args.small_flag, args.overnight_flag)
    mask = np.isin(fold_id, SELECTION_FOLDS)
    y_oof, s_oof = ydev[mask], oof[mask]
    print(f"\npooled OOF rows={mask.sum()} fraud={int(y_oof.sum())}")
    sens = []
    for cfn in SENSITIVITY_COST_FN:
        thr = best_threshold_by_cost(y_oof, s_oof, cfn, COST_FP)
        m = metrics_at_threshold(y_oof, s_oof, thr, cfn, COST_FP)
        m["cost_ratio"] = f"{cfn / COST_FP:g}:1"
        sens.append(m)
    sens_df = pd.DataFrame(sens)[["cost_ratio", "threshold", "recall", "precision",
                                  "alarms_per_1000", "saving_pct"]]
    print(sens_df.round(4).to_string(index=False))
    threshold = best_threshold_by_cost(y_oof, s_oof, COST_FN, COST_FP)
    print(f"\nFINAL threshold (cost {COST_FN:g}/{COST_FP:g}): {threshold:.6f}")

    # ---- 3. final fit on the whole development set; ONE look at the test slice ----
    pipe = fit_pipeline(build_pipeline(name, args.small_flag, args.overnight_flag), Xdev, ydev)
    Xte, yte = xy(test)
    s_te = pipe.predict_proba(Xte)[:, 1]
    test_metrics = metrics_at_threshold(yte, s_te, threshold)
    test_metrics.update({"pr_auc": pr_auc(yte, s_te),
                         "recall_at_p50": recall_at_precision(yte, s_te),
                         "test_rows": len(yte), "test_fraud": int(yte.sum())})
    rule = metrics_at_threshold(yte, rule_scores(Xte), 0.5)
    rule["pr_auc"] = pr_auc(yte, rule_scores(Xte))
    print("\nTEST (scored once):")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in test_metrics.items()}, indent=2))
    print(f"caught {test_metrics['tp']} of {test_metrics['test_fraud']} frauds, "
          f"{test_metrics['fp']} false alarms")
    print(f"rule baseline (Amount<=1): PR-AUC {rule['pr_auc']:.4f}, recall {rule['recall']:.3f}, "
          f"precision {rule['precision']:.4f}, saving {rule['saving_pct']:.1f}%")

    # ---- 4. save everything the app and README need ----
    try:
        import xgboost
        xgb_version = xgboost.__version__
    except ImportError:
        xgb_version = None
    bundle = {"pipeline": pipe, "threshold": float(threshold), "input_columns": INPUT_COLS,
              "model_name": name, "small_flag": args.small_flag,
              "overnight_flag": args.overnight_flag,
              "versions": {"sklearn": sklearn.__version__, "xgboost": xgb_version}}
    joblib.dump(bundle, MODELS_DIR / "model.joblib", compress=3)
    pd.DataFrame({"y": yte, "score": s_te}).to_csv(RESULTS_DIR / "test_scores.csv", index=False)
    demo = pd.concat([test[test["Class"] == 1],
                      test[test["Class"] == 0].sample(450, random_state=RANDOM_STATE)])
    demo.sort_values("Time")[INPUT_COLS + ["Class"]].to_csv(DATA_DIR / "demo_sample.csv", index=False)
    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump({"model": name, "cost_fn": COST_FN, "cost_fp": COST_FP,
                   "test": test_metrics, "rule_baseline": rule,
                   "cv_summary": summary.reset_index().to_dict("records"),
                   "sensitivity_oof": sens_df.to_dict("records")}, f, indent=2, default=float)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        p, r, _ = precision_recall_curve(yte, s_te)
        plt.figure(figsize=(5, 4))
        plt.plot(r, p)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Test PR curve (PR-AUC {test_metrics['pr_auc']:.3f})")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / "pr_curve.png", dpi=150); plt.close()
    except ImportError:
        pass
    save_shap(pipe, dev, RESULTS_DIR / "shap_summary.png")
    print("\nsaved: models/model.joblib, results/*, data/demo_sample.csv")


if __name__ == "__main__":
    main()