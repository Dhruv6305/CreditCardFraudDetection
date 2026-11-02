"""One evaluation module shared by every model (and the app)."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from .config import COST_FN, COST_FP, TARGET_PRECISION


def pr_auc(y, scores) -> float:
    return float(average_precision_score(y, scores))


def recall_at_precision(y, scores, target=TARGET_PRECISION) -> float:
    """Highest recall achievable while keeping precision >= target."""
    p, r, _ = precision_recall_curve(y, scores)
    ok = p >= target
    return float(r[ok].max()) if ok.any() else 0.0


def metrics_at_threshold(y, scores, threshold, cost_fn=COST_FN, cost_fp=COST_FP) -> dict:
    y = np.asarray(y)
    pred = np.asarray(scores) >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    cost = cost_fn * fn + cost_fp * fp
    cost_none = cost_fn * (tp + fn)            # baseline: flag nothing
    saving = 100 * (cost_none - cost) / cost_none if cost_none else 0.0
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "alarms_per_1000": 1000 * (tp + fp) / len(y),
        "cost": float(cost), "cost_flag_nothing": float(cost_none),
        "saving_pct": float(saving),
    }


def best_threshold_by_cost(y, scores, cost_fn=COST_FN, cost_fp=COST_FP) -> float:
    """Threshold (flag score >= t) that minimises cost_fn*FN + cost_fp*FP.
    If flagging nothing is cheapest, returns a threshold above every score."""
    y = np.asarray(y)
    s = np.asarray(scores)
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    cum_tp, cum_fp = np.cumsum(yy), np.cumsum(1 - yy)
    positives = yy.sum()
    cost = cost_fn * (positives - cum_tp) + cost_fp * cum_fp
    cuts = np.where(np.r_[ss[1:] != ss[:-1], True])[0]     # only cut between distinct scores
    best = cuts[np.argmin(cost[cuts])]
    if cost[best] >= cost_fn * positives:
        return float(ss[0]) + 1e-9
    return float(ss[best])


def rule_scores(X) -> np.ndarray:
    """Trivial baseline: flag Amount <= 1."""
    return (X["Amount"] <= 1).astype(float).to_numpy()