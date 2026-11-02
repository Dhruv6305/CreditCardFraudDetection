"""Streamlit demo for the credit card fraud model."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.evaluate import metrics_at_threshold
from src.predict import decide, load_bundle, score

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_demo():
    return pd.read_csv(ROOT / "data" / "demo_sample.csv")


@st.cache_data
def get_test_scores():
    return pd.read_csv(ROOT / "results" / "test_scores.csv")


@st.cache_data
def get_metrics():
    with open(ROOT / "results" / "test_metrics.json") as f:
        return json.load(f)


bundle = get_bundle()
saved_threshold = float(bundle["threshold"])

st.title("Credit Card Transaction Fraud Detection")
st.caption("ULB credit card dataset (anonymized PCA features). Demo only: not a production system. "
           "The score is a ranking score, not a calibrated probability.")

try:
    import sklearn
    if sklearn.__version__ != bundle["versions"]["sklearn"]:
        st.warning(f"scikit-learn {sklearn.__version__} here vs {bundle['versions']['sklearn']} "
                   "when the model was trained. Pin versions in requirements.txt.")
except Exception:
    pass

tab_check, tab_batch, tab_perf = st.tabs(["Check a transaction", "Batch scoring", "Model performance"])

# ------------------------------------------------------------------ single transaction
with tab_check:
    demo = get_demo()
    st.write("Pick a transaction from a small sample of the **held-out test period** "
             "(all of its fraud cases plus a random set of genuine ones, so fraud looks far more "
             "common here than in reality). The V1 to V28 features are anonymized, so manual entry "
             "would be meaningless.")
    show = st.radio("Show", ["All", "Fraud only", "Genuine only"], horizontal=True)
    view = demo if show == "All" else demo[demo["Class"] == (1 if show == "Fraud only" else 0)]
    idx = st.selectbox("Transaction", view.index,
                       format_func=lambda i: f"#{i}  amount={demo.loc[i, 'Amount']:.2f}")
    thr = st.slider("Decision threshold", 0.0, 1.0, min(saved_threshold, 1.0), 0.001, key="thr1")
    row = demo.loc[[idx]]
    s = float(score(bundle, row)[0])
    flagged = bool(decide([s], thr)[0])
    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud score", f"{s:.4f}")
    c2.metric("Threshold", f"{thr:.3f}")
    c3.metric("Decision", "FLAG / review" if flagged else "Approve")
    with st.expander("Ground truth and raw values"):
        st.write(f"True label: {'FRAUD' if int(row['Class'].iloc[0]) == 1 else 'genuine'}")
        st.dataframe(row.T.rename(columns={idx: "value"}))

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write("Upload a CSV with columns: " + ", ".join(bundle["input_columns"][:3]) + ", ... , Amount "
             "(Time, V1 to V28, Amount).")
    up = st.file_uploader("CSV file", type="csv")
    thr_b = st.slider("Decision threshold", 0.0, 1.0, min(saved_threshold, 1.0), 0.001, key="thr2")
    if up is not None:
        data = pd.read_csv(up)
        try:
            out = data.copy()
            out["fraud_score"] = score(bundle, data)
            out["flagged"] = decide(out["fraud_score"], thr_b)
            st.write(f"{int(out['flagged'].sum())} of {len(out)} transactions flagged.")
            st.dataframe(out.sort_values("fraud_score", ascending=False).head(200))
            st.download_button("Download scored CSV", out.to_csv(index=False), "scored.csv", "text/csv")
        except ValueError as e:
            st.error(str(e))

# ------------------------------------------------------------------ performance
with tab_perf:
    m = get_metrics()
    ts = get_test_scores()
    st.subheader("Held-out test slice (scored once)")
    st.write(f"Model: **{m['model']}**. Test rows: {m['test']['test_rows']:,}, "
             f"fraud cases: {m['test']['test_fraud']}. With so few fraud cases, small differences "
             "are noise.")
    c1, c2 = st.columns(2)
    cost_fn = c1.number_input("Cost of a missed fraud", value=float(m["cost_fn"]), min_value=0.0)
    cost_fp = c2.number_input("Cost of a false alarm", value=float(m["cost_fp"]), min_value=0.0)
    thr_p = st.slider("Decision threshold", 0.0, 1.0, min(saved_threshold, 1.0), 0.001, key="thr3")
    r = metrics_at_threshold(ts["y"], ts["score"], thr_p, cost_fn, cost_fp)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recall", f"{r['recall']:.3f}", f"{r['tp']} of {r['tp'] + r['fn']} frauds")
    k2.metric("Precision", f"{r['precision']:.3f}")
    k3.metric("Alarms / 1,000", f"{r['alarms_per_1000']:.2f}")
    k4.metric("Cost saving vs flagging nothing", f"{r['saving_pct']:.1f}%")
    st.dataframe(pd.DataFrame([[r["tn"], r["fp"]], [r["fn"], r["tp"]]],
                              index=["actual genuine", "actual fraud"],
                              columns=["predicted genuine", "predicted fraud"]))
    col_a, col_b = st.columns(2)
    if (ROOT / "results" / "pr_curve.png").exists():
        col_a.image(str(ROOT / "results" / "pr_curve.png"), caption="Precision-recall curve (test)")
    if (ROOT / "results" / "shap_summary.png").exists():
        col_b.image(str(ROOT / "results" / "shap_summary.png"), caption="Which features drive the score (SHAP)")
    st.subheader("Cross-validation summary (folds 2 to 5)")
    st.dataframe(pd.DataFrame(m["cv_summary"]))
    st.caption("Limitations: two days of data, 52 test fraud cases, anonymized features, no card IDs, "
               "and a placeholder cost assumption.")