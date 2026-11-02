"""Model zoo: a small, fixed set of configurations (no big search)."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import RANDOM_STATE, SCALE_COLS
from .features import FeatureBuilder

MODEL_NAMES = ["logreg", "rf", "xgb_d3", "xgb_d5"]


def _make_model(name):
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
    if name.startswith("xgb"):
        from xgboost import XGBClassifier
        depth = int(name.split("_d")[1])
        return XGBClassifier(
            n_estimators=300, max_depth=depth, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=RANDOM_STATE, scale_pos_weight=1.0)
    raise ValueError(name)


def build_pipeline(name, small_flag=False, overnight_flag=False) -> Pipeline:
    steps = [("features", FeatureBuilder(small_flag, overnight_flag))]
    if name == "logreg":   # only the linear model needs scaling; flags pass through
        steps.append(("scale", ColumnTransformer(
            [("robust", RobustScaler(), SCALE_COLS)], remainder="passthrough")))
    steps.append(("model", _make_model(name)))
    return Pipeline(steps)


def fit_pipeline(pipe, X, y):
    """Fit; for XGBoost set scale_pos_weight = sqrt(neg/pos) from THIS training part only."""
    if "scale_pos_weight" in pipe.named_steps["model"].get_params():
        pos = max(int(y.sum()), 1)
        neg = len(y) - pos
        pipe.set_params(model__scale_pos_weight=(neg / pos) ** 0.5)
    return pipe.fit(X, y)