"""Stateless feature builder (so it cannot leak) used inside every pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import OVERNIGHT_MAX_HOUR


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Drops raw Time and Amount; adds log_amount and optional flags."""

    def __init__(self, small_flag=False, overnight_flag=False):
        self.small_flag = small_flag
        self.overnight_flag = overnight_flag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = X.drop(columns=["Time", "Amount"]).copy()
        out["log_amount"] = np.log1p(X["Amount"])
        if self.small_flag:
            out["small_amount"] = (X["Amount"] <= 1).astype(float)
        if self.overnight_flag:
            hour_of_day = (X["Time"] // 3600) % 24
            out["overnight"] = (hour_of_day <= OVERNIGHT_MAX_HOUR).astype(float)
        return out