"""The prediction contract: raw transaction rows in, fraud score out."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


def load_bundle(path=MODEL_PATH) -> dict:
    return joblib.load(path)


def score(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Fraud score in [0, 1]. Not a calibrated probability (class weights were used)."""
    missing = [c for c in bundle["input_columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return bundle["pipeline"].predict_proba(df[bundle["input_columns"]])[:, 1]


def decide(scores, threshold: float) -> np.ndarray:
    return np.asarray(scores) >= threshold