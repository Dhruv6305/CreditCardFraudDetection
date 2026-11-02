"""Load, clean and split the ULB credit card data."""
import os
from pathlib import Path

import pandas as pd

from .config import INPUT_COLS, TEST_FRAC


def load_raw(csv_path=None) -> pd.DataFrame:
    """Load creditcard.csv from a path, the FRAUD_CSV env var, or kagglehub."""
    csv_path = csv_path or os.environ.get("FRAUD_CSV")
    if csv_path is None:
        import kagglehub
        folder = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        csv_path = Path(folder) / "creditcard.csv"
    return pd.read_csv(csv_path)


def prepare(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """Drop exact duplicates, sort by Time, and verify the order."""
    before, fraud_before = len(df), int(df["Class"].sum())
    df = df.drop_duplicates().sort_values("Time", kind="stable").reset_index(drop=True)
    assert df["Time"].is_monotonic_increasing
    if verbose:
        print(f"Rows {before} -> {len(df)} | fraud {fraud_before} -> {int(df['Class'].sum())}")
    return df


def split_dev_test(df: pd.DataFrame, test_frac: float = TEST_FRAC):
    """Cut by row position (data is time-ordered): dev = early rows, test = latest rows."""
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def xy(df: pd.DataFrame):
    return df[INPUT_COLS], df["Class"].to_numpy()