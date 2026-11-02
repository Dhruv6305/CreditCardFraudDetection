"""Project-wide constants. Every decision from the EDA report lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

V_COLS = [f"V{i}" for i in range(1, 29)]
INPUT_COLS = ["Time"] + V_COLS + ["Amount"]       # what the model receives
SCALE_COLS = V_COLS + ["log_amount"]              # what the robust scaler touches

RANDOM_STATE = 42
TEST_FRAC = 0.15                 # last 15% of rows (by time) = final test slice
N_SPLITS = 5                     # TimeSeriesSplit folds on the development set
SELECTION_FOLDS = (2, 3, 4, 5)   # fold 1 is reported but not used for decisions

# Placeholder cost assumption (units are arbitrary currency units)
COST_FN = 100.0                  # a missed fraud
COST_FP = 5.0                    # a false alarm
SENSITIVITY_COST_FN = (25.0, 100.0, 500.0)   # 5:1, 20:1, 100:1 with COST_FP = 5

TARGET_PRECISION = 0.5           # for "recall at precision >= 50%"
OVERNIGHT_MAX_HOUR = 7           # elapsed-hour-of-day 0..7 = the low-volume stretch