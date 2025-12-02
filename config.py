# config.py
from pathlib import Path

# Binning thresholds (AUD)
PRICE_BINS = [0, 20000, 40000, 60000, float("inf")]
PRICE_LABELS = ["Murah", "Sedang", "Mahal", "Sangat Mahal"]

# Random seeds and splits
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3

# Paths
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "rf_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
ENCODER_PATH = ARTIFACTS_DIR / "encoders.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"
