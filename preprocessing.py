# preprocessing.py
import re
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import PRICE_BINS, PRICE_LABELS

def clean_numeric_str(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^0-9.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def extract_float(series: pd.Series) -> pd.Series:
    # e.g., "7.4 L/100km" -> 7.4
    s = series.astype(str).str.extract(r"(\d+\.?\d*)", expand=False)
    return pd.to_numeric(s, errors="coerce")

def extract_engine_size(series: pd.Series) -> pd.Series:
    # e.g., "2.0L Turbo" -> 2.0
    s = series.astype(str).str.extract(r"(\d+\.\d+)", expand=False)
    return pd.to_numeric(s, errors="coerce")

def extract_cylinders(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(s, errors="coerce")

def bin_price(price_series: pd.Series) -> pd.Series:
    return pd.cut(price_series, bins=PRICE_BINS, labels=PRICE_LABELS)

def fit_encoders(df: pd.DataFrame, cat_cols: list) -> dict:
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return encoders

def transform_with_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    for col, le in encoders.items():
        df[col] = df[col].astype(str)
        # Unseen categories handling: map unknowns to a special index if needed
        known = set(le.classes_)
        df[col] = df[col].map(lambda x: x if x in known else None)
        # Refit to include None if present (optional), or fill None with mode
        df[col] = df[col].fillna(df[col].mode().iloc[0])
        df[col] = le.transform(df[col])
    return df

def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)

def save_feature_names(path, cols):
    with open(path, "w") as f:
        json.dump(list(cols), f)

def load_feature_names(path):
    import json
    with open(path, "r") as f:
        return json.load(f)
