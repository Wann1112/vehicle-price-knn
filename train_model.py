# train_model.py
import os
import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint, uniform

from config import (
    ARTIFACTS_DIR, MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURES_PATH,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, PRICE_BINS, PRICE_LABELS
)
from preprocessing import (
    clean_numeric_str, extract_float, extract_engine_size, extract_cylinders,
    bin_price, fit_encoders, transform_with_encoders,
    fit_scaler, apply_scaler, save_feature_names
)

def ensure_artifacts_dir():
    ARTIFACTS_DIR.mkdir(exist_ok=True)

def load_data(local_path=None):
    if local_path and Path(local_path).exists():
        return pd.read_csv(local_path)
    # Fallback: instruct user to provide local CSV path
    raise FileNotFoundError("Provide local CSV path (e.g., data/Australian Vehicle Prices.csv).")

def preprocess(df: pd.DataFrame):
    # Clean numeric-like columns
    df["Price"] = clean_numeric_str(df["Price"])
    df["Kilometres"] = clean_numeric_str(df["Kilometres"])

    # FuelConsumption (if present)
    if "FuelConsumption" in df.columns:
        df["FuelConsumption"] = extract_float(df["FuelConsumption"])

    # Engine size (from Engine or Engine_Size)
    if "Engine" in df.columns:
        df["EngineSize"] = extract_engine_size(df["Engine"])
    elif "Engine_Size" in df.columns:
        df["EngineSize"] = pd.to_numeric(df["Engine_Size"], errors="coerce")
    else:
        df["EngineSize"] = np.nan

    # Cylinders
    if "CylindersinEngine" in df.columns:
        df["Cylinders"] = extract_cylinders(df["CylindersinEngine"])
    elif "Cylinders" in df.columns:
        df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce")
    else:
        df["Cylinders"] = np.nan

    # Drop rows with critical NaNs
    df = df.dropna(subset=["Price", "Kilometres"])

    # Target
    df["Price_Category"] = bin_price(df["Price"])

    # Basic imputation for remaining NaNs
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df

def build_features(df: pd.DataFrame):
    # Drop raw price and target from features
    y = df["Price_Category"]
    X = df.drop(columns=["Price", "Price_Category"], errors="ignore")

    # Identify categorical columns for encoding
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Fit encoders on training data
    encoders = fit_encoders(X, cat_cols)

    # Transform categoricals
    X_enc = transform_with_encoders(X.copy(), encoders)

    # Scale numerics
    scaler = fit_scaler(X_enc)
    X_scaled = apply_scaler(X_enc, scaler)

    return X, y, encoders, scaler, X_scaled

def randomized_rf():
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    # Randomized search space (wider than grid)
    param_dist = {
        "n_estimators": randint(150, 400),
        "max_depth": [None] + list(range(8, 24, 4)),
        "min_samples_split": randint(2, 8),
        "min_samples_leaf": randint(1, 4),
        "max_features": ["sqrt", "log2"]
    }
    rs = RandomizedSearchCV(
        rf, param_dist, n_iter=25, cv=CV_FOLDS, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=1
    )
    return rs

def main():
    ensure_artifacts_dir()

    # Load local CSV
    df = load_data(local_path="data/Australian Vehicle Prices.csv")

    # Preprocess
    df = preprocess(df)

    # Train/test split (stratify by target)
    X_raw, y, encoders, scaler, X_scaled = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Randomized search RandomForest
    search = randomized_rf()
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Best Params:", search.best_params_)
    print("CV Best Score:", search.best_score_)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Persist artifacts
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    save_feature_names(FEATURES_PATH, X_raw.columns)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")
    print(f"Saved encoders to: {ENCODER_PATH}")
    print(f"Saved feature names to: {FEATURES_PATH}")

if __name__ == "__main__":
    main()
