# app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURES_PATH, PRICE_LABELS
from preprocessing import transform_with_encoders, apply_scaler

st.set_page_config(page_title="Klasifikasi Harga Kendaraan", page_icon="🚗", layout="centered")
st.title("🚗 Klasifikasi Tingkat Harga Kendaraan")
st.caption("Random Forest dengan Optimasi Hyperparameter (RandomizedSearchCV)")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)
    return model, scaler, encoders, feature_names

def preprocess_inference(df: pd.DataFrame, encoders, scaler, feature_names):
    # Ensure required columns exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan

    # Basic imputation (keep simple for app)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Transform categoricals
    df_enc = transform_with_encoders(df.copy(), encoders)

    # Align column order
    df_enc = df_enc[feature_names]

    # Scale
    X_scaled = apply_scaler(df_enc, scaler)
    return X_scaled

# Load artifacts
try:
    model, scaler, encoders, feature_names = load_artifacts()
except Exception as e:
    st.error("Model belum tersedia. Jalankan train_model.py dan commit artifacts ke repo.")
    st.stop()

tab1, tab2 = st.tabs(["Prediksi satu kendaraan", "Upload CSV untuk batch prediksi"])

with tab1:
    st.subheader("Input spesifikasi kendaraan")
    # Minimal set; adjust to your dataset columns present in feature_names
    # Use text inputs for categoricals and numbers; Streamlit will coerce
    inputs = {}
    for col in feature_names:
        if col.lower() in ["price", "price_category"]:
            continue
        # Heuristic: numeric or text input
        if "year" in col.lower() or "kilometres" in col.lower() or "engine" in col.lower() or "cylinders" in col.lower() or "capacity" in col.lower():
            val = st.number_input(f"{col}", value=0.0, step=1.0)
        else:
            val = st.text_input(f"{col}", value="")
        inputs[col] = val

    if st.button("Prediksi kategori harga"):
        df_in = pd.DataFrame([inputs])
        X_scaled = preprocess_inference(df_in, encoders, scaler, feature_names)
        pred = model.predict(X_scaled)[0]
        st.success(f"Prediksi kategori harga: {pred}")

with tab2:
    st.subheader("Unggah CSV dengan kolom fitur")
    file = st.file_uploader("Pilih file CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview data:", df.head())
        X_scaled = preprocess_inference(df, encoders, scaler, feature_names)
        preds = model.predict(X_scaled)
        out = df.copy()
        out["Predicted_Price_Category"] = preds
        st.write("Hasil prediksi:", out.head())
        st.download_button(
            "Unduh hasil (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="prediksi_harga_kendaraan.csv",
            mime="text/csv"
        )
