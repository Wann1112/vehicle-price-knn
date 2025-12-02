import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("xgb_model.pkl")
label_y = joblib.load("label_encoder.pkl")
feature_template = joblib.load("feature_columns.pkl")

st.title("🚗 Prediksi Kategori Harga Mobil")
st.write("Model: XGBoost Classifier")

# Ambil kolom fitur
feature_names = feature_template.columns

# Form Input
st.subheader("Masukkan Data Mobil")

inputs = {}

for col in feature_names:
    if feature_template[col].dtype == "object":
        inputs[col] = st.text_input(col)
    else:
        inputs[col] = st.number_input(col, value=0)

# Button prediksi
if st.button("Prediksi Harga"):
    df_input = pd.DataFrame([inputs])

    # Encoding kategori (match training)
    for col in df_input.columns:
        if feature_template[col].dtype == "int64" and df_input[col].dtype == "object":
            df_input[col] = df_input[col].astype("category").cat.codes

    pred = model.predict(df_input)[0]
    pred_label = label_y.inverse_transform([pred])[0]

    st.success(f"Kategori Harga: **{pred_label}**")
