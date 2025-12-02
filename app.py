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

feature_names = feature_template.columns
inputs = {}

st.subheader("Masukkan Data Kendaraan")

for col in feature_names:
    if feature_template[col].dtype == "object":
        inputs[col] = st.text_input(col)
    else:
        inputs[col] = st.number_input(col, value=0)

if st.button("Prediksi"):
    df_input = pd.DataFrame([inputs])

    for col in df_input.columns:
        try:
            df_input[col] = df_input[col].astype(int)
        except:
            df_input[col] = df_input[col].astype("category").cat.codes

    pred = model.predict(df_input)[0]
    label = label_y.inverse_transform([pred])[0]

    st.success(f"Kategori Harga: **{label}**")
