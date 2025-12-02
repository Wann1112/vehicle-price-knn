# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Distribusi Harga Kendaraan", page_icon="🚗", layout="wide")

st.title("🚗 Klasifikasi Distribusi Harga Kendaraan")
st.write("Aplikasi ini menampilkan distribusi harga kendaraan berdasarkan dataset Australian Vehicle Prices.")

# Upload dataset
uploaded_file = st.file_uploader("Unggah file CSV dataset kendaraan", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Dataset")
    st.write(df.head())

    # Bersihkan kolom Price
    df['Price'] = df['Price'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Buat kategori harga
    bins = [0, 20000, 40000, 60000, float("inf")]
    labels = ["Murah", "Sedang", "Mahal", "Sangat Mahal"]
    df['Price_Category'] = pd.cut(df['Price'], bins=bins, labels=labels)

    st.subheader("Distribusi Kategori Harga")
    st.write(df['Price_Category'].value_counts())

    # Visualisasi distribusi
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Price_Category", data=df, order=labels, palette="viridis", ax=ax)
    ax.set_title("Distribusi Kategori Harga Kendaraan")
    ax.set_xlabel("Kategori Harga")
    ax.set_ylabel("Jumlah Kendaraan")
    st.pyplot(fig)

    # Histogram harga
    st.subheader("Histogram Harga Kendaraan")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Price'], bins=50, kde=True, color="blue", ax=ax2)
    ax2.set_title("Distribusi Harga Kendaraan (AUD)")
    ax2.set_xlabel("Harga (AUD)")
    ax2.set_ylabel("Frekuensi")
    st.pyplot(fig2)

else:
    st.info("Silakan unggah file CSV dataset kendaraan terlebih dahulu.")
