import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Judul aplikasi
st.title("Klasifikasi Kategori Harga Kendaraan - KNN")

# Upload file CSV
uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview Data:", df.head())

    # --- Preprocessing ---
    st.subheader("Preprocessing Data")
    st.write("Jumlah Data:", df.shape)

    # Misalnya kolom target adalah 'PriceCategory'
    if 'PriceCategory' in df.columns:
        X = df.drop('PriceCategory', axis=1)
        y = df['PriceCategory']

        # Encode target jika kategori
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Handle data numerik
        X = pd.get_dummies(X)  # one-hot encoding untuk kolom kategorikal
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # --- Model KNN ---
        st.subheader("Training Model KNN")
        k = st.slider("Pilih jumlah tetangga (k)", 1, 20, 5)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Prediksi
        y_pred = knn.predict(X_test)

        # Evaluasi
        acc = accuracy_score(y_test, y_pred)
        st.write(f"🎯 Akurasi Model: {acc:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=le.classes_))

        # --- Prediksi Input Baru ---
        st.subheader("Prediksi Data Baru")
        st.write("Masukkan fitur sesuai dataset:")
        input_data = {}
        for col in df.drop('PriceCategory', axis=1).columns:
            val = st.text_input(f"{col}")
            input_data[col] = val

        if st.button("Prediksi"):
            try:
                input_df = pd.DataFrame([input_data])
                input_df = pd.get_dummies(input_df)
                input_df = input_df.reindex(columns=X.columns, fill_value=0)
                input_scaled = scaler.transform(input_df)
                pred = knn.predict(input_scaled)
                st.success(f"Kategori Harga Prediksi: {le.inverse_transform(pred)[0]}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Kolom 'PriceCategory' tidak ditemukan. Pastikan dataset memiliki target kategori harga.")
