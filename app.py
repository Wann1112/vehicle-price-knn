import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- Fungsi untuk klasifikasi harga ---
def price_category(price):
    if 0 <= price <= 20000:
        return "Murah"
    elif 20000 < price <= 40000:
        return "Sedang"
    elif 40000 < price <= 60000:
        return "Mahal"
    else:
        return "Sangat Mahal"

# --- Streamlit UI ---
st.title("🚗 Vehicle Price Classification (AUD)")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Data")
    st.write(df.head())

    # Pastikan ada kolom harga
    if "Price" not in df.columns:
        st.error("Dataset harus memiliki kolom 'Price'")
    else:
        # Tambahkan kategori harga
        df["Category"] = df["Price"].apply(price_category)

        st.subheader("Data dengan Kategori Harga")
        st.write(df[["Price", "Category"]].head(20))

        # --- Preprocessing ---
        # Encode kategori harga
        le = LabelEncoder()
        df["CategoryEncoded"] = le.fit_transform(df["Category"])

        # Fitur sederhana: hanya harga
        X = df[["Price"]]
        y = df["CategoryEncoded"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Model KNN ---
        k = st.slider("Pilih jumlah tetangga (k)", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("📊 Evaluasi Model")
        st.write(f"Akurasi: {acc:.2f}")
        st.text(classification_report(y_test, y_pred, target_names=le.classes_))

        # --- Prediksi harga baru ---
        st.subheader("🔮 Prediksi Harga Baru")
        new_price = st.number_input("Masukkan harga kendaraan (AUD)", min_value=0, step=1000)
        if new_price > 0:
            pred_class = model.predict([[new_price]])[0]
            st.success(f"Harga {new_price} AUD diklasifikasikan sebagai: **{le.inverse_transform([pred_class])[0]}**")
