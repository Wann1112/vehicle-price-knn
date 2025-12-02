import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# Page config
st.set_page_config(
    page_title="Australian Vehicle Price Classifier",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .murah { background-color: #dcfce7; border-left: 4px solid #22c55e; }
    .sedang { background-color: #dbeafe; border-left: 4px solid #3b82f6; }
    .mahal { background-color: #fed7aa; border-left: 4px solid #f97316; }
    .sangat-mahal { background-color: #fecaca; border-left: 4px solid #ef4444; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🚗 Australian Vehicle Price Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Menggunakan Gradient Boosting Algorithm untuk Klasifikasi Harga Kendaraan</p>', unsafe_allow_html=True)

# Function to create price categories
def create_price_category(price):
    if price <= 20000:
        return 'Murah'
    elif price <= 40000:
        return 'Sedang'
    elif price <= 60000:
        return 'Mahal'
    else:
        return 'Sangat Mahal'

# Function to load and preprocess data
@st.cache_data
def load_data():
    # This assumes the CSV file is in the same directory
    try:
        df = pd.read_csv('australian_vehicle_prices.csv')
        
        # Create price category
        df['PriceCategory'] = df['Price'].apply(create_price_category)
        
        return df
    except FileNotFoundError:
        return None

# Function to train model
@st.cache_resource
def train_model(df):
    # Select features for training
    feature_columns = ['Brand', 'Year', 'Model', 'Car/Suv', 'Transmission', 
                      'Engine', 'DriveType', 'FuelType', 'FuelConsumption', 
                      'Kilometres', 'CylindersinEngine', 'Doors', 'Seats']
    
    # Handle missing values
    df_clean = df.dropna(subset=feature_columns + ['PriceCategory'])
    
    X = df_clean[feature_columns].copy()
    y = df_clean['PriceCategory']
    
    # Label encoding for categorical variables
    label_encoders = {}
    categorical_cols = ['Brand', 'Model', 'Car/Suv', 'Transmission', 'DriveType', 'FuelType']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Convert numeric columns
    numeric_cols = ['Year', 'Engine', 'FuelConsumption', 'Kilometres', 'CylindersinEngine', 'Doors', 'Seats']
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(X.median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, label_encoders, feature_columns, accuracy, y_test, y_pred

# Sidebar - Show price categories
with st.sidebar:
    st.header("📊 Kategori Harga (AUD)")
    
    st.markdown("""
    <div class="category-box murah">
        <h3>💚 Murah</h3>
        <p><b>Range:</b> $88 - $20,000</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="category-box sedang">
        <h3>💙 Sedang</h3>
        <p><b>Range:</b> $20,090 - $39,999</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="category-box mahal">
        <h3>🧡 Mahal</h3>
        <p><b>Range:</b> $40,110 - $59,999</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="category-box sangat-mahal">
        <h3>❤️ Sangat Mahal</h3>
        <p><b>Range:</b> $60,150 - $1,500,000</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Prediksi", "📈 Model Performance", "📊 Data Exploration"])

# Load data
df = load_data()

if df is None:
    st.error("⚠️ File 'australian_vehicle_prices.csv' tidak ditemukan. Silakan upload dataset terlebih dahulu.")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['PriceCategory'] = df['Price'].apply(create_price_category)
        st.success("✅ Dataset berhasil di-upload!")
        st.rerun()
else:
    # Train model
    with st.spinner('Training model...'):
        model, scaler, label_encoders, feature_columns, accuracy, y_test, y_pred = train_model(df)
    
    # TAB 1: PREDICTION
    with tab1:
        st.header("🎯 Prediksi Harga Kendaraan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.selectbox("Brand", sorted(df['Brand'].unique()))
            year = st.number_input("Year", min_value=1990, max_value=2025, value=2020)
            model = st.text_input("Model", "Corolla")
            car_type = st.selectbox("Car Type", sorted(df['Car/Suv'].unique()))
            transmission = st.selectbox("Transmission", sorted(df['Transmission'].unique()))
            engine = st.number_input("Engine (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
            drive_type = st.selectbox("Drive Type", sorted(df['DriveType'].unique()))
        
        with col2:
            fuel_type = st.selectbox("Fuel Type", sorted(df['FuelType'].unique()))
            fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=1.0, max_value=30.0, value=7.5, step=0.1)
            kilometres = st.number_input("Kilometres", min_value=0, max_value=500000, value=50000, step=1000)
            cylinders = st.number_input("Cylinders", min_value=2, max_value=12, value=4)
            doors = st.number_input("Doors", min_value=2, max_value=5, value=4)
            seats = st.number_input("Seats", min_value=2, max_value=9, value=5)
        
        if st.button("🔮 Prediksi Harga", type="primary", use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame({
                'Brand': [brand],
                'Year': [year],
                'Model': [model],
                'Car/Suv': [car_type],
                'Transmission': [transmission],
                'Engine': [engine],
                'DriveType': [drive_type],
                'FuelType': [fuel_type],
                'FuelConsumption': [fuel_consumption],
                'Kilometres': [kilometres],
                'CylindersinEngine': [cylinders],
                'Doors': [doors],
                'Seats': [seats]
            })
            
            # Encode categorical variables
            for col in ['Brand', 'Model', 'Car/Suv', 'Transmission', 'DriveType', 'FuelType']:
                le = label_encoders[col]
                try:
                    input_data[col] = le.transform(input_data[col].astype(str))
                except:
                    # If new value, use the most common encoded value
                    input_data[col] = 0
            
            # Scale
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Get confidence
            confidence = max(prediction_proba) * 100
            
            # Display result
            st.markdown("---")
            st.subheader("📊 Hasil Prediksi")
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                category_colors = {
                    'Murah': '🟢',
                    'Sedang': '🔵',
                    'Mahal': '🟠',
                    'Sangat Mahal': '🔴'
                }
                st.metric("Kategori Harga", f"{category_colors[prediction]} {prediction}")
            
            with result_col2:
                # Estimate price based on category
                price_ranges = {
                    'Murah': (88, 20000),
                    'Sedang': (20090, 39999),
                    'Mahal': (40110, 59999),
                    'Sangat Mahal': (60150, 150000)
                }
                est_price = sum(price_ranges[prediction]) / 2
                st.metric("Estimasi Harga", f"${est_price:,.0f}")
            
            with result_col3:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show probability distribution
            st.subheader("📈 Distribusi Probabilitas")
            fig_proba = go.Figure(data=[
                go.Bar(
                    x=model.classes_,
                    y=prediction_proba,
                    marker_color=['#22c55e', '#3b82f6', '#f97316', '#ef4444']
                )
            ])
            fig_proba.update_layout(
                xaxis_title="Kategori",
                yaxis_title="Probabilitas",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_proba, use_container_width=True)
    
    # TAB 2: MODEL PERFORMANCE
    with tab2:
        st.header("📈 Performance Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Akurasi Model", f"{accuracy*100:.2f}%")
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))
        
        with col2:
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=model.classes_,
                y=model.classes_,
                color_continuous_scale="Blues",
                text_auto=True
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # Feature Importance
        st.subheader("🎯 Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(height=500)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # TAB 3: DATA EXPLORATION
    with tab3:
        st.header("📊 Explorasi Data")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data", f"{len(df):,}")
        col2.metric("Brands", f"{df['Brand'].nunique()}")
        col3.metric("Models", f"{df['Model'].nunique()}")
        col4.metric("Price Range", f"${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}")
        
        # Distribution by category
        st.subheader("Distribusi Kategori Harga")
        category_dist = df['PriceCategory'].value_counts()
        fig_dist = px.pie(
            values=category_dist.values,
            names=category_dist.index,
            color=category_dist.index,
            color_discrete_map={
                'Murah': '#22c55e',
                'Sedang': '#3b82f6',
                'Mahal': '#f97316',
                'Sangat Mahal': '#ef4444'
            }
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Price distribution
        st.subheader("Distribusi Harga")
        fig_price = px.histogram(df, x='Price', nbins=50, color='PriceCategory',
                                color_discrete_map={
                                    'Murah': '#22c55e',
                                    'Sedang': '#3b82f6',
                                    'Mahal': '#f97316',
                                    'Sangat Mahal': '#ef4444'
                                })
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Top brands
        st.subheader("Top 10 Brands")
        top_brands = df['Brand'].value_counts().head(10)
        fig_brands = px.bar(top_brands, orientation='h')
        st.plotly_chart(fig_brands, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280;'>
    <p>🚗 Australian Vehicle Price Classifier | Powered by Gradient Boosting</p>
    <p>Dataset: <a href='https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices' target='_blank'>Kaggle - Australian Vehicle Prices</a></p>
</div>
""", unsafe_allow_html=True)
