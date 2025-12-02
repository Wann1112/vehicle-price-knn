"""
Script untuk training model secara terpisah dan menyimpannya
Opsional: Bisa digunakan untuk pre-train model sebelum deploy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def create_price_category(price):
    """Membuat kategori harga berdasarkan nilai"""
    if price <= 20000:
        return 'Murah'
    elif price <= 40000:
        return 'Sedang'
    elif price <= 60000:
        return 'Mahal'
    else:
        return 'Sangat Mahal'

def load_and_preprocess_data(file_path):
    """Load dan preprocess dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Buat kategori harga
    df['PriceCategory'] = df['Price'].apply(create_price_category)
    
    print(f"Total data: {len(df)}")
    print(f"Distribusi kategori:\n{df['PriceCategory'].value_counts()}\n")
    
    return df

def prepare_features(df):
    """Persiapkan features untuk training"""
    print("Preparing features...")
    
    # Pilih kolom features
    feature_columns = ['Brand', 'Year', 'Model', 'Car/Suv', 'Transmission', 
                      'Engine', 'DriveType', 'FuelType', 'FuelConsumption', 
                      'Kilometres', 'CylindersinEngine', 'Doors', 'Seats']
    
    # Drop missing values
    df_clean = df.dropna(subset=feature_columns + ['PriceCategory'])
    print(f"Data setelah cleaning: {len(df_clean)}")
    
    X = df_clean[feature_columns].copy()
    y = df_clean['PriceCategory']
    
    # Label encoding untuk categorical variables
    label_encoders = {}
    categorical_cols = ['Brand', 'Model', 'Car/Suv', 'Transmission', 'DriveType', 'FuelType']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} unique values")
    
    # Convert numeric columns
    numeric_cols = ['Year', 'Engine', 'FuelConsumption', 'Kilometres', 
                   'CylindersinEngine', 'Doors', 'Seats']
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing numeric values dengan median
    X = X.fillna(X.median())
    
    return X, y, label_encoders, feature_columns

def train_gradient_boosting(X, y):
    """Train Gradient Boosting Classifier"""
    print("\nTraining Gradient Boosting Classifier...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Training accuracy
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"\nTraining Accuracy: {train_accuracy*100:.2f}%")
    
    # Test accuracy
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print(cm)
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = sorted(
        zip(X.columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (feature, importance) in enumerate(feature_importance[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    return model, scaler, X_test, y_test, y_pred

def save_model(model, scaler, label_encoders, feature_columns, filename='vehicle_model.pkl'):
    """Simpan model dan preprocessors"""
    print(f"\nSaving model to {filename}...")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully!")

def main():
    """Main function"""
    print("="*50)
    print("AUSTRALIAN VEHICLE PRICE CLASSIFIER")
    print("Training dengan Gradient Boosting")
    print("="*50 + "\n")
    
    # Load data
    df = load_and_preprocess_data('australian_vehicle_prices.csv')
    
    # Prepare features
    X, y, label_encoders, feature_columns = prepare_features(df)
    
    # Train model
    model, scaler, X_test, y_test, y_pred = train_gradient_boosting(X, y)
    
    # Save model (opsional)
    save_choice = input("\nSimpan model? (y/n): ")
    if save_choice.lower() == 'y':
        save_model(model, scaler, label_encoders, feature_columns)
    
    print("\n" + "="*50)
    print("TRAINING SELESAI!")
    print("="*50)

if __name__ == "__main__":
    main()
