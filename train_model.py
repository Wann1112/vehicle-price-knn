import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv("Australian Vehicle Prices.csv")

def categorize_price(price):
    if price < 15000:
        return "Murah"
    elif price < 30000:
        return "Sedang"
    elif price < 50000:
        return "Mahal"
    else:
        return "Sangat Mahal"

df["Price_Category"] = df["Price"].apply(categorize_price)

X = df.drop(["Price", "Price_Category"], axis=1)
y = df["Price_Category"]

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

label_y = LabelEncoder()
y = label_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

joblib.dump(model, "xgb_model.pkl")
joblib.dump(label_y, "label_encoder.pkl")
joblib.dump(X, "feature_columns.pkl")

print("Model saved!")
