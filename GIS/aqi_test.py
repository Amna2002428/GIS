import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

model = load_model("aqi_ann_model.keras")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

df = pd.read_csv("Data.csv")

samples = df[feature_names].iloc[0:10].copy()

for col in samples.columns:
    col_min = df[col].min()
    col_max = df[col].max()
    if col_min >= 0 and col_max > 100:
        samples[col] = np.log1p(samples[col])

X_scaled = x_scaler.transform(samples)
y_pred_scaled = model.predict(X_scaled, verbose=0)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

print("Predictions:")
print(y_pred.ravel())

print("\nActual AQI:")
print(df["AQI"].iloc[0:10].values)