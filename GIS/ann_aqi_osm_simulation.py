# -*- coding: utf-8 -*-
"""
ANN-based AQI prediction + simulated 5 IoT devices + OpenStreetMap heatmap
with side info box next to each station
"""

import os
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# -----------------------------
# 1) CONFIG
# -----------------------------
DATA_PATH = "Data.csv"
MODEL_PATH = "ann_aqi_model.joblib"
MAP_PATH = "aqi_heatmap_osm.html"
CSV_SIM_PATH = "simulated_devices_aqi.csv"
RANDOM_SEED = 20

DEVICE_COORDS = [
    {"device_id": "DEV-01", "lat": 32.8872, "lon": 13.1913},
    {"device_id": "DEV-02", "lat": 32.8010, "lon": 13.2100},
    {"device_id": "DEV-03", "lat": 32.8750, "lon": 13.2350},
    {"device_id": "DEV-04", "lat": 32.8600, "lon": 13.1800},
    {"device_id": "DEV-05", "lat": 32.8150, "lon": 13.2600},
    {"device_id": "DEV-05", "lat": 32.5258, "lon": 13.157613},
]

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET = "AQI"


# -----------------------------
# 2) HELPERS
# -----------------------------
def aqi_category(aqi_value: float) -> str:
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def aqi_color(aqi_value: float) -> str:
    if aqi_value <= 50:
        return "#00e400"   # Green
    elif aqi_value <= 100:
        return "#ffff00"   # Yellow
    elif aqi_value <= 150:
        return "#ff7e00"   # Orange
    elif aqi_value <= 200:
        return "#ff0000"   # Red
    elif aqi_value <= 300:
        return "#8f3f97"   # Purple
    return "#7e0023"       # Maroon


def sample_realistic_values(df: pd.DataFrame, random_state: int | None = None) -> dict:
    rng = np.random.default_rng(random_state)
    out = {}
    for col in FEATURES:
        low = float(df[col].quantile(0.10))
        high = float(df[col].quantile(0.90))
        out[col] = round(float(rng.uniform(low, high)), 2)
    return out


def build_ann_pipeline() -> Pipeline:
    ann = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        random_state=RANDOM_SEED,
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ann),
    ])
    return pipe


# -----------------------------
# 3) LOAD DATA
# -----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in dataset: {missing_cols}")

df = df[FEATURES + [TARGET]].copy()
df = df.dropna().reset_index(drop=True)

X = df[FEATURES]
y = df[TARGET]


# -----------------------------
# 4) TRAIN / TEST
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED
)

pipeline = build_ann_pipeline()
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

metrics = {
    "MAE": round(float(mae), 3),
    "RMSE": round(float(rmse), 3),
    "R2": round(float(r2), 3),
    "Train samples": int(len(X_train)),
    "Test samples": int(len(X_test)),
}

joblib.dump(pipeline, MODEL_PATH)


# -----------------------------
# 5) SIMULATE 5 DEVICES
# -----------------------------
rows = []
for i, loc in enumerate(DEVICE_COORDS, start=1):
    sample = sample_realistic_values(df, random_state=RANDOM_SEED + i)
    x_new = pd.DataFrame([sample], columns=FEATURES)
    pred_aqi = float(pipeline.predict(x_new)[0])

    row = {
        "device_id": loc["device_id"],
        "lat": loc["lat"],
        "lon": loc["lon"],
        **sample,
        "predicted_AQI": round(pred_aqi, 2),
        "AQI_category": aqi_category(pred_aqi),
    }
    rows.append(row)

sim_df = pd.DataFrame(rows)
sim_df.to_csv(CSV_SIM_PATH, index=False, encoding="utf-8-sig")


# -----------------------------
# 6) BUILD OPENSTREETMAP HEATMAP
# -----------------------------
center_lat = float(sim_df["lat"].mean())
center_lon = float(sim_df["lon"].mean())

m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

heat_data = sim_df[["lat", "lon", "predicted_AQI"]].values.tolist()
HeatMap(
    heat_data,
    min_opacity=0.35,
    radius=35,
    blur=25,
    max_zoom=18,
).add_to(m)

for _, row in sim_df.iterrows():
    marker_color = aqi_color(row["predicted_AQI"])

    popup_html = f"""
    <div style="font-family:Arial; font-size:13px; width:220px;">
        <b>Device:</b> {row['device_id']}<br>
        <b>AQI:</b> {row['predicted_AQI']}<br>
        <b>Category:</b> {row['AQI_category']}<br><hr>
        <b>PM2.5:</b> {row['PM2.5']}<br>
        <b>PM10:</b> {row['PM10']}<br>
        <b>NO2:</b> {row['NO2']}<br>
        <b>SO2:</b> {row['SO2']}<br>
        <b>CO:</b> {row['CO']}<br>
        <b>O3:</b> {row['O3']}<br>
    </div>
    """

    # دائرة المحطة
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=8,
        color="black",
        weight=1,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.95,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{row['device_id']} | AQI={row['predicted_AQI']}",
    ).add_to(m)

    # خط قصير بين المحطة والبوكس
    line_end_lat = row["lat"] + 0.0020
    line_end_lon = row["lon"] + 0.0030

    folium.PolyLine(
        locations=[
            [row["lat"], row["lon"]],
            [line_end_lat, line_end_lon]
        ],
        color="gray",
        weight=1.5,
        opacity=0.8
    ).add_to(m)

    # بوكس جانبي ثابت
    info_box_html = f"""
    <div style="
        width: 190px;
        background-color: white;
        border: 2px solid {marker_color};
        border-radius: 8px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
        padding: 8px;
        font-size: 11px;
        font-family: Arial;
        line-height: 1.4;
    ">
        <div style="font-weight:bold; color:{marker_color}; font-size:12px; margin-bottom:4px;">
            {row['device_id']}
        </div>
        <b>AQI:</b> {row['predicted_AQI']}<br>
        <b>Cat:</b> {row['AQI_category']}<br>
        <hr style="margin:4px 0;">
        <b>PM2.5:</b> {row['PM2.5']}<br>
        <b>PM10:</b> {row['PM10']}<br>
        <b>NO2:</b> {row['NO2']}<br>
        <b>SO2:</b> {row['SO2']}<br>
        <b>CO:</b> {row['CO']}<br>
        <b>O3:</b> {row['O3']}<br>
    </div>
    """

    folium.Marker(
        location=[line_end_lat, line_end_lon],
        icon=folium.DivIcon(
            html=info_box_html,
            icon_size=(200, 140),
            icon_anchor=(0, 0)
        )
    ).add_to(m)

title_html = """
<h3 align="center" style="font-size:18px">
    <b>Predicted AQI Heatmap for 5 Simulated Devices</b>
</h3>
"""
m.get_root().html.add_child(folium.Element(title_html))
m.save(MAP_PATH)


# -----------------------------
# 7) PRINT SUMMARY
# -----------------------------
print("=== ANN AQI MODEL RESULTS ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

print("\n=== Simulated Device Readings ===")
print(sim_df.to_string(index=False))

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Simulated CSV saved to: {CSV_SIM_PATH}")
print(f"OpenStreetMap heatmap saved to: {MAP_PATH}")