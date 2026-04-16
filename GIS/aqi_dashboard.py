# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, json
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

DATA_PATH    = "Data.csv"
MODEL_PATH   = "ann_aqi_model.joblib"
MAP_PATH     = "index.html"
CSV_SIM_PATH = "simulated_devices_aqi.csv"
RANDOM_SEED  = 60

# Check for existence of essential files to prevent deployment crashes
for path in [DATA_PATH, MODEL_PATH]:
    if not os.path.exists(path):
        print(f"⚠️ Warning: Missing {path}. Site may not generate correctly on server.")

DEVICE_COORDS = [
    {"device_id": "DEV-01", "name": "طرابلس - تاجوراء  ",   "lat": 32.858581, "lon":  13.379024},
    {"device_id": "DEV-02", "name": "طرابلس - وادي الربيع",   "lat": 32.8030, "lon": 13.3100},
    {"device_id": "DEV-03", "name": "طرابلس - سيدي السايح", "lat": 32.6950, "lon": 13.1850},
    {"device_id": "DEV-04", "name": "طرابلس - طريق المطار ",   "lat": 32.7795, "lon": 13.1500},
    {"device_id": "DEV-05", "name": "طرابلس - جنزور",   "lat": 32.834500, "lon": 13.075556},
    {"device_id": "DEV-06", "name": "طرابلس - الكريمية",   "lat": 32.733701, "lon": 13.071811},
    {"device_id": "DEV-07", "name": "طرابلس - حي الأندلس",   "lat": 32.874777, "lon": 13.121812},

]

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET   = "AQI"

def aqi_category(v):
    if v <= 50:  return "Good"
    if v <= 100: return "Moderate"
    if v <= 150: return "Unhealthy for Sensitive Groups"
    if v <= 200: return "Unhealthy"
    if v <= 300: return "Very Unhealthy"
    return "Hazardous"

def aqi_color(v):
    if v <= 50:  return "#8cc63f"
    if v <= 100: return "#fdb913"
    if v <= 150: return "#f37021"
    if v <= 200: return "#ed1c24"
    if v <= 300: return "#9e005d"
    return "#790000"

def sample_realistic_values(df, random_state=None):
    rng = np.random.default_rng(random_state)
    return {col: round(float(rng.uniform(float(df[col].quantile(.1)), float(df[col].quantile(.9)))), 2) for col in FEATURES}

def build_ann():
    return Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(
        hidden_layer_sizes=(64,32), activation="relu", solver="adam",
        alpha=0.0005, learning_rate="adaptive", learning_rate_init=0.001,
        max_iter=2000, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=40, random_state=RANDOM_SEED))])

# Load & train
if not os.path.exists(DATA_PATH): raise FileNotFoundError(DATA_PATH)
df = pd.read_csv(DATA_PATH)
df = df[FEATURES + [TARGET]].copy().dropna().reset_index(drop=True)
X, y = df[FEATURES], df[TARGET]
# Load trained model (NO TRAINING HERE)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found! Please train it once first.")

pipe = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")

# Simulate devices
rows = []
for i, loc in enumerate(DEVICE_COORDS, start=1):
    s = sample_realistic_values(df, RANDOM_SEED + i)

    # نطاقات مؤشر جودة الهواء المستهدفة لكل منطقة (حسب طلب المستخدم)
    target_aqi_ranges = {
        "DEV-01": (80, 140),   # تاجوراء
        "DEV-02": (90, 150),   # وادي الربيع
        "DEV-03": (100, 170),  # سيدي السايح
        "DEV-04": (70, 120),   # طريق المطار
        "DEV-05": (80, 140),   # جنزور
        "DEV-06": (90, 160),   # الكريمية
        "DEV-07": (60, 110)    # حي الاندلس
    }

    if loc["device_id"] in target_aqi_ranges:
        min_aqi, max_aqi = target_aqi_ranges[loc["device_id"]]
        # خوارزمية بحث ثنائي لايجاد قيم المدخلات التي تنتج مؤشر جودة ضمن النطاق المطلوب
        low, high = 0.01, 10.0
        best_s = s.copy()
        best_p = 0
        for _ in range(20):
            mid = (low + high) / 2
            test_s = {k: round(v * mid, 2) for k, v in s.items()}
            p = float(pipe.predict(pd.DataFrame([test_s], columns=FEATURES))[0])
            best_s = test_s
            best_p = p
            if min_aqi <= p <= max_aqi:
                break
            elif p > max_aqi:
                high = mid
            else:
                low = mid
        s = best_s
        p = round(best_p, 2)
    else:
        # التنبؤ الحقيقي بالنموذج للمحطات الاخرى ان وجدت
        p = round(float(pipe.predict(pd.DataFrame([s], columns=FEATURES))[0]), 2)

    rows.append({
        "device_id": loc["device_id"],
        "name": loc["name"],
        "lat": loc["lat"],
        "lon": loc["lon"],
        **s,
        "predicted_AQI": p,
        "AQI_category": aqi_category(p)
    })

sim_df = pd.DataFrame(rows)
sim_df.to_csv(CSV_SIM_PATH, index=False, encoding="utf-8-sig")

center_lat = float(sim_df["lat"].mean())
center_lon = float(sim_df["lon"].mean())
devices_json = json.dumps(rows, ensure_ascii=False)

# ═══════════════════════════════════════════════════════
# BUILD COMPLETE STANDALONE HTML  (no Folium injection)
# ═══════════════════════════════════════════════════════
html = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,shrink-to-fit=no"/>
<title data-i18n="site_title">قياس جودة الهواء</title>

<!-- Leaflet -->
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<!-- Leaflet HeatMap plugin -->
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<!-- Google Font -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Kufi+Arabic:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<!-- QR Code Library -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>

<style>
/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --nav-height: 58px;
  --panel-width: 270px;
  --sidebar-width: 385px;
  --transition-speed: 0.35s;
}
html, body {
  height: 100%; width: 100%;
  font-family: 'Noto Kufi Arabic', sans-serif;
  background: #0f1623;
  overflow: hidden;
  transition: direction var(--transition-speed);
}

/* ─────────── NAVBAR ─────────── */
#navbar {
  position: fixed; top: 0; right: 0; left: 0; height: var(--nav-height);
  background: #1b2232;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 15px;
  z-index: 9999;
  box-shadow: 0 2px 12px rgba(0,0,0,.6);
}
@media (max-width: 1024px) {
  #navbar { padding: 0 10px; }
  .nav-logo { font-size: 14px; gap: 6px; }
  .nav-logo-icon { width: 30px; height: 30px; font-size: 12px; }
  .nav-links { 
    position: fixed; bottom: 0; left: 0; right: 0; 
    background: inherit; height: 50px; 
    border-top: 1px solid rgba(255,255,255,0.1);
    justify-content: space-around;
    padding: 0 5px;
  }
  .nav-link { font-size: 11px; padding: 8px 5px; }
  body.light-theme .nav-links { border-top-color: rgba(0,0,0,0.1); }
}
.nav-logo {
  display: flex; align-items: center; gap: 10px;
  color: #fff; font-size: 16px; font-weight: 700;
}
.nav-logo-icon {
  width: 34px; height: 34px; border-radius: 50%;
  background: linear-gradient(135deg, #4ade80, #22d3ee);
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; font-weight: 800; color: #fff;
}
.nav-links { 
  display: flex; gap: 2px;
  overflow-x: auto; -webkit-overflow-scrolling: touch;
  scrollbar-width: none;
}
.nav-links::-webkit-scrollbar { display: none; }
.nav-link {
  color: #9aa5be; font-size: 13px; padding: 6px 14px;
  border-radius: 6px; cursor: pointer; transition: all .2s; white-space: nowrap;
}
.nav-link:hover { color: #fff; background: rgba(255,255,255,.08); }
.nav-link.active { color: #fff; background: rgba(255,255,255,.13); font-weight: 600; }
.nav-right { display: flex; gap: 8px; }
.nav-btn {
  background: rgba(255,255,255,.08); border: none; color: #9aa5be;
  padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px;
  font-family: 'Noto Kufi Arabic', sans-serif; transition: all .2s;
}
.nav-btn:hover { background: rgba(255,255,255,.15); color: #fff; }

/* ─────────── RIGHT DEVICE PANEL ─────────── */
#device-panel {
  position: fixed; top: var(--nav-height); right: 0;
  width: var(--panel-width); height: calc(100vh - var(--nav-height));
  background: #1b2232;
  z-index: 8000;
  overflow-y: auto;
  padding: 14px 8px;
  box-shadow: -3px 0 14px rgba(0,0,0,.6);
  transition: right var(--transition-speed) ease;
}
@media (max-width: 1024px) {
  #device-panel {
    right: calc(-1 * var(--panel-width));
    width: 85%;
    max-width: 300px;
  }
  #device-panel.open { right: 0 !important; }
}
#device-panel::-webkit-scrollbar { width: 3px; }
#device-panel::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
.panel-heading {
  color: #5c6b8a; font-size: 10px; font-weight: 700; letter-spacing: 1.2px;
  padding: 4px 8px 10px; border-bottom: 1px solid #242e42; margin-bottom: 8px;
}
.device-card {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 8px; border-radius: 8px; cursor: pointer;
  transition: all .2s; margin-bottom: 3px; border: 1px solid transparent;
}
.device-card:hover { background: rgba(255,255,255,.05); border-color: rgba(255,255,255,.09); }
.device-card.selected { background: rgba(255,255,255,.09); border-color: rgba(255,255,255,.16); }
.dev-circle {
  width: 42px; height: 42px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700;
  box-shadow: 0 2px 8px rgba(0,0,0,.4);
}
.dev-body { flex: 1; min-width: 0; }
.dev-title {
  color: #dde3f0; font-size: 12px; font-weight: 600;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.dev-sub { color: #5c6b8a; font-size: 10px; margin-top: 2px; }
.dev-badge {
  display: inline-block; font-size: 10px; padding: 2px 8px;
  border-radius: 10px; margin-top: 4px; font-weight: 600;
}

/* ─────────── LEFT DETAIL SIDEBAR ─────────── */
#detail-sidebar {
  position: fixed; top: var(--nav-height); left: calc(-1.1 * var(--sidebar-width));
  width: var(--sidebar-width); height: calc(100vh - var(--nav-height));
  background: #f2f4f8;
  z-index: 8500;
  overflow-y: auto;
  transition: left var(--transition-speed) cubic-bezier(.4,0,.2,1);
  box-shadow: 5px 0 24px rgba(0,0,0,.35);
}
@media (max-width: 1024px) {
  #detail-sidebar {
    width: 100%;
    left: -100%;
    height: calc(100vh - var(--nav-height) - 50px); /* Adjust for mobile nav */
  }
}
#detail-sidebar.open { left: 0 !important; }

/* Sidebar cards */
.sb-card {
  background: #fff; border-radius: 10px;
  margin: 12px; padding: 16px 18px;
  box-shadow: 0 1px 5px rgba(0,0,0,.07);
}
.sb-top-row {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 14px;
}
.sb-dev-num {
  font-size: 11px; color: #555; background: #f3f4f6;
  border: 1px solid #e2e5eb; padding: 4px 12px; border-radius: 6px; font-weight: 500;
}
.sb-close-btn {
  background: none; border: none; font-size: 24px; color: #aab0bc;
  cursor: pointer; line-height: 1; padding: 0 2px; transition: color .2s;
}
.sb-close-btn:hover { color: #111; }
.sb-city-row {
  display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;
}
.sb-city-name { font-size: 20px; font-weight: 700; color: #1a2236; }
.sb-status-pill {
  font-size: 12px; font-weight: 700; padding: 5px 14px; border-radius: 6px;
}
.sb-meta-row {
  display: flex; justify-content: space-between;
  font-size: 11px; color: #9ca3b0; margin-top: 4px;
}
.sb-aqi-display { text-align: center; margin: 24px 0 10px; }
.sb-aqi-value {
  font-size: 92px; font-weight: 300; line-height: 1;
  color: #1a2236; letter-spacing: -4px;
}
.sb-aqi-caption { font-size: 13px; color: #9ca3b0; margin-top: 8px; }

/* Scale bar */
.scale-container { position: relative; margin: 28px 0 20px; padding: 0 4px; }
.scale-track {
  height: 10px; border-radius: 6px;
  background: linear-gradient(to left,
    #8cc63f 0%, #fdb913 30%, #f37021 48%, #ed1c24 64%, #9e005d 82%, #790000 100%);
}
.scale-pointer { position: absolute; top: -18px; font-size: 14px; }
.scale-ticks {
  display: flex; justify-content: space-between;
  font-size: 10px; color: #9ca3b0; margin-top: 5px;
}

/* Info description box */
.info-box {
  background: #f9fafb; border: 1px dashed #d1d5db; border-radius: 8px;
  padding: 13px; margin: 0 12px 10px;
  display: flex; gap: 10px; align-items: flex-start;
}
.info-icon { font-size: 20px; margin-top: 1px; flex-shrink: 0; }
.info-heading { font-size: 13px; font-weight: 700; color: #1a2236; margin-bottom: 4px; }
.info-body { font-size: 12px; color: #4b5563; line-height: 1.7; }

/* Metric rows */
.metrics-list { padding: 0 12px 20px; display: flex; flex-direction: column; gap: 4px; }
.metric-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 14px; background: #fff; border-radius: 8px;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.metric-left { display: flex; align-items: center; gap: 10px; }
.metric-accent { width: 4px; height: 22px; border-radius: 3px; }
.metric-key { font-size: 13px; font-weight: 600; color: #374151; font-family: monospace; }
.metric-unit { font-size: 10px; color: #9ca3b0; margin-top: 1px; }
.metric-value {
  font-size: 13px; font-weight: 700; padding: 5px 14px;
  border-radius: 6px; min-width: 56px; text-align: center;
}

/* ─────────── MAP CONTAINER ─────────── */
#map {
  position: fixed;
  top: var(--nav-height); bottom: 0; left: 0; right: var(--panel-width);
  z-index: 100;
  transition: right var(--transition-speed) ease;
}
@media (max-width: 1024px) {
  #map { right: 0; bottom: 50px; }
}

/* Panel Toggle Button (Mobile Only) */
#panel-toggle {
  display: none;
  position: fixed; bottom: 70px; right: 20px;
  width: 48px; height: 48px; border-radius: 50%;
  background: #22d3ee; color: #fff;
  border: none; cursor: pointer; z-index: 9000;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  align-items: center; justify-content: center; font-size: 20px;
}
@media (max-width: 1024px) {
  #panel-toggle { display: flex; }
}

/* ─────────── LEGEND ─────────── */
#legend {
  position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
  margin-left: calc(-1 * var(--panel-width) / 2);
  background: rgba(27,34,50,.92); backdrop-filter: blur(8px);
  border-radius: 10px; padding: 10px 18px; z-index: 8000;
  display: flex; gap: 14px; align-items: center;
  box-shadow: 0 4px 16px rgba(0,0,0,.5);
  transition: margin-left var(--transition-speed) ease;
}
@media (max-width: 1024px) {
  #legend { 
    bottom: 60px; left: 5px; right: 5px; transform: none; 
    margin-left: 0; flex-wrap: wrap; justify-content: center; gap: 8px;
    padding: 8px; font-size: 9px;
  }
}
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }
.legend-label { font-size: 10px; color: #9aa5be; display: flex; align-items: center; gap: 5px; }
/* ─────────── LIGHT THEME ─────────── */
body.light-theme { background: #f2f4f8; }
body.light-theme #navbar { background: #ffffff; box-shadow: 0 2px 12px rgba(0,0,0,.08); }
body.light-theme .nav-link { color: #4b5563; }
body.light-theme .nav-link:hover { background: #f3f4f6; color: #111827; }
body.light-theme .nav-link.active { background: #e5e7eb; color: #111827; }
body.light-theme .nav-logo { color: #111827; }
body.light-theme .nav-btn { background: #f3f4f6; color: #4b5563; }
body.light-theme .nav-btn:hover { background: #e5e7eb; color: #111827; }
body.light-theme #device-panel { background: #ffffff; box-shadow: -3px 0 14px rgba(0,0,0,.06); }
body.light-theme .panel-heading { color: #6b7280; border-color: #e5e7eb; }
body.light-theme .device-card { border-color: #f3f4f6; }
body.light-theme .device-card:hover { background: #f9fafb; border-color: #e5e7eb; }
body.light-theme .device-card.selected { background: #f3f4f6; border-color: #d1d5db; }
body.light-theme .dev-title { color: #111827; }
body.light-theme #legend { background: rgba(255,255,255,.92); box-shadow: 0 4px 16px rgba(0,0,0,.1); }
body.light-theme .legend-label { color: #4b5563; }
/* ─────────── TAB VIEWS ─────────── */
.tab-view {
  position: fixed; top: var(--nav-height); left: 0; right: 0; bottom: 0;
  background: inherit; z-index: 50; overflow-y: auto; padding: 25px;
}
@media (max-width: 1024px) {
  .tab-view { padding: 15px; bottom: 50px; }
}
.hidden { display: none !important; }
.view-header { font-size: 22px; font-weight: 700; color: #fff; margin-bottom: 25px; }
body.light-theme .view-header { color: #111; }

/* Sensors Grid */
.sensors-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
.sensor-card { background: #1b2232; border-radius: 8px; padding: 16px; border-right: 4px solid #8cc63f; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
body.light-theme .sensor-card { background: #fff; border-color: #8cc63f; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.s-card-title { font-weight: 700; color: #fff; margin-bottom: 10px; }
body.light-theme .s-card-title { color: #111; }
.sensor-row { display: flex; justify-content: space-between; margin-top: 8px; font-size: 13px; color: #9aa5be; border-bottom: 1px dashed rgba(255,255,255,0.05); padding-bottom: 4px; }
body.light-theme .sensor-row { color: #444; border-bottom-color: rgba(0,0,0,0.05); }

/* Readings Table */
.table-wrapper { width: 100%; overflow-x: auto; }
.readings-table { width: 100%; border-collapse: collapse; background: #1b2232; border-radius: 8px; overflow: hidden; color: #fff; box-shadow: 0 4px 6px rgba(0,0,0,0.2); min-width: 600px; }
body.light-theme .readings-table { background: #fff; color: #111; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.readings-table th, .readings-table td { padding: 14px 18px; text-align: right; border-bottom: 1px solid #2a3441; font-size: 14px; }
body.light-theme .readings-table th, body.light-theme .readings-table td { border-bottom-color: #eee; }
.readings-table th { background: #242e42; font-size: 13px; color: #9aa5be; }
body.light-theme .readings-table th { background: #f9fafb; color: #6b7280; }

/* Alerts */
.alerts-container { display: flex; flex-direction: column; gap: 12px; max-width: 800px; }
.alert-item { background: #1b2232; border-right: 4px solid #f37021; padding: 16px 20px; border-radius: 6px; display: flex; gap: 15px; color: #fff; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
body.light-theme .alert-item { background: #fff; color: #111; border-color: #f37021; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.alert-time { font-size: 13px; color: #9aa5be; width: 70px; font-weight: 600; }
body.light-theme .alert-time { color: #888; }
.alert-msg { font-size: 14px; line-height: 1.5; }

/* ── QR Modal ── */
#qr-modal {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(0,0,0,0.85); backdrop-filter: blur(5px);
  z-index: 10000; display: none; align-items: center; justify-content: center;
}
#qr-modal.open { display: flex; }
.qr-container {
  background: #1b2232; padding: 30px; border-radius: 12px;
  text-align: center; max-width: 320px; width: 90%;
  box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
body.light-theme .qr-container { background: #fff; }
.qr-title { color: #fff; margin-bottom: 20px; font-weight: 700; font-size: 18px; }
body.light-theme .qr-title { color: #111; }
#qrcode { 
  background: #fff; padding: 15px; border-radius: 8px; 
  display: inline-block; margin-bottom: 20px;
}
.qr-desc { color: #9aa5be; font-size: 13px; line-height: 1.6; }
body.light-theme .qr-desc { color: #555; }
.qr-close {
  margin-top: 20px; background: #22d3ee; color: #fff;
  border: none; padding: 10px 25px; border-radius: 6px;
  cursor: pointer; font-weight: 600; font-family: inherit;
}
</style>
</head>
<body class="light-theme">

<!-- NAVBAR -->
<nav id="navbar">
  <div class="nav-links">
    <div class="nav-link active" id="nav-main" onclick="switchTab('main')" data-i18n="nav_main">الرئيسية (الخريطة)</div>
    <div class="nav-link" id="nav-sensors" onclick="switchTab('sensors')" data-i18n="nav_sensors">أجهزة الاستشعار</div>
    <div class="nav-link" id="nav-readings" onclick="switchTab('readings')" data-i18n="nav_readings">القراءات</div>
    <div class="nav-link" id="nav-alerts" onclick="switchTab('alerts')" data-i18n="nav_alerts">التنبيهات</div>
  </div>
  <div class="nav-logo">
    <div class="nav-logo-icon">AQ</div>
    <span data-i18n="logo_text">نظام مراقبة جودة الهواء اللحظي</span>
  </div>
  <div class="nav-right">
    <button class="nav-btn" id="lang-btn" onclick="toggleLanguage()">English</button>
    <button class="nav-btn" id="share-btn" onclick="toggleQR()">🔗 <span data-i18n="share">مشاركة</span></button>
    <button class="nav-btn" id="theme-btn">🌙 <span data-i18n="dark">داكن</span></button>
  </div>
</nav>

<!-- QR MODAL -->
<div id="qr-modal">
  <div class="qr-container">
    <div class="qr-title" data-i18n="qr_title">مسح لفتح الموقع على الهاتف</div>
    <div id="qrcode"></div>
    <div class="qr-desc" data-i18n="qr_desc">امسح الكود أعلاه لمشاركة لوحة البيانات أو لفتحها بسرعة على جهازك المحمول.</div>
    <button class="qr-close" onclick="toggleQR()" data-i18n="close">إغلاق</button>
  </div>
</div>

<!-- RIGHT DEVICE PANEL -->
<aside id="device-panel">
  <div class="panel-heading" data-i18n="monitors">محطات المراقبة</div>
  <div id="device-list"></div>
</aside>

<!-- LEFT DETAIL SIDEBAR -->
<aside id="detail-sidebar">
  <div id="sidebar-body"></div>
</aside>

<!-- MAP -->
<div id="map"></div>

<!-- Mobile Toggle Panel Button -->
<button id="panel-toggle" title="قائمة المحطات">☰</button>

<!-- LEGEND -->
<div id="legend">
  <div class="legend-label"><div class="legend-dot" style="background:#8cc63f"></div> <span data-i18n="cat_good">جيد</span> (0–50)</div>
  <div class="legend-label"><div class="legend-dot" style="background:#fdb913"></div> <span data-i18n="cat_mod">معتدل</span> (51–100)</div>
  <div class="legend-label"><div class="legend-dot" style="background:#f37021"></div> <span data-i18n="cat_sens">حساس</span> (101–150)</div>
  <div class="legend-label"><div class="legend-dot" style="background:#ed1c24"></div> <span data-i18n="cat_unh">غير صحي</span> (151–200)</div>
  <div class="legend-label"><div class="legend-dot" style="background:#9e005d"></div> <span data-i18n="cat_vun">خطير</span> (201–300)</div>
  <div class="legend-label"><div class="legend-dot" style="background:#790000"></div> <span data-i18n="cat_haz">طارئ</span> (300+)</div>
</div>

<!-- TAB VIEWS (NOT MAP) -->
<div id="view-sensors" class="tab-view hidden">
  <div class="view-header" data-i18n="sensor_status">حالة الأجهزة والمستشعرات</div>
  <div class="sensors-grid" id="sensors-content"></div>
</div>

<div id="view-readings" class="tab-view hidden">
  <div class="view-header" data-i18n="live_readings">سجل القراءات الحية</div>
  <div class="table-wrapper">
    <table class="readings-table">
       <thead>
          <tr><th data-i18n="table_station">المحطة</th><th>AQI</th><th>PM2.5</th><th>PM10</th><th>NO2</th><th>SO2</th><th>CO</th><th>O3</th></tr>
       </thead>
       <tbody id="readings-tbody"></tbody>
    </table>
  </div>
</div>

<div id="view-alerts" class="tab-view hidden">
  <div class="view-header" data-i18n="system_alerts">التنبيهات النظامية</div>
  <div class="alerts-container" id="alerts-content"></div>
</div>

<script>
// ── Device data from Python ──
var DEVICES = {devices_json};
var CENTER  = [{center_lat}, {center_lon}];

// ── I18n Data ──
var currentLang = 'ar';
const I18N = {
  ar: {
    site_title: "قياس جودة الهواء",
    nav_main: "الرئيسية (الخريطة)",
    nav_sensors: "أجهزة الاستشعار",
    nav_readings: "القراءات",
    nav_alerts: "التنبيهات",
    logo_text: "نظام مراقبة جودة الهواء اللحظي",
    share: "مشاركة",
    dark: "داكن",
    light: "مضيء",
    qr_title: "مسح لفتح الموقع على الهاتف",
    qr_desc: "امسح الكود أعلاه لمشاركة لوحة البيانات أو لفتحها بسرعة على جهازك المحمول.",
    close: "إغلاق",
    monitors: "محطات المراقبة",
    cat_good: "جيد",
    cat_mod: "معتدل",
    cat_sens: "حساس",
    cat_unh: "غير صحي",
    cat_vun: "خطير جداً",
    cat_haz: "طارئ",
    sensor_status: "حالة الأجهزة والمستشعرات",
    live_readings: "سجل القراءات الحية",
    table_station: "المحطة",
    system_alerts: "التنبيهات النظامية",
    device_num: "رقم الجهاز",
    aqi_label: "مؤشر جودة الهواء",
    general_public: "عموم الجمهور",
    sensitive_groups: "الفئات الحساسة",
    online: "متصل 🟢",
    offline: "غير متصل 🔴",
    battery: "مستوى البطارية المستشعر",
    last_read: "وقت آخر قراءة",
    alert_warning: "⚠️ تحذير: تم تسجيل جودة هواء",
    alert_station: "في محطة",
    alert_index: "بمؤشر",
    all_clear: "جميع المحطات تعمل بشكل ممتاز والمؤشرات ضمن المعدلات الطبيعية.",
    station_label: "المحطة",
    device_label: "الجهاز",
    rec_pm25: "😷 ينصح بارتداء كمامة N95 لارتفاع الجزيئات الدقيقة.",
    rec_o3: "☀️ تجنب الأنشطة الخارجية الشاقة حالياً لارتفاع الأوزون.",
    rec_co: "🏠 يرجى الحرص على التهوية الجيدة (ارتفاع CO).",
    rec_so2: "🌬️ تجنب الخروج إذا كنت تعاني من الربو (ارتفاع SO2).",
    rec_no2: "🚗 ابتعد عن الطرق المزدحمة لارتفاع أكاسيد النيتروجين.",
    advice: {
      'Good': {
         gen: "✅ الهواء جيد، استمتع بالأنشطة الخارجية.",
         sens: "✅ جودة هواء مثالية لجميع الفئات."
      },
      'Moderate': {
         gen: "🚶 جودة الهواء مقبولة للجميع.",
         sens: "⚠️ راقب أعراضك إذا كنت تعاني من حساسية مفرطة."
      },
      'Unhealthy for Sensitive Groups': {
         gen: "🚶 قلل من المجهود البدني الطويل في الخارج.",
         sens: "😷 قلل المجهود الشاق، وابقَ قريباً من مكان مغلق."
      },
      'Unhealthy': {
         gen: "🏠 حد من الوقت المقضي في الخارج، وأغلق النوافذ.",
         sens: "🚫 تجنب الأنشطة الخارجية تماماً، وقم بتنقية الهواء داخلياً."
      },
      'Very Unhealthy': {
         gen: "🚫 تجنب النشاط البدني الخارجي كلياً.",
         sens: "🚨 ابقَ في الداخل، وقلل المجهود البدني لأدنى مستوى."
      },
      'Hazardous': {
         gen: "🚨 طوارئ بيئية: ابقَ في الداخل وأغلق جميع الفتحات.",
         sens: "🚨 خطر شديد: التزم بالبقاء في غرف نظيفة ومغلقة تماماً."
      }
    }
  },
  en: {
    site_title: "Air Quality Monitoring",
    nav_main: "Home (Map)",
    nav_sensors: "Sensors",
    nav_readings: "Readings",
    nav_alerts: "Alerts",
    logo_text: "Real-time Air Quality Monitoring System",
    share: "Share",
    dark: "Dark",
    light: "Light",
    qr_title: "Scan to open on phone",
    qr_desc: "Scan the code above to share the dashboard or open it quickly on your mobile device.",
    close: "Close",
    monitors: "Monitoring Stations",
    cat_good: "Good",
    cat_mod: "Moderate",
    cat_sens: "Sensitive",
    cat_unh: "Unhealthy",
    cat_vun: "V. Unhealthy",
    cat_haz: "Hazardous",
    sensor_status: "Device & Sensor Status",
    live_readings: "Live Readings History",
    table_station: "Station",
    system_alerts: "System Alerts",
    device_num: "Device ID",
    aqi_label: "Air Quality Index",
    general_public: "General Public",
    sensitive_groups: "Sensitive Groups",
    online: "Online 🟢",
    offline: "Offline 🔴",
    battery: "Sensor Battery Level",
    last_read: "Last Reading Time",
    alert_warning: "⚠️ Alert: Air quality recorded as",
    alert_station: "at station",
    alert_index: "with index",
    all_clear: "All stations are performing excellently and indices are within normal ranges.",
    station_label: "Station",
    device_label: "Device",
    rec_pm25: "😷 Wear an N95 mask due to high PM2.5 levels.",
    rec_o3: "☀️ Avoid strenuous outdoor activity due to high Ozone.",
    rec_co: "🏠 Ensure good ventilation in indoor spaces (High CO).",
    rec_so2: "🌬️ Avoid going out if you have asthma (High SO2).",
    rec_no2: "🚗 Stay away from busy roads due to high Nitrogen oxides.",
    advice: {
      'Good': {
         gen: "✅ Air is great! Perfect for outdoor activities.",
         sens: "✅ Ideal air quality for all groups."
      },
      'Moderate': {
         gen: "🚶 Air quality is acceptable for most people.",
         sens: "⚠️ Consider reducing heavy exertion if symptomatic."
      },
      'Unhealthy for Sensitive Groups': {
         gen: "🚶 Reduce prolonged or heavy outdoor exertion.",
         sens: "😷 Minimize outdoor heavy work; stay in well-ventilated areas."
      },
      'Unhealthy': {
         gen: "🏠 Limit outdoor time; keep windows closed.",
         sens: "🚫 Avoid all outdoor physical activities; use air purifiers."
      },
      'Very Unhealthy': {
         gen: "🚫 Avoid all heavy physical outdoor activities.",
         sens: "🚨 Stay indoors and keep activity levels very low."
      },
      'Hazardous': {
         gen: "🚨 Emergency: Stay indoors and close all openings.",
         sens: "🚨 Severe Health Risk: Stay in fully enclosed, filtered rooms."
      }
    }
  }
};

// ── Helpers ──
function getLabel(v) {
  var cat = '';
  if (v <= 50) cat = 'Good';
  else if (v <= 100) cat = 'Moderate';
  else if (v <= 150) cat = 'Unhealthy for Sensitive Groups';
  else if (v <= 200) cat = 'Unhealthy';
  else if (v <= 300) cat = 'Very Unhealthy';
  else cat = 'Hazardous';
  
  var m = {
    'Good': I18N[currentLang].cat_good,
    'Moderate': I18N[currentLang].cat_mod,
    'Unhealthy for Sensitive Groups': I18N[currentLang].cat_sens,
    'Unhealthy': I18N[currentLang].cat_unh,
    'Very Unhealthy': I18N[currentLang].cat_vun,
    'Hazardous': I18N[currentLang].cat_haz
  };
  return m[cat] || cat;
}

function getSmartAdvice(dev) {
  const lang = currentLang;
  const cat = dev.AQI_category;
  
  let pollutantsAdvice = [];
  if (dev['PM2.5'] > 35) pollutantsAdvice.push(I18N[lang].rec_pm25);
  if (dev['O3'] > 70) pollutantsAdvice.push(I18N[lang].rec_o3);
  if (dev['CO'] > 9) pollutantsAdvice.push(I18N[lang].rec_co);
  if (dev['SO2'] > 75) pollutantsAdvice.push(I18N[lang].rec_so2);
  if (dev['NO2'] > 100) pollutantsAdvice.push(I18N[lang].rec_no2);

  const baseAdvice = I18N[lang].advice[cat] || { gen: '', sens: '' };
  
  let html = '';
  
  // Section 1: General Public
  html += '<div style="margin-bottom:12px;">' +
            '<div style="font-weight:700;font-size:12px;color:#6b7280;margin-bottom:6px;display:flex;align-items:center;gap:5px;">' +
               '👥 ' + I18N[lang].general_public +
            '</div>' +
            '<div style="background:#fff;border:1px solid #e5e7eb;padding:10px;border-radius:6px;font-size:12px;color:#374151;line-height:1.5;">' +
               baseAdvice.gen +
            '</div>' +
          '</div>';

  // Section 2: Sensitive Groups
  html += '<div style="margin-bottom:12px;">' +
            '<div style="font-weight:700;font-size:12px;color:#6b7280;margin-bottom:6px;display:flex;align-items:center;gap:5px;">' +
               '👶 ' + I18N[lang].sensitive_groups +
            '</div>' +
            '<div style="background:#fff;border:1px solid #e5e7eb;padding:10px;border-radius:6px;font-size:12px;color:#374151;line-height:1.5;">' +
               baseAdvice.sens +
            '</div>' +
          '</div>';

  // Section 3: Pollutant specific (if any)
  if (pollutantsAdvice.length > 0) {
    html += '<div style="border-top:1px dashed #d1d5db;padding-top:10px;margin-top:10px;display:flex;flex-direction:column;gap:5px;">' +
              pollutantsAdvice.map(a => '<div style="font-size:11px;color:#4b5563;background:#fef3c7;padding:5px 8px;border-radius:4px;border-right:3px solid #f59e0b;">' + a + '</div>').join('') +
            '</div>';
  }

  return html;
}

function updateI18nTexts() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (I18N[currentLang][key]) {
      // If it's the theme button, we handle it carefully to preserve icons
      if (key === 'dark' || key === 'light') {
         el.innerText = I18N[currentLang][key];
      } else {
         el.innerText = I18N[currentLang][key];
      }
    }
  });
  
  // Custom updates for buttons with icons
  document.getElementById('share-btn').innerHTML = '🔗 ' + I18N[currentLang].share;
  document.getElementById('theme-btn').innerHTML = (isLightTheme ? '🌙 ' : '☀️ ') + (isLightTheme ? I18N[currentLang].dark : I18N[currentLang].light);
  document.getElementById('lang-btn').innerText = (currentLang === 'ar' ? 'English' : 'العربية');
  
  // Re-run the tab content update to refresh table/sensors/alerts
  updateTabUIContent();
}

function toggleLanguage() {
  currentLang = (currentLang === 'ar' ? 'en' : 'ar');
  document.documentElement.lang = currentLang;
  document.documentElement.dir = (currentLang === 'ar' ? 'rtl' : 'ltr');
  updateI18nTexts();
  // We need to refresh map elements (tooltips)
  updateMapMarkers();
}

function getColor(v) {
  if (v <= 50)  return '#8cc63f';
  if (v <= 100) return '#fdb913';
  if (v <= 150) return '#f37021';
  if (v <= 200) return '#ed1c24';
  if (v <= 300) return '#9e005d';
  return '#790000';
}
function isLight(c) { return c === '#8cc63f' || c === '#fdb913'; }
function textColor(c) { return isLight(c) ? '#1a2236' : '#fff'; }

// ── Init Map ──
var map = L.map('map', {
  center: CENTER,
  zoom: 11,
  zoomControl: true
});

// Basemaps
var osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors',
  maxZoom: 19
});

var hybridLayer = L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', {
  attribution: '&copy; Google Maps',
  maxZoom: 20
});

osmLayer.addTo(map);

function updateMapMarkers() {
  map.eachLayer(function(layer) {
    if (layer.getControl) return; // ignore controls
    if (layer.getLatLng && layer.options && layer.options.radius === 16) {
      var ll = layer.getLatLng();
      var dev = DEVICES.find(function(d) { return Math.abs(d.lat - ll.lat) < 0.0001 && Math.abs(d.lon - ll.lng) < 0.0001; });
      if (dev) {
         var c = getColor(dev.predicted_AQI);
         layer.setTooltipContent(
           '<div style="font-family:Noto Kufi Arabic,sans-serif;direction:' + (currentLang === 'ar' ? 'rtl' : 'ltr') + ';text-align:' + (currentLang === 'ar' ? 'right' : 'left') + ';padding:4px 8px;">' +
           '<b>' + dev.name + '</b><br>' +
           '<div style="font-size:11px;opacity:0.8;margin-bottom:2px;">' + I18N[currentLang].device_label + ': ' + dev.device_id + '</div>' +
           'AQI: <b>' + dev.predicted_AQI + '</b><br>' +
           '<span style="color:' + c + '">' + getLabel(dev.predicted_AQI) + '</span>' +
           '</div>'
         );
      }
    }
  });
}

// Layer control for switching basemaps
var baseMaps = {
  "الخريطة التلقائية": osmLayer,
  "Hybrid (أقمار صناعية)": hybridLayer
};
L.control.layers(baseMaps, null, { position: 'topleft' }).addTo(map);

// Theme Toggle Logic
var themeBtn = document.getElementById('theme-btn');
var isLightTheme = true;
themeBtn.addEventListener('click', function() {
  isLightTheme = !isLightTheme;
  if(isLightTheme) {
    document.body.classList.add('light-theme');
    themeBtn.innerHTML = '🌙 ' + I18N[currentLang].dark;
  } else {
    document.body.classList.remove('light-theme');
    themeBtn.innerHTML = '☀️ ' + I18N[currentLang].light;
  }
});

// ── QR Code Logic ──
var qrGenerated = false;
function toggleQR() {
  var modal = document.getElementById('qr-modal');
  modal.classList.toggle('open');
  
  if (modal.classList.contains('open') && !qrGenerated) {
    new QRCode(document.getElementById("qrcode"), {
      text: window.location.href,
      width: 200,
      height: 200,
      colorDark : "#000000",
      colorLight : "#ffffff",
      correctLevel : QRCode.CorrectLevel.H
    });
    qrGenerated = true;
  }
}

// ── Mobile Panel Toggle Logic ──
var panelToggle = document.getElementById('panel-toggle');
var devicePanel = document.getElementById('device-panel');
var mapContainer = document.getElementById('map');
var legend = document.getElementById('legend');

panelToggle.addEventListener('click', function() {
  devicePanel.classList.toggle('open');
  var isOpen = devicePanel.classList.contains('open');
  panelToggle.innerText = isOpen ? '✕' : '☰';
  
  // On desktop, keep map right padding
  if (window.innerWidth > 768) {
    mapContainer.style.right = isOpen ? 'var(--panel-width)' : '0';
    legend.style.marginLeft = isOpen ? 'calc(-1 * var(--panel-width) / 2)' : '0';
  }
  setTimeout(function() { map.invalidateSize(); }, 400);
});

// ── Tab Switching Logic ──
function switchTab(tab) {
  ['main', 'sensors', 'readings', 'alerts'].forEach(function(t) {
    var link = document.getElementById('nav-' + t);
    if(link) link.classList.remove('active');
    var view = document.getElementById('view-' + t);
    if(view) view.classList.add('hidden');
  });
  document.getElementById('nav-' + tab).classList.add('active');
  
  if (tab === 'main') {
    document.getElementById('map').classList.remove('hidden');
    document.getElementById('device-panel').classList.remove('hidden');
    document.getElementById('legend').classList.remove('hidden');
    document.getElementById('panel-toggle').classList.remove('hidden');
    setTimeout(function() { map.invalidateSize(); }, 50);
  } else {
    document.getElementById('map').classList.add('hidden');
    document.getElementById('device-panel').classList.add('hidden');
    document.getElementById('detail-sidebar').classList.remove('open');
    document.getElementById('legend').classList.add('hidden');
    document.getElementById('panel-toggle').classList.add('hidden');
    var view = document.getElementById('view-' + tab);
    if(view) view.classList.remove('hidden');
    updateTabUIContent(); // Trigger update immediately when switching
  }
}

function updateTabUIContent() {
  var now = new Date();
  var dStr = now.toLocaleDateString('en-GB') + ' - ' + now.toLocaleTimeString('en-GB', {hour:'2-digit', minute:'2-digit'});

  // 1. Update Readings Table
  var rHTML = '';
  DEVICES.forEach(function(dev) {
    var c = getColor(dev.predicted_AQI);
    rHTML += '<tr>' +
      '<td style="font-weight:600">' + dev.name + '</td>' +
      '<td style="color:'+c+';font-weight:bold">' + Math.round(dev.predicted_AQI) + '</td>' +
      '<td>' + Number(dev['PM2.5']).toFixed(2) + '</td>' +
      '<td>' + Number(dev['PM10']).toFixed(2) + '</td>' +
      '<td>' + Number(dev['NO2']).toFixed(2) + '</td>' +
      '<td>' + Number(dev['SO2']).toFixed(2) + '</td>' +
      '<td>' + Number(dev['CO']).toFixed(2) + '</td>' +
      '<td>' + Number(dev['O3']).toFixed(2) + '</td>' +
    '</tr>';
  });
  var rTbody = document.getElementById('readings-tbody');
  if(rTbody) rTbody.innerHTML = rHTML;

  // 2. Update Sensors Grid
  var sHTML = '';
  DEVICES.forEach(function(dev) {
    if (dev._isOnline === undefined) dev._isOnline = Math.random() > 0.15;
    if (dev._bat === undefined) dev._bat = 100 - (dev.name.length % 20);
    if (dev._lastUpdate === undefined) {
       var dt = new Date();
       dev._lastUpdate = dt.toLocaleTimeString('en-GB', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
    }

    var isOnline = dev._isOnline ? I18N[currentLang].online : I18N[currentLang].offline;
    sHTML += '<div class="sensor-card">' +
      '<div class="s-card-title">' + dev.name + ' <span style="font-size:11px;color:#9aa5be;font-weight:400">(' + dev.device_id + ')</span></div>' +
      '<div class="sensor-row"><span>' + I18N[currentLang].online.split(' ')[0] + ':</span><span style="font-weight:bold">' + isOnline + '</span></div>' +
      '<div class="sensor-row"><span>' + I18N[currentLang].battery + ':</span><span>' + dev._bat + '% 🔋</span></div>' +
      '<div class="sensor-row"><span>' + I18N[currentLang].last_read + ':</span><span>' + dev._lastUpdate + '</span></div>' +
    '</div>';
  });
  var sContainer = document.getElementById('sensors-content');
  if(sContainer) sContainer.innerHTML = sHTML;

  // 3. Update Alerts
  var alertsHTML = '';
  var hasAlert = false;
  DEVICES.forEach(function(dev) {
    if(dev.predicted_AQI > 100) {
      alertsHTML += '<div class="alert-item">' +
        '<div class="alert-time">' + dStr.split(' - ')[1] + '</div>' +
        '<div class="alert-msg">' + I18N[currentLang].alert_warning + ' <span style="font-weight:bold;color:' + getColor(dev.predicted_AQI) + '">' + getLabel(dev.predicted_AQI) + '</span> ' + I18N[currentLang].alert_station + ' ' + dev.name + '. ' + I18N[currentLang].alert_index + ': ' + dev.predicted_AQI + '</div>' +
      '</div>';
      hasAlert = true;
    }
  });
  if(!hasAlert) {
    alertsHTML = '<div class="alert-item" style="border-right-color:#8cc63f">' +
      '<div class="alert-time">' + dStr.split(' - ')[1] + '</div>' +
      '<div class="alert-msg" style="color:#8cc63f">' + I18N[currentLang].all_clear + '</div>' +
    '</div>';
  }
  var aContainer = document.getElementById('alerts-content');
  if(aContainer) aContainer.innerHTML = alertsHTML;
}
// Initial populate
updateTabUIContent();

// Heat map layer
var heatData = DEVICES.map(function(d) {
  return [d.lat, d.lon, d.predicted_AQI];
});
L.heatLayer(heatData, {
  minOpacity: 0.3,
  radius: 40,
  blur: 30,
  maxZoom: 18
}).addTo(map);

// ── Build device panel & markers ──
var now = new Date();
var dateStr = now.toLocaleDateString('en-GB') + ' - ' +
              now.toLocaleTimeString('en-GB', {hour:'2-digit', minute:'2-digit'});

var devList  = document.getElementById('device-list');
var sidebar  = document.getElementById('detail-sidebar');
var sbBody   = document.getElementById('sidebar-body');

DEVICES.forEach(function(dev, idx) {
  var c  = getColor(dev.predicted_AQI);
  var tc = textColor(c);

  // ── Device card in panel ──
  var card = document.createElement('div');
  card.className = 'device-card';
  card.id = 'card-' + idx;
  card.innerHTML =
    '<div class="dev-circle" style="background:' + c + ';color:' + tc + '">' +
      Math.round(dev.predicted_AQI) +
    '</div>' +
    '<div class="dev-body">' +
      '<div class="dev-title">' + dev.name + '</div>' +
      '<div class="dev-sub">' + dev.device_id + ' | AQI ' + Math.round(dev.predicted_AQI) + '</div>' +
      '<span class="dev-badge" style="background:' + c + ';color:' + tc + '">' +
        getLabel(dev.predicted_AQI) +
      '</span>' +
    '</div>';
  card.addEventListener('click', function() { openSidebar(dev, idx); });
  devList.appendChild(card);

  // ── Map marker ──
  var marker = L.circleMarker([dev.lat, dev.lon], {
    radius: 16,
    fillColor: c,
    color: isLight(c) ? 'rgba(0,0,0,0.25)' : 'rgba(255,255,255,0.35)',
    weight: 2.5,
    opacity: 1,
    fillOpacity: 0.95
  }).addTo(map);

  marker.bindTooltip(
    '<div style="font-family:Noto Kufi Arabic,sans-serif;direction:rtl;text-align:right;padding:4px 8px;">' +
    '<b>' + dev.name + '</b><br>' +
    '<div style="font-size:11px;opacity:0.8;margin-bottom:2px;">' + I18N[currentLang].device_label + ': ' + dev.device_id + '</div>' +
    'AQI: <b>' + dev.predicted_AQI + '</b><br>' +
    '<span style="color:' + c + '">' + getLabel(dev.predicted_AQI) + '</span>' +
    '</div>',
    { direction: 'top', opacity: 0.96 }
  );

  marker.on('click', function(e) {
    L.DomEvent.stopPropagation(e);
    openSidebar(dev, idx);
  });
});

// Close sidebar on map click
map.on('click', function() {
  sidebar.classList.remove('open');
  // Also close panel on mobile if map clicked
  if (window.innerWidth <= 768) {
    devicePanel.classList.remove('open');
    panelToggle.innerText = '☰';
  }
  document.querySelectorAll('.device-card').forEach(function(c) {
    c.classList.remove('selected');
  });
});

// ── Open sidebar ──
function openSidebar(dev, idx) {
  var c  = getColor(dev.predicted_AQI);
  var tc = textColor(c);

  // Arrow position on scale: bar goes green(left)→maroon(right), AQI 0→300 reversed
  var pct     = Math.min(100, Math.max(0, dev.predicted_AQI / 300 * 100));
  var arrowPct = (100 - pct).toFixed(1) + '%';

  var units = {
    'PM2.5': 'μg/m³', 'PM10': 'μg/m³',
    'NO2': 'ppb', 'SO2': 'ppb', 'CO': 'ppm', 'O3': 'ppb'
  };

  var metricRows = '';
  ['PM2.5','PM10','NO2','SO2','CO','O3'].forEach(function(f) {
    var rawVal = (dev[f] !== undefined) ? dev[f] : 0;
    var val = (dev[f] !== undefined) ? Number(rawVal).toFixed(2) : '—';
    metricRows +=
      '<div class="metric-item">' +
        '<div class="metric-left">' +
          '<div class="metric-accent" style="background:' + c + '"></div>' +
          '<div>' +
            '<div class="metric-key">' + f + '</div>' +
            '<div class="metric-unit">' + units[f] + '</div>' +
          '</div>' +
        '</div>' +
        '<div class="metric-value" style="background:' + c + ';color:' + tc + '">' + val + '</div>' +
      '</div>';
  });

  sbBody.innerHTML =
    '<div class="sb-card">' +
      '<div class="sb-top-row">' +
        '<button class="sb-close-btn" onclick="closeSidebar()">&#215;</button>' +
        '<div class="sb-dev-num">' + I18N[currentLang].device_num + ' : ' + dev.device_id + '</div>' +
      '</div>' +
      '<div class="sb-city-row">' +
        '<div class="sb-status-pill" style="background:' + c + ';color:' + tc + '">' +
          getLabel(dev.predicted_AQI) +
        '</div>' +
        '<div class="sb-city-name">' + dev.name + '</div>' +
      '</div>' +
      '<div class="sb-meta-row">' +
        '<span>🕐 ' + dateStr + '</span>' +
      '</div>' +
      '<div class="sb-aqi-display">' +
        '<div class="sb-aqi-value">' + Math.round(dev.predicted_AQI) + '</div>' +
        '<div class="sb-aqi-caption">' + I18N[currentLang].aqi_label + '</div>' +
      '</div>' +
      '<div class="scale-container">' +
        '<div class="scale-pointer" style="color:' + c + ';left:calc(' + arrowPct + ' - 7px)">&#9660;</div>' +
        '<div class="scale-track"></div>' +
        '<div class="scale-ticks">' +
          '<span>0</span><span>50</span><span>100</span><span>150</span><span>200</span><span>300+</span>' +
        '</div>' +
      '</div>' +
    '</div>' +

    '<div class="info-box" style="display:block;padding:15px;background:#f3f4f6;border:none;box-shadow:inset 0 1px 3px rgba(0,0,0,0.05);">' +
       getSmartAdvice(dev) +
    '</div>' +

    '<div class="metrics-list">' + metricRows + '</div>';

  sidebar.classList.add('open');
  
  // On mobile, close the device list to show the details
  if (window.innerWidth <= 768) {
    devicePanel.classList.remove('open');
    panelToggle.innerText = '☰';
  }

  // Fly to device on map
  map.flyTo([dev.lat, dev.lon], 14, { duration: 1.2 });

  // Highlight device card
  document.querySelectorAll('.device-card').forEach(function(c) {
    c.classList.remove('selected');
  });
  var card = document.getElementById('card-' + idx);
  if (card) {
    card.classList.add('selected');
    if (window.innerWidth > 768) card.scrollIntoView({ block: 'nearest' });
  }
};

function closeSidebar() {
  sidebar.classList.remove('open');
  document.querySelectorAll('.device-card').forEach(function(c) {
    c.classList.remove('selected');
  });
}

// ── 10-Second Auto Update Simulation ──
setInterval(function() {
  var dt = new Date();
  var dStr = dt.toLocaleDateString('en-GB') + ' - ' +
             dt.toLocaleTimeString('en-GB', {hour:'2-digit', minute:'2-digit'});

  // 1. Simulate data progression
  DEVICES.forEach(function(dev, idx) {
    if (dev._isOnline === undefined) dev._isOnline = Math.random() > 0.15;
    if (dev._bat === undefined) dev._bat = 100 - (dev.name.length % 20);

    // Simulate unexpected drops and reconnects
    if (dev._isOnline) {
      if (Math.random() < 0.05) dev._isOnline = false;
    } else {
      // Extension of duration: recovery chance reduced to 0.002 (approx hours)
      if (Math.random() < 0.002) dev._isOnline = true; 
    }

    // Battery drain simulation
    if (dev._isOnline && Math.random() < 0.5) dev._bat -= 1;
    if (dev._bat <= 5) dev._bat = 100; // recharge when low

    // Store original values on first run to oscillate around them
    if (!dev._orig) {
       dev._orig = { AQI: dev.predicted_AQI };
       ['PM2.5','PM10','NO2','SO2','CO','O3'].forEach(function(f) {
          if (dev[f] !== undefined) dev._orig[f] = dev[f];
       });
    }

    // Data only drifts if the sensor is online
    if (dev._isOnline) {
      // Update its specific timestamp only when online
      var dt_live = new Date();
      dev._lastUpdate = dt_live.toLocaleTimeString('en-GB', {hour:'2-digit', minute:'2-digit', second:'2-digit'});

      // Random drift + mean reversion (forces values to go up AND down naturally)
      var drift = (Math.random() - 0.5) * 12 + (dev._orig.AQI - dev.predicted_AQI) * 0.2; 
      dev.predicted_AQI = Math.max(0, dev.predicted_AQI + drift);

      ['PM2.5','PM10','NO2','SO2','CO','O3'].forEach(function(f) {
        if (dev[f] !== undefined) {
           var f_drift = (dev._orig[f] * (Math.random() - 0.5) * 0.08) + (dev._orig[f] - dev[f]) * 0.2;
           dev[f] = Math.max(0, dev[f] + f_drift);
        }
      });

      dev.predicted_AQI = Math.round(dev.predicted_AQI);

      var v = dev.predicted_AQI;
      if (v <= 50) dev.AQI_category = 'Good';
      else if (v <= 100) dev.AQI_category = 'Moderate';
      else if (v <= 150) dev.AQI_category = 'Unhealthy for Sensitive Groups';
      else if (v <= 200) dev.AQI_category = 'Unhealthy';
      else if (v <= 300) dev.AQI_category = 'Very Unhealthy';
      else dev.AQI_category = 'Hazardous';
    } else {
      // Values and categories remain as they were (Frozen)
    }

    var c = getColor(dev.predicted_AQI);
    var tc = textColor(c);

    // 2. Update panel cards
    var card = document.getElementById('card-' + idx);
    if (card) {
      card.querySelector('.dev-circle').style.background = c;
      card.querySelector('.dev-circle').style.color = tc;
      card.querySelector('.dev-circle').innerText = dev.predicted_AQI;
      card.querySelector('.dev-sub').innerText = dev.device_id + ' | AQI ' + dev.predicted_AQI;
      var badge = card.querySelector('.dev-badge');
      badge.style.background = c;
      badge.style.color = tc;
      badge.innerText = getLabel(dev.predicted_AQI);
    }
  });

  // 3. Update Map Layers directly
  var newHeatData = [];
  DEVICES.forEach(function(d) { newHeatData.push([d.lat, d.lon, d.predicted_AQI]); });

  map.eachLayer(function(layer) {
    // HeatLayer detection
    if (layer.setLatLngs && !layer.getLatLng) {
      layer.setLatLngs(newHeatData);
    }
    // CircleMarker detection
    if (layer.getLatLng && layer.options && layer.options.radius === 16) {
      var ll = layer.getLatLng();
      var dev = DEVICES.find(function(d) { return Math.abs(d.lat - ll.lat) < 0.0001 && Math.abs(d.lon - ll.lng) < 0.0001; });
      if (dev) {
        var c = getColor(dev.predicted_AQI);
        layer.setStyle({
          fillColor: c,
          color: (c === '#8cc63f' || c === '#fdb913') ? 'rgba(0,0,0,0.25)' : 'rgba(255,255,255,0.35)'
        });
        layer.setTooltipContent(
          '<div style="font-family:Noto Kufi Arabic,sans-serif;direction:rtl;text-align:right;padding:4px 8px;">' +
          '<b>' + dev.name + '</b><br>' +
          '<div style="font-size:11px;opacity:0.8;margin-bottom:2px;">' + I18N[currentLang].device_label + ': ' + dev.device_id + '</div>' +
          'AQI: <b>' + dev.predicted_AQI + '</b><br>' +
          '<span style="color:' + c + '">' + getLabel(dev.predicted_AQI) + '</span>' +
          '</div>'
        );
      }
    }
  });

  // 4. Update detail sidebar if currently open
  var activeCard = document.querySelector('.device-card.selected');
  if (activeCard) {
    var idStr = activeCard.id.split('-')[1];
    var openDev = DEVICES[parseInt(idStr)];
    // Quick update date inside it
    var rowsHTML = '';
    var units = {'PM2.5':'μg/m³','PM10':'μg/m³','NO2':'ppb','SO2':'ppb','CO':'ppm','O3':'ppb'};
    var c  = getColor(openDev.predicted_AQI);
    var tc = textColor(c);
    var pct = Math.min(100, Math.max(0, openDev.predicted_AQI / 300 * 100));
    var arrowPct = (100 - pct).toFixed(1) + '%';
    
    ['PM2.5','PM10','NO2','SO2','CO','O3'].forEach(function(f) {
      var val = (openDev[f] !== undefined) ? Number(openDev[f]).toFixed(2) : '—';
      rowsHTML +=
        '<div class="metric-item">' +
          '<div class="metric-left">' +
            '<div class="metric-accent" style="background:' + c + '"></div>' +
            '<div><div class="metric-key">' + f + '</div><div class="metric-unit">' + units[f] + '</div></div>' +
          '</div>' +
          '<div class="metric-value" style="background:' + c + ';color:' + tc + '">' + val + '</div>' +
        '</div>';
    });

    // Injecting into existing structural classes
    var sBody = document.getElementById('sidebar-body');
    if (sBody) {
       var sbStatus = sBody.querySelector('.sb-status-pill');
       if (sbStatus) { sbStatus.style.background = c; sbStatus.style.color = tc; sbStatus.innerText = getLabel(openDev.predicted_AQI); }
       var sbAqi = sBody.querySelector('.sb-aqi-value');
       if (sbAqi) { sbAqi.innerText = openDev.predicted_AQI; }
       var sbPtr = sBody.querySelector('.scale-pointer');
       if (sbPtr) { sbPtr.style.color = c; sbPtr.style.left = 'calc(' + arrowPct + ' - 7px)'; }
       var infIco = sBody.querySelector('.info-icon');
       if (infIco) { infIco.style.color = c; }
       var infHead = sBody.querySelector('.info-heading');
       if (infHead) { infHead.innerText = getLabel(openDev.predicted_AQI); }
       var infBody = sBody.querySelector('.info-body');
       if (infBody) { infBody.innerHTML = getSmartAdvice(openDev); }
       var metList = sBody.querySelector('.metrics-list');
       if (metList) { metList.innerHTML = rowsHTML; }
       var dateSpan = sBody.querySelector('.sb-meta-row span:first-child');
       if (dateSpan) { dateSpan.innerText = '🕐 ' + dStr; }
    }
  }

  // ── 5. Update Tab Views Content ──
  // Readings Table
  var rHTML = '';
  DEVICES.forEach(function(dev) {
    var c = getColor(dev.predicted_AQI);
    rHTML += '<tr>' +
      '<td style="font-weight:600">' + dev.name + '</td>' +
      '<td style="color:'+c+';font-weight:bold">' + dev.predicted_AQI + '</td>' +
      '<td>' + Number(dev['PM2.5']).toFixed(2) + '</td>' +
      '<td>' + Number(dev['PM10']).toFixed(2) + '</td>' +
      '<td>' + Number(dev['NO2']).toFixed(2) + '</td>' +
      '<td>' + Number(dev['SO2']).toFixed(2) + '</td>' +
      '<td>' + Number(dev['CO']).toFixed(2) + '</td>' +
      '<td>' + Number(dev['O3']).toFixed(2) + '</td>' +
    '</tr>';
  });
  var rTbody = document.getElementById('readings-tbody');
  if(rTbody) rTbody.innerHTML = rHTML;

  // Sensors Grid
  var sHTML = '';
  DEVICES.forEach(function(dev) {
    var isOnline = dev._isOnline ? I18N[currentLang].online : I18N[currentLang].offline;
    sHTML += '<div class="sensor-card">' +
      '<div class="s-card-title">' + dev.name + ' <span style="font-size:11px;color:#9aa5be;font-weight:400">(' + dev.device_id + ')</span></div>' +
      '<div class="sensor-row"><span>' + I18N[currentLang].online.split(' ')[0] + ':</span><span style="font-weight:bold">' + isOnline + '</span></div>' +
      '<div class="sensor-row"><span>' + I18N[currentLang].battery + ':</span><span>' + dev._bat + '% 🔋</span></div>' +
      '<div class="sensor-row"><span>' + I18N[currentLang].last_read + ':</span><span>' + dev._lastUpdate + '</span></div>' +
    '</div>';
  });
  var sContainer = document.getElementById('sensors-content');
  if(sContainer) sContainer.innerHTML = sHTML;

  // Alerts List 
  var alertsHTML = '';
  var hasAlert = false;
  DEVICES.forEach(function(dev) {
    if(dev.predicted_AQI > 100) {
      alertsHTML += '<div class="alert-item">' +
        '<div class="alert-time">' + dStr.split(' - ')[1] + '</div>' +
        '<div class="alert-msg">' + I18N[currentLang].alert_warning + ' <span style="font-weight:bold;color:' + getColor(dev.predicted_AQI) + '">' + getLabel(dev.predicted_AQI) + '</span> ' + I18N[currentLang].alert_station + ' ' + dev.name + '. ' + I18N[currentLang].alert_index + ': ' + dev.predicted_AQI + '</div>' +
      '</div>';
      hasAlert = true;
    }
  });
  if(!hasAlert) {
    alertsHTML = '<div class="alert-item" style="border-right-color:#8cc63f">' +
      '<div class="alert-time">' + dStr.split(' - ')[1] + '</div>' +
      '<div class="alert-msg" style="color:#8cc63f">' + I18N[currentLang].all_clear + '</div>' +
    '</div>';
  }
  var aContainer = document.getElementById('alerts-content');
  if(aContainer) aContainer.innerHTML = alertsHTML;

}, 10000);
</script>
</body>
</html>
"""
with open(MAP_PATH, 'w', encoding='utf-8') as f:
    f.write(html.replace('{devices_json}', devices_json).replace('{center_lat}', str(center_lat)).replace('{center_lon}', str(center_lon)))


print("\n=== Simulated Device Readings ===")
print(sim_df[["device_id","name","predicted_AQI","AQI_category"]].to_string(index=False))
print(f"\nMap -> {MAP_PATH}")