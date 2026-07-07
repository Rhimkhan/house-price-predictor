"""
app.py — House Price Prediction Flask Web Application
================================================
Run:  python app.py
Then: http://localhost:5000
"""

import os
import sys
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "house-price-2024-secret"

ROOT    = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")

# ── Load model artefacts ──────────────────────────────────────
MODEL_LOADED = False
model = scaler = label_encoders = None

try:
    model          = joblib.load(os.path.join(SRC_DIR, "model.pkl"))
    scaler         = joblib.load(os.path.join(SRC_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(SRC_DIR, "label_encoders.pkl"))
    MODEL_LOADED   = True
    logger.info("✅ Model artefacts loaded successfully")
except Exception as e:
    logger.warning(f"⚠ Model not found: {e}. Run model_training.py first.")

# ── Feature definitions ───────────────────────────────────────
FEATURES = {
    "OverallQual" : {"label": "Overall Quality",         "min": 1,    "max": 10,     "default": 5,    "step": 1,    "unit": "/ 10",   "icon": "⭐", "desc": "Overall material & finish quality"},
    "GrLivArea"   : {"label": "Living Area",             "min": 300,  "max": 5000,   "default": 1500, "step": 50,   "unit": "sq ft",  "icon": "📐", "desc": "Above-ground living area in sq ft"},
    "TotalBsmtSF" : {"label": "Basement Area",           "min": 0,    "max": 3000,   "default": 800,  "step": 50,   "unit": "sq ft",  "icon": "🏗️", "desc": "Total basement area in sq ft"},
    "GarageArea"  : {"label": "Garage Area",             "min": 0,    "max": 1500,   "default": 400,  "step": 25,   "unit": "sq ft",  "icon": "🚗", "desc": "Garage area in sq ft"},
    "YearBuilt"   : {"label": "Year Built",              "min": 1850, "max": 2024,   "default": 1990, "step": 1,    "unit": "",       "icon": "📅", "desc": "Original construction year"},
    "YearRemodAdd": {"label": "Year Remodelled",         "min": 1850, "max": 2024,   "default": 1990, "step": 1,    "unit": "",       "icon": "🔨", "desc": "Year of remodel (= built if none)"},
    "FullBath"    : {"label": "Full Bathrooms",          "min": 0,    "max": 5,      "default": 2,    "step": 1,    "unit": "",       "icon": "🛁", "desc": "Full bathrooms above ground"},
    "HalfBath"    : {"label": "Half Bathrooms",          "min": 0,    "max": 3,      "default": 0,    "step": 1,    "unit": "",       "icon": "🚿", "desc": "Half bathrooms above ground"},
    "BedroomAbvGr": {"label": "Bedrooms",                "min": 0,    "max": 8,      "default": 3,    "step": 1,    "unit": "",       "icon": "🛏️", "desc": "Bedrooms above ground"},
    "Fireplaces"  : {"label": "Fireplaces",              "min": 0,    "max": 4,      "default": 1,    "step": 1,    "unit": "",       "icon": "🔥", "desc": "Number of fireplaces"},
    "GarageCars"  : {"label": "Garage Car Capacity",     "min": 0,    "max": 5,      "default": 2,    "step": 1,    "unit": "cars",   "icon": "🏎️", "desc": "Garage capacity in car spaces"},
    "LotArea"     : {"label": "Lot Area",                "min": 1000, "max": 200000, "default": 8000, "step": 500,  "unit": "sq ft",  "icon": "🌿", "desc": "Lot size in sq ft"},
    "OverallCond" : {"label": "Overall Condition",       "min": 1,    "max": 9,      "default": 5,    "step": 1,    "unit": "/ 9",    "icon": "🏠", "desc": "Overall condition rating"},
    "TotRmsAbvGrd": {"label": "Total Rooms",             "min": 2,    "max": 14,     "default": 7,    "step": 1,    "unit": "rooms",  "icon": "🚪", "desc": "Total rooms above ground (excl. baths)"},
    "1stFlrSF"    : {"label": "1st Floor Area",          "min": 300,  "max": 4000,   "default": 900,  "step": 50,   "unit": "sq ft",  "icon": "1️⃣", "desc": "First floor area in sq ft"},
    "2ndFlrSF"    : {"label": "2nd Floor Area",          "min": 0,    "max": 2000,   "default": 0,    "step": 50,   "unit": "sq ft",  "icon": "2️⃣", "desc": "Second floor area (0 if single storey)"},
}

INTEGER_FEATURES = {"OverallQual","OverallCond","FullBath","HalfBath",
                    "BedroomAbvGr","Fireplaces","GarageCars","TotRmsAbvGrd",
                    "YearBuilt","YearRemodAdd"}

# ── Feature engineering (mirrors training) ────────────────────
def _apply_feature_engineering(row: dict) -> dict:
    r = row.copy()
    r["TotalSF"]         = r.get("TotalBsmtSF",0) + r.get("1stFlrSF",0) + r.get("2ndFlrSF",0)
    r["TotalPorchSF"]    = 0
    r["TotalLivingArea"] = r.get("GrLivArea",0) + r.get("TotalBsmtSF",0)
    r["HouseAge"]        = max(2010 - r.get("YearBuilt",1990), 0)
    r["RemodAge"]        = max(2010 - r.get("YearRemodAdd",1990), 0)
    r["IsRemodeled"]     = int(r.get("YearRemodAdd",0) > r.get("YearBuilt",0))
    r["IsNew"]           = 0
    r["HasBasement"]     = int(r.get("TotalBsmtSF",0) > 0)
    r["HasGarage"]       = int(r.get("GarageArea",0) > 0)
    r["HasPool"]         = 0
    r["HasFireplace"]    = int(r.get("Fireplaces",0) > 0)
    r["HasDeck"]         = 0
    r["TotalBath"]       = r.get("FullBath",0) + 0.5*r.get("HalfBath",0)
    r["TotalQual"]       = r.get("OverallQual",5) * r.get("OverallCond",5)
    r["GrLivArea_Qual"]  = r.get("GrLivArea",1500) * r.get("OverallQual",5)
    r["QualCondRatio"]   = r.get("OverallQual",5) / (r.get("OverallCond",5) + 1)
    r["PricePerSF"]      = r.get("OverallQual",5) / max(r.get("TotalSF",1), 1)
    r["Neighborhood_Quality"] = 0
    yr = r.get("YearBuilt", 1990)
    era_bins   = [0,1900,1950,1970,1990,2000,2010,2030]
    era_labels = [0,1,2,3,4,5,6]
    r["YearBuilt_Era"] = next(
        (era_labels[i] for i,(lo,hi) in enumerate(zip(era_bins,era_bins[1:])) if lo<yr<=hi), 0
    )
    return r


def predict_price(features: dict) -> dict:
    """Run prediction and return price + confidence band."""
    if not MODEL_LOADED:
        raise RuntimeError("Model not loaded — run model_training.py first")

    features = _apply_feature_engineering(features)

    # Encode categoricals
    for col, le in label_encoders.items():
        if col in features:
            val = str(features[col])
            val = val if val in le.classes_ else le.classes_[0]
            features[col] = int(le.transform([val])[0])

    df = pd.DataFrame([features])
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Align columns
    n = scaler.n_features_in_
    while df.shape[1] < n:
        df[f"_pad_{df.shape[1]}"] = 0
    df = df.iloc[:, :n]

    X = scaler.transform(df.values.astype(float))
    price = float(model.predict(X)[0])

    return {
        "price"    : price,
        "formatted": f"${price:,.0f}",
        "lower"    : f"${price*0.90:,.0f}",
        "upper"    : f"${price*1.10:,.0f}",
        "lower_raw": price * 0.90,
        "upper_raw": price * 1.10,
    }


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           features=FEATURES,
                           model_loaded=MODEL_LOADED)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = {}
        for key, meta in FEATURES.items():
            val = request.form.get(key, meta["default"])
            features[key] = int(val) if key in INTEGER_FEATURES else float(val)

        result   = predict_price(features)
        features_display = {FEATURES[k]["label"]: v for k, v in features.items() if k in FEATURES}

        return render_template("result.html",
                               result=result,
                               features=features_display,
                               raw_features=features,
                               timestamp=datetime.now().strftime("%B %d, %Y at %H:%M"))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("index.html",
                               features=FEATURES,
                               model_loaded=MODEL_LOADED,
                               error=str(e))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API — JSON in, JSON out."""
    try:
        data = request.get_json(force=True) or {}
        features = {}
        for key, meta in FEATURES.items():
            features[key] = data.get(key, meta["default"])

        result = predict_price(features)
        return jsonify({"success": True, **result,
                        "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": MODEL_LOADED,
                    "timestamp": datetime.now().isoformat()})


@app.route("/api/features")
def api_features():
    return jsonify(FEATURES)


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": str(e)}), 500


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False)
