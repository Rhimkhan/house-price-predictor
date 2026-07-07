from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import traceback

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path    = os.path.join(BASE_DIR, "model.pkl")
scaler_path   = os.path.join(BASE_DIR, "scaler.pkl")
encoders_path = os.path.join(BASE_DIR, "label_encoders.pkl")

model_load_error = "Not attempted"
model = None
scaler = None
label_encoders = None

try:
    model = joblib.load(model_path)
    model_load_error = "None - loaded OK"
except Exception as e:
    model_load_error = str(e)

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    pass

try:
    label_encoders = joblib.load(encoders_path)
except Exception as e:
    pass

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #667eea, #764ba2); min-height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; }
        .container { background: white; padding: 40px; border-radius: 20px; width: 100%; max-width: 500px; }
        h1 { text-align: center; color: #333; }
        p { text-align: center; color: #666; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #444; }
        input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 8px; box-sizing: border-box; font-size: 14px; }
        .btn-predict { width: 100%; padding: 15px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; border-radius: 10px; font-size: 18px; font-weight: bold; cursor: pointer; margin-top: 10px; }
        .btn-predict:hover { opacity: 0.9; }
        .result { text-align: center; margin-top: 20px; padding: 20px; background: #f0f9ff; border-radius: 10px; display: none; }
        .price { font-size: 36px; font-weight: bold; color: #667eea; }
        .error { color: red; text-align: center; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏠 House Price Predictor</h1>
        <p>Enter house details to get estimated price</p>
        <form id="predictForm">
            <div class="form-group"><label>Lot Area (sq ft)</label><input type="number" name="LotArea" value="10000"></div>
            <div class="form-group"><label>Overall Quality (1-10)</label><input type="number" name="OverallQual" value="6" min="1" max="10"></div>
            <div class="form-group"><label>Year Built</label><input type="number" name="YearBuilt" value="2000"></div>
            <div class="form-group"><label>Total Basement SF</label><input type="number" name="TotalBsmtSF" value="800"></div>
            <div class="form-group"><label>1st Floor SF</label><input type="number" name="X1stFlrSF" value="1000"></div>
            <div class="form-group"><label>2nd Floor SF</label><input type="number" name="X2ndFlrSF" value="0"></div>
            <div class="form-group"><label>Garage Area (sq ft)</label><input type="number" name="GarageArea" value="400"></div>
            <div class="form-group"><label>Bedrooms</label><input type="number" name="BedroomAbvGr" value="3"></div>
            <div class="form-group"><label>Full Bathrooms</label><input type="number" name="FullBath" value="2"></div>
            <div class="form-group"><label>Half Bathrooms</label><input type="number" name="HalfBath" value="1"></div>
            <div class="form-group"><label>Fireplaces</label><input type="number" name="Fireplaces" value="0"></div>
            <button type="submit" class="btn-predict">🔮 Predict Price</button>
        </form>
        <div id="result" class="result">
            <p style="color:#666;font-size:14px;">Estimated Price</p>
            <div class="price" id="priceDisplay">$0</div>
        </div>
        <div id="error" class="error"></div>
    </div>
    <script>
        document.getElementById("predictForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => { data[key] = value; });
            document.getElementById("result").style.display = "none";
            document.getElementById("error").textContent = "";
            try {
                const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
                const result = await response.json();
                if (result.success) {
                    document.getElementById("priceDisplay").textContent = "$" + result.price.toLocaleString();
                    document.getElementById("result").style.display = "block";
                } else {
                    document.getElementById("error").textContent = "Error: " + result.error;
                }
            } catch (err) {
                document.getElementById("error").textContent = "Error: Could not connect to server";
            }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return HTML_TEMPLATE

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or label_encoders is None:
        return jsonify({"success": False, "error": "Model files are not loaded. Please try again later."}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        input_df = pd.DataFrame([{
            "LotArea": float(data.get("LotArea", 10000)),
            "OverallQual": float(data.get("OverallQual", 6)),
            "YearBuilt": float(data.get("YearBuilt", 2000)),
            "TotalBsmtSF": float(data.get("TotalBsmtSF", 0)),
            "1stFlrSF": float(data.get("X1stFlrSF", 0)),
            "2ndFlrSF": float(data.get("X2ndFlrSF", 0)),
            "GrLivArea": float(data.get("X1stFlrSF", 0)) + float(data.get("X2ndFlrSF", 0)),
            "GarageArea": float(data.get("GarageArea", 0)),
            "Fireplaces": float(data.get("Fireplaces", 0)),
            "FullBath": float(data.get("FullBath", 0)),
            "HalfBath": float(data.get("HalfBath", 0)),
            "BedroomAbvGr": float(data.get("BedroomAbvGr", 0)),
            "Neighborhood": data.get("Neighborhood", "NAmes")
        }])
        input_df = input_df.fillna(0)
        for col in label_encoders:
            if col in input_df.columns:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                except:
                    input_df[col] = 0
        input_scaled = scaler.transform(input_df)
        prediction = float(model.predict(input_scaled)[0])
        prediction = max(prediction, 10000.0)
        return jsonify({"success": True, "price": round(prediction, 2)})
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_error": model_load_error,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": label_encoders is not None,
        "model_path": model_path,
        "model_exists_on_disk": os.path.exists(model_path),
        "base_dir": BASE_DIR,
        "base_dir_files": os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else []
    })

app = app
