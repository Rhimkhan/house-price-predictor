from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# ------------------------------------------------------------------
# 1. LOAD MODELS WITH ERROR HANDLING
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
encoders_path = os.path.join(BASE_DIR, "label_encoders.pkl")

print(f"ðŸ“ Looking for models in: {BASE_DIR}")

model = None
scaler = None
label_encoders = None

try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("âœ… Model loaded successfully")
    else:
        print(f"âŒ Model not found at: {model_path}")
except Exception as e:
    print(f"âŒ Error loading model: {e}")

try:
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("âœ… Scaler loaded successfully")
    else:
        print(f"âŒ Scaler not found at: {scaler_path}")
except Exception as e:
    print(f"âŒ Error loading scaler: {e}")

try:
    if os.path.exists(encoders_path):
        label_encoders = joblib.load(encoders_path)
        print("âœ… Label encoders loaded successfully")
    else:
        print(f"âŒ Label encoders not found at: {encoders_path}")
except Exception as e:
    print(f"âŒ Error loading label encoders: {e}")

# ------------------------------------------------------------------
# 2. HTML TEMPLATE
# ------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ðŸ  House Price Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { text-align: center; color: #333; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; font-weight: 600; color: #555; margin-bottom: 5px; }
        input, select {
            width: 100%;
            padding: 10px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s;
        }
        input:focus, select:focus {
            border-color: #667eea;
            outline: none;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.2);
        }
        .btn-predict {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 10px;
        }
        .btn-predict:hover {
            transform: scale(1.02);
            box-shadow: 0 10px 30px rgba(102,126,234,0.4);
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            background: #f8f9fa;
            text-align: center;
            display: none;
        }
        .result.show { display: block; animation: fadeIn 0.5s; }
        .price {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .error { color: #dc3545; text-align: center; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ðŸ  House Price Predictor</h1>
        <p class="subtitle">Enter house details to get estimated price</p>

        <form id="predictForm">
            <div class="form-group">
                <label>Lot Area (sq ft)</label>
                <input type="number" name="LotArea" value="10000" required>
            </div>
            <div class="form-group">
                <label>Overall Quality (1-10)</label>
                <input type="number" name="OverallQual" value="6" min="1" max="10" required>
            </div>
            <div class="form-group">
                <label>Year Built</label>
                <input type="number" name="YearBuilt" value="2000" required>
            </div>
            <div class="form-group">
                <label>Total Basement SF</label>
                <input type="number" name="TotalBsmtSF" value="800">
            </div>
            <div class="form-group">
                <label>1st Floor SF</label>
                <input type="number" name="X1stFlrSF" value="1000">
            </div>
            <div class="form-group">
                <label>2nd Floor SF</label>
                <input type="number" name="X2ndFlrSF" value="0">
            </div>
            <div class="form-group">
                <label>Garage Area (sq ft)</label>
                <input type="number" name="GarageArea" value="400">
            </div>
            <div class="form-group">
                <label>Bedrooms</label>
                <input type="number" name="BedroomAbvGr" value="3" required>
            </div>
            <div class="form-group">
                <label>Full Bathrooms</label>
                <input type="number" name="FullBath" value="2" required>
            </div>
            <div class="form-group">
                <label>Half Bathrooms</label>
                <input type="number" name="HalfBath" value="1">
            </div>
            <div class="form-group">
                <label>Fireplaces</label>
                <input type="number" name="Fireplaces" value="0">
            </div>
            <button type="submit" class="btn-predict">ðŸ”® Predict Price</button>
        </form>

        <div id="result" class="result">
            <p style="color: #666; font-size: 14px;">Estimated Price</p>
            <div class="price" id="priceDisplay">$0</div>
        </div>
        <div id="error" class="error"></div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = value;
            });

            const resultDiv = document.getElementById('result');
            const errorDiv = document.getElementById('error');
            const priceDisplay = document.getElementById('priceDisplay');

            resultDiv.classList.remove('show');
            errorDiv.textContent = '';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.success) {
                    priceDisplay.textContent = '$' + result.price.toLocaleString();
                    resultDiv.classList.add('show');
                } else {
                    errorDiv.textContent = 'Error: ' + result.error;
                }
            } catch (err) {
                errorDiv.textContent = 'Error: Could not connect to server';
            }
        });
    </script>
</body>
</html>
"""

# ------------------------------------------------------------------
# 3. ROUTES
# ------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return HTML_TEMPLATE

@app.route('/api/predict', methods=['POST'])
def predict():
    # --- 3a. Check if models are loaded ---
    if model is None or scaler is None or label_encoders is None:
        return jsonify({
            'success': False,
            'error': 'Model files are not loaded. Please try again later.'
        }), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # --- 3b. Build input DataFrame ---
        input_df = pd.DataFrame([{
            'LotArea': float(data.get('LotArea', 10000)),
            'OverallQual': float(data.get('OverallQual', 6)),
            'YearBuilt': float(data.get('YearBuilt', 2000)),
            'TotalBsmtSF': float(data.get('TotalBsmtSF', 0)),
            '1stFlrSF': float(data.get('X1stFlrSF', 0)),
            '2ndFlrSF': float(data.get('X2ndFlrSF', 0)),
            'GrLivArea': float(data.get('X1stFlrSF', 0)) + float(data.get('X2ndFlrSF', 0)),
            'GarageArea': float(data.get('GarageArea', 0)),
            'Fireplaces': float(data.get('Fireplaces', 0)),
            'FullBath': float(data.get('FullBath', 0)),
            'HalfBath': float(data.get('HalfBath', 0)),
            'BedroomAbvGr': float(data.get('BedroomAbvGr', 0)),
            'Neighborhood': data.get('Neighborhood', 'NAmes')
        }])

        input_df = input_df.fillna(0)

        # --- 3c. Encode categorical (safe) ---
        if label_encoders is not None:
            for col in label_encoders:
                if col in input_df.columns:
                    try:
                        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                    except Exception as e:
                        print(f"âš ï¸ Encoding error for {col}: {e}")

        # --- 3d. Scale & Predict ---
        if scaler is None:
            return jsonify({'success': False, 'error': 'Scaler not loaded'}), 500
        input_scaled = scaler.transform(input_df)

        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        prediction = float(model.predict(input_scaled)[0])
        prediction = max(prediction, 10000.0)

        return jsonify({
            'success': True,
            'price': round(prediction, 2)
        })

    except Exception as e:
        print(f"âŒ Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

# ------------------------------------------------------------------
# 4. HEALTH CHECK ENDPOINT
# ------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'encoders_loaded': label_encoders is not None,
        'model_path': model_path,
        'model_exists_on_disk': os.path.exists(model_path),
        'scaler_exists_on_disk': os.path.exists(scaler_path),
        'base_dir': BASE_DIR,
        'base_dir_files': os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else []
    })

# ------------------------------------------------------------------
# 5. REQUIRED FOR VERCEL
# ------------------------------------------------------------------
app = app

