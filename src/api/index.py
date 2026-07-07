from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
import traceback

app = Flask(__name__)

# Absolute path — Vercel pe /var/task/src/model.pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level: src/api -> src
BASE_DIR = os.path.dirname(BASE_DIR)

model_path    = os.path.join(BASE_DIR, 'model.pkl')
scaler_path   = os.path.join(BASE_DIR, 'scaler.pkl')
encoders_path = os.path.join(BASE_DIR, 'label_encoders.pkl')

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