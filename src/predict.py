"""
predict.py
----------
Interactive CLI for predicting house sale price by entering feature values.

Usage (from project root):
    python src/predict.py

The script loads the trained model (src/model.pkl), scaler (src/scaler.pkl),
and label-encoders (src/label_encoders.pkl), then prompts the user for key
house features and outputs the predicted SalePrice.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT    = os.path.join(os.path.dirname(__file__), "..")
SRC_DIR = os.path.dirname(__file__)

# ─────────────────────────────────────────────────────────────
# Feature template (same columns the model was trained on)
# ─────────────────────────────────────────────────────────────

# We ask the user about the MOST IMPORTANT features only,
# and fill the rest with sensible defaults.

IMPORTANT_FEATURES = [
    # (display_name,  internal_col,   dtype,  default,  hint)
    ("Overall Quality (1-10)",          "OverallQual",    int,   5,     "1=Poor … 10=Excellent"),
    ("Overall Condition (1-10)",        "OverallCond",    int,   5,     "1=Very Poor … 9=Excellent"),
    ("Above-Ground Living Area (sqft)", "GrLivArea",      float, 1500,  "e.g. 1500"),
    ("Total Basement Area (sqft)",      "TotalBsmtSF",    float, 800,   "0 if no basement"),
    ("1st Floor Area (sqft)",           "1stFlrSF",       float, 900,   "e.g. 900"),
    ("2nd Floor Area (sqft)",           "2ndFlrSF",       float, 0,     "0 if single-storey"),
    ("Garage Area (sqft)",              "GarageArea",     float, 400,   "0 if no garage"),
    ("Year Built",                      "YearBuilt",      int,   1990,  "e.g. 1990"),
    ("Year Remodelled / Added",         "YearRemodAdd",   int,   1990,  "same as YearBuilt if no remodel"),
    ("Year Sold",                       "YrSold",         int,   2010,  "e.g. 2010"),
    ("Full Baths (above ground)",       "FullBath",       int,   2,     "e.g. 2"),
    ("Half Baths (above ground)",       "HalfBath",       int,   0,     "e.g. 0 or 1"),
    ("Basement Full Baths",             "BsmtFullBath",   int,   0,     "e.g. 0 or 1"),
    ("Basement Half Baths",             "BsmtHalfBath",   int,   0,     "e.g. 0 or 1"),
    ("Number of Fireplaces",            "Fireplaces",     int,   1,     "e.g. 0, 1, 2"),
    ("Garage Cars Capacity",            "GarageCars",     int,   2,     "e.g. 2"),
    ("Pool Area (sqft)",                "PoolArea",       float, 0,     "0 if no pool"),
    ("Lot Area (sqft)",                 "LotArea",        float, 8000,  "e.g. 8000"),
    ("Open Porch Area (sqft)",          "OpenPorchSF",    float, 0,     "e.g. 0 or 60"),
    ("Wood Deck Area (sqft)",           "WoodDeckSF",     float, 0,     "e.g. 0 or 200"),
    ("Neighbourhood (text)",            "Neighborhood",   str,   "NAmes","e.g. NAmes, CollgCr, OldTown"),
    ("MS Zoning",                       "MSZoning",       str,   "RL",  "RL=Residential Low Density"),
    ("Sale Condition",                  "SaleCondition",  str,   "Normal","Normal, Abnorml, Partial …"),
]

# Numeric defaults for all OTHER columns not asked above
OTHER_NUMERIC_DEFAULTS = {
    "MSSubClass"    : 20,
    "LotFrontage"   : 70,
    "OverallQual"   : 5,
    "MasVnrArea"    : 0,
    "BsmtFinSF1"    : 0,
    "BsmtFinSF2"    : 0,
    "BsmtUnfSF"     : 0,
    "Enclosed Porch": 0,
    "3SsnPorch"     : 0,
    "ScreenPorch"   : 0,
    "MiscVal"       : 0,
    "MoSold"        : 6,
    "TotRmsAbvGrd"  : 6,
    "BedroomAbvGr"  : 3,
    "KitchenAbvGr"  : 1,
}

OTHER_CAT_DEFAULTS = {
    "Street"       : "Pave",
    "Alley"        : "None",
    "LotShape"     : "Reg",
    "LandContour"  : "Lvl",
    "Utilities"    : "AllPub",
    "LotConfig"    : "Inside",
    "LandSlope"    : "Gtl",
    "Condition1"   : "Norm",
    "Condition2"   : "Norm",
    "BldgType"     : "1Fam",
    "HouseStyle"   : "1Story",
    "RoofStyle"    : "Gable",
    "RoofMatl"     : "CompShg",
    "Exterior1st"  : "VinylSd",
    "Exterior2nd"  : "VinylSd",
    "MasVnrType"   : "None",
    "ExterQual"    : "TA",
    "ExterCond"    : "TA",
    "Foundation"   : "PConc",
    "BsmtQual"     : "TA",
    "BsmtCond"     : "TA",
    "BsmtExposure" : "No",
    "BsmtFinType1" : "GLQ",
    "BsmtFinType2" : "Unf",
    "Heating"      : "GasA",
    "HeatingQC"    : "Ex",
    "CentralAir"   : "Y",
    "Electrical"   : "SBrkr",
    "KitchenQual"  : "TA",
    "Functional"   : "Typ",
    "FireplaceQu"  : "TA",
    "GarageType"   : "Attchd",
    "GarageFinish" : "Unf",
    "GarageQual"   : "TA",
    "GarageCond"   : "TA",
    "PavedDrive"   : "Y",
    "SaleType"     : "WD",
    "PoolQC"       : "None",
    "Fence"        : "None",
    "MiscFeature"  : "None",
}


# ─────────────────────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────────────────────

def load_model_artefacts():
    model_path  = os.path.join(SRC_DIR, "model.pkl")
    scaler_path = os.path.join(SRC_DIR, "scaler.pkl")
    le_path     = os.path.join(SRC_DIR, "label_encoders.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, le_path]):
        print("\n[✗] Model files not found.")
        print("    Please run:  python src/model_training.py\n")
        sys.exit(1)

    model          = joblib.load(model_path)
    scaler         = joblib.load(scaler_path)
    label_encoders = joblib.load(le_path)
    return model, scaler, label_encoders


# ─────────────────────────────────────────────────────────────
# Interactive input
# ─────────────────────────────────────────────────────────────

def _prompt(display_name: str, dtype, default, hint: str):
    """Prompt user for a single feature value with a default fallback."""
    hint_str = f"  [{hint}]" if hint else ""
    try:
        raw = input(f"  {display_name}{hint_str} (default={default}): ").strip()
        if raw == "":
            return default
        return dtype(raw)
    except (ValueError, EOFError):
        print(f"    ⚠ Invalid input — using default ({default})")
        return default


def gather_user_inputs() -> dict:
    """Collect feature values from the user interactively."""
    print("\n" + "═" * 60)
    print("  HOUSE PRICE PREDICTION — Enter House Details")
    print("  (Press Enter to accept the default value shown)")
    print("═" * 60 + "\n")

    row = {}
    for display_name, col, dtype, default, hint in IMPORTANT_FEATURES:
        row[col] = _prompt(display_name, dtype, default, hint)

    # Fill other numeric defaults
    for col, val in OTHER_NUMERIC_DEFAULTS.items():
        if col not in row:
            row[col] = val

    # Fill other categorical defaults
    for col, val in OTHER_CAT_DEFAULTS.items():
        if col not in row:
            row[col] = val

    return row


# ─────────────────────────────────────────────────────────────
# Feature engineering (mirrors what was done during training)
# ─────────────────────────────────────────────────────────────

def apply_feature_engineering(row: dict) -> dict:
    """Compute derived features on the raw input dict."""
    r = row.copy()
    r["TotalSF"]          = r.get("TotalBsmtSF", 0) + r.get("1stFlrSF", 0) + r.get("2ndFlrSF", 0)
    r["TotalPorchSF"]     = r.get("OpenPorchSF", 0) + r.get("ScreenPorch", 0)
    r["TotalLivingArea"]  = r.get("GrLivArea", 0)   + r.get("TotalBsmtSF", 0)
    r["HouseAge"]         = max(r.get("YrSold", 2010) - r.get("YearBuilt", 1990), 0)
    r["RemodAge"]         = max(r.get("YrSold", 2010) - r.get("YearRemodAdd", 1990), 0)
    r["IsRemodeled"]      = int(r.get("YearRemodAdd", 0) > r.get("YearBuilt", 0))
    r["IsNew"]            = int(r.get("YrSold", 0) == r.get("YearBuilt", 0))
    r["HasBasement"]      = int(r.get("TotalBsmtSF", 0) > 0)
    r["HasGarage"]        = int(r.get("GarageArea", 0) > 0)
    r["HasPool"]          = int(r.get("PoolArea", 0) > 0)
    r["HasFireplace"]     = int(r.get("Fireplaces", 0) > 0)
    r["HasDeck"]          = int((r.get("WoodDeckSF", 0) + r.get("OpenPorchSF", 0)) > 0)
    r["TotalBath"]        = (r.get("FullBath", 0) + 0.5 * r.get("HalfBath", 0) +
                             r.get("BsmtFullBath", 0) + 0.5 * r.get("BsmtHalfBath", 0))
    r["TotalQual"]        = r.get("OverallQual", 5) * r.get("OverallCond", 5)
    r["GrLivArea_Qual"]   = r.get("GrLivArea", 1500) * r.get("OverallQual", 5)
    r["QualCondRatio"]    = r.get("OverallQual", 5) / (r.get("OverallCond", 5) + 1)
    r["PricePerSF"]       = r.get("OverallQual", 5) / max(r.get("TotalSF", 1), 1)
    r["Neighborhood_Quality"] = 0    # unknown for inference
    # Year built era
    yr = r.get("YearBuilt", 1990)
    era_bins   = [0, 1900, 1950, 1970, 1990, 2000, 2010, 2030]
    era_labels = [0,    1,    2,    3,    4,    5,    6]
    for i, (lo, hi) in enumerate(zip(era_bins[:-1], era_bins[1:])):
        if lo < yr <= hi:
            r["YearBuilt_Era"] = era_labels[i]
            break
    else:
        r["YearBuilt_Era"] = 0
    return r


# ─────────────────────────────────────────────────────────────
# Preprocess & Predict
# ─────────────────────────────────────────────────────────────

def predict_price(row: dict, model, scaler, label_encoders) -> float:
    """Transform one input dict into a price prediction."""
    row = apply_feature_engineering(row)

    # Encode categorical columns using saved encoders
    for col, le in label_encoders.items():
        if col in row:
            val = str(row[col])
            if val not in le.classes_:
                val = le.classes_[0]   # fallback to first known class
            row[col] = le.transform([val])[0]

    # Build a DataFrame with ONE row
    df = pd.DataFrame([row])

    # Drop target-like columns if accidentally present
    for col in ["SalePrice", "Id", "SalePrice_log"]:
        df.drop(columns=[col], errors="ignore", inplace=True)

    # Ensure column order matches scaler's expected input
    n_features = scaler.n_features_in_
    # Pad or trim columns
    if df.shape[1] < n_features:
        for _ in range(n_features - df.shape[1]):
            df[f"_pad_{_}"] = 0
    df = df.iloc[:, :n_features]

    # Convert to numeric, fill NaN
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_scaled = scaler.transform(df.values)
    price    = model.predict(X_scaled)[0]
    return float(price)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  🏠  HOUSE PRICE PREDICTION")
    print("═" * 60)

    # Load model artefacts
    model, scaler, label_encoders = load_model_artefacts()
    print("[✓] Model loaded successfully.\n")

    while True:
        # Gather inputs
        user_input = gather_user_inputs()

        # Predict
        try:
            predicted_price = predict_price(user_input, model, scaler, label_encoders)

            print("\n" + "═" * 60)
            print(f"  💰  PREDICTED SALE PRICE : ${predicted_price:>12,.0f}")
            print("═" * 60 + "\n")
        except Exception as e:
            print(f"\n[✗] Prediction failed: {e}\n")

        # Ask to predict another
        again = input("Predict another house? (y/n): ").strip().lower()
        if again != "y":
            print("\n[✓] Thank you! Goodbye.\n")
            break


if __name__ == "__main__":
    main()
