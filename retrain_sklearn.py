"""
Retrain model using sklearn GradientBoostingRegressor (no XGBoost).
This produces a model.pkl that Vercel can load WITHOUT the 191MB xgboost package.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Retraining with sklearn GradientBoostingRegressor")
print("  (Vercel-compatible — no XGBoost needed)")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
print(f"[OK] Train: {train.shape}, Test: {test.shape}")

test_ids = test['Id'].values

# ── Drop high-missing columns ──────────────────────────────────
def drop_high_missing(df, thresh=50):
    pct = df.isnull().sum() / len(df) * 100
    cols = pct[pct > thresh].index.tolist()
    df = df.drop(columns=cols)
    if cols:
        print(f"[OK] Dropped: {cols}")
    return df

train = drop_high_missing(train)
test  = drop_high_missing(test)

# ── Fill missing ───────────────────────────────────────────────
for col in train.select_dtypes(include=[np.number]).columns:
    train[col] = train[col].fillna(train[col].median())
for col in train.select_dtypes(include='object').columns:
    train[col] = train[col].fillna(train[col].mode()[0])

for col in test.select_dtypes(include=[np.number]).columns:
    test[col] = test[col].fillna(test[col].median())
for col in test.select_dtypes(include='object').columns:
    test[col] = test[col].fillna(test[col].mode()[0])

# ── Encode categoricals ────────────────────────────────────────
label_encoders = {}
cat_cols = train.select_dtypes(include='object').columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col] if col in test.columns else train[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    if col in test.columns:
        test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

print(f"[OK] Encoded {len(cat_cols)} categorical columns")

# ── Prepare X, y ──────────────────────────────────────────────
y = train['SalePrice'].values
drop = [c for c in ['SalePrice', 'Id', 'SalePrice_log'] if c in train.columns]
X = train.drop(columns=drop)

test_drop = [c for c in ['Id'] if c in test.columns]
X_test = test.drop(columns=test_drop, errors='ignore')
X_test = X_test.reindex(columns=X.columns, fill_value=0)

feature_names = X.columns.tolist()

# ── Scale ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled      = scaler.fit_transform(X.values)
X_test_scaled = scaler.transform(X_test.values)
print("[OK] Features scaled")

# ── Train / evaluate ───────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42,
    verbose=0
)
print("\n[...] Training GradientBoostingRegressor (this takes ~2 min)...")
model.fit(X_tr, y_tr)

y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2   = r2_score(y_val, y_pred)
print(f"[OK] Validation  RMSE = ${rmse:,.0f}  |  R² = {r2:.4f}")

# ── Retrain on full data ───────────────────────────────────────
print("[...] Retraining on full dataset...")
model.fit(X_scaled, y)

# ── Save artefacts ─────────────────────────────────────────────
os.makedirs('src', exist_ok=True)
joblib.dump(model,          'src/model.pkl')
joblib.dump(scaler,         'src/scaler.pkl')
joblib.dump(label_encoders, 'src/label_encoders.pkl')
joblib.dump(feature_names,  'src/feature_names.pkl')

print(f"\n[OK] Saved src/model.pkl         ({os.path.getsize('src/model.pkl')//1024} KB)")
print(f"[OK] Saved src/scaler.pkl         ({os.path.getsize('src/scaler.pkl')//1024} KB)")
print(f"[OK] Saved src/label_encoders.pkl ({os.path.getsize('src/label_encoders.pkl')//1024} KB)")

# ── Submission ─────────────────────────────────────────────────
preds = model.predict(X_test_scaled)
sub = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
os.makedirs('outputs', exist_ok=True)
sub.to_csv('outputs/submission_sklearn.csv', index=False)
print(f"[OK] Saved outputs/submission_sklearn.csv")

print("\n" + "=" * 60)
print("  DONE — sklearn model saved. Ready for Vercel deployment.")
print("  Bundle will be ~280 MB (no XGBoost needed)")
print("=" * 60)
