import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Features jo form mein hain (sirf 13)
FEATURES = [
    'LotArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF',
    '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea',
    'Fireplaces', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'Neighborhood'
]

# Data load karo
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
df = pd.read_csv(data_path)

# GrLivArea = 1stFlrSF + 2ndFlrSF
df['GrLivArea'] = df['1stFlrSF'] + df['2ndFlrSF']

# Sirf ye columns lo
df = df[FEATURES + ['SalePrice']].copy()
df = df.fillna(0)

# Label encode Neighborhood
label_encoders = {}
le = LabelEncoder()
df['Neighborhood'] = le.fit_transform(df['Neighborhood'].astype(str))
label_encoders['Neighborhood'] = le

# X aur y
X = df[FEATURES]
y = df['SalePrice']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale karo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model train karo
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train_scaled, y_train)

score = model.score(X_test_scaled, y_test)
print(f"Model R2 Score: {score:.4f}")

# Save karo
save_dir = os.path.dirname(__file__)
joblib.dump(model, os.path.join(save_dir, 'model.pkl'))
joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
joblib.dump(label_encoders, os.path.join(save_dir, 'label_encoders.pkl'))
print("Files saved: model.pkl, scaler.pkl, label_encoders.pkl")
