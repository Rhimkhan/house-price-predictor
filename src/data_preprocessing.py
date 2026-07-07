import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(train_path='data/train.csv', test_path='data/test.csv'):
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        print(f"[OK] Train: {train.shape}, Test: {test.shape}")
        return train, test
    except FileNotFoundError:
        print("[ERROR] Dataset not found! Download from Kaggle.")
        return None, None

def handle_missing_values(df):
    df_copy = df.copy()
    missing_pct = df_copy.isnull().sum() / len(df_copy) * 100
    cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
    if cols_to_drop:
        df_copy = df_copy.drop(columns=cols_to_drop)
        print(f"[OK] Dropped columns: {cols_to_drop}")

    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_copy[col].isnull().any():
            median_val = df_copy[col].median()
            df_copy[col] = df_copy[col].fillna(median_val)

    categorical_cols = df_copy.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_copy[col].isnull().any():
            mode_val = df_copy[col].mode()[0]
            df_copy[col] = df_copy[col].fillna(mode_val)

    return df_copy

def encode_categorical(train_df, test_df=None, label_encoders=None):
    train_copy = train_df.copy()
    test_copy = test_df.copy() if test_df is not None else None

    categorical_cols = train_copy.select_dtypes(include=['object']).columns

    if label_encoders is None:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            train_copy[col] = le.fit_transform(train_copy[col].astype(str))
            label_encoders[col] = le

        if test_copy is not None:
            for col in categorical_cols:
                if col in label_encoders and col in test_copy.columns:
                    le = label_encoders[col]
                    test_copy[col] = test_copy[col].astype(str)
                    for label in test_copy[col].unique():
                        if label not in le.classes_:
                            le.classes_ = np.append(le.classes_, label)
                    test_copy[col] = le.transform(test_copy[col])

        return train_copy, test_copy, label_encoders
    else:
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                train_copy[col] = train_copy[col].astype(str)
                for label in train_copy[col].unique():
                    if label not in le.classes_:
                        le.classes_ = np.append(le.classes_, label)
                train_copy[col] = le.transform(train_copy[col])

        if test_copy is not None:
            for col in categorical_cols:
                if col in label_encoders and col in test_copy.columns:
                    le = label_encoders[col]
                    test_copy[col] = test_copy[col].astype(str)
                    for label in test_copy[col].unique():
                        if label not in le.classes_:
                            le.classes_ = np.append(le.classes_, label)
                    test_copy[col] = le.transform(test_copy[col])

        return train_copy, test_copy, label_encoders

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[OK] Features scaled")
    return X_train_scaled, X_test_scaled, scaler

def run_preprocessing(train_path='data/train.csv', test_path='data/test.csv', target_col='SalePrice'):
    train, test = load_data(train_path, test_path)

    if train is None or test is None:
        return None, None, None, None, None, None, None

    test_ids = test['Id'].values if 'Id' in test.columns else None

    train_clean = handle_missing_values(train)
    test_clean = handle_missing_values(test)

    train_enc, test_enc, label_encoders = encode_categorical(train_clean, test_clean)

    y_train = train_enc[target_col].values if target_col in train_enc.columns else None

    drop_cols = [c for c in [target_col, 'Id', 'SalePrice_log'] if c in train_enc.columns]
    X_train = train_enc.drop(columns=drop_cols)

    test_drop_cols = [c for c in ['Id'] if c in test_enc.columns]
    X_test = test_enc.drop(columns=test_drop_cols, errors='ignore')

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    feature_names = X_train.columns.tolist()

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train.values, X_test.values)

    print(f"[OK] Preprocessing complete: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test")

    return (X_train_scaled, X_test_scaled, y_train, test_ids, feature_names, scaler, label_encoders)

def save_artefacts(scaler, label_encoders, scaler_path='src/scaler.pkl', le_path='src/label_encoders.pkl'):
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoders, le_path)
    print(f"[OK] Saved artefacts")

def load_artefacts(scaler_path='src/scaler.pkl', le_path='src/label_encoders.pkl'):
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(le_path)
    return scaler, label_encoders

if __name__ == '__main__':
    result = run_preprocessing()
    if result[0] is not None:
        save_artefacts(result[5], result[6])
        print("[OK] Preprocessing completed successfully!")
