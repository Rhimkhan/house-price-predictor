"""
tests/test_model.py
===================
Unit & integration tests for the House Price Prediction project.

Run with:
    pytest tests/test_model.py -v
    pytest tests/test_model.py -v --tb=short
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.join(os.path.dirname(__file__), "..")

# ════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    """Small synthetic dataframe resembling Ames housing data."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "Id"          : range(1, n+1),
        "OverallQual" : np.random.randint(1, 11, n),
        "OverallCond" : np.random.randint(1, 10, n),
        "GrLivArea"   : np.random.randint(800, 3000, n),
        "TotalBsmtSF" : np.random.randint(0, 2000, n).astype(float),
        "1stFlrSF"    : np.random.randint(400, 2000, n),
        "2ndFlrSF"    : np.random.randint(0, 1000, n),
        "GarageArea"  : np.random.randint(0, 900, n).astype(float),
        "YearBuilt"   : np.random.randint(1920, 2020, n),
        "YearRemodAdd": np.random.randint(1960, 2022, n),
        "YrSold"      : np.random.randint(2006, 2011, n),
        "FullBath"    : np.random.randint(0, 4, n),
        "HalfBath"    : np.random.randint(0, 2, n),
        "BsmtFullBath": np.random.randint(0, 2, n),
        "BsmtHalfBath": np.random.randint(0, 1, n),
        "Fireplaces"  : np.random.randint(0, 3, n),
        "GarageCars"  : np.random.randint(0, 4, n).astype(float),
        "LotArea"     : np.random.randint(3000, 50000, n),
        "PoolArea"    : np.zeros(n),
        "WoodDeckSF"  : np.random.randint(0, 400, n),
        "OpenPorchSF" : np.random.randint(0, 200, n),
        "EnclosedPorch": np.zeros(n),
        "ScreenPorch" : np.zeros(n),
        "Neighborhood": np.random.choice(["NAmes","CollgCr","OldTown"], n),
        "SalePrice"   : np.random.randint(80000, 450000, n),
    })


@pytest.fixture
def sample_train_path(tmp_path, sample_df):
    path = tmp_path / "train.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_test_path(tmp_path, sample_df):
    test_df = sample_df.drop(columns=["SalePrice"])
    path = tmp_path / "test.csv"
    test_df.to_csv(path, index=False)
    return str(path)


# ════════════════════════════════════════════════════════════
# Feature Engineering Tests
# ════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def test_imports(self):
        from feature_engineering import engineer_features
        assert callable(engineer_features)

    def test_totalsf_created(self, sample_df):
        from feature_engineering import engineer_features
        result = engineer_features(sample_df.drop(columns=["SalePrice"]))
        assert "TotalSF" in result.columns

    def test_totalSF_values_positive(self, sample_df):
        from feature_engineering import engineer_features
        result = engineer_features(sample_df.drop(columns=["SalePrice"]))
        assert (result["TotalSF"] >= 0).all()

    def test_houseage_non_negative(self, sample_df):
        from feature_engineering import engineer_features
        result = engineer_features(sample_df.drop(columns=["SalePrice"]))
        if "HouseAge" in result.columns:
            assert (result["HouseAge"] >= 0).all()

    def test_boolean_flags_binary(self, sample_df):
        from feature_engineering import engineer_features
        result = engineer_features(sample_df.drop(columns=["SalePrice"]))
        for flag in ["HasBasement","HasGarage","HasPool","HasFireplace"]:
            if flag in result.columns:
                assert result[flag].isin([0, 1]).all(), f"{flag} is not binary"

    def test_no_nan_after_engineering(self, sample_df):
        from feature_engineering import engineer_features
        result = engineer_features(sample_df.drop(columns=["SalePrice"]))
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        nan_count = result[numeric_cols].isnull().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values after feature engineering"

    def test_shape_increases(self, sample_df):
        from feature_engineering import engineer_features
        original_cols = sample_df.shape[1] - 1  # remove SalePrice
        result = engineer_features(sample_df.drop(columns=["SalePrice"]))
        assert result.shape[1] > original_cols, "Feature engineering should add columns"


# ════════════════════════════════════════════════════════════
# Data Preprocessing Tests
# ════════════════════════════════════════════════════════════

class TestDataPreprocessing:

    def test_preprocessing_runs(self, sample_train_path, sample_test_path):
        from data_preprocessing import run_preprocessing
        result = run_preprocessing(sample_train_path, sample_test_path)
        assert result is not None
        assert len(result) >= 5

    def test_output_shapes(self, sample_train_path, sample_test_path):
        from data_preprocessing import run_preprocessing
        (X_train, X_test, y, test_ids, feature_names, scaler, le) = run_preprocessing(
            sample_train_path, sample_test_path)
        assert X_train.ndim == 2
        assert len(y) == X_train.shape[0]

    def test_no_nans_in_output(self, sample_train_path, sample_test_path):
        from data_preprocessing import run_preprocessing
        (X_train, X_test, *_) = run_preprocessing(sample_train_path, sample_test_path)
        assert not np.isnan(X_train).any(), "NaN in X_train after preprocessing"
        assert not np.isnan(X_test).any(), "NaN in X_test after preprocessing"

    def test_scaler_fitted(self, sample_train_path, sample_test_path):
        from data_preprocessing import run_preprocessing
        from sklearn.preprocessing import StandardScaler
        (*_, scaler, _) = run_preprocessing(sample_train_path, sample_test_path)
        assert hasattr(scaler, "mean_"), "Scaler was not fitted"


# ════════════════════════════════════════════════════════════
# Model Tests
# ════════════════════════════════════════════════════════════

class TestModel:

    def test_xgboost_trains(self, sample_train_path, sample_test_path):
        from data_preprocessing import run_preprocessing
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        (X, X_test, y, *_) = run_preprocessing(sample_train_path, sample_test_path)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        model = XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        assert preds.shape == y_val.shape
        r2 = r2_score(y_val, preds)
        assert r2 > 0.0, f"R² should be > 0, got {r2:.4f}"

    def test_prediction_range(self, sample_train_path, sample_test_path):
        """Predictions should be within a plausible price range."""
        from data_preprocessing import run_preprocessing
        from xgboost import XGBRegressor

        (X, X_test, y, *_) = run_preprocessing(sample_train_path, sample_test_path)
        model = XGBRegressor(n_estimators=30, random_state=42, verbosity=0)
        model.fit(X, y)
        preds = model.predict(X_test)

        assert (preds > 0).all(), "All predictions should be positive"
        assert (preds < 10_000_000).all(), "Predictions too large"

    def test_model_saved_and_loaded(self, tmp_path, sample_train_path, sample_test_path):
        from data_preprocessing import run_preprocessing
        from xgboost import XGBRegressor
        import joblib

        (X, _, y, *_) = run_preprocessing(sample_train_path, sample_test_path)
        model = XGBRegressor(n_estimators=20, random_state=42, verbosity=0)
        model.fit(X, y)

        path = str(tmp_path / "model.pkl")
        joblib.dump(model, path)

        loaded = joblib.load(path)
        preds = loaded.predict(X)
        assert preds.shape[0] == X.shape[0]


# ════════════════════════════════════════════════════════════
# Performance Monitor Tests
# ════════════════════════════════════════════════════════════

class TestPerformanceMonitor:

    def test_log_prediction(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from performance_monitor import PerformanceMonitor
        monitor = PerformanceMonitor()
        monitor.log_file = str(tmp_path / "pred_log.jsonl")
        monitor._entries = []

        entry_id = monitor.log_prediction(
            features={"OverallQual": 7, "GrLivArea": 1800},
            predicted_price=250000.0
        )
        assert isinstance(entry_id, str)
        assert len(monitor._entries) == 1

    def test_get_stats_empty(self):
        from performance_monitor import PerformanceMonitor
        monitor = PerformanceMonitor()
        monitor._entries = []
        stats = monitor.get_stats()
        assert stats["total_predictions"] == 0

    def test_drift_not_enough_data(self):
        from performance_monitor import PerformanceMonitor
        monitor = PerformanceMonitor()
        monitor._entries = [{"predicted_price": 200000}] * 5
        result = monitor.detect_drift()
        assert result["drift_detected"] is False


# ════════════════════════════════════════════════════════════
# Edge Case Tests
# ════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_row_prediction(self, sample_train_path, sample_test_path):
        from data_preprocessing import run_preprocessing
        from xgboost import XGBRegressor

        (X, X_test, y, *_) = run_preprocessing(sample_train_path, sample_test_path)
        model = XGBRegressor(n_estimators=20, random_state=42, verbosity=0)
        model.fit(X, y)

        single = X_test[:1]
        pred   = model.predict(single)
        assert pred.shape == (1,)
        assert not np.isnan(pred[0])

    def test_all_default_features(self, sample_train_path, sample_test_path):
        """Prediction with all-default feature values should not crash."""
        from data_preprocessing import run_preprocessing
        from xgboost import XGBRegressor
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        (X, _, y, _, feature_names, scaler, _) = run_preprocessing(
            sample_train_path, sample_test_path)
        model = XGBRegressor(n_estimators=20, random_state=42, verbosity=0)
        model.fit(X, y)

        default_row = np.zeros((1, X.shape[1]))
        pred = model.predict(default_row)
        assert not np.isnan(pred[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
