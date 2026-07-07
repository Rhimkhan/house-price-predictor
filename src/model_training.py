"""
model_training.py
-----------------
Trains, evaluates, tunes, and saves the best model for House Price Prediction.

Usage (from project root):
    python src/model_training.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection   import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model      import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics           import mean_squared_error, mean_absolute_error, r2_score
from xgboost                   import XGBRegressor
from lightgbm                  import LGBMRegressor
from catboost                  import CatBoostRegressor

# Local modules
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing  import run_preprocessing, save_artefacts
from feature_engineering import engineer_features, build_neighborhood_rank

warnings.filterwarnings("ignore")
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("seaborn-darkgrid")
sns.set_theme(style="darkgrid", palette="husl")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

ROOT        = os.path.join(os.path.dirname(__file__), "..")
EDA_DIR     = os.path.join(ROOT, "outputs", "eda_plots")
MODEL_DIR   = os.path.join(ROOT, "outputs", "model_results")
SRC_DIR     = os.path.join(ROOT, "src")
OUTPUT_DIR  = os.path.join(ROOT, "outputs")

for d in [EDA_DIR, MODEL_DIR, SRC_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. Load & Preprocess
# ─────────────────────────────────────────────────────────────

def load_and_prepare():
    train_path = os.path.join(ROOT, "data", "train.csv")
    test_path  = os.path.join(ROOT, "data", "test.csv")

    train_raw = pd.read_csv(train_path)
    test_raw  = pd.read_csv(test_path)

    # Build neighborhood rank BEFORE encoding (still has string Neighborhood)
    nbr_rank = build_neighborhood_rank(train_raw)

    (X_train_scaled, X_test_scaled, y,
     test_ids, feature_names,
     scaler, label_encoders) = run_preprocessing(train_path, test_path)

    print(f"\n[✓] X_train: {X_train_scaled.shape}, y: {y.shape}")
    print(f"[✓] X_test : {X_test_scaled.shape}")
    return (X_train_scaled, X_test_scaled, y,
            test_ids, feature_names,
            scaler, label_encoders, nbr_rank, test_raw)



# ─────────────────────────────────────────────────────────────
# 1b. Generate EDA Plots from raw train data
# ─────────────────────────────────────────────────────────────

def generate_eda_plots(train_raw: pd.DataFrame):
    """Generate and save all 5 required EDA plots."""
    print("\n[⏳] Generating EDA plots...")

    # ── missing_values.png ──────────────────────────────────────
    missing = train_raw.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Missing Values Analysis", fontsize=14, fontweight="bold")
    missing.head(20).plot(kind="bar", ax=axes[0], color="#e74c3c", edgecolor="white")
    axes[0].set_title("Top-20 Columns with Missing Values")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Missing Count")
    axes[0].tick_params(axis="x", rotation=60)
    cols_to_show = missing.index[:20]
    sns.heatmap(train_raw[cols_to_show].isnull(), cbar=True,
                yticklabels=False, cmap="Reds", ax=axes[1])
    axes[1].set_title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "missing_values.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] missing_values.png")

    # ── target_distribution.png ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SalePrice Distribution", fontsize=14, fontweight="bold")
    axes[0].hist(train_raw["SalePrice"], bins=40, color="#3498db", edgecolor="white")
    axes[0].set_title("Original Distribution")
    axes[0].set_xlabel("SalePrice ($)")
    log_price = np.log1p(train_raw["SalePrice"])
    axes[1].hist(log_price, bins=40, color="#2ecc71", edgecolor="white")
    axes[1].set_title("Log-Transformed (more normal)")
    axes[1].set_xlabel("log(1 + SalePrice)")
    axes[2].boxplot(train_raw["SalePrice"], patch_artist=True,
                    boxprops=dict(facecolor="#e67e22", alpha=0.7))
    axes[2].set_title("Box Plot — outliers visible")
    axes[2].set_ylabel("SalePrice ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "target_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] target_distribution.png")

    # ── top_correlations.png ────────────────────────────────────
    num_cols = train_raw.select_dtypes(include=[np.number]).columns
    corr = train_raw[num_cols].corr()["SalePrice"].sort_values(ascending=False)
    top_feats = [f for f in corr.head(10).index if f != "SalePrice"][:9]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Top Feature Correlations with SalePrice", fontsize=14, fontweight="bold")
    axes = axes.flatten()
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(top_feats)))
    for i, (feat, c) in enumerate(zip(top_feats, colors)):
        axes[i].scatter(train_raw[feat], train_raw["SalePrice"],
                        alpha=0.35, s=15, color=c)
        axes[i].set_xlabel(feat, fontsize=9)
        axes[i].set_ylabel("SalePrice", fontsize=9)
        axes[i].set_title(f"{feat}  (r={corr[feat]:.3f})", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "top_correlations.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] top_correlations.png")

    # ── categorical_analysis.png ────────────────────────────────
    key_cats = [c for c in ["MSZoning","Street","Neighborhood","SaleType",
                              "SaleCondition","BldgType"] if c in train_raw.columns]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Average SalePrice by Categorical Feature", fontsize=14, fontweight="bold")
    axes = axes.flatten()
    for i, cat in enumerate(key_cats[:6]):
        means = train_raw.groupby(cat)["SalePrice"].mean().sort_values(ascending=False)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(means)))
        axes[i].bar(means.index, means.values, color=colors, edgecolor="white")
        axes[i].set_title(f"Avg SalePrice by {cat}", fontsize=11)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_ylabel("Avg SalePrice ($)")
        axes[i].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "categorical_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] categorical_analysis.png")

    # ── outliers.png ────────────────────────────────────────────
    important = [c for c in ["GrLivArea","TotalBsmtSF","LotArea","GarageArea"]
                 if c in train_raw.columns]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Outlier Detection — Box Plots", fontsize=14, fontweight="bold")
    axes = axes.flatten()
    for i, col in enumerate(important):
        axes[i].boxplot(train_raw[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="#3498db", alpha=0.7),
                        flierprops=dict(marker="o", color="red", alpha=0.4))
        axes[i].set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "outliers.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] outliers.png")

    print(f"[✓] All EDA plots saved → {EDA_DIR}")


# ─────────────────────────────────────────────────────────────
# 2. Define Models
# ─────────────────────────────────────────────────────────────

def get_models():
    return {
        "Linear Regression"  : LinearRegression(),
        "Ridge"              : Ridge(alpha=10),
        "Lasso"              : Lasso(alpha=0.01),
        "ElasticNet"         : ElasticNet(alpha=0.01, l1_ratio=0.5),
        "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost"            : XGBRegressor(n_estimators=100, learning_rate=0.1,
                                             random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM"           : LGBMRegressor(n_estimators=100, learning_rate=0.1,
                                              random_state=42, n_jobs=-1, verbose=-1),
        "CatBoost"           : CatBoostRegressor(iterations=100, learning_rate=0.1,
                                                  random_state=42, verbose=False),
    }



# ─────────────────────────────────────────────────────────────
# 3. Train & Evaluate
# ─────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_val, y_train, y_val, models):
    results = {}
    feat_importances = {}

    for name, model in models.items():
        print(f"\n{'─'*50}\nTraining: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae  = mean_absolute_error(y_val, y_pred)
        r2   = r2_score(y_val, y_pred)

        results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}
        print(f"  RMSE={rmse:,.0f}  MAE={mae:,.0f}  R²={r2:.4f}")

        if hasattr(model, "feature_importances_"):
            feat_importances[name] = model.feature_importances_

    return results, feat_importances


# ─────────────────────────────────────────────────────────────
# 4. Plot Model Comparison
# ─────────────────────────────────────────────────────────────

def plot_model_comparison(results: dict):
    df = pd.DataFrame(results).T
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")

    for ax, metric, color in zip(axes,
                                  ["RMSE", "MAE", "R²"],
                                  ["#e74c3c", "#e67e22", "#2ecc71"]):
        ascending = metric != "R²"
        sorted_df = df.sort_values(metric, ascending=ascending)
        bars = ax.bar(sorted_df.index, sorted_df[metric], color=color, alpha=0.85)
        ax.set_title(f"{metric} (lower better)" if metric != "R²" else "R² (higher better)",
                     fontsize=12)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7)

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved → {path}")


# ─────────────────────────────────────────────────────────────
# 5. Plot Feature Importance
# ─────────────────────────────────────────────────────────────

def plot_feature_importance(feat_importances: dict, feature_names: list):
    model_name = "Random Forest" if "Random Forest" in feat_importances else list(feat_importances.keys())[0]
    importance = feat_importances[model_name]
    indices    = np.argsort(importance)[::-1][:20]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(range(len(indices)),
                   importance[indices],
                   color=plt.cm.viridis(np.linspace(0, 1, len(indices))))
    plt.xticks(range(len(indices)),
               [feature_names[i] for i in indices],
               rotation=45, ha="right", fontsize=9)
    plt.title(f"Top-20 Feature Importances ({model_name})", fontsize=14, fontweight="bold")
    plt.ylabel("Importance")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved → {path}")


# ─────────────────────────────────────────────────────────────
# 6. Plot Actual vs Predicted & Residuals
# ─────────────────────────────────────────────────────────────

def plot_predictions(model, X_val, y_val):
    y_pred = model.predict(X_val)

    # Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(y_val, y_pred, alpha=0.45, color="#3498db", s=20)
    lim = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", lw=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual SalePrice ($)")
    axes[0].set_ylabel("Predicted SalePrice ($)")
    axes[0].set_title("Actual vs Predicted SalePrice", fontsize=13, fontweight="bold")
    axes[0].legend()

    # Residuals
    residuals = y_val - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.45, color="#e74c3c", s=20)
    axes[1].axhline(0, color="black", lw=2, linestyle="--")
    axes[1].set_xlabel("Predicted SalePrice ($)")
    axes[1].set_ylabel("Residuals ($)")
    axes[1].set_title("Residual Plot", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path1 = os.path.join(MODEL_DIR, "actual_vs_predicted.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved → {path1}")


# ─────────────────────────────────────────────────────────────
# 7. Hyperparameter Tuning (XGBoost)
# ─────────────────────────────────────────────────────────────

def tune_xgboost(X_train, y_train):
    print("\n[⏳] Hyperparameter tuning XGBoost (this may take a few minutes)…")
    param_grid = {
        "n_estimators"    : [200, 300],
        "max_depth"       : [3, 5],
        "learning_rate"   : [0.05, 0.1],
        "subsample"       : [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    gs  = GridSearchCV(xgb, param_grid, cv=5,
                       scoring="neg_mean_squared_error",
                       n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"[✓] Best params : {gs.best_params_}")
    print(f"[✓] Best CV RMSE: {np.sqrt(-gs.best_score_):,.0f}")
    return gs.best_estimator_, gs.best_params_


# ─────────────────────────────────────────────────────────────
# 8. Cross-Validation
# ─────────────────────────────────────────────────────────────

def cross_validate(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5,
                                scoring="neg_mean_squared_error", n_jobs=-1)
    cv_rmse = np.sqrt(-cv_scores)
    print(f"\n[CV] RMSE per fold : {cv_rmse.round(0)}")
    print(f"[CV] Mean  RMSE    : {cv_rmse.mean():,.0f}")
    print(f"[CV] Std   RMSE    : {cv_rmse.std():,.0f}")
    return cv_rmse


# ─────────────────────────────────────────────────────────────
# 9. Generate Submission
# ─────────────────────────────────────────────────────────────

def generate_submission(model, X_test_scaled, test_ids):
    preds = model.predict(X_test_scaled)
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
    path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(path, index=False)
    print(f"\n[✓] Submission saved → {path}")
    print(submission.head(10).to_string(index=False))
    return submission


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   HOUSE PRICE PREDICTION — MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load & preprocess
    (X_scaled, X_test_scaled, y,
     test_ids, feature_names,
     scaler, label_encoders,
     nbr_rank, test_raw) = load_and_prepare()

    # 1b. Generate EDA plots from raw training data
    train_raw_path = os.path.join(ROOT, "data", "train.csv")
    train_raw      = pd.read_csv(train_raw_path)
    generate_eda_plots(train_raw)

    # 2. Train / Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"\n[✓] Train: {X_train.shape}  Val: {X_val.shape}")


    # 3. Train all baseline models
    models  = get_models()
    results, feat_importances = train_and_evaluate(X_train, X_val, y_train, y_val, models)

    # 4. Print comparison table
    df_res = pd.DataFrame(results).T.sort_values("RMSE")
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(df_res.to_string())

    # 5. Plots
    plot_model_comparison(results)
    if feat_importances:
        plot_feature_importance(feat_importances, feature_names)

    # 6. Tune best model
    best_model, best_params = tune_xgboost(X_train, y_train)
    plot_predictions(best_model, X_val, y_val)

    # 7. Cross-validation
    cross_validate(best_model, X_scaled, y)

    # 8. Final training on full data
    print("\n[⏳] Training final model on full dataset…")
    final_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
    final_model.fit(X_scaled, y)
    print("[✓] Final model trained.")

    # 9. Save artefacts
    model_path = os.path.join(SRC_DIR, "model.pkl")
    joblib.dump(final_model, model_path)
    print(f"[✓] Model saved → {model_path}")
    save_artefacts(scaler, label_encoders,
                   scaler_path=os.path.join(SRC_DIR, "scaler.pkl"),
                   le_path=os.path.join(SRC_DIR, "label_encoders.pkl"))

    # 10. Generate submission
    generate_submission(final_model, X_test_scaled, test_ids)

    print("\n" + "=" * 60)
    print("   PIPELINE COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
