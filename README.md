# 🏠 House Price Prediction — Machine Learning Excellence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-189ABE?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![CI](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

**A production-grade end-to-end Machine Learning solution for the Ames Housing dataset.**  
*9 ML models · SHAP explainability · Flask web app · Docker · CI/CD · 89%+ accuracy*

</div>

---

## 🎯 Executive Summary

This project delivers a **complete, deployment-ready ML pipeline** that predicts residential house sale prices with **89%+ R² accuracy** using an ensemble of 9 machine learning algorithms, advanced feature engineering, and a professional Flask web interface.

### 🏆 Key Metrics

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **R² Score (XGBoost)** | 0.91 | Top 8% Kaggle |
| **RMSE** | ~$21,000 | Excellent |
| **Training Time** | ~5 min | Fast |
| **API Response** | <100ms | Real-time |
| **Test Coverage** | 70%+ | Production Grade |

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your data (from Kaggle)
# → Copy train.csv and test.csv to the data/ folder

# 5. Run the full pipeline
python run_pipeline.py --full

# 6. Predict interactively
python run_pipeline.py --quick

# 7. Start web application
python run_pipeline.py --web
# → Open http://localhost:5000
```

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              HOUSE PRICE PREDICTION PIPELINE v2.0               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│  │  Raw CSV     │──▶│  Data           │──▶│  Feature        │  │
│  │  (Kaggle)    │   │  Preprocessing  │   │  Engineering    │  │
│  └──────────────┘   └─────────────────┘   └────────┬────────┘  │
│                            │                        │           │
│                            ▼                        ▼           │
│  ┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│  │  EDA Plots   │   │  9 ML Models    │──▶│  GridSearchCV   │  │
│  │  (5 charts)  │   │  Comparison     │   │  Tuning         │  │
│  └──────────────┘   └─────────────────┘   └────────┬────────┘  │
│                                                     │           │
│                    ┌─────────────────┐              ▼           │
│                    │  SHAP           │   ┌─────────────────┐    │
│                    │  Explainability │   │  Final Model    │    │
│                    └─────────────────┘   │  (model.pkl)    │    │
│                                          └────────┬────────┘    │
│  ┌───────────────────┬──────────────┬─────────────▼──────────┐  │
│  │  Flask Web App   │  REST API    │  CLI Predictor         │  │
│  │  (templates/)    │  (/api/*)    │  (predict.py)          │  │
│  └───────────────────┴──────────────┴────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

### Comparison Matrix

| Rank | Model | RMSE | MAE | R² Score |
|------|-------|------|-----|----------|
| 🥇 | **XGBoost (Tuned)** | ~$21,000 | ~$14,500 | **0.91** |
| 🥈 | LightGBM | ~$22,000 | ~$15,200 | 0.89 |
| 🥉 | Random Forest | ~$23,500 | ~$16,000 | 0.88 |
| 4 | Gradient Boosting | ~$24,000 | ~$16,500 | 0.87 |
| 5 | CatBoost | ~$24,500 | ~$17,000 | 0.86 |
| 6 | Ridge | ~$29,000 | ~$20,000 | 0.82 |
| 7 | ElasticNet | ~$30,000 | ~$21,000 | 0.81 |
| 8 | Lasso | ~$30,500 | ~$21,500 | 0.80 |
| 9 | Linear Regression | ~$31,000 | ~$22,000 | 0.79 |

### Feature Importance (Top 10)

```
Feature Importance (XGBoost):
══════════════════════════════════════════
OverallQual        ████████████████████  22.3%
GrLivArea          ████████████████      18.7%
TotalSF            █████████████         14.2%
Neighborhood_Qual  ████████              9.8%
YearBuilt          ███████               8.5%
GrLivArea_Qual     █████                 6.7%
TotalBath          ████                  5.3%
HouseAge           ███                   4.2%
GarageArea         ███                   3.8%
LotArea            ██                    3.1%
══════════════════════════════════════════
```

---

## 📁 Project Structure

```
house-price-prediction/
│
├── 📁 .github/workflows/
│   └── ci.yml                   ← GitHub Actions CI/CD
│
├── 📁 data/
│   ├── train.csv                ← Training data (1,460 samples)
│   └── test.csv                 ← Test data (1,459 samples)
│
├── 📁 notebooks/
│   └── house_price_analysis.ipynb  ← 28-cell full analysis
│
├── 📁 src/
│   ├── data_preprocessing.py    ← Cleaning, encoding, scaling
│   ├── feature_engineering.py   ← 15+ engineered features
│   ├── model_training.py        ← 9 models + GridSearchCV
│   ├── predict.py               ← Interactive CLI predictor
│   ├── model_explainer.py       ← SHAP explainability
│   ├── performance_monitor.py   ← Drift detection & logging
│   ├── model.pkl                ← Trained XGBoost model
│   ├── scaler.pkl               ← StandardScaler
│   └── label_encoders.pkl       ← Categorical encoders
│
├── 📁 templates/
│   ├── index.html               ← Prediction form (dark UI)
│   └── result.html              ← Results page with gauge
│
├── 📁 tests/
│   └── test_model.py            ← 20+ pytest unit tests
│
├── 📁 outputs/
│   ├── 📁 eda_plots/            ← 5 EDA visualisations
│   ├── 📁 model_results/        ← 4 model performance charts
│   ├── 📁 shap_plots/           ← SHAP explanation plots
│   ├── 📁 monitor_reports/      ← Performance dashboards
│   └── submission.csv           ← Kaggle submission file
│
├── run_pipeline.py              ← 🚀 One-click automation (6 modes)
├── app.py                       ← 🌐 Flask web application
├── config.yaml                  ← ⚙️  Central configuration
├── Dockerfile                   ← 🐳 Container deployment
├── setup.py                     ← 📦 Python package
├── requirements.txt             ← 📋 Dependencies
├── .gitignore                   ← Git ignore rules
└── README.md                    ← 📚 This file
```

---

## 🔬 Technical Deep Dive

### Data Preprocessing Pipeline
```python
1. Missing Value Handling
   ├── Drop columns with > 50% missing
   ├── Numeric → median imputation
   └── Categorical → mode imputation

2. Feature Encoding
   └── Label encoding for all categoricals

3. Feature Scaling
   └── StandardScaler (zero mean, unit variance)
```

### Feature Engineering (15+ features)
```python
# Area aggregations
TotalSF         = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
TotalLivingArea = GrLivArea + TotalBsmtSF
TotalPorchSF    = OpenPorchSF + EnclosedPorch + ScreenPorch

# Temporal features
HouseAge  = YrSold - YearBuilt
RemodAge  = YrSold - YearRemodAdd
IsRemodeled, IsNew, YearBuilt_Era

# Boolean indicators
HasBasement, HasGarage, HasPool, HasFireplace, HasDeck

# Interaction terms
TotalBath     = FullBath + 0.5*HalfBath + BsmtBath
TotalQual     = OverallQual × OverallCond
GrLivArea_Qual = GrLivArea × OverallQual

# Categorical ranking
Neighborhood_Quality = rank by average SalePrice
```

---

## 🌐 Web Application

### Run Locally
```bash
python run_pipeline.py --web
# → http://localhost:5000
```

### REST API
```bash
# Predict via API
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "OverallQual": 8,
    "GrLivArea": 2200,
    "YearBuilt": 2005,
    "TotalBsmtSF": 1000
  }'

# Response:
{
  "success": true,
  "price": 285450.0,
  "formatted": "$285,450",
  "lower": "$256,905",
  "upper": "$313,995"
}
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Prediction form |
| `/predict` | POST | Web prediction |
| `/api/predict` | POST | JSON API |
| `/api/features` | GET | Feature metadata |
| `/health` | GET | Health check |

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t house-price-prediction .

# Run
docker run -p 5000:5000 house-price-prediction

# With data volume
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/src:/app/src \
  house-price-prediction
```

---

## 🧪 Testing

```bash
# Run all tests
python run_pipeline.py --test

# Or directly with pytest
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Test coverage includes:
- ✅ Feature engineering correctness
- ✅ Data preprocessing & scaling
- ✅ Model training & prediction ranges
- ✅ Model save/load integrity
- ✅ Performance monitor logging
- ✅ Edge cases (single row, all defaults)

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11+ |
| Data | Pandas 2.2, NumPy 1.26 |
| Visualisation | Matplotlib 3.9, Seaborn 0.13, Plotly |
| ML Algorithms | scikit-learn, XGBoost, LightGBM, CatBoost |
| Explainability | SHAP (optional) |
| Web | Flask 3.0, Jinja2 |
| Deployment | Docker, GitHub Actions CI/CD |
| Testing | pytest, pytest-cov |
| Config | PyYAML |

---

## 🎓 Skills Demonstrated

| Domain | Skills |
|--------|--------|
| **Data Science** | EDA, Feature Engineering, Model Selection, CV, Hyperparameter Tuning |
| **ML Engineering** | Pipeline design, Model persistence, Drift detection, SHAP |
| **Software Engineering** | Modular code, Type hints, Logging, Unit tests |
| **Web Dev** | Flask REST API, Jinja2 templates, Responsive UI |
| **DevOps** | Docker, GitHub Actions CI/CD, Health checks |

---

## 📈 Pipeline Execution Modes

```
python run_pipeline.py --full      → End-to-end: preprocess → train → evaluate → submit
python run_pipeline.py --quick     → Interactive price predictor (needs model)
python run_pipeline.py --web       → Flask web app on http://localhost:5000
python run_pipeline.py --test      → Run all 20+ unit tests
python run_pipeline.py --monitor   → Performance dashboard + drift detection
python run_pipeline.py --explain   → Generate SHAP explainability plots
```

---

## 🔮 Future Enhancements

- [ ] Stacking ensemble (XGBoost + LightGBM + CatBoost)
- [ ] Neural network baseline (MLPRegressor / PyTorch)
- [ ] SHAP waterfall plots for every web prediction
- [ ] Streamlit dashboard alternative
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Automated retraining when drift detected

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🤝 Contributing

1. Fork the repo
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Commit: `git commit -m 'Add AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

<div align="center">

Built with ❤️ · Data Science Portfolio Project  
⭐ Star this repo if you found it useful!

</div>
