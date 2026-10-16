# 🌾 AgroPrice AI — Smart Crop Price Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-EB6C34.svg)](https://xgboost.readthedocs.io/)
[![Pytest](https://img.shields.io/badge/Tests-26%20Passed-22C55E.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-14532D.svg)](LICENSE)

**AgroPrice AI** is a production-grade, full-stack machine learning web application tailored for the Indian agricultural ecosystem. It empowers farmers, agricultural students, traders, APMC commission agents, and agri-businesses to forecast future crop market prices in **Indian Rupees (₹) per quintal (100 kg)**, explore multi-year mandi arrival dynamics, and compare regional markets across India.

---

## 🎨 Four-Color Green UI Theme

AgroPrice AI adheres strictly to an intentional, accessible **four-color green design system**:

| Color Role | Hex Code | Purpose in Application |
|---|---|---|
| **Forest Green** | `#14532D` | Sidebar background, primary titles, section headers, and high-emphasis borders |
| **Fresh Green** | `#22C55E` | Primary action buttons, active navigation, KPI accents, and positive price movements |
| **Sage Green** | `#86A789` | Secondary cards, chart highlights, supporting subtitles, and confidence bands |
| **Soft Mint** | `#F0FDF4` | Application canvas background, card surfaces, and subtle container panels |

*Clean white cards (`#FFFFFF`) and subtle slate dividers (`#E2E8F0`) provide crisp contrast without introducing extraneous brand colors.*

---

## 🚀 Key Features

1. **Intelligent Crop Price Prediction (`pages/1_Crop_Prediction.py`):**
   - Interactive, cascading selectors: *Crop → State → District → Mandi → Month → Year*.
   - Evaluated by validated ML pipelines (**Ridge Regression**, **HistGradientBoosting**, **Random Forest**, **XGBoost**).
   - Point estimates in **₹ per quintal**.
   - **Empirical 90% Confidence Interval** calibrated from out-of-time test residuals.
   - Interactive comparison against the most recent recorded historical market price.
   - Automatic local SQLite logging.

2. **Market Price Dashboard (`pages/2_Market_Dashboard.py`):**
   - High-level KPIs: Minimum, Maximum, and Modal price spreads across Indian mandis.
   - Multi-parameter filtering by Commodity, State, District, and Mandi search.
   - **Live CSV Upload:** Upload any official Agmarknet / OGD India CSV callset with automatic schema validation and instant visual updates.
   - Transparent date and source tags distinguishing historical demonstration data from uploaded files.

3. **Crop Insights & Seasonality (`pages/3_Crop_Insights.py`):**
   - Multi-year price trajectories with min-max spread ribbons.
   - Month-of-year seasonal cycles showing harvest influx dips and off-season storage peaks.
   - Month-over-month (MoM) percentage delta indicators.
   - One-click CSV export of filtered historical series.

4. **Compare Crops & Mandis (`pages/4_Market_Comparison.py`):**
   - Side-by-side bar charts and historical trend overlays across multiple APMC mandis.
   - Price differences in Indian Rupees (₹).
   - **Critical Net Profit Warning:** Honest agricultural reminders factoring in transport freight, diesel, APMC cess (1-2%), loading labor, and transit shrinkage.

5. **Prediction History (`pages/5_Prediction_History.py`):**
   - Local SQLite persistence of previous user predictions.
   - Search by mandi, state, or crop with instant table filtering.
   - Individual record deletion and complete history clearing with confirmation protection.

6. **Model Insights (`pages/6_Model_Insights.py`):**
   - Student-friendly educational guide explaining MAE, RMSE, R², and Time-Aware Splitting.
   - Candidate benchmark table across Ridge, HistGradientBoosting, Random Forest, and XGBoost.
   - Actual vs. Predicted scatter plot with 45° ideal regression reference line.
   - Crop-by-crop performance and error breakdown.
   - Relative feature importance breakdown.

7. **About & Educational Help (`pages/7_About.py`):**
   - Core concepts of agricultural price forecasting.
   - Required prediction inputs explained for first-time farmers.
   - Frequently asked questions (FAQs) and legal disclaimer.

---

## 🏛️ System Architecture & Directory Structure

```
AgroPrice-AI-Smart-Crop-Price-Prediction-Platform/
│
├── app.py                      # Main landing page & hero dashboard
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerized deployment specification
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git exclusions
├── README.md                   # Comprehensive documentation
│
├── data/
│   ├── sample/
│   │   └── agmarknet_crop_prices_sample.csv  # 13,000+ Agmarknet historical records
│   └── database.sqlite         # SQLite database for prediction history
│
├── models/
│   ├── crop_price_pipeline.joblib  # Trained scikit-learn pipeline artifact
│   └── model_metadata.json     # Architecture specs, validation scores & hierarchy
│
├── reports/
│   └── model_evaluation_summary.json  # Comprehensive evaluation report
│
├── scripts/
│   └── generate_sample_data.py # Reproducible multi-year Agmarknet data generator
│
├── src/
│   ├── __init__.py
│   ├── config.py               # 4-Color theme, paths, Indian agricultural constants
│   ├── data_cleaning.py        # Schema validation, crop alias normalization, bounds checking
│   ├── data_loader.py          # CSV ingestion, upload handling, dataset summaries
│   ├── feature_engineering.py  # Cyclical sin/cos month, seasons (Kharif/Rabi/Zaid)
│   ├── train.py                # Time-aware split, candidate model benchmarking, model saving
│   ├── evaluate.py             # MAE, RMSE, R2, MAPE, prediction intervals, crop breakdowns
│   ├── predict.py              # Input validation, inference service, uncertainty bounds
│   ├── market_data_service.py  # Mandi aggregations, time series, seasonality & comparisons
│   ├── database.py             # SQLite CRUD operations
│   └── model_utils.py          # Model artifact loading and validation
│
├── pages/
│   ├── 1_Crop_Prediction.py    # Prediction form & empirical price bands
│   ├── 2_Market_Dashboard.py   # Price spreads, mandi explorer & CSV upload
│   ├── 3_Crop_Insights.py      # Multi-year historical trends & seasonal cycles
│   ├── 4_Market_Comparison.py  # Side-by-side mandi comparison & logistics warning
│   ├── 5_Prediction_History.py # SQLite history manager & search
│   ├── 6_Model_Insights.py     # Student ML guide, model benchmark & diagnostics
│   └── 7_About.py              # Platform documentation, FAQs & disclaimers
│
└── tests/
    ├── __init__.py
    ├── test_data_loader.py     # Ingestion, validation, upload & error handling tests
    ├── test_preprocessing.py   # Crop alias standardization, anomaly cleaning & features
    ├── test_training.py        # Time-aware split, metric calculation & persistence tests
    ├── test_prediction.py      # Inference validation, bounds & unsupported rejection tests
    ├── test_market_service.py  # Time series, seasonal & comparison query tests
    └── test_database.py        # SQLite insert, query, delete & clear tests
```

---

## 🌾 Supported Indian Crops & Mandis

AgroPrice AI provides deep historical training coverage for 10 primary Indian commodities across major agricultural states:

| Commodity | Primary Seasons | Key Covered Mandis & States | Typical Price Base (₹/Qtl) |
|---|---|---|---|
| **Wheat** | Rabi | Khanna (Punjab), Ujjain (MP), Kanpur (UP), Kota (Rajasthan) | ₹ 2,200 – ₹ 2,800 |
| **Rice (Paddy)** | Kharif | Amritsar (Punjab), Meerut (UP), Burdwan (WB), Warangal (Telangana) | ₹ 2,800 – ₹ 3,600 |
| **Onion** | All Season | Lasalgaon & Pune (Maharashtra), Indore (MP), Bengaluru (Karnataka) | ₹ 1,500 – ₹ 3,800 |
| **Tomato** | All Season | Pimpalgaon (Maharashtra), Kolar (Karnataka), Madanapalle (AP) | ₹ 1,200 – ₹ 4,500 |
| **Potato** | Rabi | Agra (UP), Jalandhar (Punjab), Hooghly (WB), Deesa (Gujarat) | ₹ 1,000 – ₹ 2,200 |
| **Soybean** | Kharif | Indore & Ujjain (MP), Nagpur & Jalgaon (Maharashtra), Kota (Rajasthan) | ₹ 4,200 – ₹ 5,600 |
| **Cotton** | Kharif | Rajkot & Gondal (Gujarat), Nagpur (Maharashtra), Adilabad (Telangana) | ₹ 6,400 – ₹ 8,500 |
| **Maize** | Kharif / Zaid | Davanagere (Karnataka), Chhindwara (MP), Gulabbagh (Bihar) | ₹ 1,800 – ₹ 2,500 |
| **Turmeric** | Kharif | Sangli (Maharashtra), Nizamabad (Telangana), Erode (Tamil Nadu) | ₹ 7,500 – ₹ 12,000 |
| **Gram (Chana)** | Rabi | Bhopal (MP), Bikaner (Rajasthan), Latur (Maharashtra), Kanpur (UP) | ₹ 4,800 – ₹ 6,200 |

---

## 💻 Installation & Setup Guide (Windows & VS Code)

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14 installed on your machine.
- Git installed.
- Visual Studio Code (recommended).

### Step 1: Open Terminal in VS Code
Clone the repository or navigate to the project directory:
```powershell
cd "d:\project_aiml\AI\New folder\AgroPrice-AI-Smart-Crop-Price-Prediction-Platform"
```

### Step 2: Create and Activate a Virtual Environment
```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1
```
*(If you encounter execution policy restrictions on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Required Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Generate Training Dataset (Optional if already generated)
```powershell
python scripts/generate_sample_data.py
```

### Step 5: Train and Evaluate the ML Models
Executes time-aware chronological splitting (2021-2024 train, 2025 test), candidate model benchmarking, and saves the champion pipeline:
```powershell
python src/train.py
```

### Step 6: Run Automated Test Suite
Verify that all 26 unit tests pass:
```powershell
pytest -v
```

### Step 7: Launch the Web Application
```powershell
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 🐳 Docker Deployment

To build and run AgroPrice AI inside a lightweight container:

```bash
# Build the Docker container image
docker build -t agroprice-ai:latest .

# Run container on port 8501
docker run -d -p 8501:8501 --name agroprice_app agroprice-ai:latest
```
Access the application at `http://localhost:8501`.

---

## 🔬 Machine Learning Pipeline Details

### Time-Aware Data Splitting (No Data Leakage)
Agricultural price records represent a temporal time series. Traditional random k-fold cross-validation erroneously leaks future market prices into past training sets. AgroPrice AI employs **strict chronological splitting**:
- **Training Period:** 2021-01 to 2024-12 (80% of series)
- **Holdout Test Period:** 2025-01 to 2025-12 (20% out-of-time holdout)

### Candidate Model Benchmarking
We benchmark 4 regression architectures:
1. **Ridge Regression:** Regularized linear estimator capturing baseline price differentials.
2. **HistGradientBoosting Regressor:** Fast histogram-based gradient boosting suited for non-linear seasonal interactions.
3. **Random Forest Regressor:** Multi-tree bagging regressor robust to localized market noise.
4. **XGBoost Regressor:** Extreme gradient boosting optimizing depth-6 trees with sub-sampling.

### Statistically Reliable Uncertainty Bands
Instead of reporting single deterministic point estimates, AgroPrice AI computes an empirical 90% confidence interval:
$$\text{Expected Price Band} = \text{Predicted Modal Price} \pm (1.645 \times \text{Crop Residual Standard Error})$$
This reflects realistic volatility in Indian APMC mandis.

---

## 🧪 Testing Summary

Our test suite in `tests/` contains **26 automated tests** covering:
- Dataset loading, schema validation, and CSV upload parsing (`test_data_loader.py`)
- Commodity name alias standardization, anomaly removal, and cyclical feature engineering (`test_preprocessing.py`)
- Chronological time-series splitting, metric calculation, and pipeline persistence (`test_training.py`)
- Prediction engine validation, statistical bounds, and rejection of unsupported inputs (`test_prediction.py`)
- Market data service time-series, monthly seasonality, and comparative queries (`test_market_service.py`)
- SQLite database insertion, filtering, single-row deletion, and full clearing (`test_database.py`)

Run tests with:
```bash
pytest
```

---

## ⚠️ Advisory & Legal Notice

AgroPrice AI provides statistical price forecasts based on historical market trends and machine learning algorithms. Real-time agricultural prices in India depend on instantaneous variables including daily truck arrivals, weather shocks, export policies, and mandi bidding dynamics. These forecasts are indicative estimates and do not constitute guaranteed financial returns.

---

## 📄 License
This project is open-source and distributed under the MIT License.