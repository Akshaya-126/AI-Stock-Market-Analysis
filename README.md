# 📈 AI Stock Market Analysis & Forecasting System

**AI Stock Market Analysis & Forecasting System** is an end-to-end AI-powered financial analytics application that collects historical stock-market data, generates technical indicators, performs machine-learning-based market direction classification, evaluates deep-learning time-series forecasting models, analyzes historical risk, and presents the results through an interactive **Streamlit dashboard**.

The system combines:

**yfinance · Technical Analysis · Machine Learning · XGBoost · LSTM · GRU · Risk Analysis · Plotly · Streamlit · GitHub · Pytest**

The project is designed for **experimental market analytics, educational purposes, and AI/ML portfolio demonstration**.

---

# 🚀 Live Dashboard

https://ai-stock-market-analysis.streamlit.app

---

#  What the System Does

The system automates the complete stock-analysis workflow:

```text
        Stock Selection
              │
              ▼
        Yahoo Finance
              │
              ▼
     Market Data Collection
              │
              ▼
    Data Validation & Cleaning
              │
              ▼
   Technical Feature Engineering
              │
      ┌───────┼────────┐
      │       │        │
      ▼       ▼        ▼
     ML   Forecasting  Risk
Classification         Analysis
      │       │        │
      │   ┌───┼───┐    │
      │   │   │   │    │
      │   ▼   ▼   ▼    │
      │ Naive LSTM GRU  │
      │       │        │
      └───────┼────────┘
              │
              ▼
      Signal Generation
              │
              ▼
      Model Comparison
              │
              ▼
      AI Market Summary
              │
              ▼
      Streamlit Dashboard
```

---

# ✨ Key Features

## 📊 Historical Market Data

The system retrieves historical daily stock-market data using **Yahoo Finance through `yfinance`**.

The collected market data includes:

- Open
- High
- Low
- Close
- Volume

The dashboard supports the following historical periods:

- 3 Months
- 6 Months
- 1 Year
- 2 Years
- 5 Years
- Maximum Available

The latest available trading-day data is retrieved when the dashboard requests fresh market data.

Because the system uses daily market data, the latest available observation may correspond to the most recent trading day rather than the current calendar date when the market is closed.

---

# 📈 Technical Indicator Analysis

The system generates technical indicators from historical price and volume data.

## Moving Averages

- SMA 20
- SMA 50
- EMA 20

## Momentum Indicators

- RSI 14
- MACD
- MACD Signal
- MACD Histogram

## Market Features

- Daily Return
- 20-Day Annualized Volatility
- 20-Day Average Volume
- High-Low Price Range

These features are used across:

- Market-analysis layer
- Machine-learning pipeline
- Risk-analysis layer
- Experimental signal engine

### Feature Engineering Workflow

```text
Historical OHLCV Data
        │
        ▼
Moving Averages
        │
        ├── SMA20
        ├── SMA50
        └── EMA20
        │
        ▼
Momentum Indicators
        │
        ├── RSI14
        ├── MACD
        ├── MACD Signal
        └── MACD Histogram
        │
        ▼
Market Features
        │
        ├── Daily Return
        ├── Volatility20
        ├── VolumeSMA20
        └── HighLowRange
```

---

# 🤖 Machine Learning Market Direction Classification

The system performs **next-day market direction classification**.

Three machine-learning models are trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The target is created from the next day's closing price:

```text
Current Close
      │
      ▼
Next-Day Close
      │
      ▼
Next-Day Close > Current Close?
      │
      ├───────────────┐
      │               │
     TRUE            FALSE
      │               │
      ▼               ▼
      1               0
```

Therefore:

- `1` → Next-day price direction is **UP**
- `0` → Next-day price direction is **DOWN**

---

# 🔮 Time-Series Forecasting

In addition to direction classification, the system evaluates next-day stock-price forecasting.

The forecasting approaches are:

- Naive Persistence
- LSTM
- GRU

These models are evaluated independently from the classification models.

```text
Historical Closing Prices
          │
          ▼
Chronological Train/Test Split
          │
          ▼
     ┌────┼────┐
     │    │    │
     ▼    ▼    ▼
   Naive LSTM  GRU
     │    │    │
     └────┼────┘
          │
          ▼
    Predicted Prices
          │
          ▼
     MAE / RMSE / MAPE
```

---


# 📊 Model Evaluation

The system evaluates both classification and forecasting models.

## Classification Metrics

The classification models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Results are stored in:

```text
results/classification_results.csv
```

---

## Forecasting Metrics

Forecasting models are evaluated using:

- MAE
- RMSE
- MAPE

Results are stored in:

```text
results/forecasting_results.csv
```

The forecasting evaluation compares:

```text
Naive Persistence
       │
       ├── LSTM
       │
       └── GRU
```

---

# 🛡️ Risk Analysis

The system includes a dedicated risk-analysis module.

The following historical metrics are calculated:

- Annualized Volatility
- Average Daily Return
- Cumulative Return
- Maximum Drawdown

---

# 📱 Streamlit Dashboard

The project provides an interactive Streamlit application called:

## **AI Stock Insights**

The dashboard is designed as a decision-support interface rather than exposing the underlying implementation as the main user experience.

The user primarily selects:

- Stock
- Historical period

The technical implementation remains behind the dashboard.

---


# 🏗️ System Architecture

```text
                              User
                                │
                                ▼
                       Streamlit Dashboard
                                │
                                ▼
                         Stock Selection
                                │
                                ▼
                            yfinance
                                │
                                ▼
                      Market Data Collection
                                │
                                ▼
                    Validation & Preprocessing
                                │
                                ▼
                  Technical Feature Engineering
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
     ML Classification      Forecasting       Risk Analysis
            │                   │                   │
      ┌─────┼─────┐       ┌────┼────┐             │
      │     │     │       │    │    │             │
      ▼     ▼     ▼       ▼    ▼    ▼             ▼
   Logistic Random XGB   Naive LSTM GRU       Volatility
  Regression Forest                          Drawdown
                                                Returns
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                                ▼
                       Signal Generation
                                │
                                ▼
                     Model Disagreement
                                │
                                ▼
                       AI Market Summary
                                │
                                ▼
                         Dashboard Output
```

---

# 🗂️ Project Structure

```text
AI-Stock-Market-Analysis/
│
├── app/
│   └── dashboard.py
│
├── models/
│   ├── BHARTIARTL_NS/
│   ├── HDFCBANK_NS/
│   ├── ICICIBANK_NS/
│   ├── INFY_NS/
│   ├── ITC_NS/
│   ├── LT_NS/
│   ├── MARUTI_NS/
│   ├── RELIANCE_NS/
│   ├── SBIN_NS/
│   └── TCS_NS/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── results/
│   ├── classification_results.csv
│   └── forecasting_results.csv
│
├── scripts/
│   ├── train_models.py
│   └── evaluate_models.py
│
├── src/
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── insights.py
│   │   └── risk_analysis.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── technical_indicators.py
│   │
│   └── models/
│       ├── __init__.py
│       ├── forecasting.py
│       ├── model_loader.py
│       └── trend_model.py
│
├── tests/
│   └── test_core.py
│
├── .streamlit/
│   └── config.toml
│
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# 🧪 Testing

The project includes automated tests using **Pytest**.

Run:

```bash
pytest
```

The current core test suite validates:

- Technical-indicator generation
- Risk-analysis output

Expected result:

```text
2 passed
```

---

# 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Market Data | yfinance |
| Data Processing | Pandas, NumPy |
| Technical Analysis | TA |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Deep Learning | TensorFlow / Keras |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Model Persistence | Joblib, Keras |
| Testing | Pytest |
| Version Control | Git |
| Repository | GitHub |

---
# ⚙️ Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Akshaya-126/AI-Stock-Market-Analysis.git
cd AI-Stock-Market-Analysis
```

---

## 2. Create a Virtual Environment

On Windows:

```bash
python -m venv venv
```

Activate:

```cmd
venv\Scripts\activate
```

---

## 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Dashboard

From the project root:

```bash
streamlit run app/dashboard.py
```

The Streamlit dashboard will open in the browser.

The application uses the saved models from the `models/` directory for inference.

---

# 🧪 Run Tests

```bash
pytest
```

---

# 🏋️ Train the Models

Run:

```bash
python scripts/train_models.py
```

This trains the configured stock models and stores them under:

```text
models/
```

---

# 📊 Evaluate the Models

Run:

```bash
python scripts/evaluate_models.py
```

Evaluation results are stored under:

```text
results/
```

---

# ⚠️ Current Limitations

Financial-market prediction is inherently difficult and uncertain.

The current system has several limitations:

- Model predictions can be incorrect.
- Historical patterns may not continue in future market conditions.
- Technical indicators cannot capture every market factor.
- External events can significantly affect prices.
- News and market sentiment are not currently incorporated.
- The classification models show limited predictive power on the evaluated historical datasets.
- The deep-learning forecasting models do not consistently outperform the naive baseline.
- Model probability estimates are not necessarily calibrated real-world probabilities.
- The rule-based BUY / HOLD / SELL signal is experimental.
- The system does not account for an individual's financial situation or risk tolerance.
- The system does not guarantee future returns.
- The current system primarily uses historical price and volume information.
- The current dashboard does not provide personalized investment recommendations.

---

# 🔮 Future Improvements

Possible future improvements include:

- Walk-forward validation
- Rolling-window evaluation
- Hyperparameter optimization
- Probability calibration
- SHAP-based model explainability
- Additional technical indicators
- Market-index features
- News sentiment analysis
- Social-media sentiment analysis
- Fundamental financial data
- Improved time-series architectures
- Automated model retraining
- Model drift monitoring
- Backtesting framework
- Portfolio-level analysis
- Additional stocks and asset classes
- Real-time/intraday data support

---

# 🎯 Project Objective

The objective of this project is to build a complete AI-powered stock-market analytics pipeline rather than only training a prediction model.

The system combines:

```text
Market Data
     ↓
Data Engineering
     ↓
Technical Analysis
     ↓
Machine Learning
     ↓
Deep Learning
     ↓
Risk Analysis
     ↓
Signal Generation
     ↓
Interactive Dashboard
```

This demonstrates the complete process of transforming raw financial data into an interactive AI/ML application.

---

# ⚠️ Disclaimer

**Experimental market analytics — not financial advice.**

This project is developed for educational, research, and portfolio demonstration purposes.

The technical indicators, machine-learning predictions, deep-learning forecasts, risk classifications, and BUY / HOLD / SELL signals are experimental outputs.

They should not be interpreted as personalized financial advice or guarantees of future market performance.

Users should independently evaluate financial decisions and, where appropriate, consult a qualified financial professional.

---

# 👩‍💻 Author

**Akshaya P.**

B.E. Computer Science and Engineering

Kumaraguru College of Technology, Coimbatore

GitHub: **Akshaya-126**
