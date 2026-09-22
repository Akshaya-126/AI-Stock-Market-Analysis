📈 AI Stock Market Analysis & Forecasting System

An end-to-end AI/ML stock-market analytics application that collects historical market data, performs technical analysis, predicts market direction, evaluates deep-learning forecasting models, analyzes risk, and presents insights through an interactive Streamlit dashboard.

Tech Stack: Python · Pandas · NumPy · yfinance · Scikit-learn · XGBoost · TensorFlow/Keras · TA · Plotly · Streamlit · Pytest

🚀 Live Demo

Open the Streamlit Dashboard

✨ Key Features
📊 Historical stock-market data using yfinance
📈 Technical indicators including SMA, EMA, RSI and MACD
🤖 Next-day market-direction classification
🔮 Next-day price forecasting
📉 Historical risk and drawdown analysis
📌 Experimental BUY / HOLD / SELL technical signal
🔀 Model disagreement detection
📊 Interactive Plotly charts
🌐 Interactive Streamlit dashboard
🧪 Automated testing with Pytest
🧠 Machine Learning

The system performs next-day UP/DOWN market-direction classification using:

Logistic Regression
Random Forest
XGBoost

The target is based on whether the following day's closing price is higher than the current closing price.

A chronological 80/20 train-test split is used to preserve the temporal structure of financial data.

🔮 Time-Series Forecasting

The project evaluates:

Naive Persistence — tomorrow's price is assumed to equal today's price
LSTM
GRU

The deep-learning models use a 60-day historical lookback and are evaluated against the naive baseline.

This provides a realistic comparison instead of assuming that complex neural networks automatically perform better.

📈 Technical Indicators

The feature-engineering pipeline generates:

SMA 20
SMA 50
EMA 20
RSI 14
MACD
MACD Signal
MACD Histogram
Daily Return
20-Day Annualized Volatility
20-Day Average Volume
High-Low Price Range
📊 Model Evaluation
Classification Metrics
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
Forecasting Metrics
MAE
RMSE
MAPE

Evaluation results are stored in:

results/
├── classification_results.csv
└── forecasting_results.csv

The evaluation shows that LSTM and GRU do not consistently outperform the naive baseline across the evaluated stocks.

📉 Risk Analysis

The system calculates:

Annualized volatility
Average daily return
Cumulative return
Maximum drawdown
Experimental risk level

These metrics provide historical context for the selected stock.

📱 Streamlit Dashboard — AI Stock Insights

The dashboard provides:

Current Market
Latest price
Daily change
Weekly change
Weekly trend
Latest trading date
Market Indicators
RSI
MACD
Volatility
Trading volume
Next-Day Outlook
GRU forecast price
Expected direction
Model-implied change
Direction-model probability estimate
Risk Snapshot
Risk level
Annualized volatility
Maximum drawdown
Average daily return
Technical Signal
BUY / HOLD / SELL
Signal score
Signal strength
Explanation of current technical conditions

The direction model and GRU forecast are evaluated independently. If their outputs disagree, the dashboard explicitly highlights the disagreement.

🏗️ System Architecture
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
       Data Validation & Cleaning
                  │
                  ▼
       Technical Feature Engineering
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
       ML     Forecasting   Risk
        │         │         │
   Logistic    Naive       Volatility
   Random      LSTM        Drawdown
   XGBoost     GRU         Returns
        │         │         │
        └─────────┼─────────┘
                  ▼
        Signal & Insight Layer
                  │
                  ▼
         Dashboard Presentation
📚 Trained Stocks
Company	NSE Ticker
Reliance Industries	RELIANCE.NS
TCS	TCS.NS
Infosys	INFY.NS
HDFC Bank	HDFCBANK.NS
ICICI Bank	ICICIBANK.NS
State Bank of India	SBIN.NS
ITC	ITC.NS
Larsen & Toubro	LT.NS
Bharti Airtel	BHARTIARTL.NS
Maruti Suzuki	MARUTI.NS
🗂️ Project Structure
AI-Stock-Market-Analysis/
│
├── app/
│   └── dashboard.py
│
├── models/
├── notebooks/
├── results/
│   ├── classification_results.csv
│   └── forecasting_results.csv
│
├── scripts/
│   ├── train_models.py
│   └── evaluate_models.py
│
├── src/
│   ├── analysis/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   └── models/
│
├── tests/
├── .streamlit/
├── requirements.txt
├── pytest.ini
└── README.md
🛠️ Technology Stack
Area	Technology
Programming	Python
Market Data	yfinance
Data Processing	Pandas, NumPy
Technical Analysis	TA
Machine Learning	Scikit-learn
Boosting	XGBoost
Deep Learning	TensorFlow / Keras
Visualization	Plotly
Dashboard	Streamlit
Model Storage	Joblib, Keras
Testing	Pytest
Version Control	Git & GitHub
🧪 Testing

The project includes automated tests for core functionality.

pytest

Current test result:

2 passed
🎯 Engineering Highlights
Time-series-aware train/test splitting
Training-only scaling for forecasting models
Baseline comparison with Naive Persistence
Separate classification and forecasting tasks
Offline model training with saved-model inference
Explicit model-disagreement handling
Modular project architecture
Reusable evaluation and analysis modules
Interactive deployment-ready dashboard
⚠️ Limitations
Financial markets are highly uncertain.
Predictions can be incorrect.
Historical patterns may not continue in the future.
News, sentiment and fundamental financial data are not currently incorporated.
Deep-learning models do not consistently outperform the naive baseline.
Model probability estimates are not guaranteed real-world probabilities.
BUY / HOLD / SELL signals are experimental.
The system does not provide personalized investment recommendations.
🔮 Future Improvements
Walk-forward validation
Hyperparameter optimization
Probability calibration
SHAP explainability
News and sentiment analysis
Fundamental-data integration
Backtesting
Automated model retraining
Model-drift monitoring
Portfolio-level analysis
Additional stocks and asset classes
🎯 Project Objective

The objective is to demonstrate a complete AI/ML financial analytics pipeline:

Market Data
     ↓
Preprocessing
     ↓
Feature Engineering
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
⚠️ Disclaimer

Experimental market analytics — not financial advice.

This project is developed for educational, research, and portfolio demonstration purposes. Predictions, forecasts, risk classifications, and trading signals are experimental outputs and do not guarantee future market performance.

👩‍💻 Author

Akshaya P.
B.E. Computer Science and Engineering
Kumaraguru College of Technology, Coimbatore

GitHub: Akshaya-126
