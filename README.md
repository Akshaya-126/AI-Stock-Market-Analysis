# AI Stock Market Analysis & Forecasting System

A deployment-ready Streamlit dashboard for historical stock analysis, technical indicators, risk analysis and experimental AI market signals.

## User Features
- Stock and period selection
- Interactive price chart
- SMA/EMA, RSI and MACD
- Volatility, returns and maximum drawdown
- Next-day experimental direction
- Confidence estimate
- Next-day price outlook
- Positive / Neutral / Negative investment signal
- Human-readable market summary

## Technical Architecture
Data collection (yfinance) → preprocessing → technical feature engineering → risk analysis → signal engine → Streamlit dashboard.

Optional ML/time-series utilities are included under `src/models/` for offline experimentation. Heavy models should be trained offline rather than on every dashboard visit.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## GitHub
```bash
git init
git add .
git commit -m "Initial AI stock market dashboard"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Streamlit deployment
Push to GitHub, create a Streamlit app, select the `main` branch and use `app/dashboard.py` as the main file.

## Important
The Positive/Neutral/Negative signal is an experimental, non-personalized market signal. It is not a buy/sell recommendation and does not consider personal financial goals or risk tolerance. Confidence is not a calibrated probability. Historical performance cannot guarantee future returns.
