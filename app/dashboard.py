# ============================================================
# AI STOCK INSIGHTS - STREAMLIT DASHBOARD
# ============================================================

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime


# ------------------------------------------------------------
# PROJECT IMPORTS
# ------------------------------------------------------------

from src.data.data_loader import download_stock_data
from src.data.preprocessing import clean_market_data
from src.features.technical_indicators import add_indicators

from src.analysis.risk_analysis import calculate_risk
from src.analysis.insights import build_signal

from src.models.trend_model import (
    FEATURES,
    predict_latest
)

from src.models.model_loader import (
    load_model,
    load_forecasting_model,
    load_forecasting_scaler
)

from src.models.forecasting import forecast_next_day


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Stock Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# INDIAN STOCKS
# ============================================================

INDIAN_STOCKS = {

    "Reliance Industries": "RELIANCE.NS",

    "TCS": "TCS.NS",

    "Infosys": "INFY.NS",

    "HDFC Bank": "HDFCBANK.NS",

    "ICICI Bank": "ICICIBANK.NS",

    "State Bank of India": "SBIN.NS",

    "ITC": "ITC.NS",

    "Larsen & Toubro": "LT.NS",

    "Bharti Airtel": "BHARTIARTL.NS",

    "Maruti Suzuki": "MARUTI.NS",

    "HCL Technologies": "HCLTECH.NS",

    "Wipro": "WIPRO.NS",

    "Axis Bank": "AXISBANK.NS",

    "Sun Pharma": "SUNPHARMA.NS",

    "Tata Motors": "TATAMOTORS.NS",

    "Tata Steel": "TATASTEEL.NS",

    "Adani Enterprises": "ADANIENT.NS",

    "Adani Ports": "ADANIPORTS.NS",

    "HUL": "HINDUNILVR.NS",

    "UltraTech Cement": "ULTRACEMCO.NS"

}


# ============================================================
# HISTORICAL PERIODS
# ============================================================

HISTORICAL_PERIODS = {

    "3 Months": "3mo",

    "6 Months": "6mo",

    "1 Year": "1y",

    "2 Years": "2y",

    "5 Years": "5y",

    "Maximum Available": "max"

}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=900)
def get_stock_data(
    ticker,
    period="1y"
):

    df = download_stock_data(
        ticker=ticker,
        period=period
    )

    return clean_market_data(
        df
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_weekly_change(df):
    """
    Calculate approximately 5-trading-day percentage change.
    """

    if len(df) < 2:
        return 0.0

    latest_price = float(
        df["Close"].iloc[-1]
    )

    if len(df) >= 6:

        previous_price = float(
            df["Close"].iloc[-6]
        )

    else:

        previous_price = float(
            df["Close"].iloc[0]
        )

    if previous_price == 0:
        return 0.0

    return (
        (latest_price - previous_price)
        / previous_price
    ) * 100


def calculate_trend(weekly_change):

    if weekly_change > 1:
        return "Upward"

    elif weekly_change < -1:
        return "Downward"

    else:
        return "Mostly Stable"


def get_forecast_direction(forecast_change):

    if forecast_change > 0:
        return "Up"

    elif forecast_change < 0:
        return "Down"

    else:
        return "Mostly Unchanged"


def models_disagree(
    model_direction,
    forecast_change
):
    """
    Check whether the classification direction and
    GRU price forecast point in different directions.
    """

    if model_direction is None or forecast_change is None:
        return False

    if (
        model_direction.upper() == "UP"
        and forecast_change < 0
    ):
        return True

    if (
        model_direction.upper() == "DOWN"
        and forecast_change > 0
    ):
        return True

    return False


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Analysis Settings"
)


selected_stock = st.sidebar.selectbox(
    "Select Stock",
    list(INDIAN_STOCKS.keys())
)


ticker = INDIAN_STOCKS[
    selected_stock
]


st.sidebar.caption(
    "Market data is obtained from Yahoo Finance. "
    "Daily data represents the latest available "
    "trading session."
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "📈 AI Stock Insights"
)


st.write(
    "Understand recent Indian stock-market conditions "
    "using price trends, technical indicators, "
    "forecasting and risk analysis."
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = get_stock_data(
        ticker=ticker,
        period="1y"
    )

except Exception as e:

    st.error(
        f"Unable to load market data for {selected_stock}."
    )

    st.caption(
        "Please check your internet connection or try again later."
    )

    with st.expander("Technical details"):

        st.exception(e)

    st.stop()


if df.empty:

    st.error(
        "No market data is available for this stock."
    )

    st.stop()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

try:

    df = add_indicators(
        df
    )

except Exception as e:

    st.error(
        "Unable to calculate technical indicators."
    )

    with st.expander("Technical details"):

        st.exception(e)

    st.stop()


if len(df) < 2:

    st.error(
        "Not enough data is available for analysis."
    )

    st.stop()


# ============================================================
# LATEST MARKET DATA
# ============================================================

latest = df.iloc[-1]


latest_price = float(
    latest["Close"]
)


previous_price = float(
    df["Close"].iloc[-2]
)


if previous_price == 0:

    daily_change = 0.0

else:

    daily_change = (
        (latest_price - previous_price)
        / previous_price
    ) * 100


weekly_change = calculate_weekly_change(
    df
)


trend = calculate_trend(
    weekly_change
)


latest_date = df.index[-1]


refresh_date = datetime.now()


# ============================================================
# CURRENT INDICATORS
# ============================================================

rsi = float(
    latest["RSI14"]
)


macd = float(
    latest["MACD"]
)


macd_signal = float(
    latest["MACD_SIGNAL"]
)


volatility_20 = float(
    latest["Volatility20"]
)


volume = int(
    latest["Volume"]
)


# ============================================================
# RISK ANALYSIS
# ============================================================

risk = calculate_risk(
    df
)


risk_level = risk["risk_level"]


historical_volatility = risk[
    "volatility"
]


historical_drawdown = risk[
    "max_drawdown"
]


average_return = risk[
    "average_return"
]


# ============================================================
# XGBOOST DIRECTION MODEL
# ============================================================

model_direction = None

model_confidence = None

model_available = False


try:

    model = load_model(
        model_name="xgboost",
        ticker=ticker
    )


    latest_features = (
        df[FEATURES]
        .iloc[-1:]
        .copy()
    )


    prediction_result = predict_latest(
        model,
        latest_features
    )


    model_direction = (
        prediction_result["direction"]
    )


    model_confidence = (
        prediction_result["confidence"]
    )


    model_available = True


except Exception:

    model_available = False


# ============================================================
# NEXT-DAY GRU FORECAST
# ============================================================

forecast_price = None

forecast_change = None

forecast_direction = None

forecast_available = False


try:

    forecast_model = load_forecasting_model(
        model_name="gru",
        ticker=ticker
    )


    forecast_scaler = load_forecasting_scaler(
        model_name="gru",
        ticker=ticker
    )


    forecast_price = float(
        forecast_next_day(
            model=forecast_model,
            df=df,
            scaler=forecast_scaler
        )
    )


    if latest_price != 0:

        forecast_change = (
            (forecast_price - latest_price)
            / latest_price
        ) * 100

    else:

        forecast_change = 0.0


    forecast_direction = get_forecast_direction(
        forecast_change
    )


    forecast_available = True


except Exception:

    forecast_available = False


# ============================================================
# CHECK MODEL AGREEMENT
# ============================================================

model_disagreement = models_disagree(
    model_direction,
    forecast_change
)


# ============================================================
# EXPERIMENTAL MARKET SIGNAL
# ============================================================

try:

    signal_result = build_signal(
        df
    )


    signal = signal_result[
        "label"
    ]


    signal_score = signal_result[
        "score"
    ]


    signal_confidence = (
        signal_result["confidence"]
    )


    signal_summary = (
        signal_result["summary"]
    )


    signal_explanation = (
        signal_result["explanation"]
    )


except Exception:

    signal = "HOLD"

    signal_score = 50.0

    signal_confidence = 50.0

    signal_summary = (
        "The current market conditions could not "
        "be fully evaluated."
    )

    signal_explanation = (
        "Insufficient indicator information is available."
    )


# ============================================================
# LATEST MARKET
# ============================================================

st.subheader(
    f"📅 {selected_stock} — Latest Market"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Latest Price",
        f"₹{latest_price:,.2f}"
    )


with col2:

    st.metric(
        "Today's Change",
        f"{daily_change:+.2f}%"
    )


with col3:

    st.metric(
        "1-Week Change",
        f"{weekly_change:+.2f}%"
    )


with col4:

    st.metric(
        "1-Week Trend",
        trend
    )


st.caption(
    f"Latest trading data: "
    f"{latest_date.strftime('%d %b %Y')} | "
    f"Dashboard refreshed: "
    f"{refresh_date.strftime('%d %b %Y, %I:%M %p')}"
)


# ============================================================
# CURRENT MARKET SIGNAL
# ============================================================

st.subheader(
    "🎯 Current Market Signal"
)


if signal == "BUY":

    st.success(
        f"### 🟢 {signal}\n\n"
        f"{signal_summary}"
    )


elif signal == "SELL":

    st.error(
        f"### 🔴 {signal}\n\n"
        f"{signal_summary}"
    )


else:

    st.warning(
        f"### 🟡 {signal}\n\n"
        f"{signal_summary}"
    )


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Signal Score",
        f"{signal_score:.0f}/100"
    )


with col2:

    st.metric(
        "Signal Strength",
        f"{signal_confidence:.0f}%"
    )


st.caption(
    "The signal strength reflects how strongly the "
    "current technical conditions support the rule-based "
    "signal. It is not a probability of future returns "
    "and does not guarantee future performance."
)


# ============================================================
# THIS WEEK AT A GLANCE
# ============================================================

st.subheader(
    "📊 This Week at a Glance"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Weekly Performance",
        f"{weekly_change:+.2f}%"
    )


with col2:

    st.metric(
        "Trend",
        trend
    )


with col3:

    st.metric(
        "RSI",
        f"{rsi:.2f}"
    )


with col4:

    st.metric(
        "Risk",
        risk_level
    )


# ============================================================
# RECENT PRICE MOVEMENT
# ============================================================

st.subheader(
    "📈 Recent Price Movement"
)


recent_df = df.tail(30)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=recent_df.index,
        y=recent_df["Close"],
        mode="lines",
        name="Price"
    )
)


fig.update_layout(
    height=420,
    margin=dict(
        l=20,
        r=20,
        t=20,
        b=20
    ),
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# KEY MARKET INDICATORS
# ============================================================

st.subheader(
    "📌 Key Market Indicators"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "RSI",
        f"{rsi:.2f}"
    )


    if rsi >= 70:

        st.caption(
            "Strong / potentially stretched"
        )

    elif rsi <= 30:

        st.caption(
            "Weak / potentially oversold"
        )

    elif rsi >= 50:

        st.caption(
            "Positive momentum"
        )

    else:

        st.caption(
            "Below neutral momentum"
        )


with col2:

    st.metric(
        "MACD",
        f"{macd:.2f}"
    )


    if macd > macd_signal:

        st.caption(
            "Positive trend momentum"
        )

    else:

        st.caption(
            "Cautious trend momentum"
        )


with col3:

    st.metric(
        "20-Day Volatility",
        f"{volatility_20 * 100:.2f}%"
    )


    st.caption(
        "Annualized from recent daily returns"
    )


with col4:

    st.metric(
        "Latest Volume",
        f"{volume:,}"
    )


    st.caption(
        "Shares traded on latest session"
    )


# ============================================================
# NEXT-DAY OUTLOOK
# ============================================================

st.subheader(
    "🔮 Next-Day Outlook"
)


if forecast_available:

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "GRU Forecast Price",
            f"₹{forecast_price:,.2f}"
        )


    with col2:

        st.metric(
            "Expected Direction",
            forecast_direction
        )


    with col3:

        st.metric(
            "Model-Implied Change",
            f"{forecast_change:+.2f}%"
        )


    st.caption(
        "This is an experimental neural-network forecast "
        "based on historical price patterns. It may differ "
        "substantially from the actual next trading price."
    )


    if model_disagreement:

        st.warning(
            "The direction model and GRU forecast currently "
            f"disagree: the direction model estimates "
            f"{model_direction}, while the GRU forecast "
            f"indicates a {forecast_change:+.2f}% price movement. "
            "These are independent model outputs and may disagree."
        )


else:

    st.info(
        "The GRU forecasting model is not available "
        "for this stock yet. Train the forecasting "
        "models before using the next-day forecast."
    )


# ============================================================
# DIRECTION MODEL INFORMATION
# ============================================================

if model_available:

    st.caption(
        f"Direction model estimate: "
        f"{model_direction} "
        f"({model_confidence:.1%} model probability estimate)."
    )


# ============================================================
# RISK SNAPSHOT
# ============================================================

st.subheader(
    "⚠️ Risk Snapshot"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Risk Level",
        risk_level
    )


with col2:

    st.metric(
        "Annualized Volatility",
        f"{historical_volatility * 100:.2f}%"
    )


with col3:

    st.metric(
        "Maximum Drawdown",
        f"{historical_drawdown * 100:.2f}%"
    )


st.caption(
    f"Average daily return: "
    f"{average_return * 100:+.3f}%"
)


# ============================================================
# SIGNAL EXPLANATION
# ============================================================

st.subheader(
    "🔎 Signal Explanation"
)


st.write(
    signal_explanation
)


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================

st.subheader(
    "📊 Historical Analysis"
)


historical_period = st.selectbox(
    "Historical Period",
    list(HISTORICAL_PERIODS.keys()),
    index=1
)


historical_code = HISTORICAL_PERIODS[
    historical_period
]


try:

    historical_df = get_stock_data(
        ticker=ticker,
        period=historical_code
    )

except Exception:

    historical_df = df.copy()


if len(historical_df) > 1:

    start_price = float(
        historical_df["Close"].iloc[0]
    )


    end_price = float(
        historical_df["Close"].iloc[-1]
    )


    if start_price != 0:

        historical_return = (
            (end_price - start_price)
            / start_price
        ) * 100

    else:

        historical_return = 0.0


    historical_returns = (
        historical_df["Close"]
        .pct_change()
        .dropna()
    )


    if not historical_returns.empty:

        historical_average_return = (
            historical_returns.mean()
            * 100
        )


        historical_volatility_value = (
            historical_returns.std()
            * np.sqrt(252)
            * 100
        )


        cumulative = (
            1 + historical_returns
        ).cumprod()


        running_max = (
            cumulative.cummax()
        )


        historical_drawdown_period = (
            (cumulative - running_max)
            / running_max
        ).min() * 100

    else:

        historical_average_return = 0.0

        historical_volatility_value = 0.0

        historical_drawdown_period = 0.0


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Return",
            f"{historical_return:+.2f}%"
        )


    with col2:

        st.metric(
            "Average Daily Return",
            f"{historical_average_return:+.3f}%"
        )


    with col3:

        st.metric(
            "Annualized Volatility",
            f"{historical_volatility_value:.2f}%"
        )


    with col4:

        st.metric(
            "Maximum Drawdown",
            f"{historical_drawdown_period:.2f}%"
        )


    # --------------------------------------------------------
    # HISTORICAL CHART
    # --------------------------------------------------------

    fig_hist = go.Figure()


    fig_hist.add_trace(
        go.Scatter(
            x=historical_df.index,
            y=historical_df["Close"],
            mode="lines",
            name="Price"
        )
    )


    fig_hist.update_layout(
        height=420,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode="x unified"
    )


    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )


# ============================================================
# AI MARKET SUMMARY
# ============================================================

st.subheader(
    "🤖 AI Market Summary"
)


summary_parts = []


# ------------------------------------------------------------
# WEEKLY PERFORMANCE
# ------------------------------------------------------------

if weekly_change > 0:

    summary_parts.append(
        f"{selected_stock} gained "
        f"{weekly_change:.2f}% over the past week."
    )

elif weekly_change < 0:

    summary_parts.append(
        f"{selected_stock} declined "
        f"{abs(weekly_change):.2f}% over the past week."
    )

else:

    summary_parts.append(
        f"{selected_stock} remained broadly unchanged "
        "over the past week."
    )


# ------------------------------------------------------------
# RSI
# ------------------------------------------------------------

if rsi >= 70:

    summary_parts.append(
        f"RSI is {rsi:.2f}, indicating strong and potentially "
        "stretched momentum."
    )

elif rsi <= 30:

    summary_parts.append(
        f"RSI is {rsi:.2f}, indicating relatively weak momentum."
    )

elif rsi >= 50:

    summary_parts.append(
        f"RSI is {rsi:.2f}, indicating positive but "
        "not extreme momentum."
    )

else:

    summary_parts.append(
        f"RSI is {rsi:.2f}, indicating momentum below "
        "the neutral level."
    )


# ------------------------------------------------------------
# MACD
# ------------------------------------------------------------

if macd > macd_signal:

    summary_parts.append(
        "MACD is above its signal line, indicating "
        "positive short-term momentum."
    )

else:

    summary_parts.append(
        "MACD is below its signal line, indicating "
        "cautious short-term momentum."
    )


# ------------------------------------------------------------
# RISK
# ------------------------------------------------------------

summary_parts.append(
    f"Historical risk is currently classified as "
    f"{risk_level.lower()} based on volatility and drawdown."
)


# ------------------------------------------------------------
# DIRECTION MODEL
# ------------------------------------------------------------

if model_available:

    summary_parts.append(
        f"The direction model estimates "
        f"{model_direction.lower()} next-day movement "
        f"with a {model_confidence:.1%} model probability estimate."
    )


# ------------------------------------------------------------
# GRU FORECAST
# ------------------------------------------------------------

if forecast_available:

    summary_parts.append(
        f"The GRU forecasting model produces an experimental "
        f"next-day forecast of ₹{forecast_price:,.2f}, "
        f"corresponding to a model-implied "
        f"{forecast_change:+.2f}% change from the latest price."
    )


# ------------------------------------------------------------
# MODEL DISAGREEMENT
# ------------------------------------------------------------

if model_disagreement:

    summary_parts.append(
        f"The direction model and GRU forecast currently "
        f"disagree: the direction model estimates "
        f"{model_direction.lower()}, while the GRU forecast "
        f"indicates a {forecast_change:+.2f}% price movement. "
        "They are independent model outputs and may disagree."
    )

elif model_available and forecast_available:

    summary_parts.append(
        "The direction model and GRU forecast are currently "
        "aligned in their directional estimate."
    )


# ------------------------------------------------------------
# TECHNICAL SIGNAL
# ------------------------------------------------------------

summary_parts.append(
    f"The experimental technical signal is currently "
    f"{signal}."
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

ai_summary = " ".join(
    summary_parts
)


st.info(
    ai_summary
)


# ============================================================
# ABOUT
# ============================================================

st.subheader(
    "ℹ️ About"
)


st.write(
    "AI Stock Insights combines historical market data, "
    "technical indicators, machine-learning analysis, "
    "short-term forecasting and historical risk analysis "
    "to help users understand stock-market conditions."
)


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()


st.warning(
    """
    **Disclaimer**

    This application is an experimental stock-market analysis system.

    BUY, HOLD and SELL are system-generated analytical signals based
    on recent market data and model outputs. They are not personalized
    financial advice, guarantees of future performance, or recommendations
    based on an individual's financial situation.

    Market prices can change rapidly, and model forecasts may be inaccurate.
    Users should independently evaluate financial information before making
    investment decisions.
    """
)