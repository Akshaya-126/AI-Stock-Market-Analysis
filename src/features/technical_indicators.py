import ta


def add_indicators(df):
    """
    Calculate technical indicators for stock market data.

    Indicators:
    - SMA 20
    - SMA 50
    - EMA 20
    - RSI 14
    - MACD
    - Daily Return
    - 20-day annualized volatility
    - 20-day average volume
    - High-Low price range
    """

    out = df.copy()

    # -----------------------------
    # Moving Averages
    # -----------------------------

    out["SMA20"] = (
        out["Close"]
        .rolling(window=20)
        .mean()
    )

    out["SMA50"] = (
        out["Close"]
        .rolling(window=50)
        .mean()
    )

    out["EMA20"] = (
        out["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # -----------------------------
    # RSI
    # -----------------------------

    rsi = ta.momentum.RSIIndicator(
        close=out["Close"],
        window=14
    )

    out["RSI14"] = rsi.rsi()

    # -----------------------------
    # MACD
    # -----------------------------

    macd = ta.trend.MACD(
        close=out["Close"]
    )

    out["MACD"] = macd.macd()

    out["MACD_SIGNAL"] = (
        macd.macd_signal()
    )

    out["MACD_HIST"] = (
        macd.macd_diff()
    )

    # -----------------------------
    # Daily Return
    # -----------------------------

    out["DailyReturn"] = (
        out["Close"].pct_change()
    )

    # -----------------------------
    # 20-Day Annualized Volatility
    # -----------------------------

    out["Volatility20"] = (
        out["DailyReturn"]
        .rolling(window=20)
        .std()
        * (252 ** 0.5)
    )

    # -----------------------------
    # Average Volume
    # -----------------------------

    out["VolumeSMA20"] = (
        out["Volume"]
        .rolling(window=20)
        .mean()
    )

    # -----------------------------
    # High-Low Price Range
    # -----------------------------

    out["HighLowRange"] = (
        (out["High"] - out["Low"])
        / out["Close"]
    )

    # -----------------------------
    # Remove rows created with
    # insufficient historical data
    # -----------------------------

    out = out.dropna()

    return out