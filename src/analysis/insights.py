import pandas as pd


# ============================================================
# BUILD MARKET SIGNAL
# ============================================================

def build_signal(df):
    """
    Generate an experimental short-term market signal
    using technical indicators and recent momentum.

    The signal is based on:
        - Price vs SMA20
        - SMA20 vs SMA50
        - RSI14
        - MACD vs MACD Signal
        - Recent price momentum

    Returns:
        dict containing:
        - score
        - label
        - trend
        - confidence
        - forecast_price
        - summary
        - explanation
    """

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------

    required_columns = [
        "Close",
        "SMA20",
        "SMA50",
        "RSI14",
        "MACD",
        "MACD_SIGNAL"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(df) < 2:
        raise ValueError(
            "At least two rows of data are required "
            "to calculate momentum."
        )

    # ---------------------------------------------------------
    # Latest and previous trading day
    # ---------------------------------------------------------

    current = df.iloc[-1]
    previous = df.iloc[-2]

    # ---------------------------------------------------------
    # Start with neutral score
    # ---------------------------------------------------------

    score = 50.0

    # =========================================================
    # 1. PRICE VS SMA20
    # =========================================================

    if current["Close"] > current["SMA20"]:
        score += 10
    else:
        score -= 10

    # =========================================================
    # 2. SMA20 VS SMA50
    # =========================================================

    if current["SMA20"] > current["SMA50"]:
        score += 10
    else:
        score -= 10

    # =========================================================
    # 3. RSI
    # =========================================================

    rsi = float(current["RSI14"])

    if 50 <= rsi <= 70:
        score += 10

    elif rsi < 30:
        score += 3

    elif rsi > 75:
        score -= 8

    else:
        score -= 2

    # =========================================================
    # 4. MACD
    # =========================================================

    if current["MACD"] > current["MACD_SIGNAL"]:
        score += 10
    else:
        score -= 10

    # =========================================================
    # 5. RECENT MOMENTUM
    # =========================================================

    previous_close = float(previous["Close"])
    current_close = float(current["Close"])

    if previous_close == 0:
        momentum = 0.0
    else:
        momentum = (
            current_close / previous_close
        ) - 1

    if momentum > 0:
        score += 5
    else:
        score -= 5

    # ---------------------------------------------------------
    # Keep score between 0 and 100
    # ---------------------------------------------------------

    score = max(
        0.0,
        min(100.0, score)
    )

    # =========================================================
    # SIGNAL CLASSIFICATION
    # =========================================================

    if score >= 60:

        label = "BUY"
        trend = "▲ UP"

    elif score >= 40:

        label = "HOLD"
        trend = "→ NEUTRAL"

    else:

        label = "SELL"
        trend = "▼ DOWN"

    # =========================================================
    # CONFIDENCE ESTIMATE
    # =========================================================

    confidence = min(
        95.0,
        max(
            50.0,
            50.0 + abs(score - 50.0) * 1.6
        )
    )

    # =========================================================
    # SIMPLE NEXT-DAY PRICE ESTIMATE
    # =========================================================

    forecast_price = (
        current_close *
        (1 + momentum * 0.5)
    )

    # =========================================================
    # SUMMARY
    # =========================================================

    if score >= 60:

        summary = (
            "Current indicators show relatively "
            "favorable short-term conditions."
        )

    elif score >= 40:

        summary = (
            "Current indicators are mixed, suggesting "
            "no strong short-term direction."
        )

    else:

        summary = (
            "Current indicators show relatively "
            "unfavorable short-term conditions."
        )

    # =========================================================
    # EXPLANATION
    # =========================================================

    explanation = (
        f"The current experimental market signal is "
        f"{label}. It considers the stock's position "
        f"relative to its 20-day and 50-day moving "
        f"averages, RSI, MACD and recent price momentum. "
        f"Historical risk is evaluated separately using "
        f"volatility and maximum drawdown. The signal strength "
        f"reflects the strength of the rule-based "
        f"signal and is not a guarantee of future returns."
    )

    # =========================================================
    # RETURN
    # =========================================================

    return {
        "score": float(score),
        "label": label,
        "trend": trend,
        "confidence": float(confidence),
        "forecast_price": float(forecast_price),
        "summary": summary,
        "explanation": explanation
    }