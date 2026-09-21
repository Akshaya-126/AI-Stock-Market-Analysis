import numpy as np
import pandas as pd


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(df):
    """
    Calculate historical stock risk and performance metrics.

    Returns:
        dict containing:
        - volatility
        - max_drawdown
        - average_return
        - risk_level
    """

    data = df.copy()

    # ---------------------------------------------------------
    # Validate required column
    # ---------------------------------------------------------

    if "Close" not in data.columns:
        raise ValueError(
            "DataFrame must contain a 'Close' column."
        )

    # ---------------------------------------------------------
    # Clean Close prices
    # ---------------------------------------------------------

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    data = data.dropna(subset=["Close"])

    if len(data) < 2:
        return {
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "average_return": 0.0,
            "risk_level": "Low"
        }

    # ---------------------------------------------------------
    # Calculate daily returns
    # ---------------------------------------------------------

    data["DailyReturn"] = data["Close"].pct_change()

    returns = data["DailyReturn"].dropna()

    if returns.empty:
        return {
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "average_return": 0.0,
            "risk_level": "Low"
        }

    # ---------------------------------------------------------
    # Annualized volatility
    #
    # Daily standard deviation × sqrt(252)
    # ---------------------------------------------------------

    volatility = returns.std() * np.sqrt(252)

    # ---------------------------------------------------------
    # Average daily return
    # ---------------------------------------------------------

    average_return = returns.mean()

    # ---------------------------------------------------------
    # Cumulative growth of ₹1
    # ---------------------------------------------------------

    cumulative_returns = (1 + returns).cumprod()

    # ---------------------------------------------------------
    # Running historical maximum
    # ---------------------------------------------------------

    running_max = cumulative_returns.cummax()

    # ---------------------------------------------------------
    # Drawdown
    # ---------------------------------------------------------

    drawdown = (
        cumulative_returns - running_max
    ) / running_max

    # ---------------------------------------------------------
    # Maximum drawdown
    # ---------------------------------------------------------

    max_drawdown = drawdown.min()

    # ---------------------------------------------------------
    # Risk classification
    # ---------------------------------------------------------

    risk_level = calculate_risk_level(
        volatility,
        max_drawdown
    )

    return {
        "volatility": float(volatility),
        "max_drawdown": float(max_drawdown),
        "average_return": float(average_return),
        "risk_level": risk_level
    }


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk_level(volatility, max_drawdown):
    """
    Convert historical volatility and maximum drawdown
    into an experimental historical-risk category.

    This classification is for dashboard visualization
    and is not financial advice.
    """

    volatility = abs(float(volatility))
    max_drawdown = abs(float(max_drawdown))

    # ---------------------------------------------------------
    # High historical risk
    # ---------------------------------------------------------

    if volatility >= 0.50 or max_drawdown >= 0.30:
        return "High"

    # ---------------------------------------------------------
    # Moderate historical risk
    # ---------------------------------------------------------

    elif volatility >= 0.25 or max_drawdown >= 0.15:
        return "Moderate"

    # ---------------------------------------------------------
    # Lower historical risk
    # ---------------------------------------------------------

    else:
        return "Low"


# ============================================================
# RISK SUMMARY
# ============================================================

def get_risk_summary(risk):
    """
    Convert the risk dictionary into a clean dashboard summary.
    """

    return {
        "risk_level": risk.get(
            "risk_level",
            "Unknown"
        ),

        "volatility": float(
            risk.get("volatility", 0.0)
        ),

        "max_drawdown": float(
            risk.get("max_drawdown", 0.0)
        ),

        "average_return": float(
            risk.get("average_return", 0.0)
        )
    }