from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[2]


def download_stock_data(ticker, period="5y"):
    """
    Download historical stock data from Yahoo Finance.

    Examples:
        RELIANCE.NS
        TCS.NS
        INFY.NS
        HDFCBANK.NS
        ICICIBANK.NS
    """

    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(
            f"No market data returned for {ticker}. "
            "Check whether the ticker is valid."
        )

    # yfinance can sometimes return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns for {ticker}: {missing_columns}"
        )

    # Keep only required market columns
    df = df[required_columns].copy()

    # Remove missing rows
    df = df.dropna()

    # Make sure index is datetime
    df.index = pd.to_datetime(df.index)

    # Remove duplicate dates
    df = df[~df.index.duplicated(keep="last")]

    # Sort chronologically
    df = df.sort_index()

    return df