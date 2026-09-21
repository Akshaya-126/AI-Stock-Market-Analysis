import pandas as pd

from src.features.technical_indicators import add_indicators
from src.analysis.risk_analysis import calculate_risk


def create_sample_market_data():
    idx = pd.date_range(
        "2020-01-01",
        periods=120
    )

    close = pd.Series(
        range(100, 220),
        index=idx,
        dtype=float
    )

    df = pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": 1000
        },
        index=idx
    )

    return df


def test_technical_indicators():

    df = create_sample_market_data()

    output = add_indicators(df)

    assert not output.empty

    assert "SMA20" in output.columns
    assert "SMA50" in output.columns
    assert "EMA20" in output.columns
    assert "RSI14" in output.columns
    assert "MACD" in output.columns


def test_risk_analysis():

    df = create_sample_market_data()

    output = add_indicators(df)

    risk = calculate_risk(output)

    assert "risk_level" in risk