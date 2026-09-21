import pandas as pd


def clean_market_data(df):
    """
    Clean historical stock market data.

    Steps:
    1. Create a copy of the DataFrame.
    2. Sort data by date.
    3. Remove duplicate dates.
    4. Remove completely empty rows.
    5. Remove rows where essential market values are missing.
    """

    # Create a copy so the original DataFrame is not modified
    df = df.copy()

    # Sort data chronologically
    df = df.sort_index()

    # Remove duplicate dates
    df = df[~df.index.duplicated(keep="last")]

    # Remove rows where every value is missing
    df = df.dropna(how="all")

    # Required market columns
    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    # Remove rows with missing essential market values
    existing_columns = [
        column for column in required_columns
        if column in df.columns
    ]

    if existing_columns:
        df = df.dropna(subset=existing_columns)

    # Make sure numerical columns are numeric
    for column in existing_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove rows that became invalid after numeric conversion
    if existing_columns:
        df = df.dropna(subset=existing_columns)

    return df