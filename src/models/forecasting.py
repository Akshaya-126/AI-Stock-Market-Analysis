"""
Time-series forecasting utilities.

Includes:
- Naive persistence baseline
- LSTM
- GRU
- MAE, RMSE and MAPE evaluation

Uses chronological splitting to avoid time-series leakage.

Forecasting models and scalers are stored separately
for each stock ticker.
"""

from pathlib import Path

import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf


# ================================================================
# TENSORFLOW COMPONENTS
# ================================================================

Sequential = tf.keras.Sequential
load_model = tf.keras.models.load_model

Input = tf.keras.layers.Input
LSTM = tf.keras.layers.LSTM
GRU = tf.keras.layers.GRU
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

EarlyStopping = tf.keras.callbacks.EarlyStopping


# ================================================================
# PROJECT PATHS
# ================================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# SETTINGS
# ================================================================

LOOKBACK = 60


# ================================================================
# TICKER MODEL DIRECTORY
# ================================================================

def get_ticker_model_dir(ticker):
    """
    Return the model directory for a ticker.

    Example:
        RELIANCE.NS
        ->
        models/RELIANCE_NS/
    """

    ticker_folder = ticker.replace(".", "_")

    ticker_dir = MODEL_DIR / ticker_folder

    ticker_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return ticker_dir


# ================================================================
# PREPARE FORECASTING DATA
# ================================================================

def prepare_forecasting_data(
    df,
    train_ratio=0.8,
    lookback=LOOKBACK
):
    """
    Prepare chronological LSTM/GRU sequences.

    The scaler is fitted ONLY on training prices
    to prevent test-data leakage.
    """

    prices = (
        df["Close"]
        .values
        .reshape(-1, 1)
    )

    if len(prices) <= lookback:
        raise ValueError(
            f"At least {lookback + 1} "
            "price observations are required."
        )

    # ------------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # ------------------------------------------------------------

    split_index = int(
        len(prices) * train_ratio
    )

    train_prices = prices[:split_index]
    test_prices = prices[split_index:]

    if len(train_prices) <= lookback:
        raise ValueError(
            "Training data is too small "
            f"for a lookback of {lookback}."
        )

    if len(test_prices) == 0:
        raise ValueError(
            "Test dataset is empty."
        )

    # ------------------------------------------------------------
    # FIT SCALER ONLY ON TRAINING DATA
    # ------------------------------------------------------------

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_prices
    )

    test_scaled = scaler.transform(
        test_prices
    )

    # ------------------------------------------------------------
    # TRAIN SEQUENCES
    # ------------------------------------------------------------

    X_train = []
    y_train = []

    for i in range(
        lookback,
        len(train_scaled)
    ):

        X_train.append(
            train_scaled[
                i - lookback:i,
                0
            ]
        )

        y_train.append(
            train_scaled[i, 0]
        )

    # ------------------------------------------------------------
    # TEST SEQUENCES
    # ------------------------------------------------------------

    test_context = np.concatenate(
        [
            train_scaled[-lookback:],
            test_scaled
        ]
    )

    X_test = []
    y_test = []

    for i in range(
        lookback,
        len(test_context)
    ):

        X_test.append(
            test_context[
                i - lookback:i,
                0
            ]
        )

        y_test.append(
            test_context[i, 0]
        )

    # ------------------------------------------------------------
    # NUMPY ARRAYS
    # ------------------------------------------------------------

    X_train = np.asarray(
        X_train,
        dtype=np.float32
    )

    y_train = np.asarray(
        y_train,
        dtype=np.float32
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float32
    )

    y_test = np.asarray(
        y_test,
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # LSTM / GRU INPUT SHAPE
    #
    # (samples, timesteps, features)
    # ------------------------------------------------------------

    X_train = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    X_test = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )


# ================================================================
# NAIVE BASELINE
# ================================================================

def naive_forecast(
    df,
    train_ratio=0.8
):
    """
    Naive persistence forecast.

    Tomorrow's predicted price =
    today's actual price.
    """

    prices = (
        df["Close"]
        .values
        .reshape(-1)
    )

    split_index = int(
        len(prices) * train_ratio
    )

    if split_index <= 0:
        raise ValueError(
            "Not enough data for forecasting."
        )

    test_prices = prices[split_index:]

    previous_price = prices[
        split_index - 1
    ]

    predictions = []
    actual = []

    for price in test_prices:

        predictions.append(
            previous_price
        )

        actual.append(
            price
        )

        previous_price = price

    return (
        np.asarray(predictions),
        np.asarray(actual)
    )


# ================================================================
# BUILD LSTM
# ================================================================

def build_lstm(input_shape):
    """
    Build LSTM forecasting model.
    """

    model = Sequential([

        Input(
            shape=input_shape
        ),

        LSTM(
            64,
            return_sequences=True
        ),

        Dropout(
            0.2
        ),

        LSTM(
            32
        ),

        Dropout(
            0.2
        ),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1
        )

    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# ================================================================
# BUILD GRU
# ================================================================

def build_gru(input_shape):
    """
    Build GRU forecasting model.
    """

    model = Sequential([

        Input(
            shape=input_shape
        ),

        GRU(
            64,
            return_sequences=True
        ),

        Dropout(
            0.2
        ),

        GRU(
            32
        ),

        Dropout(
            0.2
        ),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1
        )

    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# ================================================================
# TRAIN LSTM
# ================================================================

def train_lstm(
    X_train,
    y_train,
    epochs=50,
    batch_size=32
):
    """
    Train LSTM with early stopping.
    """

    model = build_lstm(
        (
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        callbacks=[
            early_stopping
        ],
        verbose=1
    )

    return (
        model,
        history
    )


# ================================================================
# TRAIN GRU
# ================================================================

def train_gru(
    X_train,
    y_train,
    epochs=50,
    batch_size=32
):
    """
    Train GRU with early stopping.
    """

    model = build_gru(
        (
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        callbacks=[
            early_stopping
        ],
        verbose=1
    )

    return (
        model,
        history
    )


# ================================================================
# PREDICT PRICES
# ================================================================

def predict_prices(
    model,
    X_test,
    scaler
):
    """
    Generate predictions and convert them
    back to the original price scale.
    """

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    )

    predictions = scaler.inverse_transform(
        predictions_scaled
    ).reshape(-1)

    return predictions


# ================================================================
# FORECAST METRICS
# ================================================================

def calculate_forecast_metrics(
    actual,
    predicted
):
    """
    Calculate:

    - MAE
    - RMSE
    - MAPE
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    non_zero = actual != 0

    if not np.any(non_zero):

        mape = 0.0

    else:

        mape = np.mean(
            np.abs(
                (
                    actual[non_zero]
                    - predicted[non_zero]
                )
                / actual[non_zero]
            )
        ) * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape)
    }


# ================================================================
# SAVE FORECASTING MODEL
# ================================================================

def save_forecasting_model(
    model,
    model_name,
    ticker
):
    """
    Save forecasting model.

    Example:

        models/
        └── RELIANCE_NS/
            ├── gru_model.keras
            └── lstm_model.keras
    """

    ticker_dir = get_ticker_model_dir(
        ticker
    )

    path = (
        ticker_dir
        / f"{model_name}_model.keras"
    )

    model.save(
        path
    )

    print(
        f"Saved forecasting model -> {path}"
    )

    return path


# ================================================================
# SAVE SCALER
# ================================================================

def save_scaler(
    scaler,
    model_name,
    ticker
):
    """
    Save ticker-specific forecasting scaler.

    Example:

        models/
        └── RELIANCE_NS/
            └── scalers/
                └── gru_scaler.pkl
    """

    ticker_dir = get_ticker_model_dir(
        ticker
    )

    scaler_dir = (
        ticker_dir
        / "scalers"
    )

    scaler_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        scaler_dir
        / f"{model_name}_scaler.pkl"
    )

    joblib.dump(
        scaler,
        path
    )

    print(
        f"Saved scaler -> {path}"
    )

    return path


# ================================================================
# LOAD FORECASTING MODEL
# ================================================================

def load_forecasting_model(
    model_name,
    ticker
):
    """
    Load a forecasting model.

    Example:

        load_forecasting_model(
            "gru",
            "RELIANCE.NS"
        )
    """

    ticker_dir = get_ticker_model_dir(
        ticker
    )

    path = (
        ticker_dir
        / f"{model_name}_model.keras"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Forecasting model not found "
            f"for {ticker}: {path}"
        )

    return load_model(
        path
    )


# ================================================================
# LOAD SCALER
# ================================================================

def load_scaler(
    model_name,
    ticker
):
    """
    Load a ticker-specific scaler.
    """

    ticker_dir = get_ticker_model_dir(
        ticker
    )

    scaler_dir = (
        ticker_dir
        / "scalers"
    )

    path = (
        scaler_dir
        / f"{model_name}_scaler.pkl"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Scaler not found "
            f"for {ticker}: {path}"
        )

    return joblib.load(
        path
    )


# ================================================================
# FORECAST NEXT DAY
# ================================================================

def forecast_next_day(
    model,
    df,
    scaler,
    lookback=LOOKBACK
):
    """
    Forecast the next available trading day's
    closing price.
    """

    prices = (
        df["Close"]
        .values
        .reshape(-1, 1)
    )

    if len(prices) < lookback:

        raise ValueError(
            f"At least {lookback} "
            "price observations are required."
        )

    recent_prices = prices[-lookback:]

    scaled_prices = scaler.transform(
        recent_prices
    )

    X = scaled_prices.reshape(
        1,
        lookback,
        1
    )

    prediction_scaled = model.predict(
        X,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction_scaled
    )[0][0]

    return float(
        prediction
    )