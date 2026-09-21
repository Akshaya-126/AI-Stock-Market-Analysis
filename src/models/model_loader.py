from pathlib import Path

import joblib
from tensorflow.keras.models import load_model as keras_load_model


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models"


# ============================================================
# TICKER DIRECTORY
# ============================================================

def get_ticker_model_dir(ticker):
    """
    Return the model directory for a specific stock ticker.

    Examples:
        RELIANCE.NS -> models/RELIANCE_NS
        TCS.NS      -> models/TCS_NS
        INFY.NS     -> models/INFY_NS
    """

    ticker_folder = ticker.replace(".", "_")

    ticker_dir = MODEL_DIR / ticker_folder

    return ticker_dir


# ============================================================
# CLASSIFICATION MODEL LOADER
# ============================================================

def load_model(
    model_name="xgboost",
    ticker="RELIANCE.NS"
):
    """
    Load a trained classification model.

    Supported examples:
        logistic_regression.pkl
        random_forest.pkl
        xgboost.pkl
    """

    ticker_dir = get_ticker_model_dir(ticker)

    model_path = (
        ticker_dir
        / f"{model_name}.pkl"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Classification model not found:\n"
            f"{model_path}\n\n"
            f"Train the {model_name} model "
            f"for {ticker} first."
        )

    return joblib.load(model_path)


# ============================================================
# FORECASTING MODEL LOADER
# ============================================================

def load_forecasting_model(
    model_name="gru",
    ticker="RELIANCE.NS"
):
    """
    Load a trained LSTM or GRU forecasting model.

    Expected files:

        models/RELIANCE_NS/
            gru_model.keras
            lstm_model.keras
    """

    ticker_dir = get_ticker_model_dir(ticker)

    model_path = (
        ticker_dir
        / f"{model_name}_model.keras"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Forecasting model not found:\n"
            f"{model_path}\n\n"
            f"Train the {model_name} model "
            f"for {ticker} first."
        )

    return keras_load_model(
        model_path
    )


# ============================================================
# FORECASTING SCALER LOADER
# ============================================================

def load_scaler(
    model_name="gru",
    ticker="RELIANCE.NS"
):
    """
    Load the scaler corresponding to
    an LSTM or GRU forecasting model.

    Expected files:

        models/RELIANCE_NS/
            scalers/
                gru_scaler.pkl
                lstm_scaler.pkl
    """

    ticker_dir = get_ticker_model_dir(ticker)

    scaler_path = (
        ticker_dir
        / "scalers"
        / f"{model_name}_scaler.pkl"
    )

    if not scaler_path.exists():

        raise FileNotFoundError(
            f"Forecasting scaler not found:\n"
            f"{scaler_path}\n\n"
            f"Train the {model_name} model "
            f"for {ticker} first."
        )

    return joblib.load(
        scaler_path
    )


# ============================================================
# BACKWARD-COMPATIBILITY ALIAS
# ============================================================

def load_forecasting_scaler(
    model_name="gru",
    ticker="RELIANCE.NS"
):
    """
    Backward-compatible alias.

    Allows older code using:

        load_forecasting_scaler(...)

    to continue working.
    """

    return load_scaler(
        model_name=model_name,
        ticker=ticker
    )


# ============================================================
# CLASSIFICATION PREDICTION
# ============================================================

def predict_latest(
    model,
    features
):
    """
    Generate the latest UP/DOWN prediction.

    Returns:
        prediction
        direction
        probability_up
        probability_down
        confidence
    """

    prediction = int(
        model.predict(features)[0]
    )

    probabilities = model.predict_proba(
        features
    )[0]

    probability_up = float(
        probabilities[1]
    )

    probability_down = float(
        probabilities[0]
    )

    if prediction == 1:

        direction = "UP"
        confidence = probability_up

    else:

        direction = "DOWN"
        confidence = probability_down

    return {

        "prediction": prediction,

        "direction": direction,

        "probability_up": probability_up,

        "probability_down": probability_down,

        "confidence": confidence,

    }