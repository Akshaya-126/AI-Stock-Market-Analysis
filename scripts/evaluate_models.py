# ============================================================
# AI STOCK MARKET ANALYSIS
# MODEL EVALUATION SCRIPT
# ============================================================
#
# This script evaluates already-trained models.
#
# Classification:
#   - Logistic Regression
#   - Random Forest
#   - XGBoost
#
# Forecasting:
#   - Naive Persistence Baseline
#   - LSTM
#   - GRU
#
# Results:
#   results/classification_results.csv
#   results/forecasting_results.csv
#
# IMPORTANT:
#   This script does NOT train or retrain models.
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

import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras.models import load_model


# ------------------------------------------------------------
# PROJECT IMPORTS
# ------------------------------------------------------------

from src.data.data_loader import download_stock_data
from src.data.preprocessing import clean_market_data
from src.features.technical_indicators import add_indicators


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.80

LOOKBACK = 60

DATA_PERIOD = "5y"

RESULTS_DIR = ROOT / "results"

MODELS_DIR = ROOT / "models"


# ============================================================
# STOCKS
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

    "Maruti Suzuki": "MARUTI.NS"

}


# ============================================================
# CLASSIFICATION FEATURES
# ============================================================

FEATURES = [

    "SMA20",

    "SMA50",

    "EMA20",

    "RSI14",

    "MACD",

    "MACD_SIGNAL",

    "MACD_HIST",

    "DailyReturn",

    "Volatility20",

    "VolumeSMA20",

    "HighLowRange"

]


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPER: MODEL DIRECTORY
# ============================================================

def get_model_dir(ticker):

    ticker_folder = ticker.replace(
        ".",
        "_"
    )

    return MODELS_DIR / ticker_folder


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_stock_data(ticker):

    print(
        f"\nDownloading data for {ticker}..."
    )

    df = download_stock_data(
        ticker=ticker,
        period=DATA_PERIOD
    )

    df = clean_market_data(
        df
    )

    df = add_indicators(
        df
    )

    if df.empty:

        raise ValueError(
            f"No usable data available for {ticker}."
        )

    return df


# ============================================================
# PREPARE CLASSIFICATION DATA
# ============================================================

def prepare_classification_data(df):

    data = df.copy()

    data["TomorrowClose"] = (
        data["Close"].shift(-1)
    )

    data["Target"] = (
        data["TomorrowClose"]
        > data["Close"]
    ).astype(int)

    data = data.dropna(
        subset=FEATURES + [
            "Target"
        ]
    )

    X = data[FEATURES]

    y = data["Target"]

    split_index = int(
        len(data) * TRAIN_RATIO
    )

    X_test = X.iloc[
        split_index:
    ]

    y_test = y.iloc[
        split_index:
    ]

    return X_test, y_test


# ============================================================
# EVALUATE CLASSIFICATION MODEL
# ============================================================

def evaluate_classification_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    predictions = np.asarray(
        predictions
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = np.nan

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        if y_test.nunique() >= 2:

            roc_auc = roc_auc_score(
                y_test,
                probabilities
            )

    return {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc

    }


# ============================================================
# LOAD CLASSIFICATION MODEL
# ============================================================

def load_classification_model(
    ticker,
    model_name
):

    model_dir = get_model_dir(
        ticker
    )

    model_path = (
        model_dir
        / f"{model_name}.pkl"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(
        model_path
    )


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_classification_for_stock(
    stock_name,
    ticker,
    df
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"CLASSIFICATION EVALUATION: "
        f"{stock_name} ({ticker})"
    )

    print(
        f"{'=' * 60}"
    )

    X_test, y_test = (
        prepare_classification_data(
            df
        )
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    model_names = [

        "logistic_regression",

        "random_forest",

        "xgboost"

    ]

    results = []

    for model_name in model_names:

        try:

            print(
                f"\nEvaluating {model_name}..."
            )

            model = (
                load_classification_model(
                    ticker,
                    model_name
                )
            )

            metrics = (
                evaluate_classification_model(
                    model,
                    X_test,
                    y_test
                )
            )

            result = {

                "Stock": stock_name,

                "Ticker": ticker,

                "Model": model_name,

                "Test_Samples": len(X_test),

                "Accuracy": metrics[
                    "Accuracy"
                ],

                "Precision": metrics[
                    "Precision"
                ],

                "Recall": metrics[
                    "Recall"
                ],

                "F1": metrics[
                    "F1"
                ],

                "ROC_AUC": metrics[
                    "ROC_AUC"
                ]

            }

            results.append(
                result
            )

            print(
                f"Accuracy : "
                f"{metrics['Accuracy']:.4f}"
            )

            print(
                f"Precision: "
                f"{metrics['Precision']:.4f}"
            )

            print(
                f"Recall   : "
                f"{metrics['Recall']:.4f}"
            )

            print(
                f"F1       : "
                f"{metrics['F1']:.4f}"
            )

            if pd.isna(
                metrics["ROC_AUC"]
            ):

                print(
                    "ROC-AUC  : N/A"
                )

            else:

                print(
                    f"ROC-AUC  : "
                    f"{metrics['ROC_AUC']:.4f}"
                )

        except Exception as e:

            print(
                f"ERROR evaluating "
                f"{model_name}: {e}"
            )

    return results


# ============================================================
# PREPARE FORECASTING DATA
# ============================================================

def prepare_forecasting_data(
    df
):

    prices = (
        df["Close"]
        .values
        .astype(float)
    )

    total_samples = len(
        prices
    )

    split_index = int(
        total_samples
        * TRAIN_RATIO
    )

    if split_index <= LOOKBACK:

        raise ValueError(
            "Not enough data for "
            "forecasting evaluation."
        )

    train_prices = prices[
        :split_index
    ]

    test_prices = prices[
        split_index:
    ]

    return (
        prices,
        train_prices,
        test_prices,
        split_index
    )


# ============================================================
# BUILD TEST SEQUENCES
# ============================================================

def build_test_sequences(
    prices,
    scaler,
    split_index
):

    scaled_prices = scaler.transform(
        prices.reshape(-1, 1)
    )

    X_test = []

    y_test = []

    actual_test_prices = []

    start_index = split_index

    for i in range(
        start_index,
        len(prices)
    ):

        if i < LOOKBACK:

            continue

        sequence = (
            scaled_prices[
                i - LOOKBACK:i
            ]
        )

        X_test.append(
            sequence
        )

        y_test.append(
            scaled_prices[i]
        )

        actual_test_prices.append(
            prices[i]
        )

    X_test = np.asarray(
        X_test
    )

    y_test = np.asarray(
        y_test
    )

    actual_test_prices = np.asarray(
        actual_test_prices
    )

    return (
        X_test,
        y_test,
        actual_test_prices
    )


# ============================================================
# INVERSE TRANSFORM PREDICTIONS
# ============================================================

def inverse_transform_prices(
    scaled_values,
    scaler
):

    scaled_values = np.asarray(
        scaled_values
    ).reshape(-1, 1)

    return scaler.inverse_transform(
        scaled_values
    ).flatten()


# ============================================================
# FORECASTING METRICS
# ============================================================

def calculate_forecasting_metrics(
    actual,
    predicted
):

    actual = np.asarray(
        actual
    )

    predicted = np.asarray(
        predicted
    )

    mae = np.mean(
        np.abs(
            actual - predicted
        )
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    non_zero_mask = (
        actual != 0
    )

    if np.any(
        non_zero_mask
    ):

        mape = np.mean(
            np.abs(
                (
                    actual[
                        non_zero_mask
                    ]
                    -
                    predicted[
                        non_zero_mask
                    ]
                )
                /
                actual[
                    non_zero_mask
                ]
            )
        ) * 100

    else:

        mape = np.nan

    return {

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape

    }


# ============================================================
# NAIVE FORECAST
# ============================================================

def naive_forecast(
    prices,
    split_index
):

    actual_test_prices = (
        prices[split_index:]
    )

    previous_prices = (
        prices[split_index - 1:-1]
    )

    return (
        actual_test_prices,
        previous_prices
    )


# ============================================================
# LOAD FORECASTING MODEL
# ============================================================

def load_forecasting_model(
    ticker,
    model_name
):

    model_dir = get_model_dir(
        ticker
    )

    model_path = (
        model_dir
        / f"{model_name}_model.keras"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Forecasting model not found: "
            f"{model_path}"
        )

    return load_model(
        model_path
    )


# ============================================================
# LOAD FORECASTING SCALER
# ============================================================

def load_forecasting_scaler(
    ticker,
    model_name
):

    model_dir = get_model_dir(
        ticker
    )

    scaler_path = (
        model_dir
        / "scalers"
        / f"{model_name}_scaler.pkl"
    )

    if not scaler_path.exists():

        raise FileNotFoundError(
            f"Forecasting scaler not found: "
            f"{scaler_path}"
        )

    return joblib.load(
        scaler_path
    )


# ============================================================
# EVALUATE LSTM / GRU
# ============================================================

def evaluate_neural_forecast(
    ticker,
    model_name,
    prices,
    split_index
):

    model = load_forecasting_model(
        ticker,
        model_name
    )

    scaler = load_forecasting_scaler(
        ticker,
        model_name
    )

    (
        X_test,
        y_test,
        actual_prices
    ) = build_test_sequences(
        prices,
        scaler,
        split_index
    )

    if len(X_test) == 0:

        raise ValueError(
            "No test sequences were created."
        )

    scaled_predictions = (
        model.predict(
            X_test,
            verbose=0
        )
    )

    predicted_prices = (
        inverse_transform_prices(
            scaled_predictions,
            scaler
        )
    )

    metrics = (
        calculate_forecasting_metrics(
            actual_prices,
            predicted_prices
        )
    )

    return (
        metrics,
        len(actual_prices)
    )


# ============================================================
# FORECASTING EVALUATION
# ============================================================

def evaluate_forecasting_for_stock(
    stock_name,
    ticker,
    df
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"FORECASTING EVALUATION: "
        f"{stock_name} ({ticker})"
    )

    print(
        f"{'=' * 60}"
    )

    (
        prices,
        train_prices,
        test_prices,
        split_index
    ) = prepare_forecasting_data(
        df
    )

    results = []

    # --------------------------------------------------------
    # NAIVE BASELINE
    # --------------------------------------------------------

    print(
        "\nEvaluating Naive Persistence Baseline..."
    )

    (
        naive_actual,
        naive_predictions
    ) = naive_forecast(
        prices,
        split_index
    )

    naive_metrics = (
        calculate_forecasting_metrics(
            naive_actual,
            naive_predictions
        )
    )

    naive_result = {

        "Stock": stock_name,

        "Ticker": ticker,

        "Model": "Naive",

        "Test_Samples": len(
            naive_actual
        ),

        "MAE": naive_metrics[
            "MAE"
        ],

        "RMSE": naive_metrics[
            "RMSE"
        ],

        "MAPE": naive_metrics[
            "MAPE"
        ]

    }

    results.append(
        naive_result
    )

    print(
        f"MAE : "
        f"{naive_metrics['MAE']:.4f}"
    )

    print(
        f"RMSE: "
        f"{naive_metrics['RMSE']:.4f}"
    )

    print(
        f"MAPE: "
        f"{naive_metrics['MAPE']:.4f}%"
    )

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    try:

        print(
            "\nEvaluating LSTM..."
        )

        (
            lstm_metrics,
            lstm_samples
        ) = evaluate_neural_forecast(
            ticker,
            "lstm",
            prices,
            split_index
        )

        lstm_result = {

            "Stock": stock_name,

            "Ticker": ticker,

            "Model": "LSTM",

            "Test_Samples": lstm_samples,

            "MAE": lstm_metrics[
                "MAE"
            ],

            "RMSE": lstm_metrics[
                "RMSE"
            ],

            "MAPE": lstm_metrics[
                "MAPE"
            ]

        }

        results.append(
            lstm_result
        )

        print(
            f"MAE : "
            f"{lstm_metrics['MAE']:.4f}"
        )

        print(
            f"RMSE: "
            f"{lstm_metrics['RMSE']:.4f}"
        )

        print(
            f"MAPE: "
            f"{lstm_metrics['MAPE']:.4f}%"
        )

    except Exception as e:

        print(
            f"ERROR evaluating LSTM: {e}"
        )

    # --------------------------------------------------------
    # GRU
    # --------------------------------------------------------

    try:

        print(
            "\nEvaluating GRU..."
        )

        (
            gru_metrics,
            gru_samples
        ) = evaluate_neural_forecast(
            ticker,
            "gru",
            prices,
            split_index
        )

        gru_result = {

            "Stock": stock_name,

            "Ticker": ticker,

            "Model": "GRU",

            "Test_Samples": gru_samples,

            "MAE": gru_metrics[
                "MAE"
            ],

            "RMSE": gru_metrics[
                "RMSE"
            ],

            "MAPE": gru_metrics[
                "MAPE"
            ]

        }

        results.append(
            gru_result
        )

        print(
            f"MAE : "
            f"{gru_metrics['MAE']:.4f}"
        )

        print(
            f"RMSE: "
            f"{gru_metrics['RMSE']:.4f}"
        )

        print(
            f"MAPE: "
            f"{gru_metrics['MAPE']:.4f}%"
        )

    except Exception as e:

        print(
            f"ERROR evaluating GRU: {e}"
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "AI STOCK MARKET ANALYSIS"
    )

    print(
        "FINAL MODEL EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This script evaluates existing trained models."
    )

    print(
        "No models will be retrained."
    )

    print(
        f"Data period : {DATA_PERIOD}"
    )

    print(
        f"Train ratio : {TRAIN_RATIO}"
    )

    print(
        f"Lookback    : {LOOKBACK}"
    )

    classification_results = []

    forecasting_results = []

    failed_stocks = []

    # --------------------------------------------------------
    # PROCESS EACH STOCK
    # --------------------------------------------------------

    for stock_name, ticker in INDIAN_STOCKS.items():

        print(
            "\n"
            + "#" * 70
        )

        print(
            f"PROCESSING: {stock_name} "
            f"({ticker})"
        )

        print(
            "#" * 70
        )

        try:

            df = load_stock_data(
                ticker
            )

            print(
                f"Usable rows: {len(df)}"
            )

            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

            classification_stock_results = (
                evaluate_classification_for_stock(
                    stock_name,
                    ticker,
                    df
                )
            )

            classification_results.extend(
                classification_stock_results
            )

            # ------------------------------------------------
            # FORECASTING
            # ------------------------------------------------

            forecasting_stock_results = (
                evaluate_forecasting_for_stock(
                    stock_name,
                    ticker,
                    df
                )
            )

            forecasting_results.extend(
                forecasting_stock_results
            )

            print(
                f"\n✓ Completed: {ticker}"
            )

        except Exception as e:

            print(
                f"\n✗ FAILED: {ticker}"
            )

            print(
                f"Reason: {e}"
            )

            failed_stocks.append({

                "Stock": stock_name,

                "Ticker": ticker,

                "Error": str(e)

            })

    # ========================================================
    # SAVE CLASSIFICATION RESULTS
    # ========================================================

    if classification_results:

        classification_df = pd.DataFrame(
            classification_results
        )

        classification_path = (
            RESULTS_DIR
            / "classification_results.csv"
        )

        classification_df.to_csv(
            classification_path,
            index=False
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CLASSIFICATION RESULTS SAVED"
        )

        print(
            classification_path
        )

    else:

        print(
            "\nNo classification results generated."
        )

    # ========================================================
    # SAVE FORECASTING RESULTS
    # ========================================================

    if forecasting_results:

        forecasting_df = pd.DataFrame(
            forecasting_results
        )

        forecasting_path = (
            RESULTS_DIR
            / "forecasting_results.csv"
        )

        forecasting_df.to_csv(
            forecasting_path,
            index=False
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FORECASTING RESULTS SAVED"
        )

        print(
            forecasting_path
        )

    else:

        print(
            "\nNo forecasting results generated."
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nStocks processed: "
        f"{len(INDIAN_STOCKS)}"
    )

    print(
        f"Stocks failed: "
        f"{len(failed_stocks)}"
    )

    if failed_stocks:

        print(
            "\nFailed stocks:"
        )

        for item in failed_stocks:

            print(
                f"  ✗ {item['Ticker']}: "
                f"{item['Error']}"
            )

    else:

        print(
            "\n✓ All stocks evaluated successfully."
        )

    print(
        "\nGenerated files:"
    )

    print(
        "  results/classification_results.csv"
    )

    print(
        "  results/forecasting_results.csv"
    )

    print(
        "\nEvaluation completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()