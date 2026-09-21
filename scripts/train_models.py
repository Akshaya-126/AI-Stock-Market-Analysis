import sys
from pathlib import Path

import numpy as np


# ================================================================
# PROJECT ROOT
# ================================================================

# train_models.py is inside:
# AI-Stock-Market-Analysis/scripts/
#
# parents[1] points to:
# AI-Stock-Market-Analysis/

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ================================================================
# PROJECT IMPORTS
# ================================================================

from src.data.data_loader import download_stock_data
from src.data.preprocessing import clean_market_data
from src.features.technical_indicators import add_indicators

from src.models.trend_model import (
    train_models,
    save_models
)

from src.models.forecasting import (
    prepare_forecasting_data,
    naive_forecast,
    train_lstm,
    train_gru,
    predict_prices,
    calculate_forecast_metrics,
    save_forecasting_model,
    save_scaler
)


# ================================================================
# INDIAN STOCKS
# ================================================================

TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
    "BHARTIARTL.NS",
    "MARUTI.NS",
]


# ================================================================
# SETTINGS
# ================================================================

# Historical data used for training
DATA_PERIOD = "5y"

# Chronological train/test split
TRAIN_RATIO = 0.8

# Actual training
# Earlier 5 epochs was only for the smoke test.
FORECAST_EPOCHS = 50

FORECAST_BATCH_SIZE = 32


# ================================================================
# MODEL DIRECTORY
# ================================================================

MODEL_DIR = ROOT / "models"


# ================================================================
# TRAIN ONE STOCK
# ================================================================

def train_stock(ticker):

    print("\n" + "=" * 80)
    print(f"TRAINING STOCK: {ticker}")
    print("=" * 80)

    ticker_folder = ticker.replace(".", "_")

    stock_model_dir = MODEL_DIR / ticker_folder

    print(
        f"\nModel output directory: {stock_model_dir}"
    )

    # ============================================================
    # 1. DOWNLOAD DATA
    # ============================================================

    print(
        f"\n[1/5] Downloading {ticker} historical data..."
    )

    df = download_stock_data(
        ticker=ticker,
        period=DATA_PERIOD
    )

    print(
        f"Raw data shape: {df.shape}"
    )

    # ============================================================
    # 2. CLEAN DATA
    # ============================================================

    print(
        "\n[2/5] Cleaning market data..."
    )

    df = clean_market_data(
        df
    )

    print(
        f"Clean data shape: {df.shape}"
    )

    # ============================================================
    # 3. TECHNICAL INDICATORS
    # ============================================================

    print(
        "\n[3/5] Creating technical indicators..."
    )

    feature_df = add_indicators(
        df
    )

    print(
        f"Feature data shape: {feature_df.shape}"
    )

    # ============================================================
    # 4. CLASSIFICATION MODELS
    # ============================================================

    print(
        "\n[4/5] Training classification models..."
    )

    (
        trained_models,
        classification_results
    ) = train_models(
        feature_df,
        train_ratio=TRAIN_RATIO
    )

    # ------------------------------------------------------------
    # DISPLAY CLASSIFICATION RESULTS
    # ------------------------------------------------------------

    print(
        "\n" + "-" * 80
    )

    print(
        f"CLASSIFICATION RESULTS - {ticker}"
    )

    print(
        "-" * 80
    )

    for name, metrics in classification_results.items():

        print(
            f"\n{name.upper()}"
        )

        print(
            f"Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score : "
            f"{metrics['f1']:.4f}"
        )

        roc_auc = metrics.get(
            "roc_auc"
        )

        if roc_auc is not None:

            print(
                f"ROC-AUC  : "
                f"{roc_auc:.4f}"
            )

        else:

            print(
                "ROC-AUC  : N/A"
            )

        print(
            "Confusion Matrix:"
        )

        print(
            np.array(
                metrics["confusion_matrix"]
            )
        )

    # ------------------------------------------------------------
    # SAVE CLASSIFICATION MODELS
    # ------------------------------------------------------------

    print(
        f"\nSaving classification models for {ticker}..."
    )

    save_models(
        trained_models,
        ticker
    )

    print(
        f"Classification models saved in:"
        f"\n{stock_model_dir}"
    )

    # ============================================================
    # 5. TIME-SERIES FORECASTING
    # ============================================================

    print(
        "\n[5/5] Training time-series forecasting models..."
    )

    # ------------------------------------------------------------
    # Prepare LSTM / GRU data
    # ------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = prepare_forecasting_data(
        df,
        train_ratio=TRAIN_RATIO
    )

    print(
        f"\nForecasting training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Forecasting test samples: "
        f"{len(X_test)}"
    )

    # ============================================================
    # NAIVE BASELINE
    # ============================================================

    print(
        "\nTraining/evaluating naive persistence baseline..."
    )

    (
        naive_predictions,
        naive_actual
    ) = naive_forecast(
        df,
        train_ratio=TRAIN_RATIO
    )

    naive_metrics = calculate_forecast_metrics(
        naive_actual,
        naive_predictions
    )

    print(
        "\nNAIVE BASELINE"
    )

    print(
        f"MAE : {naive_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {naive_metrics['rmse']:.4f}"
    )

    print(
        f"MAPE: {naive_metrics['mape']:.4f}%"
    )

    # ============================================================
    # LSTM
    # ============================================================

    print(
        "\nTraining LSTM..."
    )

    (
        lstm_model,
        lstm_history
    ) = train_lstm(
        X_train,
        y_train,
        epochs=FORECAST_EPOCHS,
        batch_size=FORECAST_BATCH_SIZE
    )

    # ------------------------------------------------------------
    # LSTM predictions
    # ------------------------------------------------------------

    lstm_predictions = predict_prices(
        lstm_model,
        X_test,
        scaler
    )

    lstm_actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).reshape(-1)

    lstm_metrics = calculate_forecast_metrics(
        lstm_actual,
        lstm_predictions
    )

    print(
        "\nLSTM RESULTS"
    )

    print(
        f"MAE : {lstm_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {lstm_metrics['rmse']:.4f}"
    )

    print(
        f"MAPE: {lstm_metrics['mape']:.4f}%"
    )

    # ------------------------------------------------------------
    # Save LSTM model and scaler
    # ------------------------------------------------------------

    print(
        "\nSaving LSTM model..."
    )

    save_forecasting_model(
        lstm_model,
        "lstm",
        ticker
    )

    save_scaler(
        scaler,
        "lstm",
        ticker
    )

    print(
        "LSTM model and scaler saved."
    )

    # ============================================================
    # GRU
    # ============================================================

    print(
        "\nTraining GRU..."
    )

    (
        gru_model,
        gru_history
    ) = train_gru(
        X_train,
        y_train,
        epochs=FORECAST_EPOCHS,
        batch_size=FORECAST_BATCH_SIZE
    )

    # ------------------------------------------------------------
    # GRU predictions
    # ------------------------------------------------------------

    gru_predictions = predict_prices(
        gru_model,
        X_test,
        scaler
    )

    gru_actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).reshape(-1)

    gru_metrics = calculate_forecast_metrics(
        gru_actual,
        gru_predictions
    )

    print(
        "\nGRU RESULTS"
    )

    print(
        f"MAE : {gru_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {gru_metrics['rmse']:.4f}"
    )

    print(
        f"MAPE: {gru_metrics['mape']:.4f}%"
    )

    # ------------------------------------------------------------
    # Save GRU model and scaler
    # ------------------------------------------------------------

    print(
        "\nSaving GRU model..."
    )

    save_forecasting_model(
        gru_model,
        "gru",
        ticker
    )

    save_scaler(
        scaler,
        "gru",
        ticker
    )

    print(
        "GRU model and scaler saved."
    )

    # ============================================================
    # FINAL FORECASTING COMPARISON
    # ============================================================

    print(
        "\n" + "-" * 80
    )

    print(
        f"FORECASTING COMPARISON - {ticker}"
    )

    print(
        "-" * 80
    )

    print(
        "\nModel              MAE          RMSE         MAPE"
    )

    print(
        "-" * 60
    )

    print(
        f"Naive              "
        f"{naive_metrics['mae']:.4f}       "
        f"{naive_metrics['rmse']:.4f}       "
        f"{naive_metrics['mape']:.4f}%"
    )

    print(
        f"LSTM               "
        f"{lstm_metrics['mae']:.4f}       "
        f"{lstm_metrics['rmse']:.4f}       "
        f"{lstm_metrics['mape']:.4f}%"
    )

    print(
        f"GRU                "
        f"{gru_metrics['mae']:.4f}       "
        f"{gru_metrics['rmse']:.4f}       "
        f"{gru_metrics['mape']:.4f}%"
    )

    # ============================================================
    # VERIFY FILES
    # ============================================================

    print(
        "\nVerifying saved model files..."
    )

    expected_files = [
        stock_model_dir / "logistic_regression.pkl",
        stock_model_dir / "random_forest.pkl",
        stock_model_dir / "xgboost.pkl",
        stock_model_dir / "lstm_model.keras",
        stock_model_dir / "gru_model.keras",
        stock_model_dir / "scalers" / "lstm_scaler.pkl",
        stock_model_dir / "scalers" / "gru_scaler.pkl",
    ]

    all_files_exist = True

    for file_path in expected_files:

        if file_path.exists():

            print(
                f"  ✓ {file_path.name}"
            )

        else:

            print(
                f"  ✗ MISSING: {file_path}"
            )

            all_files_exist = False

    if not all_files_exist:

        raise FileNotFoundError(
            f"One or more model files are missing for {ticker}."
        )

    # ============================================================
    # COMPLETION
    # ============================================================

    print(
        "\n" + "=" * 80
    )

    print(
        f"{ticker} TRAINING COMPLETED SUCCESSFULLY"
    )

    print(
        f"Models saved to: {stock_model_dir}"
    )

    print(
        "=" * 80
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "AI STOCK MARKET"
    )

    print(
        "INDIAN STOCK MULTI-MODEL TRAINING PIPELINE"
    )

    print(
        "=" * 80
    )

    print(
        "\nStocks to train:"
    )

    for ticker in TICKERS:

        print(
            f"  - {ticker}"
        )

    print(
        f"\nHistorical period: {DATA_PERIOD}"
    )

    print(
        "Classification split: 80/20 chronological"
    )

    print(
        "Forecasting split: 80/20 chronological"
    )

    print(
        f"LSTM/GRU epochs: {FORECAST_EPOCHS}"
    )

    print(
        f"Batch size: {FORECAST_BATCH_SIZE}"
    )

    print(
        f"Model directory: {MODEL_DIR}"
    )

    # ============================================================
    # TRAIN ALL STOCKS
    # ============================================================

    successful = []

    failed = []

    for ticker in TICKERS:

        try:

            train_stock(
                ticker
            )

            successful.append(
                ticker
            )

        except Exception as e:

            print(
                "\n" + "!" * 80
            )

            print(
                f"ERROR WHILE TRAINING {ticker}"
            )

            print(
                "!" * 80
            )

            print(
                f"Error type: {type(e).__name__}"
            )

            print(
                f"Error message: {e}"
            )

            failed.append(
                ticker
            )

            print(
                f"\nSkipping {ticker} "
                "and continuing..."
            )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "MULTI-STOCK TRAINING SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "\nSuccessfully trained:"
    )

    if successful:

        for ticker in successful:

            print(
                f"  ✓ {ticker}"
            )

    else:

        print(
            "  None"
        )

    print(
        "\nFailed:"
    )

    if failed:

        for ticker in failed:

            print(
                f"  ✗ {ticker}"
            )

    else:

        print(
            "  None"
        )

    print(
        "\nTraining pipeline completed."
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()