"""
Supervised stock trend prediction utilities.

Uses chronological train/test splitting to avoid time-series leakage.
Models are saved separately for each stock ticker.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


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
# FEATURES
# ================================================================

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
    "HighLowRange",
]


# ================================================================
# TARGET CREATION
# ================================================================

def prepare_target(df):
    """
    Create the next-day UP/DOWN target.

    1 -> tomorrow's closing price is higher
    0 -> tomorrow's closing price is equal/lower
    """

    data = df.copy()

    data["TomorrowClose"] = data["Close"].shift(-1)

    data["Target"] = (
        data["TomorrowClose"] > data["Close"]
    ).astype(int)

    data = data.dropna(
        subset=["TomorrowClose"]
    )

    return data


# ================================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ================================================================

def prepare_train_test(
    df,
    train_ratio=0.8
):
    """
    Prepare features and split chronologically.

    No random shuffling is used because
    stock data is time-series data.
    """

    data = prepare_target(df)

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = data[FEATURES]

    y = data["Target"]

    split_index = int(
        len(data) * train_ratio
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ================================================================
# CREATE MODELS
# ================================================================

def create_models():

    models = {

        "logistic_regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            ),
        ]),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    return models


# ================================================================
# MODEL EVALUATION
# ================================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {

        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions
        ),
    }

    return metrics


# ================================================================
# TRAIN ALL MODELS
# ================================================================

def train_models(
    df,
    train_ratio=0.8
):

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_train_test(
        df,
        train_ratio=train_ratio
    )

    models = create_models()

    trained_models = {}

    results = {}

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        trained_models[name] = model

        results[name] = metrics

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score : {metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )

        print(
            "Confusion Matrix:"
        )

        print(
            metrics["confusion_matrix"]
        )

    return (
        trained_models,
        results
    )


# ================================================================
# SAVE MODELS
# ================================================================

def save_models(
    trained_models,
    ticker
):
    """
    Save models for a specific stock.

    Example:

    RELIANCE.NS

    becomes:

    models/
        RELIANCE_NS/
            logistic_regression.pkl
            random_forest.pkl
            xgboost.pkl
    """

    ticker_folder = ticker.replace(
        ".",
        "_"
    )

    ticker_model_dir = (
        MODEL_DIR / ticker_folder
    )

    ticker_model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_paths = {}

    for name, model in trained_models.items():

        path = (
            ticker_model_dir
            / f"{name}.pkl"
        )

        joblib.dump(
            model,
            path
        )

        saved_paths[name] = path

        print(
            f"Saved {name} -> {path}"
        )

    return saved_paths


# ================================================================
# LOAD TICKER-SPECIFIC MODEL
# ================================================================

def load_model(
    model_name,
    ticker
):
    """
    Load a model for a specific stock.

    Example:

    load_model(
        "xgboost",
        "RELIANCE.NS"
    )
    """

    ticker_folder = ticker.replace(
        ".",
        "_"
    )

    path = (
        MODEL_DIR
        / ticker_folder
        / f"{model_name}.pkl"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Model not found for {ticker}: {path}"
        )

    return joblib.load(
        path
    )


# ================================================================
# LATEST MARKET DIRECTION
# ================================================================

def predict_latest(
    model,
    df
):
    """
    Predict the latest available market direction.

    Returns:
        UP/DOWN
        probability_up
        probability_down
        confidence
    """

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    latest_features = (
        df[FEATURES]
        .iloc[[-1]]
    )

    prediction = int(
        model.predict(
            latest_features
        )[0]
    )

    probability_up = float(
        model.predict_proba(
            latest_features
        )[0][1]
    )

    if prediction == 1:

        direction = "UP"
        confidence = probability_up

    else:

        direction = "DOWN"
        confidence = 1 - probability_up

    return {

        "prediction": prediction,

        "direction": direction,

        "probability_up": probability_up,

        "probability_down": (
            1 - probability_up
        ),

        "confidence": confidence,
    }


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

def get_feature_importance(
    model
):
    """
    Return feature importance for
    tree-based models such as
    Random Forest and XGBoost.
    """

    if not hasattr(
        model,
        "feature_importances_"
    ):

        raise ValueError(
            "This model does not provide "
            "tree-based feature importance."
        )

    importance = pd.DataFrame({

        "feature": FEATURES,

        "importance":
            model.feature_importances_,
    })

    return (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )