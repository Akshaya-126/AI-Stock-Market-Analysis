import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(y_true, y_pred, y_probability=None):
    """
    Calculate classification evaluation metrics.

    Metrics:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC-AUC (when probabilities are provided)

    Args:
        y_true: Actual binary labels.
        y_pred: Predicted binary labels.
        y_probability: Predicted probability for the positive class.

    Returns:
        Dictionary containing evaluation metrics.
    """

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        )
    }

    # ---------------------------------------------------------
    # ROC-AUC
    # ---------------------------------------------------------

    if y_probability is not None:

        try:
            metrics["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    y_probability
                )
            )

        except ValueError:
            # ROC-AUC cannot be calculated when
            # only one class is present.
            metrics["roc_auc"] = None

    return metrics


# ============================================================
# REGRESSION / FORECASTING METRICS
# ============================================================

def regression_metrics(y_true, y_pred):
    """
    Calculate forecasting/regression evaluation metrics.

    Metrics:
        - MAE
        - RMSE
        - MAPE

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary containing evaluation metrics.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # ---------------------------------------------------------
    # Mean Absolute Error
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    # ---------------------------------------------------------
    # Root Mean Squared Error
    # ---------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    # ---------------------------------------------------------
    # Mean Absolute Percentage Error
    #
    # Avoid division by zero.
    # ---------------------------------------------------------

    denominator = np.maximum(
        np.abs(y_true),
        1e-8
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred) / denominator
        )
    ) * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape)
    }