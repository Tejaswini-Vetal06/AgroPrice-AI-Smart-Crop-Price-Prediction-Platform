"""
Model evaluation module for AgroPrice AI.
Computes MAE, RMSE, R2, MAPE, crop-level performance breakdowns, and prediction intervals.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates standard regression performance metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # Avoid zero division in MAPE
    non_zero = y_true != 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
    else:
        mape = 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2)
    }


def evaluate_crop_breakdown(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Evaluates prediction metrics broken down by individual commodity."""
    eval_df = df_test.copy()
    eval_df["y_true"] = y_true
    eval_df["y_pred"] = y_pred

    breakdown = {}
    for crop, group in eval_df.groupby("Commodity"):
        metrics = calculate_metrics(group["y_true"].values, group["y_pred"].values)
        metrics["count"] = int(len(group))
        metrics["mean_price"] = float(round(group["y_true"].mean(), 2))
        breakdown[crop] = metrics

    return breakdown


def compute_prediction_interval(
    residuals: np.ndarray,
    confidence_level: float = 0.90
) -> Tuple[float, float, float]:
    """
    Computes residual standard deviation and empirical error margins for price bounds.
    Returns (residual_std, margin_error, mae).
    """
    residual_std = float(np.std(residuals))
    mae = float(np.mean(np.abs(residuals)))

    # Use standard normal multiplier: 1.645 for 90%, 1.96 for 95%
    if confidence_level >= 0.95:
        z = 1.96
    else:
        z = 1.645

    margin_error = round(z * residual_std, 2)
    return round(residual_std, 2), margin_error, round(mae, 2)
