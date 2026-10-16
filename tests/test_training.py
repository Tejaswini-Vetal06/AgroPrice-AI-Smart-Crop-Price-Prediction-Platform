"""
Unit tests for model training, time-aware splitting, evaluation, and persistence.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_MODEL_PATH, METADATA_PATH
from src.data_loader import DataLoader
from src.evaluate import calculate_metrics, compute_prediction_interval
from src.train import split_time_aware, train_pipeline
from src.model_utils import load_trained_model, load_model_metadata, is_model_ready


def test_split_time_aware_chronological():
    """Verify that split_time_aware enforces strict chronological partitioning without leakage."""
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="W")
    dummy_df = pd.DataFrame({
        "Arrival_Date": dates,
        "Commodity": ["Wheat"] * len(dates),
        "State": ["Punjab"] * len(dates),
        "District": ["Ludhiana"] * len(dates),
        "Market": ["Khanna"] * len(dates),
        "Modal_Price": np.random.uniform(2000, 2500, len(dates))
    })

    train_df, test_df = split_time_aware(dummy_df, test_ratio=0.25)

    assert len(train_df) + len(test_df) == len(dummy_df)
    assert train_df["Arrival_Date"].max() <= test_df["Arrival_Date"].min()


def test_calculate_metrics():
    """Verify calculation of MAE, RMSE, and R2 regression metrics."""
    y_true = np.array([2000.0, 2500.0, 3000.0, 3500.0])
    y_pred = np.array([2050.0, 2480.0, 2950.0, 3520.0])

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]
    assert 0.9 <= metrics["r2"] <= 1.0
    assert metrics["mape"] > 0


def test_compute_prediction_interval():
    """Verify residual standard deviation and empirical error bound calculation."""
    residuals = np.random.normal(0, 150, 1000)
    res_std, margin_error, mae = compute_prediction_interval(residuals, confidence_level=0.90)

    assert 130 < res_std < 170
    assert margin_error > res_std
    assert mae > 0


def test_trained_model_and_metadata_exist():
    """Verify trained model file and metadata JSON exist on disk and load correctly."""
    assert is_model_ready() is True

    pipeline = load_trained_model()
    assert hasattr(pipeline, "predict")

    metadata = load_model_metadata()
    assert "champion_model" in metadata
    assert "supported_crops" in metadata
    assert len(metadata["supported_crops"]) >= 10
    assert "prediction_interval" in metadata
    assert "margin_error_90" in metadata["prediction_interval"]
