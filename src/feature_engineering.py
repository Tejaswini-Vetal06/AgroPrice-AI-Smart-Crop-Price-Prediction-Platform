"""
Feature engineering module for AgroPrice AI.
Extracts calendar features, cyclical seasonality variables, agricultural seasons,
and sets up preprocessors without data leakage.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.config import CROP_SEASONS


CATEGORICAL_FEATURES = ["Commodity", "State", "District", "Market", "Variety", "Season"]
NUMERICAL_FEATURES = ["Year", "Month", "sin_month", "cos_month", "Quarter"]


def get_season_from_month(month: int) -> str:
    """
    Maps calendar month to typical Indian agricultural seasons:
    - Kharif: July (7) to October (10)
    - Rabi: November (11) to March (3)
    - Zaid: April (4) to June (6)
    """
    if month in [7, 8, 9, 10]:
        return "Kharif"
    elif month in [11, 12, 1, 2, 3]:
        return "Rabi"
    else:
        return "Zaid"


def extract_features(df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
    """
    Extracts time-based, cyclical, and agricultural features from the raw records.
    Works for both historical training datasets and inference queries.
    """
    data = df.copy()

    # If Arrival_Date is present, extract date components
    if "Arrival_Date" in data.columns and not data["Arrival_Date"].isna().all():
        data["Arrival_Date"] = pd.to_datetime(data["Arrival_Date"])
        data["Year"] = data["Arrival_Date"].dt.year
        data["Month"] = data["Arrival_Date"].dt.month
        data["Quarter"] = data["Arrival_Date"].dt.quarter
    else:
        # Expected for inference where Year and Month are passed directly
        if "Year" not in data.columns or "Month" not in data.columns:
            raise ValueError("Input data must contain either 'Arrival_Date' or ('Year' and 'Month').")
        data["Year"] = data["Year"].astype(int)
        data["Month"] = data["Month"].astype(int)
        data["Quarter"] = ((data["Month"] - 1) // 3 + 1).astype(int)

    # Cyclical seasonal transformation
    data["sin_month"] = np.sin(2 * np.pi * data["Month"] / 12.0)
    data["cos_month"] = np.cos(2 * np.pi * data["Month"] / 12.0)

    # Assign season if not present
    if "Season" not in data.columns or data["Season"].isna().all():
        data["Season"] = data["Month"].apply(get_season_from_month)

    # Ensure Variety is present
    if "Variety" not in data.columns:
        data["Variety"] = "Standard"
    else:
        data["Variety"] = data["Variety"].fillna("Standard").astype(str).str.strip().str.title()

    # Standardize string categories
    for col in ["Commodity", "State", "District", "Market"]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip().str.title()

    return data


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer that scales numerical features
    and one-hot encodes categorical features with safe unknown-handling.
    """
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts features and separates features X and target price y from cleaned dataset.
    """
    if "Modal_Price" not in df.columns:
        raise ValueError("DataFrame must contain 'Modal_Price' for model training.")

    featured_df = extract_features(df)
    
    # Feature columns
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = featured_df[feature_cols].copy()
    y = featured_df["Modal_Price"].copy()

    return X, y
