"""
Unit tests for data cleaning, normalization, and feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    clean_agricultural_data,
    standardize_crop_name,
    COMMODITY_ALIASES
)
from src.feature_engineering import (
    extract_features,
    get_season_from_month,
    build_preprocessor,
    prepare_training_data
)


def test_standardize_crop_aliases():
    """Verify alias mapping for vernacular and variant crop names."""
    assert standardize_crop_name("paddy") == "Rice"
    assert standardize_crop_name("paddy(dhan)(common)") == "Rice"
    assert standardize_crop_name("pyaz") == "Onion"
    assert standardize_crop_name("aloo") == "Potato"
    assert standardize_crop_name("chana") == "Gram"
    assert standardize_crop_name("soyabean") == "Soybean"
    assert standardize_crop_name("Wheat") == "Wheat"


def test_clean_agricultural_data_anomalies():
    """Verify handling of invalid, zero, negative prices and duplicates."""
    dirty_data = pd.DataFrame({
        "Arrival_Date": ["2024-03-01", "2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04"],
        "State": [" Punjab ", " Punjab ", "Maharashtra", "UP", "MP"],
        "District": ["Ludhiana", "Ludhiana", "Nashik", "Agra", "Indore"],
        "Market": ["Khanna", "Khanna", "Lasalgaon", "Agra", "Indore"],
        "Commodity": ["wheat", "wheat", "pyaz", "potato", "soybean"],
        "Variety": ["Lokwan", "Lokwan", "Red", "Jyoti", "Yellow"],
        "Min_Price": [2100, 2100, -500, 1000, 4000],
        "Max_Price": [2400, 2400, 2000, 1500, 5000],
        "Modal_Price": [2250, 2250, 0, 1200, 999999]  # 0 is invalid, 999999 is an outlier
    })

    cleaned = clean_agricultural_data(dirty_data)

    # Duplicate row removed, zero modal removed, extreme outlier removed
    assert len(cleaned) == 2
    assert "Wheat" in cleaned["Commodity"].values
    assert "Potato" in cleaned["Commodity"].values
    assert 0 not in cleaned["Modal_Price"].values
    assert cleaned["State"].iloc[0] == "Punjab"  # Trimmed whitespace


def test_season_mapping():
    """Verify calendar month to Indian agricultural season mapping."""
    assert get_season_from_month(7) == "Kharif"   # July
    assert get_season_from_month(8) == "Kharif"   # August
    assert get_season_from_month(12) == "Rabi"    # December
    assert get_season_from_month(2) == "Rabi"     # February
    assert get_season_from_month(5) == "Zaid"     # May


def test_extract_features_cyclical_and_calendar():
    """Verify date feature extraction and cyclical transformations."""
    df = pd.DataFrame([{
        "Arrival_Date": "2024-06-15",
        "State": "Maharashtra",
        "District": "Nashik",
        "Market": "Lasalgaon",
        "Commodity": "Onion",
        "Variety": "Red",
        "Modal_Price": 2200.0
    }])

    featured = extract_features(df)

    assert "Year" in featured.columns
    assert "Month" in featured.columns
    assert "sin_month" in featured.columns
    assert "cos_month" in featured.columns
    assert "Season" in featured.columns

    assert featured["Year"].iloc[0] == 2024
    assert featured["Month"].iloc[0] == 6
    assert featured["Season"].iloc[0] == "Zaid"

    # In June (month 6), sin(2*pi*6/12) is sin(pi) ~ 0, cos(pi) = -1
    assert np.isclose(featured["sin_month"].iloc[0], 0.0, atol=1e-5)
    assert np.isclose(featured["cos_month"].iloc[0], -1.0, atol=1e-5)


def test_preprocessor_transformation():
    """Verify ColumnTransformer transforms features without errors."""
    df = pd.DataFrame([
        {
            "Arrival_Date": "2024-01-10",
            "State": "Punjab",
            "District": "Ludhiana",
            "Market": "Khanna",
            "Commodity": "Wheat",
            "Variety": "Lokwan",
            "Modal_Price": 2200.0
        },
        {
            "Arrival_Date": "2024-07-20",
            "State": "Maharashtra",
            "District": "Nashik",
            "Market": "Lasalgaon",
            "Commodity": "Onion",
            "Variety": "Red",
            "Modal_Price": 1800.0
        }
    ])

    X, y = prepare_training_data(df)
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)

    assert X_trans.shape[0] == 2
    assert X_trans.shape[1] > 5
    assert not np.isnan(X_trans).any()
