"""
Data cleaning and validation module for AgroPrice AI.
Standardizes column names, date representations, commodity names, and cleans price anomalies.
"""

from typing import Tuple, List, Dict
import pandas as pd
import numpy as np


# Standard Agmarknet commodity alias mapping
COMMODITY_ALIASES: Dict[str, str] = {
    "wheat": "Wheat",
    "gehun": "Wheat",
    "rice": "Rice",
    "paddy": "Rice",
    "paddy(dhan)(common)": "Rice",
    "paddy(dhan)(basmati)": "Rice",
    "onion": "Onion",
    "pyaz": "Onion",
    "tomato": "Tomato",
    "tamatar": "Tomato",
    "potato": "Potato",
    "aloo": "Potato",
    "soyabean": "Soybean",
    "soybean": "Soybean",
    "cotton": "Cotton",
    "kapas": "Cotton",
    "maize": "Maize",
    "makka": "Maize",
    "turmeric": "Turmeric",
    "haldi": "Turmeric",
    "gram": "Gram",
    "chana": "Gram",
    "gram raw(chana)": "Gram",
    "bengal gram": "Gram"
}

REQUIRED_COLUMNS = [
    "Arrival_Date",
    "State",
    "District",
    "Market",
    "Commodity",
    "Modal_Price"
]


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates if the provided dataframe contains the necessary columns for price analysis and modeling.
    Returns (is_valid, list_of_errors_or_missing_columns).
    """
    if df is None or df.empty:
        return False, ["Dataset is empty or None."]

    # Normalize column names in df for flexible matching
    normalized_cols = {col.strip().lower().replace(" ", "_"): col for col in df.columns}
    missing_cols = []

    for req_col in REQUIRED_COLUMNS:
        req_norm = req_col.lower().replace(" ", "_")
        if req_norm not in normalized_cols:
            missing_cols.append(req_col)

    if missing_cols:
        return False, [f"Missing required column(s): {', '.join(missing_cols)}"]

    return True, []


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes incoming column names to the standard Agmarknet schema."""
    df_copy = df.copy()
    col_mapping = {}

    for col in df_copy.columns:
        clean_col = col.strip().lower().replace(" ", "_")
        if "arrival" in clean_col or "date" in clean_col:
            col_mapping[col] = "Arrival_Date"
        elif clean_col in ["state", "state_name"]:
            col_mapping[col] = "State"
        elif clean_col in ["district", "district_name"]:
            col_mapping[col] = "District"
        elif clean_col in ["market", "market_name", "mandi"]:
            col_mapping[col] = "Market"
        elif clean_col in ["commodity", "crop", "crop_name"]:
            col_mapping[col] = "Commodity"
        elif clean_col in ["variety", "grade"]:
            col_mapping[col] = "Variety"
        elif "min" in clean_col and "price" in clean_col:
            col_mapping[col] = "Min_Price"
        elif "max" in clean_col and "price" in clean_col:
            col_mapping[col] = "Max_Price"
        elif "modal" in clean_col and "price" in clean_col:
            col_mapping[col] = "Modal_Price"

    df_copy = df_copy.rename(columns=col_mapping)
    return df_copy


def standardize_crop_name(name: str) -> str:
    """Standardizes crop/commodity names using the alias dictionary."""
    if not isinstance(name, str):
        return "Unknown"
    cleaned = name.strip().lower()
    return COMMODITY_ALIASES.get(cleaned, name.strip().title())


def clean_agricultural_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs comprehensive data cleaning on agricultural market records:
    - Standardizes column names
    - Parses dates safely
    - Normalizes crop names and location strings
    - Cleans and casts price columns
    - Validates price ranges (Min <= Modal <= Max)
    - Removes invalid, duplicate, and null records
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df_clean = standardize_column_names(df)

    # Check minimum required columns
    is_valid, errors = validate_dataset_schema(df_clean)
    if not is_valid:
        raise ValueError(f"Schema validation failed: {'; '.join(errors)}")

    # Parse Arrival_Date
    df_clean["Arrival_Date"] = pd.to_datetime(
        df_clean["Arrival_Date"],
        errors="coerce",
        dayfirst=True
    )
    df_clean = df_clean.dropna(subset=["Arrival_Date"])

    # Clean text columns
    text_cols = ["State", "District", "Market"]
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
        else:
            df_clean[col] = "Unknown"

    if "Variety" in df_clean.columns:
        df_clean["Variety"] = df_clean["Variety"].fillna("Standard").astype(str).str.strip().str.title()
    else:
        df_clean["Variety"] = "Standard"

    # Standardize Commodity
    df_clean["Commodity"] = df_clean["Commodity"].apply(standardize_crop_name)

    # Clean price columns
    price_cols = ["Modal_Price"]
    if "Min_Price" in df_clean.columns:
        price_cols.append("Min_Price")
    if "Max_Price" in df_clean.columns:
        price_cols.append("Max_Price")

    for col in price_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Drop invalid or non-positive modal prices
    df_clean = df_clean.dropna(subset=["Modal_Price"])
    df_clean = df_clean[df_clean["Modal_Price"] > 0]

    # Fill Min_Price and Max_Price if missing
    if "Min_Price" not in df_clean.columns or df_clean["Min_Price"].isna().all():
        df_clean["Min_Price"] = df_clean["Modal_Price"] * 0.95
    else:
        df_clean["Min_Price"] = df_clean["Min_Price"].fillna(df_clean["Modal_Price"] * 0.95)

    if "Max_Price" not in df_clean.columns or df_clean["Max_Price"].isna().all():
        df_clean["Max_Price"] = df_clean["Modal_Price"] * 1.05
    else:
        df_clean["Max_Price"] = df_clean["Max_Price"].fillna(df_clean["Modal_Price"] * 1.05)

    # Sanity bounds: Min <= Modal <= Max
    df_clean["Min_Price"] = np.minimum(df_clean["Min_Price"], df_clean["Modal_Price"])
    df_clean["Max_Price"] = np.maximum(df_clean["Max_Price"], df_clean["Modal_Price"])

    # Filter out unreasonable prices (< ₹100 or > ₹100,000 per quintal)
    df_clean = df_clean[(df_clean["Modal_Price"] >= 100) & (df_clean["Modal_Price"] <= 100000)]

    # Remove exact duplicate rows
    df_clean = df_clean.drop_duplicates()

    # Sort chronologically
    df_clean = df_clean.sort_values(by=["Arrival_Date", "Commodity", "Market"]).reset_index(drop=True)

    return df_clean
