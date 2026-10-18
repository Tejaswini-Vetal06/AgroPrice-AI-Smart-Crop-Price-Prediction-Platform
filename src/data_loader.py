"""
Data loading and source management module for AgroPrice AI.
Handles CSV ingestion, upload validation, metadata extraction, and provides
an extensible interface for live Agmarknet / OGD India connectors.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, BinaryIO
import pandas as pd
import streamlit as st

from src.config import DEFAULT_DATASET_PATH
from src.data_cleaning import clean_agricultural_data, validate_dataset_schema


class DataLoader:
    """Central loader for agricultural price datasets with schema validation and caching."""

    def __init__(self, default_path: Optional[Union[str, Path]] = None):
        self.default_path = Path(default_path) if default_path else DEFAULT_DATASET_PATH

    def load_default_data(self) -> pd.DataFrame:
        """Loads and cleans the default historical Agmarknet sample dataset."""
        if not self.default_path.exists():
            raise FileNotFoundError(f"Default dataset not found at {self.default_path}")
        
        raw_df = pd.read_csv(self.default_path)
        return clean_agricultural_data(raw_df)

    def load_from_csv(self, file_source: Union[str, Path, BinaryIO]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads dataset from a file path or uploaded file object (e.g., Streamlit UploadedFile).
        Returns (cleaned_dataframe, metadata_dict).
        """
        try:
            if hasattr(file_source, "read"):
                # Handle file-like object from Streamlit
                raw_df = pd.read_csv(file_source)
            else:
                raw_df = pd.read_csv(file_source)
        except Exception as e:
            raise ValueError(f"Unable to read CSV file: {str(e)}")

        is_valid, errors = validate_dataset_schema(raw_df)
        if not is_valid:
            raise ValueError(f"CSV Validation Error: {'; '.join(errors)}")

        cleaned_df = clean_agricultural_data(raw_df)
        meta = self.get_dataset_summary(cleaned_df, is_sample=False)
        return cleaned_df, meta

    @staticmethod
    def get_dataset_summary(df: pd.DataFrame, is_sample: bool = True) -> Dict[str, Any]:
        """Generates comprehensive summary metadata for a dataset."""
        if df.empty:
            return {
                "record_count": 0,
                "crops": [],
                "states": [],
                "markets": [],
                "start_date": None,
                "end_date": None,
                "is_sample": is_sample,
                "source_label": "Empty"
            }

        start_date = df["Arrival_Date"].min().strftime("%d %b %Y") if "Arrival_Date" in df.columns else "N/A"
        end_date = df["Arrival_Date"].max().strftime("%d %b %Y") if "Arrival_Date" in df.columns else "N/A"

        return {
            "record_count": int(len(df)),
            "crop_count": int(df["Commodity"].nunique()),
            "crops": sorted(df["Commodity"].unique().tolist()),
            "state_count": int(df["State"].nunique()),
            "states": sorted(df["State"].unique().tolist()),
            "district_count": int(df["District"].nunique()),
            "market_count": int(df["Market"].nunique()),
            "markets": sorted(df["Market"].unique().tolist()),
            "start_date": start_date,
            "end_date": end_date,
            "min_modal_price": float(df["Modal_Price"].min()),
            "max_modal_price": float(df["Modal_Price"].max()),
            "avg_modal_price": float(round(df["Modal_Price"].mean(), 2)),
            "is_sample": is_sample,
            "source_label": "Agmarknet Historical (Govt of India)" if is_sample else "User Uploaded Dataset"
        }


@st.cache_data(show_spinner=False)
def get_cached_agricultural_data() -> pd.DataFrame:
    """Streamlit cached loader for fast web app responsiveness."""
    loader = DataLoader()
    return loader.load_default_data()

