"""
Unit tests for DataLoader and schema validation.
"""

from io import StringIO
import pytest
import pandas as pd
from src.data_loader import DataLoader
from src.data_cleaning import validate_dataset_schema, clean_agricultural_data


def test_load_default_data_success():
    """Verify loading of the default historical dataset."""
    loader = DataLoader()
    df = loader.load_default_data()

    assert not df.empty
    assert len(df) > 1000
    assert "Commodity" in df.columns
    assert "Modal_Price" in df.columns
    assert "Market" in df.columns
    assert "Arrival_Date" in df.columns
    assert df["Modal_Price"].min() > 0


def test_dataset_summary_metadata():
    """Verify dataset metadata summary generation."""
    loader = DataLoader()
    df = loader.load_default_data()
    summary = loader.get_dataset_summary(df, is_sample=True)

    assert summary["record_count"] == len(df)
    assert summary["crop_count"] >= 10
    assert "Wheat" in summary["crops"]
    assert summary["state_count"] > 0
    assert summary["is_sample"] is True


def test_schema_validation_missing_columns():
    """Verify validation detects missing required columns."""
    invalid_df = pd.DataFrame({
        "Arrival_Date": ["2024-01-01"],
        "State": ["Maharashtra"],
        # Missing Commodity, Modal_Price, etc.
    })
    is_valid, errors = validate_dataset_schema(invalid_df)

    assert is_valid is False
    assert len(errors) > 0
    assert any("Missing required column" in err for err in errors)


def test_csv_upload_parsing():
    """Verify CSV loading from in-memory file buffers (like Streamlit uploads)."""
    csv_content = """Arrival_Date,State,District,Market,Commodity,Variety,Min_Price,Max_Price,Modal_Price
2024-05-10,Punjab,Ludhiana,Khanna,Wheat,Lokwan,2200,2350,2280
2024-05-11,Punjab,Ludhiana,Khanna,Wheat,Lokwan,2210,2360,2290
"""
    file_obj = StringIO(csv_content)
    loader = DataLoader()
    df, meta = loader.load_from_csv(file_obj)

    assert len(df) == 2
    assert meta["record_count"] == 2
    assert "Wheat" in meta["crops"]
    assert meta["is_sample"] is False


def test_nonexistent_dataset_handling():
    """Verify exception is raised when data source file is missing."""
    loader = DataLoader(default_path="nonexistent_folder/missing_file.csv")
    with pytest.raises(FileNotFoundError):
        loader.load_default_data()
