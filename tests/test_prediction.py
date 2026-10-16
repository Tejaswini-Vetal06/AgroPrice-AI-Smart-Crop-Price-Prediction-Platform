"""
Unit tests for the prediction engine, input validation, and unsupported combination handling.
"""

import pytest
from src.predict import PredictionService


@pytest.fixture
def service():
    return PredictionService()


def test_valid_prediction(service):
    """Verify prediction generation for a supported crop and mandi combination."""
    res = service.predict_crop_price(
        crop="Wheat",
        state="Punjab",
        district="Ludhiana",
        market="Khanna",
        month=4,
        year=2026,
        variety="Lokwan / Sharbati"
    )

    assert res["success"] is True
    assert res["predicted_price"] > 1000
    assert res["price_range_low"] <= res["predicted_price"]
    assert res["predicted_price"] <= res["price_range_high"]
    assert res["unit"] == "₹ / Quintal"
    assert "disclaimer" in res


def test_unsupported_crop(service):
    """Verify graceful rejection and message for unsupported crops (e.g., Dragonfruit)."""
    res = service.predict_crop_price(
        crop="Dragonfruit",
        state="Punjab",
        district="Ludhiana",
        market="Khanna",
        month=5,
        year=2026
    )

    assert res["success"] is False
    assert "not currently supported" in res["error"]
    assert res["error_type"] == "validation_error"


def test_unsupported_market_combination(service):
    """Verify friendly error message when crop is valid but location is unsupported."""
    res = service.predict_crop_price(
        crop="Wheat",
        state="Kerala",  # Wheat is not cultivated/traded in Kerala mandis in this dataset
        district="Ernakulam",
        market="Kochi",
        month=4,
        year=2026
    )

    assert res["success"] is False
    assert res["error_type"] == "unsupported_market_combination"
    assert "No records for 'Wheat' in State 'Kerala'" in res["error"]


def test_invalid_month_and_year(service):
    """Verify rejection of invalid calendar dates."""
    res = service.predict_crop_price(
        crop="Wheat",
        state="Punjab",
        district="Ludhiana",
        market="Khanna",
        month=15,  # Invalid month
        year=2026
    )

    assert res["success"] is False
    assert "Selling month must be between 1 and 12" in res["error"]
