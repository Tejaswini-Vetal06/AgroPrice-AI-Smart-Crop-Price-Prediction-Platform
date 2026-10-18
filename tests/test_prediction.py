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


def test_forecast_multi_month_trajectory(service):
    """Verify forward price trajectory forecasting across 6 months."""
    fc_res = service.forecast_multi_month_trajectory(
        crop="Wheat",
        state="Punjab",
        district="Ludhiana",
        market="Khanna",
        start_month=4,
        start_year=2026,
        horizon_months=6
    )

    assert fc_res["success"] is True
    assert len(fc_res["trajectory"]) == 6
    assert "peak_month" in fc_res
    assert "peak_price" in fc_res
    assert "trough_month" in fc_res
    assert fc_res["peak_price"] >= fc_res["trough_price"]


def test_evaluate_storage_decision(service):
    """Verify holding vs. selling profitability evaluation."""
    decision = service.evaluate_storage_decision(
        crop="Wheat",
        current_price=2200.0,
        target_predicted_price=2600.0,
        months_to_hold=3,
        custom_storage_cost_per_month=30.0
    )

    assert "net_profit_loss" in decision
    assert "total_holding_expenditure" in decision
    assert "advisory_en" in decision
    assert "advisory_hi" in decision
    # 2600 - 2200 = 400 gross gain. Storage = 3 * 30 = 90. Shrinkage = 2200 * 1.5% = 33. Net = 400 - 123 = 277
    assert decision["net_profit_loss"] > 0
    assert decision["is_profitable"] is True


def test_get_msp_status(service):
    """Verify comparison against Government Minimum Support Price."""
    msp_info = service.get_msp_status(crop="Wheat", predicted_price=2650.0)
    assert msp_info is not None
    assert msp_info["msp_price"] == 2425
    assert msp_info["is_above_msp"] is True
    assert msp_info["difference"] > 0


def test_new_all_india_crops_prediction(service):
    """Verify prediction for newly added All-India crops."""
    # Test Mustard in Rajasthan
    mustard_res = service.predict_crop_price(
        crop="Mustard",
        state="Rajasthan",
        district="Bharatpur",
        market="Bharatpur",
        month=3,
        year=2026
    )
    assert mustard_res["success"] is True
    assert mustard_res["predicted_price"] > 3000

    # Test Apple in Jammu and Kashmir
    apple_res = service.predict_crop_price(
        crop="Apple",
        state="Jammu and Kashmir",
        district="Baramulla",
        market="Sopore",
        month=9,
        year=2026
    )
    assert apple_res["success"] is True
    assert apple_res["predicted_price"] > 4000

