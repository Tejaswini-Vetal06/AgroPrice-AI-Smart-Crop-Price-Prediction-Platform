"""
Unit tests for SQLite prediction history persistence and queries.
"""

import pytest
from src.database import (
    init_db,
    save_prediction,
    get_predictions,
    delete_prediction,
    clear_all_predictions,
    get_prediction_count
)


def setup_function():
    """Ensure clean table before each test."""
    init_db()
    clear_all_predictions()


def test_save_and_retrieve_prediction():
    """Verify saving a prediction and retrieving it from SQLite."""
    inserted_id = save_prediction(
        crop="Onion",
        state="Maharashtra",
        district="Nashik",
        market="Lasalgaon",
        month=10,
        year=2026,
        predicted_price=2450.50,
        variety="Red / Nashik",
        season="Kharif",
        price_range_low=2100.00,
        price_range_high=2800.00,
        model_version="v1.0.0"
    )

    assert inserted_id is not None
    assert get_prediction_count() == 1

    df = get_predictions()
    assert len(df) == 1
    assert df["crop"].iloc[0] == "Onion"
    assert df["market"].iloc[0] == "Lasalgaon"
    assert df["predicted_price"].iloc[0] == 2450.50


def test_filter_and_search_predictions():
    """Verify search and crop filters on prediction history."""
    save_prediction("Wheat", "Punjab", "Ludhiana", "Khanna", 4, 2026, 2300.0)
    save_prediction("Rice", "Punjab", "Amritsar", "Amritsar", 11, 2026, 3100.0)
    save_prediction("Tomato", "Karnataka", "Kolar", "Kolar", 7, 2026, 2900.0)

    assert get_prediction_count() == 3

    # Filter by crop
    wheat_df = get_predictions(crop_filter="Wheat")
    assert len(wheat_df) == 1
    assert wheat_df["crop"].iloc[0] == "Wheat"

    # Search by keyword
    search_df = get_predictions(search_query="Kolar")
    assert len(search_df) == 1
    assert search_df["market"].iloc[0] == "Kolar"


def test_delete_and_clear():
    """Verify deleting a specific record and clearing all history."""
    id1 = save_prediction("Wheat", "Punjab", "Ludhiana", "Khanna", 4, 2026, 2300.0)
    id2 = save_prediction("Rice", "Punjab", "Amritsar", "Amritsar", 11, 2026, 3100.0)

    assert get_prediction_count() == 2

    # Delete one
    deleted = delete_prediction(id1)
    assert deleted is True
    assert get_prediction_count() == 1

    # Clear remaining
    clear_all_predictions()
    assert get_prediction_count() == 0

