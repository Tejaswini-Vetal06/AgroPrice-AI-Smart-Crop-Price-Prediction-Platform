"""
Unit tests for MarketDataService analytics and queries.
"""

import pandas as pd
import pytest
from src.data_loader import DataLoader
from src.market_data_service import MarketDataService


@pytest.fixture
def market_service():
    loader = DataLoader()
    df = loader.load_default_data()
    return MarketDataService(df)


def test_get_latest_prices(market_service):
    """Verify retrieval of latest market prices with filters."""
    latest_all = market_service.get_latest_prices()
    assert not latest_all.empty
    assert "Modal_Price" in latest_all.columns

    # Filter by specific crop
    wheat_latest = market_service.get_latest_prices(crop="Wheat")
    assert not wheat_latest.empty
    assert (wheat_latest["Commodity"] == "Wheat").all()


def test_get_crop_time_series(market_service):
    """Verify time series extraction for a crop."""
    ts = market_service.get_crop_time_series(crop="Onion")
    assert not ts.empty
    assert "Arrival_Date" in ts.columns
    assert "Modal_Price" in ts.columns
    assert len(ts) > 10


def test_get_seasonal_monthly_pattern(market_service):
    """Verify monthly aggregation for seasonal cycle analysis."""
    seasonal = market_service.get_seasonal_monthly_pattern(crop="Tomato")
    assert not seasonal.empty
    assert len(seasonal) == 12  # 12 calendar months
    assert "Month_Name" in seasonal.columns
    assert "Avg_Modal" in seasonal.columns


def test_compare_markets_for_crop(market_service):
    """Verify comparative metrics across multiple mandis."""
    mandis = ["Lasalgaon", "Pune"]
    comp = market_service.compare_markets_for_crop("Onion", mandis)
    assert not comp.empty
    assert len(comp) == 2
    assert "Latest_Price" in comp.columns
    assert "Avg_Price" in comp.columns


def test_get_month_over_month_change(market_service):
    """Verify calculation of monthly price deltas and percentages."""
    mom = market_service.get_month_over_month_change(crop="Wheat")
    assert not mom.empty
    assert "MoM_Change_Pct" in mom.columns
    assert "YearMonth_Str" in mom.columns

