"""
Sample Agricultural Data Generator for AgroPrice AI.
Generates realistic, multi-year, seasonal Agmarknet market price data for Indian commodities.
All prices are in Indian Rupees (₹) per quintal.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import DEFAULT_DATASET_PATH, SAMPLE_DATA_DIR


# Define realistic parameters for Indian agricultural markets
CROP_MARKET_CONFIGS = [
    # 1. Onion
    {
        "commodity": "Onion",
        "variety": "Red / Nashik",
        "markets": [
            ("Maharashtra", "Nashik", "Lasalgaon"),
            ("Maharashtra", "Pune", "Pune"),
            ("Madhya Pradesh", "Indore", "Indore"),
            ("Karnataka", "Bengaluru", "Bengaluru"),
            ("Rajasthan", "Alwar", "Alwar")
        ],
        "base_price": 2200,
        "volatility": 0.28,
        # Spikes in Sep-Nov (months 9, 10, 11), low in Mar-May (months 3, 4, 5)
        "monthly_factors": {1: 0.95, 2: 0.90, 3: 0.80, 4: 0.75, 5: 0.85, 6: 1.00,
                            7: 1.10, 8: 1.25, 9: 1.45, 10: 1.65, 11: 1.40, 12: 1.10},
        "annual_trend": 0.04
    },
    # 2. Tomato
    {
        "commodity": "Tomato",
        "variety": "Hybrid",
        "markets": [
            ("Maharashtra", "Nashik", "Pimpalgaon"),
            ("Karnataka", "Kolar", "Kolar"),
            ("Andhra Pradesh", "Chittoor", "Madanapalle"),
            ("Uttar Pradesh", "Varanasi", "Varanasi"),
            ("Madhya Pradesh", "Shivpuri", "Shivpuri")
        ],
        "base_price": 2000,
        "volatility": 0.35,
        # Spikes in monsoon June-August, low in winter Jan-March
        "monthly_factors": {1: 0.75, 2: 0.70, 3: 0.80, 4: 0.95, 5: 1.15, 6: 1.50,
                            7: 1.80, 8: 1.60, 9: 1.20, 10: 1.05, 11: 0.85, 12: 0.75},
        "annual_trend": 0.05
    },
    # 3. Wheat
    {
        "commodity": "Wheat",
        "variety": "Lokwan / Sharbati",
        "markets": [
            ("Punjab", "Ludhiana", "Khanna"),
            ("Madhya Pradesh", "Ujjain", "Ujjain"),
            ("Uttar Pradesh", "Kanpur", "Kanpur"),
            ("Rajasthan", "Kota", "Kota"),
            ("Haryana", "Karnal", "Karnal")
        ],
        "base_price": 2350,
        "volatility": 0.08,
        # Harvest in Mar-May (low), higher in winter Nov-Feb
        "monthly_factors": {1: 1.08, 2: 1.06, 3: 0.94, 4: 0.90, 5: 0.92, 6: 0.96,
                            7: 1.00, 8: 1.02, 9: 1.04, 10: 1.06, 11: 1.08, 12: 1.10},
        "annual_trend": 0.06
    },
    # 4. Rice (Paddy)
    {
        "commodity": "Rice",
        "variety": "Common / Basmati",
        "markets": [
            ("Punjab", "Amritsar", "Amritsar"),
            ("Uttar Pradesh", "Meerut", "Meerut"),
            ("West Bengal", "Burdwan", "Burdwan"),
            ("Telangana", "Warangal", "Warangal"),
            ("Haryana", "Kurukshetra", "Kurukshetra")
        ],
        "base_price": 3100,
        "volatility": 0.10,
        # Kharif harvest in Oct-Dec (lowest), rises gradually
        "monthly_factors": {1: 0.98, 2: 1.00, 3: 1.02, 4: 1.04, 5: 1.05, 6: 1.06,
                            7: 1.07, 8: 1.08, 9: 1.05, 10: 0.93, 11: 0.91, 12: 0.94},
        "annual_trend": 0.05
    },
    # 5. Potato
    {
        "commodity": "Potato",
        "variety": "Jyoti / Kufri",
        "markets": [
            ("Uttar Pradesh", "Agra", "Agra"),
            ("Punjab", "Jalandhar", "Jalandhar"),
            ("West Bengal", "Hooghly", "Hooghly"),
            ("Gujarat", "Deesa", "Deesa"),
            ("Madhya Pradesh", "Indore", "Indore")
        ],
        "base_price": 1400,
        "volatility": 0.22,
        # Harvest Jan-Mar (low), peak storage releases Sep-Nov
        "monthly_factors": {1: 0.85, 2: 0.75, 3: 0.70, 4: 0.82, 5: 0.95, 6: 1.10,
                            7: 1.20, 8: 1.30, 9: 1.40, 10: 1.45, 11: 1.25, 12: 0.95},
        "annual_trend": 0.04
    },
    # 6. Soybean
    {
        "commodity": "Soybean",
        "variety": "Yellow",
        "markets": [
            ("Madhya Pradesh", "Indore", "Indore"),
            ("Maharashtra", "Nagpur", "Nagpur"),
            ("Maharashtra", "Jalgaon", "Jalgaon"),
            ("Rajasthan", "Kota", "Kota"),
            ("Madhya Pradesh", "Ujjain", "Ujjain")
        ],
        "base_price": 4800,
        "volatility": 0.14,
        # Harvest Sep-Nov (low), off-season peak Apr-Jul
        "monthly_factors": {1: 1.02, 2: 1.03, 3: 1.06, 4: 1.10, 5: 1.14, 6: 1.15,
                            7: 1.12, 8: 1.05, 9: 0.92, 10: 0.88, 11: 0.90, 12: 0.98},
        "annual_trend": 0.05
    },
    # 7. Cotton
    {
        "commodity": "Cotton",
        "variety": "Medium / Long Staple",
        "markets": [
            ("Gujarat", "Rajkot", "Rajkot"),
            ("Maharashtra", "Nagpur", "Nagpur"),
            ("Telangana", "Adilabad", "Adilabad"),
            ("Punjab", "Bathinda", "Bathinda"),
            ("Gujarat", "Gondal", "Gondal")
        ],
        "base_price": 7200,
        "volatility": 0.12,
        # Harvest Oct-Jan (low), rises towards summer
        "monthly_factors": {1: 0.94, 2: 0.96, 3: 1.00, 4: 1.04, 5: 1.08, 6: 1.10,
                            7: 1.11, 8: 1.08, 9: 1.02, 10: 0.92, 11: 0.91, 12: 0.93},
        "annual_trend": 0.04
    },
    # 8. Maize
    {
        "commodity": "Maize",
        "variety": "Yellow",
        "markets": [
            ("Karnataka", "Davanagere", "Davanagere"),
            ("Madhya Pradesh", "Chhindwara", "Chhindwara"),
            ("Bihar", "Purnea", "Gulabbagh"),
            ("Rajasthan", "Udaipur", "Udaipur"),
            ("Maharashtra", "Sangli", "Sangli")
        ],
        "base_price": 2100,
        "volatility": 0.11,
        # Kharif arrivals Oct-Dec
        "monthly_factors": {1: 1.00, 2: 1.02, 3: 1.05, 4: 1.08, 5: 1.06, 6: 1.04,
                            7: 1.03, 8: 1.01, 9: 0.98, 10: 0.92, 11: 0.90, 12: 0.96},
        "annual_trend": 0.05
    },
    # 9. Turmeric
    {
        "commodity": "Turmeric",
        "variety": "Finger / Nizamabad",
        "markets": [
            ("Maharashtra", "Sangli", "Sangli"),
            ("Telangana", "Nizamabad", "Nizamabad"),
            ("Tamil Nadu", "Erode", "Erode"),
            ("Andhra Pradesh", "Guntur", "Duggirala"),
            ("Karnataka", "Chamarajanagar", "Gundlupet")
        ],
        "base_price": 9500,
        "volatility": 0.16,
        # Harvest Feb-May
        "monthly_factors": {1: 1.04, 2: 0.94, 3: 0.90, 4: 0.88, 5: 0.92, 6: 0.98,
                            7: 1.02, 8: 1.06, 9: 1.10, 10: 1.12, 11: 1.10, 12: 1.08},
        "annual_trend": 0.06
    },
    # 10. Gram (Chana)
    {
        "commodity": "Gram",
        "variety": "Desi / Bengal Gram",
        "markets": [
            ("Madhya Pradesh", "Bhopal", "Bhopal"),
            ("Rajasthan", "Bikaner", "Bikaner"),
            ("Maharashtra", "Latur", "Latur"),
            ("Uttar Pradesh", "Kanpur", "Kanpur"),
            ("Karnataka", "Gulbarga", "Kalaburagi")
        ],
        "base_price": 5400,
        "volatility": 0.09,
        # Rabi harvest Feb-April
        "monthly_factors": {1: 1.04, 2: 0.98, 3: 0.92, 4: 0.90, 5: 0.95, 6: 1.00,
                            7: 1.03, 8: 1.06, 9: 1.08, 10: 1.09, 11: 1.07, 12: 1.05},
        "annual_trend": 0.05
    }
]


def generate_market_price_dataset(
    start_year: int = 2021,
    end_year: int = 2025,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a structured, realistic time series of Agmarknet mandi price records.
    Covers weekly dates for each crop and mandi combination over 5 years.
    """
    np.random.seed(random_seed)
    records = []

    # Weekly date range across the target years
    date_range = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="W-MON"
    )

    for cfg in CROP_MARKET_CONFIGS:
        commodity = cfg["commodity"]
        variety = cfg["variety"]
        base_price = cfg["base_price"]
        volatility = cfg["volatility"]
        monthly_factors = cfg["monthly_factors"]
        trend = cfg["annual_trend"]

        for state, district, market in cfg["markets"]:
            # State/mandi-specific pricing offset (-5% to +5%)
            mandi_offset = np.random.uniform(0.96, 1.05)

            for dt in date_range:
                year = dt.year
                month = dt.month
                years_passed = year - start_year

                # Calculate realistic price dynamics:
                # 1. Base price with annual inflation/trend
                # 2. Monthly seasonal multiplier
                # 3. Market-level geographic differential
                # 4. Realistic market noise & episodic weather/supply shock
                season_mult = monthly_factors.get(month, 1.0)
                trend_mult = (1.0 + trend) ** years_passed
                
                # Random market fluctuation
                noise = np.random.normal(0, volatility * 0.4)
                
                # Episodic shock (e.g. unseasonal rain or drought with 2% probability)
                shock = 0.0
                if np.random.rand() < 0.02:
                    shock = np.random.uniform(0.15, 0.35)

                modal = base_price * trend_mult * season_mult * mandi_offset * (1.0 + noise + shock)
                modal = round(max(modal, base_price * 0.4), 2)

                # Min price typically 2% to 8% below modal
                min_price = round(modal * np.random.uniform(0.92, 0.98), 2)
                # Max price typically 2% to 10% above modal
                max_price = round(modal * np.random.uniform(1.02, 1.10), 2)

                records.append({
                    "Arrival_Date": dt.strftime("%Y-%m-%d"),
                    "State": state,
                    "District": district,
                    "Market": market,
                    "Commodity": commodity,
                    "Variety": variety,
                    "Min_Price": min_price,
                    "Max_Price": max_price,
                    "Modal_Price": modal,
                    "Units": "₹ / Quintal",
                    "Source": "Agmarknet Historical (Govt of India)"
                })

    df = pd.DataFrame(records)
    return df


def main():
    print(f"Generating realistic Agmarknet crop price dataset...")
    df = generate_market_price_dataset(start_year=2021, end_year=2025)
    
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_DATASET_PATH, index=False)
    
    print(f"Successfully generated {len(df):,} records!")
    print(f"Saved to: {DEFAULT_DATASET_PATH}")
    print(f"Commodities: {df['Commodity'].nunique()} ({', '.join(df['Commodity'].unique())})")
    print(f"Mandis: {df['Market'].nunique()} across {df['State'].nunique()} Indian states.")
    print(f"Date range: {df['Arrival_Date'].min()} to {df['Arrival_Date'].max()}")


if __name__ == "__main__":
    main()
