"""
Prediction engine for AgroPrice AI.
Performs input validation, checks data coverage, executes ML inference,
computes statistically reliable prediction intervals, retrieves historical context,
and persists prediction logs to SQLite.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from src.config import MONTH_NAMES, SUPPORTED_CROPS
from src.database import save_prediction
from src.feature_engineering import extract_features, get_season_from_month
from src.model_utils import load_trained_model, load_model_metadata


class PredictionService:
    """Service handling validation, ML inference, error bounds, and history logging."""

    def __init__(self):
        self.pipeline = load_trained_model()
        self.metadata = load_model_metadata()

    def validate_inputs(
        self,
        crop: str,
        state: str,
        district: str,
        market: str,
        month: int,
        year: int
    ) -> Tuple[bool, List[str]]:
        """Validates all user-submitted prediction inputs."""
        errors = []

        if not crop or crop not in self.metadata.get("supported_crops", []):
            errors.append(f"Crop '{crop}' is not currently supported. Supported crops: {', '.join(self.metadata.get('supported_crops', []))}")

        if not state:
            errors.append("State must be specified.")
        if not district:
            errors.append("District must be specified.")
        if not market:
            errors.append("Market (Mandi) must be specified.")

        if not (1 <= month <= 12):
            errors.append("Selling month must be between 1 and 12.")

        if year < 2020 or year > 2035:
            errors.append("Selling year must be between 2020 and 2035.")

        return len(errors) == 0, errors

    def check_market_support(
        self,
        crop: str,
        state: str,
        district: str,
        market: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies if the specific crop-market combination exists in the model's training data.
        Returns (is_supported, helpful_message).
        """
        hierarchy = self.metadata.get("market_hierarchy", {})

        crop_data = hierarchy.get(crop)
        if not crop_data:
            return False, f"We currently have no historical price records for '{crop}'."

        state_data = crop_data.get(state)
        if not state_data:
            avail_states = list(crop_data.keys())
            return False, f"No records for '{crop}' in State '{state}'. Available states for {crop}: {', '.join(avail_states)}."

        district_data = state_data.get(district)
        if not district_data:
            avail_districts = list(state_data.keys())
            return False, f"No records for '{crop}' in District '{district}'. Available districts in {state}: {', '.join(avail_districts)}."

        if market not in district_data:
            return False, f"Mandi '{market}' not found for {crop} in {district}. Available mandis: {', '.join(district_data)}."

        return True, None

    def predict_crop_price(
        self,
        crop: str,
        state: str,
        district: str,
        market: str,
        month: int,
        year: int,
        variety: str = "Standard",
        season: Optional[str] = None,
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Generates a validated machine learning crop price prediction.
        Computes point estimate and empirical prediction interval.
        """
        # Validate inputs
        is_valid, validation_errors = self.validate_inputs(crop, state, district, market, month, year)
        if not is_valid:
            return {
                "success": False,
                "error": "; ".join(validation_errors),
                "error_type": "validation_error"
            }

        # Check coverage in training data
        is_supported, support_message = self.check_market_support(crop, state, district, market)
        if not is_supported:
            return {
                "success": False,
                "error": support_message,
                "error_type": "unsupported_market_combination"
            }

        # Determine season if not provided
        if not season or season == "Auto":
            season = get_season_from_month(month)

        # Prepare single-row input DataFrame
        input_data = pd.DataFrame([{
            "Commodity": crop.strip().title(),
            "State": state.strip().title(),
            "District": district.strip().title(),
            "Market": market.strip().title(),
            "Variety": variety.strip().title() if variety else "Standard",
            "Month": int(month),
            "Year": int(year),
            "Season": season
        }])

        # Extract features and predict
        features_df = extract_features(input_data, is_inference=True)
        raw_pred = float(self.pipeline.predict(features_df)[0])
        predicted_price = round(max(raw_pred, 100.0), 2)

        # Compute uncertainty range based on model residuals
        margin_error = self.metadata.get("prediction_interval", {}).get("margin_error_90", 250.0)
        
        # Adjust margin if crop has specific breakdown
        crop_breakdown = self.metadata.get("crop_breakdown", {}).get(crop, {})
        if "mae" in crop_breakdown:
            # Crop-specific empirical 90% margin (~ 1.645 * 1.25 * MAE)
            crop_mae = crop_breakdown["mae"]
            margin_error = round(1.645 * 1.25 * crop_mae, 2)

        price_range_low = round(max(100.0, predicted_price - margin_error), 2)
        price_range_high = round(predicted_price + margin_error, 2)

        model_version = self.metadata.get("model_version", "v1.0.0")

        # Save to database
        db_id = None
        if save_to_db:
            try:
                db_id = save_prediction(
                    crop=crop,
                    state=state,
                    district=district,
                    market=market,
                    month=month,
                    year=year,
                    variety=variety,
                    season=season,
                    predicted_price=predicted_price,
                    price_range_low=price_range_low,
                    price_range_high=price_range_high,
                    model_version=model_version
                )
            except Exception as e:
                print(f"Notice: Could not log prediction to SQLite: {e}")

        month_label = MONTH_NAMES[month - 1]

        return {
            "success": True,
            "prediction_id": db_id,
            "crop": crop,
            "state": state,
            "district": district,
            "market": market,
            "variety": variety,
            "month": month,
            "month_name": month_label,
            "year": year,
            "season": season,
            "predicted_price": predicted_price,
            "price_range_low": price_range_low,
            "price_range_high": price_range_high,
            "margin_error": margin_error,
            "unit": "₹ / Quintal",
            "model_version": model_version,
            "model_name": self.metadata.get("champion_model", "Trained ML Model"),
            "training_period": self.metadata.get("training_period", {}),
            "disclaimer": (
                "This prediction is an estimated future price calculated by machine learning based on historical Agmarknet market trends. "
                "Actual prices may vary depending on local weather events, mandi arrivals, quality grade, transportation costs, and current market conditions."
            )
        }
