"""
Prediction and forecasting engine for AgroPrice AI.
Performs input validation, checks data coverage, executes ML inference,
computes statistically reliable prediction intervals, generates multi-month forward price curves,
evaluates post-harvest storage holding decisions, benchmarks Government MSP,
and persists prediction logs to SQLite.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from src.config import (
    MONTH_NAMES,
    SUPPORTED_CROPS,
    GOVT_MSP_BENCHMARKS,
    STORAGE_COST_ESTIMATES,
    CROP_HINDI_NAMES
)
from src.database import save_prediction
from src.feature_engineering import extract_features, get_season_from_month
from src.model_utils import load_trained_model, load_model_metadata


class PredictionService:
    """Service handling validation, ML inference, multi-month forecasting, and storage advisory."""

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

        # Case-insensitive crop lookup
        crop_match = next((k for k in hierarchy if k.lower() == crop.strip().lower()), None)
        if not crop_match:
            return False, f"We currently have no historical price records for '{crop}'."
        crop_data = hierarchy[crop_match]

        # Case-insensitive state lookup
        state_match = next((k for k in crop_data if k.lower() == state.strip().lower()), None)
        if not state_match:
            avail_states = list(crop_data.keys())
            return False, f"No records for '{crop}' in State '{state}'. Available states for {crop}: {', '.join(avail_states)}."
        state_data = crop_data[state_match]

        # Case-insensitive district lookup
        dist_match = next((k for k in state_data if k.lower() == district.strip().lower()), None)
        if not dist_match:
            avail_districts = list(state_data.keys())
            return False, f"No records for '{crop}' in District '{district}'. Available districts in {state}: {', '.join(avail_districts)}."
        district_data = state_data[dist_match]

        # Case-insensitive market lookup
        market_match = next((k for k in district_data if k.lower() == market.strip().lower()), None)
        if not market_match:
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
        Computes point estimate, empirical prediction interval, and MSP status.
        """
        is_valid, validation_errors = self.validate_inputs(crop, state, district, market, month, year)
        if not is_valid:
            return {
                "success": False,
                "error": "; ".join(validation_errors),
                "error_type": "validation_error"
            }

        is_supported, support_message = self.check_market_support(crop, state, district, market)
        if not is_supported:
            return {
                "success": False,
                "error": support_message,
                "error_type": "unsupported_market_combination"
            }

        if not season or season == "Auto":
            season = get_season_from_month(month)

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

        features_df = extract_features(input_data, is_inference=True)
        raw_pred = float(self.pipeline.predict(features_df)[0])
        predicted_price = round(max(raw_pred, 100.0), 2)

        # Compute empirical confidence margin based on crop error breakdown
        margin_error = self.metadata.get("prediction_interval", {}).get("margin_error_90", 250.0)
        crop_breakdown = self.metadata.get("crop_breakdown", {}).get(crop, {})
        if "mae" in crop_breakdown:
            crop_mae = crop_breakdown["mae"]
            margin_error = round(1.645 * 1.25 * crop_mae, 2)

        price_range_low = round(max(100.0, predicted_price - margin_error), 2)
        price_range_high = round(predicted_price + margin_error, 2)

        # Government MSP benchmark analysis
        msp_info = self.get_msp_status(crop, predicted_price)

        model_version = self.metadata.get("model_version", "v1.0.0")

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
        hindi_name = CROP_HINDI_NAMES.get(crop, crop)

        return {
            "success": True,
            "prediction_id": db_id,
            "crop": crop,
            "crop_hindi": hindi_name,
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
            "msp_info": msp_info,
            "unit": "₹ / Quintal",
            "model_version": model_version,
            "model_name": self.metadata.get("champion_model", "Trained ML Model"),
            "training_period": self.metadata.get("training_period", {}),
            "disclaimer": (
                "This prediction is an estimated future price calculated by machine learning based on historical Agmarknet market trends. "
                "Actual prices may vary depending on local weather events, mandi arrivals, quality grade, transportation costs, and current market conditions."
            )
        }

    def forecast_multi_month_trajectory(
        self,
        crop: str,
        state: str,
        district: str,
        market: str,
        start_month: int,
        start_year: int,
        horizon_months: int = 6,
        variety: str = "Standard"
    ) -> Dict[str, Any]:
        """
        Forecasts a forward price trajectory across consecutive future months.
        Pinpoints the peak selling month (best window) and lowest price month.
        """
        trajectory = []
        cur_m = start_month
        cur_y = start_year

        for i in range(horizon_months):
            res = self.predict_crop_price(
                crop=crop,
                state=state,
                district=district,
                market=market,
                month=cur_m,
                year=cur_y,
                variety=variety,
                save_to_db=False
            )

            if not res["success"]:
                return {"success": False, "error": res["error"]}

            trajectory.append({
                "month_idx": i + 1,
                "month": cur_m,
                "year": cur_y,
                "month_label": f"{MONTH_NAMES[cur_m - 1][:3]} {cur_y}",
                "season": res["season"],
                "predicted_price": res["predicted_price"],
                "price_low": res["price_range_low"],
                "price_high": res["price_range_high"]
            })

            # Advance month
            cur_m += 1
            if cur_m > 12:
                cur_m = 1
                cur_y += 1

        df_traj = pd.DataFrame(trajectory)
        peak_row = df_traj.loc[df_traj["predicted_price"].idxmax()]
        trough_row = df_traj.loc[df_traj["predicted_price"].idxmin()]

        base_price = df_traj["predicted_price"].iloc[0]
        peak_gain = round(peak_row["predicted_price"] - base_price, 2)
        peak_gain_pct = round((peak_gain / base_price) * 100.0, 1) if base_price > 0 else 0.0

        return {
            "success": True,
            "crop": crop,
            "market": market,
            "trajectory": trajectory,
            "trajectory_df": df_traj,
            "peak_month": peak_row["month_label"],
            "peak_price": peak_row["predicted_price"],
            "peak_gain": peak_gain,
            "peak_gain_pct": peak_gain_pct,
            "trough_month": trough_row["month_label"],
            "trough_price": trough_row["predicted_price"]
        }

    def evaluate_storage_decision(
        self,
        crop: str,
        current_price: float,
        target_predicted_price: float,
        months_to_hold: int,
        custom_storage_cost_per_month: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Assesses the financial advisability of storing produce vs. selling immediately.
        Calculates godown rent, shrinkage loss, and net gain/loss per quintal.
        """
        cost_per_month = (
            custom_storage_cost_per_month
            if custom_storage_cost_per_month is not None
            else STORAGE_COST_ESTIMATES.get(crop, STORAGE_COST_ESTIMATES["default"])
        )

        total_storage_cost = round(cost_per_month * months_to_hold, 2)
        # Average weight shrinkage over storage (approx 0.5% per month)
        shrinkage_pct = min(months_to_hold * 0.5, 3.5)
        shrinkage_loss_rupees = round(current_price * (shrinkage_pct / 100.0), 2)

        total_holding_expenditure = round(total_storage_cost + shrinkage_loss_rupees, 2)
        gross_price_gain = round(target_predicted_price - current_price, 2)
        net_profit_loss = round(gross_price_gain - total_holding_expenditure, 2)

        is_profitable = net_profit_loss > 0

        if is_profitable:
            advisory_en = (
                f"Holding for {months_to_hold} months is projected to be profitable. "
                f"Estimated net profit of ₹{net_profit_loss:,.2f}/quintal after accounting for ₹{total_holding_expenditure:,.2f} in storage and shrinkage costs."
            )
            advisory_hi = f"{months_to_hold} महीने भंडारण करने से प्रति क्विंटल लगभग ₹{net_profit_loss:,.2f} का शुद्ध लाभ मिलने की संभावना है।"
        else:
            advisory_en = (
                f"Storing is not financially advantageous. Expected price appreciation (₹{gross_price_gain:,.2f}) "
                f"does not cover the estimated storage and handling expenses (₹{total_holding_expenditure:,.2f}). Selling earlier is recommended."
            )
            advisory_hi = "भंडारण करना फायदेमंद नहीं है। गोदाम किराए और वजन घटने के कारण तुरंत बेचना अधिक सुरक्षित रहेगा।"

        return {
            "months_held": months_to_hold,
            "monthly_rent": cost_per_month,
            "total_rent": total_storage_cost,
            "shrinkage_loss": shrinkage_loss_rupees,
            "total_holding_expenditure": total_holding_expenditure,
            "gross_price_gain": gross_price_gain,
            "net_profit_loss": net_profit_loss,
            "is_profitable": is_profitable,
            "advisory_en": advisory_en,
            "advisory_hi": advisory_hi
        }

    def get_msp_status(self, crop: str, predicted_price: float) -> Optional[Dict[str, Any]]:
        """Compares predicted price against official Government of India MSP."""
        msp = GOVT_MSP_BENCHMARKS.get(crop)
        if not msp:
            return None

        diff = round(predicted_price - msp, 2)
        diff_pct = round((diff / msp) * 100.0, 1)
        is_above_msp = diff >= 0

        return {
            "msp_price": msp,
            "difference": diff,
            "difference_pct": diff_pct,
            "is_above_msp": is_above_msp,
            "status_label": "Above Govt MSP (Open Market Premium)" if is_above_msp else "Below / Near Govt MSP (Procurement Alert)",
            "guidance": (
                f"Market price is trading at a premium of ₹{diff:,.2f} ({diff_pct}%) above official MSP."
                if is_above_msp else
                f"Estimated price is ₹{abs(diff):,.2f} below official MSP. Farmers are advised to register at official Government procurement centers (FCI/NAFED/CCI) for guaranteed MSP payment."
            )
        }
