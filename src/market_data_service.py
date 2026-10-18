"""
Market data analysis service for AgroPrice AI.
Provides aggregations, historical trend queries, seasonal breakdowns,
and market comparisons for the web dashboard.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class MarketDataService:
    """Service providing agricultural market queries and dashboard analytics."""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        if "Arrival_Date" in self.data.columns and not pd.api.types.is_datetime64_any_dtype(self.data["Arrival_Date"]):
            self.data["Arrival_Date"] = pd.to_datetime(self.data["Arrival_Date"])

    def get_latest_prices(
        self,
        crop: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieves the most recent record for each crop and market combination.
        """
        df = self.data.copy()
        if crop and crop != "All":
            df = df[df["Commodity"].str.lower() == crop.lower()]
        if state and state != "All":
            df = df[df["State"].str.lower() == state.lower()]
        if district and district != "All":
            df = df[df["District"].str.lower() == district.lower()]

        if df.empty:
            return pd.DataFrame()

        # Sort descending by date and pick latest for each (Commodity, Market)
        df_latest = df.sort_values(by="Arrival_Date", ascending=False).groupby(
            ["Commodity", "State", "District", "Market"], as_index=False
        ).first()

        return df_latest.sort_values(by="Modal_Price", ascending=False).reset_index(drop=True)

    def get_crop_time_series(
        self,
        crop: str,
        market: Optional[str] = None,
        state: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieves chronological price series for a given crop, with optional market filtering.
        """
        df = self.data[self.data["Commodity"].str.lower() == crop.lower()].copy()

        if state and state != "All":
            df = df[df["State"].str.lower() == state.lower()]
        if market and market != "All":
            df = df[df["Market"].str.lower() == market.lower()]

        if df.empty:
            return pd.DataFrame()

        # Aggregate by date if multiple records exist on same day
        ts = df.groupby("Arrival_Date").agg({
            "Modal_Price": "mean",
            "Min_Price": "mean",
            "Max_Price": "mean"
        }).reset_index()

        ts["Modal_Price"] = ts["Modal_Price"].round(2)
        ts["Min_Price"] = ts["Min_Price"].round(2)
        ts["Max_Price"] = ts["Max_Price"].round(2)

        return ts.sort_values(by="Arrival_Date").reset_index(drop=True)

    def get_seasonal_monthly_pattern(self, crop: str) -> pd.DataFrame:
        """
        Calculates month-of-year average price and standard deviation across all historical years.
        """
        df = self.data[self.data["Commodity"].str.lower() == crop.lower()].copy()
        if df.empty:
            return pd.DataFrame()

        df["Month"] = df["Arrival_Date"].dt.month
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }

        agg = df.groupby("Month").agg(
            Avg_Modal=("Modal_Price", "mean"),
            Min_Modal=("Modal_Price", "min"),
            Max_Modal=("Modal_Price", "max"),
            Std_Dev=("Modal_Price", "std"),
            Record_Count=("Modal_Price", "count")
        ).reset_index()

        agg["Month_Name"] = agg["Month"].map(month_names)
        agg["Avg_Modal"] = agg["Avg_Modal"].round(2)
        agg["Min_Modal"] = agg["Min_Modal"].round(2)
        agg["Max_Modal"] = agg["Max_Modal"].round(2)
        agg["Std_Dev"] = agg["Std_Dev"].fillna(0).round(2)

        return agg.sort_values(by="Month").reset_index(drop=True)

    def compare_markets_for_crop(self, crop: str, markets: List[str]) -> pd.DataFrame:
        """
        Compares average, minimum, maximum, and latest price across a list of mandis for a crop.
        """
        df = self.data[
            (self.data["Commodity"].str.lower() == crop.lower()) &
            (self.data["Market"].isin(markets))
        ].copy()

        if df.empty:
            return pd.DataFrame()

        records = []
        for mkt in markets:
            sub = df[df["Market"] == mkt]
            if sub.empty:
                continue
            latest_row = sub.sort_values(by="Arrival_Date", ascending=False).iloc[0]
            records.append({
                "Market": mkt,
                "State": latest_row["State"],
                "Latest_Date": latest_row["Arrival_Date"].strftime("%d %b %Y"),
                "Latest_Price": latest_row["Modal_Price"],
                "Avg_Price": round(sub["Modal_Price"].mean(), 2),
                "Min_Observed": round(sub["Min_Price"].min(), 2),
                "Max_Observed": round(sub["Max_Price"].max(), 2),
                "Observations": len(sub)
            })

        res_df = pd.DataFrame(records)
        if not res_df.empty:
            res_df = res_df.sort_values(by="Latest_Price", ascending=False).reset_index(drop=True)
        return res_df

    def get_month_over_month_change(self, crop: str) -> pd.DataFrame:
        """
        Computes monthly average prices and month-over-month (MoM) percentage change.
        """
        df = self.data[self.data["Commodity"].str.lower() == crop.lower()].copy()
        if df.empty:
            return pd.DataFrame()

        df["YearMonth"] = df["Arrival_Date"].dt.to_period("M")
        monthly = df.groupby("YearMonth")["Modal_Price"].mean().reset_index()
        monthly["Modal_Price"] = monthly["Modal_Price"].round(2)
        monthly["YearMonth_Str"] = monthly["YearMonth"].astype(str)
        monthly["MoM_Change_Pct"] = (monthly["Modal_Price"].pct_change() * 100.0).round(2)
        monthly["MoM_Diff_Rupees"] = (monthly["Modal_Price"].diff()).round(2)

        return monthly.sort_values(by="YearMonth").reset_index(drop=True)

