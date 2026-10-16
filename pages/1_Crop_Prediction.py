"""
AgroPrice AI — Crop Price Prediction Page.
Interactive prediction form, cascading location selectors, validated ML inference,
empirical price bands, and historical context.
"""

from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.config import (
    apply_custom_theme,
    get_plotly_layout,
    COLOR_FOREST_GREEN,
    COLOR_FRESH_GREEN,
    COLOR_SAGE_GREEN,
    COLOR_SOFT_MINT,
    COLOR_WHITE,
    MONTH_NAMES,
    SEASONS
)
from src.data_loader import get_cached_agricultural_data
from src.model_utils import is_model_ready, load_model_metadata
from src.predict import PredictionService
from src.market_data_service import MarketDataService


st.set_page_config(
    page_title="Crop Price Prediction — AgroPrice AI",
    page_icon="🔮",
    layout="wide"
)

apply_custom_theme()

st.markdown("## 🔮 Intelligent Crop Price Prediction")
st.markdown("Estimate future mandi prices in **₹ per quintal** using validated machine learning models trained on Agmarknet data.")

if not is_model_ready():
    st.error("⚠️ Trained ML model was not found on disk. Please run the training pipeline first.")
    st.stop()

metadata = load_model_metadata()
hierarchy = metadata.get("market_hierarchy", {})
supported_crops = metadata.get("supported_crops", [])

# Initialize services
pred_service = PredictionService()
raw_data = get_cached_agricultural_data()
market_service = MarketDataService(raw_data)

# Layout: Form and Results side by side or responsive
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown(
        f"""
        <div class="agro-card">
            <h3 style="margin-top: 0; color: {COLOR_FOREST_GREEN} !important;">
                📋 Mandi & Harvest Parameters
            </h3>
            <p style="color: #64748B; font-size: 0.88rem;">
                Select your crop, market center, and expected selling schedule.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Crop Selection
    selected_crop = st.selectbox(
        "1. Select Crop / Commodity 🌾",
        options=supported_crops,
        index=0,
        help="Select one of the 10 supported Indian agricultural commodities."
    )

    # Cascading States for Selected Crop
    avail_states = sorted(list(hierarchy.get(selected_crop, {}).keys()))
    if not avail_states:
        avail_states = ["All States"]

    selected_state = st.selectbox(
        "2. Select State 📍",
        options=avail_states,
        index=0,
        help="States with active trading records for the selected crop."
    )

    # Cascading Districts for Selected State
    avail_districts = sorted(list(hierarchy.get(selected_crop, {}).get(selected_state, {}).keys()))
    if not avail_districts:
        avail_districts = ["All Districts"]

    selected_district = st.selectbox(
        "3. Select District 🏛️",
        options=avail_districts,
        index=0
    )

    # Cascading Markets (Mandis) for Selected District
    avail_markets = sorted(hierarchy.get(selected_crop, {}).get(selected_state, {}).get(selected_district, []))
    if not avail_markets:
        avail_markets = ["Standard Mandi"]

    selected_market = st.selectbox(
        "4. Agricultural Market / Mandi 🏪",
        options=avail_markets,
        index=0,
        help="The specific APMC Mandi or trading center."
    )

    # Date Inputs: Month and Year
    current_year = datetime.now().year
    fcol1, fcol2 = st.columns(2)

    with fcol1:
        selected_month_name = st.selectbox(
            "5. Expected Selling Month 📅",
            options=MONTH_NAMES,
            index=3  # April default
        )
        selected_month = MONTH_NAMES.index(selected_month_name) + 1

    with fcol2:
        selected_year = st.selectbox(
            "6. Expected Selling Year 🗓️",
            options=[current_year, current_year + 1, current_year + 2],
            index=0
        )

    # Optional Variety and Season
    fcol3, fcol4 = st.columns(2)
    with fcol3:
        selected_season = st.selectbox(
            "Season (Auto-detected) ☀️",
            options=["Auto", "Kharif", "Rabi", "Zaid"],
            index=0,
            help="Select 'Auto' to infer the season automatically from the selling month."
        )

    with fcol4:
        variety_options = ["Standard", "Deshi / Local", "Hybrid", "Lokwan / Sharbati", "Jyoti / Kufri", "Yellow"]
        selected_variety = st.selectbox(
            "Crop Variety / Grade 🏷️",
            options=variety_options,
            index=0
        )

    predict_btn = st.button("🚀 Calculate Estimated Price", use_container_width=True)

with col_result:
    if predict_btn:
        with st.spinner("Analyzing historical market patterns and calculating estimate..."):
            result = pred_service.predict_crop_price(
                crop=selected_crop,
                state=selected_state,
                district=selected_district,
                market=selected_market,
                month=selected_month,
                year=selected_year,
                variety=selected_variety,
                season=None if selected_season == "Auto" else selected_season,
                save_to_db=True
            )

        if not result["success"]:
            st.error(f"❌ {result.get('error', 'Prediction could not be completed.')}")
            if result.get("error_type") == "unsupported_market_combination":
                st.info("💡 Tip: Use the cascading dropdowns on the left to select mandis with verified training records.")
        else:
            pred_price = result["predicted_price"]
            low_price = result["price_range_low"]
            high_price = result["price_range_high"]
            margin = result["margin_error"]

            st.markdown(
                f"""
                <div class="agro-card" style="border-top: 5px solid {COLOR_FRESH_GREEN};">
                    <span class="agro-badge" style="margin-bottom: 0.5rem;">VALIDATED ML ESTIMATE</span>
                    <h4 style="color: #64748B !important; margin: 0.3rem 0; font-weight: 500;">
                        Estimated Modal Price for {result['crop']}
                    </h4>
                    <div style="font-size: 2.8rem; font-weight: 800; color: {COLOR_FOREST_GREEN}; margin: 0.3rem 0;">
                        ₹ {pred_price:,.2f}
                        <span style="font-size: 1.1rem; font-weight: 500; color: #64748B;"> / quintal</span>
                    </div>
                    <div style="background-color: {COLOR_SOFT_MINT}; border: 1px solid #BBF7D0; border-radius: 8px; padding: 0.8rem 1rem; margin: 1rem 0;">
                        <div style="color: {COLOR_FOREST_GREEN}; font-weight: 700; font-size: 0.95rem;">
                            Expected Price Band (90% Confidence):
                        </div>
                        <div style="color: {COLOR_TEXT_MAIN}; font-size: 1.15rem; font-weight: 700; margin-top: 0.2rem;">
                            ₹ {low_price:,.2f} &nbsp;—&nbsp; ₹ {high_price:,.2f}
                        </div>
                        <div style="color: #64748B; font-size: 0.8rem; margin-top: 0.2rem;">
                            Calibrated from holdout residual margin (± ₹{margin:,.2f})
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; font-size: 0.85rem; color: #334155;">
                        <div><b>Location:</b> {result['market']}, {result['district']}</div>
                        <div><b>State:</b> {result['state']}</div>
                        <div><b>Target Date:</b> {result['month_name']} {result['year']}</div>
                        <div><b>Season:</b> {result['season']}</div>
                        <div><b>Variety:</b> {result['variety']}</div>
                        <div><b>Model Version:</b> {result['model_version']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Historical Context & Mini Trend Chart
            st.markdown("#### 📜 Recent Historical Mandi Context")
            hist_ts = market_service.get_crop_time_series(
                crop=selected_crop,
                market=selected_market,
                state=selected_state
            )

            if not hist_ts.empty:
                # Take recent 25 weeks of historical data
                recent_hist = hist_ts.tail(25)
                latest_hist_price = recent_hist["Modal_Price"].iloc[-1]
                latest_hist_date = pd.to_datetime(recent_hist["Arrival_Date"].iloc[-1]).strftime("%d %b %Y")

                diff_from_latest = round(pred_price - latest_hist_price, 2)
                diff_sign = "+" if diff_from_latest >= 0 else ""

                st.markdown(
                    f"""
                    <div class="agro-card-subtle" style="margin-bottom: 0.8rem;">
                        <div style="font-size: 0.85rem; color: #475569;">
                            <b>Latest recorded mandi price:</b> ₹ {latest_hist_price:,.2f}/quintal ({latest_hist_date})<br>
                            <b>Projected difference:</b> <span style="font-weight: 700; color: {'#16A34A' if diff_from_latest >= 0 else '#DC2626'}">{diff_sign}₹ {diff_from_latest:,.2f}</span> compared to latest recorded trading.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Plot historical trend + future projection point
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=recent_hist["Arrival_Date"],
                    y=recent_hist["Modal_Price"],
                    mode="lines+markers",
                    name="Historical Modal Price",
                    line=dict(color=COLOR_FOREST_GREEN, width=2.5),
                    marker=dict(size=4)
                ))

                # Projection date (15th of selected month & year)
                pred_date = pd.to_datetime(f"{selected_year}-{selected_month:02d}-15")
                fig.add_trace(go.Scatter(
                    x=[pred_date],
                    y=[pred_price],
                    mode="markers",
                    name="AI Prediction",
                    marker=dict(color=COLOR_FRESH_GREEN, size=14, symbol="diamond", line=dict(color="#14532D", width=2)),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=[high_price - pred_price],
                        arrayminus=[pred_price - low_price],
                        color=COLOR_SAGE_GREEN,
                        width=4
                    )
                ))

                fig.update_layout(
                    get_plotly_layout(f"Historical Trend vs. Future Estimate ({selected_crop} at {selected_market})", height=280)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Disclaimer
            st.caption(f"ℹ️ {result['disclaimer']}")

    else:
        # Default placeholder before prediction
        st.markdown(
            f"""
            <div class="agro-card" style="text-align: center; padding: 3rem 1.5rem;">
                <div style="font-size: 3.5rem; margin-bottom: 1rem;">🌾⚖️</div>
                <h3 style="color: {COLOR_FOREST_GREEN} !important;">Ready to Estimate Crop Price</h3>
                <p style="color: #64748B; font-size: 0.95rem; line-height: 1.5;">
                    Select your crop, mandi, and expected selling schedule on the left, then click <b>"Calculate Estimated Price"</b>.
                </p>
                <div style="background: {COLOR_SOFT_MINT}; border-radius: 8px; padding: 1rem; border: 1px dashed {COLOR_SAGE_GREEN}; display: inline-block; text-align: left; margin-top: 1rem;">
                    <div style="font-weight: 600; color: {COLOR_FOREST_GREEN}; font-size: 0.88rem;">✨ Supported Features:</div>
                    <ul style="color: #334155; font-size: 0.85rem; margin: 0.3rem 0 0 1.2rem; padding: 0;">
                        <li>Model trained on multi-year Agmarknet price histories</li>
                        <li>Empirical 90% confidence bands based on actual residual standard error</li>
                        <li>Automated logging to local SQLite prediction history</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
