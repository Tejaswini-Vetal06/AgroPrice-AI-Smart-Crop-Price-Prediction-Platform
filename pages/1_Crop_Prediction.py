"""
AgroPrice AI — All-India Crop Price Prediction & Multi-Month Forecasting Hub.
Bilingual (Hindi/English) interface designed for Indian farmers and traders.
Supports point predictions, 6-12 month forward price forecasting curves,
best selling window detection, and sell-now vs. store profitability advisory.
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
    COLOR_TEXT_MAIN,
    MONTH_NAMES,
    SEASONS,
    CROP_HINDI_NAMES
)
from src.data_loader import get_cached_agricultural_data
from src.model_utils import is_model_ready, load_model_metadata
from src.predict import PredictionService
from src.market_data_service import MarketDataService


st.set_page_config(
    page_title="Crop Price Forecasting — AgroPrice AI",
    page_icon="🔮",
    layout="wide"
)

apply_custom_theme()

# Header with Hindi / English Bilingual Context
st.markdown(
    f"""
    <div style="margin-bottom: 1.25rem;">
        <span class="agro-badge">🇮🇳 अखिल भारतीय किसान सेवा / ALL-INDIA FARMER SERVICE</span>
        <h2 style="margin: 0.4rem 0 0.2rem 0; color: {COLOR_TEXT_MAIN} !important;">
            🔮 फसल भाव भविष्यवाणी एवं मूल्य पूर्वानुमान (Crop Price Forecasting)
        </h2>
        <p style="color: #475569; font-size: 0.95rem; margin: 0;">
            आगामी महीनों के संभावित मंडी भाव (₹ प्रति क्विंटल) जानें, सही समय पर फसल बेचने का निर्णय लें और भंडारण लाभ का विश्लेषण करें।
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if not is_model_ready():
    st.error("⚠️ Trained ML model was not found on disk. Please run the training pipeline first.")
    st.stop()

metadata = load_model_metadata()
hierarchy = metadata.get("market_hierarchy", {})
supported_crops = metadata.get("supported_crops", [])

pred_service = PredictionService()
raw_data = get_cached_agricultural_data()
market_service = MarketDataService(raw_data)

# Sidebar / Top Parameters Form
st.markdown("### 🌾 1. चयन करें (Select Crop & Mandi Location)")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)

with col_p1:
    # Display formatted bilingual crop options
    crop_display_options = {c: CROP_HINDI_NAMES.get(c, c) for c in supported_crops}
    selected_crop = st.selectbox(
        "फसल चुनें (Select Crop) 🌱",
        options=supported_crops,
        format_func=lambda x: crop_display_options.get(x, x),
        index=0
    )

with col_p2:
    avail_states = sorted(list(hierarchy.get(selected_crop, {}).keys()))
    if not avail_states:
        avail_states = ["All States"]
    selected_state = st.selectbox(
        "राज्य चुनें (Select State) 📍",
        options=avail_states,
        index=0
    )

with col_p3:
    avail_districts = sorted(list(hierarchy.get(selected_crop, {}).get(selected_state, {}).keys()))
    if not avail_districts:
        avail_districts = ["All Districts"]
    selected_district = st.selectbox(
        "ज़िला चुनें (Select District) 🏛️",
        options=avail_districts,
        index=0
    )

with col_p4:
    avail_markets = sorted(hierarchy.get(selected_crop, {}).get(selected_state, {}).get(selected_district, []))
    if not avail_markets:
        avail_markets = ["Standard Mandi"]
    selected_market = st.selectbox(
        "मंडी चुनें (Select APMC Mandi) 🏪",
        options=avail_markets,
        index=0
    )

# Current Year & Month Anchor
current_year = datetime.now().year
current_month = datetime.now().month

# Tabs for Prediction vs. Multi-Month Forward Forecasting
tab_predict, tab_forecast, tab_storage = st.tabs([
    "🎯 एकल माह मूल्य अनुमान (Target Month Prediction)",
    "📈 6-12 माह मूल्य पूर्वानुमान (Forward Price Curve)",
    "📦 भंडारण लाभ कैलकुलेटर (Sell Now vs. Store Advisory)"
])

# -------------------------------------------------------------
# TAB 1: TARGET MONTH PREDICTION
# -------------------------------------------------------------
with tab_predict:
    st.markdown("#### 🎯 लक्षित माह का अनुमानित मंडी भाव (Single Month Estimate)")
    
    col_t1, col_t2, col_t3 = st.columns([1, 1, 2])
    with col_t1:
        sel_month_name = st.selectbox("बिक्री माह (Selling Month) 📅", options=MONTH_NAMES, index=(current_month + 1) % 12)
        sel_month = MONTH_NAMES.index(sel_month_name) + 1
    with col_t2:
        sel_year = st.selectbox("बिक्री वर्ष (Selling Year) 🗓️", options=[current_year, current_year + 1, current_year + 2], index=0)
    with col_t3:
        variety_options = ["Standard", "Deshi / Local", "Hybrid", "Lokwan / Sharbati", "Jyoti / Kufri", "Yellow"]
        sel_variety = st.selectbox("किस्म / ग्रेड (Variety/Grade)", options=variety_options, index=0)

    if st.button("🚀 अनुमानित भाव देखें (Calculate Price Estimate)", key="btn_single_pred"):
        with st.spinner("Analyzing all-India Agmarknet records..."):
            res = pred_service.predict_crop_price(
                crop=selected_crop,
                state=selected_state,
                district=selected_district,
                market=selected_market,
                month=sel_month,
                year=sel_year,
                variety=sel_variety,
                save_to_db=True
            )

        if not res["success"]:
            st.error(f"❌ {res.get('error')}")
        else:
            p_price = res["predicted_price"]
            p_low = res["price_range_low"]
            p_high = res["price_range_high"]
            margin = res["margin_error"]

            st.markdown(
                f"""
                <div class="agro-card" style="border-top: 5px solid {COLOR_FRESH_GREEN}; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span class="agro-badge">AI ESTIMATE / एआई अनुमान</span>
                            <h3 style="margin: 0.3rem 0; color: {COLOR_TEXT_MAIN} !important;">
                                {res['crop_hindi']} ({res['crop']}) — {res['market']} Mandi
                            </h3>
                            <div style="color: #64748B; font-size: 0.88rem;">
                                राज्य: {res['state']} &bull; ज़िला: {res['district']} &bull; लक्षित समय: <b>{res['month_name']} {res['year']}</b> ({res['season']} मौसम)
                            </div>
                        </div>
                        <div style="text-align: right; padding: 0.5rem 0;">
                            <div style="font-size: 2.6rem; font-weight: 800; color: {COLOR_FOREST_GREEN};">
                                ₹ {p_price:,.2f}
                            </div>
                            <div style="font-size: 0.95rem; color: #64748B; font-weight: 600;">प्रति क्विंटल (Per 100 kg)</div>
                        </div>
                    </div>
                    
                    <div style="background-color: {COLOR_SOFT_MINT}; border: 1px solid #BBF7D0; border-radius: 8px; padding: 0.9rem 1.2rem; margin: 1rem 0;">
                        <div style="color: {COLOR_FOREST_GREEN}; font-weight: 700; font-size: 0.95rem;">
                            📊 संभावित मूल्य सीमा (90% Confidence Interval):
                        </div>
                        <div style="color: {COLOR_TEXT_MAIN}; font-size: 1.25rem; font-weight: 800; margin-top: 0.2rem;">
                            ₹ {p_low:,.2f} &nbsp;—&nbsp; ₹ {p_high:,.2f}
                        </div>
                        <div style="color: #64748B; font-size: 0.82rem; margin-top: 0.2rem;">
                            (मंडी आवक और मौसमी उतार-चढ़ाव के आधार पर अनुमानित प्रसार: ± ₹{margin:,.2f})
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Government MSP Benchmarking Callout
            msp_data = res.get("msp_info")
            if msp_data:
                msp_val = msp_data["msp_price"]
                diff_val = msp_data["difference"]
                diff_pct = msp_data["difference_pct"]
                is_above = msp_data["is_above_msp"]
                
                banner_bg = "#DCFCE7" if is_above else "#FEF3C7"
                banner_border = "#22C55E" if is_above else "#F59E0B"
                badge_text = "MSP से अधिक (Above MSP)" if is_above else "MSP अलर्ट (Near / Below MSP)"

                st.markdown(
                    f"""
                    <div style="background-color: {banner_bg}; border: 1px solid {banner_border}; border-left: 5px solid {banner_border}; border-radius: 8px; padding: 1rem; margin-bottom: 1.25rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                            <div>
                                <b style="color: {COLOR_FOREST_GREEN}; font-size: 0.95rem;">
                                    🏛️ सरकारी न्यूनतम समर्थन मूल्य (Govt MSP Benchmark): ₹ {msp_val:,.2f} / क्विंटल
                                </b>
                                <div style="color: #334155; font-size: 0.88rem; margin-top: 0.2rem;">
                                    {msp_data['guidance']}
                                </div>
                            </div>
                            <div style="margin-top: 0.4rem;">
                                <span class="agro-badge" style="background: {'#16A34A' if is_above else '#D97706'}; color: white; border: none;">
                                    {badge_text}: {'+' if diff_val >= 0 else ''}₹{diff_val:,.2f} ({diff_pct}%)
                                </span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Recent Historical Price Context
            hist_ts = market_service.get_crop_time_series(crop=selected_crop, market=selected_market, state=selected_state)
            if not hist_ts.empty:
                recent_pt = hist_ts.iloc[-1]
                latest_price = recent_pt["Modal_Price"]
                latest_date = pd.to_datetime(recent_pt["Arrival_Date"]).strftime("%d %b %Y")
                diff_from_today = round(p_price - latest_price, 2)
                diff_sign = "+" if diff_from_today >= 0 else ""

                st.info(
                    f"📌 **पिछला दर्ज मंडी भाव:** ₹ {latest_price:,.2f}/क्विंटल ({latest_date}) &bull; "
                    f"**संभावित बदलाव:** {diff_sign}₹ {diff_from_today:,.2f} प्रति क्विंटल"
                )

# -------------------------------------------------------------
# TAB 2: MULTI-MONTH FORWARD PRICE FORECASTING CURVE
# -------------------------------------------------------------
with tab_forecast:
    st.markdown("#### 📈 आगामी 6 से 12 महीनों का भाव पूर्वानुमान (Forward Price Curve)")
    st.markdown("यह ग्राफ दिखाता है कि आने वाले महीनों में फसल के भाव कब बढ़ेंगे और कब घटेंगे, जिससे आप सही महीने में फसल बेच सकें।")

    col_fc1, col_fc2 = st.columns([1, 1])
    with col_fc1:
        forecast_horizon = st.radio(
            "पूर्वानुमान अवधि (Forecast Horizon) ⏱️",
            options=[6, 9, 12],
            format_func=lambda x: f"अगले {x} महीने (Next {x} Months)",
            horizontal=True
        )
    with col_fc2:
        start_forecast_month = st.selectbox(
            "प्रारंभिक माह (Start Month) 📅",
            options=MONTH_NAMES,
            index=(current_month - 1),
            key="start_fc_month"
        )
        fc_start_m = MONTH_NAMES.index(start_forecast_month) + 1

    if st.button("📊 बहु-मासिक पूर्वानुमान ग्राफ बनाएं (Generate Forward Curve)", key="btn_multi_fc"):
        with st.spinner("Projecting forward seasonal trajectory..."):
            fc_res = pred_service.forecast_multi_month_trajectory(
                crop=selected_crop,
                state=selected_state,
                district=selected_district,
                market=selected_market,
                start_month=fc_start_m,
                start_year=current_year,
                horizon_months=forecast_horizon
            )

        if not fc_res["success"]:
            st.error(f"❌ {fc_res.get('error')}")
        else:
            traj_df = fc_res["trajectory_df"]
            peak_m = fc_res["peak_month"]
            peak_p = fc_res["peak_price"]
            peak_gain = fc_res["peak_gain"]
            peak_gain_pct = fc_res["peak_gain_pct"]
            trough_m = fc_res["trough_month"]
            trough_p = fc_res["trough_price"]

            # Highlight Cards
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.markdown(
                    f"""
                    <div class="agro-card" style="border-left: 5px solid {COLOR_FRESH_GREEN};">
                        <span class="agro-badge" style="background-color: #22C55E; color: white; border: none;">
                            🏆 सर्वोत्तम बिक्री समय (BEST MONTH TO SELL)
                        </span>
                        <h3 style="margin: 0.4rem 0 0.1rem 0; color: {COLOR_TEXT_MAIN} !important;">
                            {peak_m}
                        </h3>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #16A34A;">
                            ₹ {peak_p:,.2f} <span style="font-size: 0.9rem; color: #64748B;">/ क्विंटल</span>
                        </div>
                        <div style="color: #475569; font-size: 0.85rem; margin-top: 0.3rem;">
                            प्रारंभिक महीने से संभावित बढ़त: <b>+{peak_gain_pct}% (+₹{peak_gain:,.2f})</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_h2:
                st.markdown(
                    f"""
                    <div class="agro-card" style="border-left: 5px solid #0284C7;">
                        <span class="agro-badge" style="background-color: #0284C7; color: white; border: none;">
                            📉 आवक का न्यूनतम भाव (HARVEST TROUGH)
                        </span>
                        <h3 style="margin: 0.4rem 0 0.1rem 0; color: {COLOR_TEXT_MAIN} !important;">
                            {trough_m}
                        </h3>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #0284C7;">
                            ₹ {trough_p:,.2f} <span style="font-size: 0.9rem; color: #64748B;">/ क्विंटल</span>
                        </div>
                        <div style="color: #475569; font-size: 0.85rem; margin-top: 0.3rem;">
                            कटाई एवं भारी आवक के कारण भाव में संभावित दबाव
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Interactive Forward Trajectory Plotly Chart
            fig_fc = go.Figure()

            # Upper Confidence Bound
            fig_fc.add_trace(go.Scatter(
                x=traj_df["month_label"],
                y=traj_df["price_high"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))

            # Lower Confidence Bound with Shading
            fig_fc.add_trace(go.Scatter(
                x=traj_df["month_label"],
                y=traj_df["price_low"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(134, 167, 137, 0.2)",
                name="90% Expected Price Spread"
            ))

            # Main Forecast Line
            fig_fc.add_trace(go.Scatter(
                x=traj_df["month_label"],
                y=traj_df["predicted_price"],
                mode="lines+markers",
                name="Forecasted Modal Price (₹/Quintal)",
                line=dict(color=COLOR_FOREST_GREEN, width=3),
                marker=dict(size=8, color=COLOR_FRESH_GREEN, line=dict(color=COLOR_FOREST_GREEN, width=1.5))
            ))

            # Mark Peak Point
            fig_fc.add_trace(go.Scatter(
                x=[peak_m],
                y=[peak_p],
                mode="markers+text",
                name="Peak Window",
                text=[f"Peak: ₹{peak_p:,.0f}"],
                textposition="top center",
                marker=dict(color="#16A34A", size=14, symbol="star")
            ))

            fig_fc.update_layout(
                get_plotly_layout(f"{selected_crop} Forward Price Forecast ({selected_market} Mandi)", height=380)
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Trajectory Table
            with st.expander("📋 मासिक पूर्वानुमान विवरण तालिका (Monthly Forecast Table)", expanded=False):
                traj_display = traj_df[["month_label", "season", "predicted_price", "price_low", "price_high"]].rename(columns={
                    "month_label": "माह एवं वर्ष (Month)",
                    "season": "मौसम (Season)",
                    "predicted_price": "अनुमानित भाव (₹/Qtl)",
                    "price_low": "न्यूनतम सीमा (₹)",
                    "price_high": "अधिकतम सीमा (₹)"
                })
                st.dataframe(traj_display, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 3: SELL NOW VS. STORE PROFITABILITY CALCULATOR
# -------------------------------------------------------------
with tab_storage:
    st.markdown("#### 📦 भंडारण लाभ कैलकुलेटर (Sell Immediately vs. Store in Warehouse)")
    st.markdown("क्या फसल को तुरंत बेचना चाहिए या 2-4 महीने गोदाम/कोल्ड स्टोरेज में रखने से मुनाफा होगा? नीचे गणना करें:")

    col_st1, col_st2, col_st3 = st.columns(3)
    with col_st1:
        current_harvest_price = st.number_input(
            "वर्तमान कटाई समय का भाव (Current Mandi Price) ₹/Qtl",
            min_value=500.0,
            max_value=100000.0,
            value=2400.0,
            step=50.0
        )
    with col_st2:
        hold_months = st.slider(
            "भंडारण अवधि (Months to Hold) ⏳",
            min_value=1,
            max_value=8,
            value=3
        )
    with col_st3:
        godown_rent = st.number_input(
            "गोदाम किराया प्रति माह (Godown Rent/Month) ₹/Qtl",
            min_value=10.0,
            max_value=200.0,
            value=35.0,
            step=5.0,
            help="साधारण अनाज गोदाम का किराया लगभग ₹25-40/क्विंटल/माह, कोल्ड स्टोरेज का ₹60-90/माह होता है।"
        )

    # Project price for target holding month
    target_mo = (current_month + hold_months - 1) % 12 + 1
    target_yr = current_year + ((current_month + hold_months - 1) // 12)

    pred_future = pred_service.predict_crop_price(
        crop=selected_crop,
        state=selected_state,
        district=selected_district,
        market=selected_market,
        month=target_mo,
        year=target_yr,
        save_to_db=False
    )

    if pred_future["success"]:
        future_price = pred_future["predicted_price"]
        decision = pred_service.evaluate_storage_decision(
            crop=selected_crop,
            current_price=current_harvest_price,
            target_predicted_price=future_price,
            months_to_hold=hold_months,
            custom_storage_cost_per_month=godown_rent
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            st.markdown(
                f"""
                <div class="agro-metric-box">
                    <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">भविष्य का अनुमानित भाव</div>
                    <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.6rem; font-weight: 800;">₹ {future_price:,.2f}</div>
                    <div style="color: #64748B; font-size: 0.8rem;">{hold_months} महीने बाद ({MONTH_NAMES[target_mo - 1]} {target_yr})</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_res2:
            st.markdown(
                f"""
                <div class="agro-metric-box">
                    <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">कुल भंडारण एवं नमी खर्च</div>
                    <div style="color: #DC2626; font-size: 1.6rem; font-weight: 800;">- ₹ {decision['total_holding_expenditure']:,.2f}</div>
                    <div style="color: #64748B; font-size: 0.8rem;">किराया: ₹{decision['total_rent']} + नमी कमी: ₹{decision['shrinkage_loss']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_res3:
            net_profit = decision["net_profit_loss"]
            profit_color = "#16A34A" if net_profit >= 0 else "#DC2626"
            st.markdown(
                f"""
                <div class="agro-metric-box">
                    <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">शुद्ध लाभ / हानि (NET PROFIT)</div>
                    <div style="color: {profit_color}; font-size: 1.6rem; font-weight: 800;">{'+' if net_profit >= 0 else ''}₹ {net_profit:,.2f}</div>
                    <div style="color: #64748B; font-size: 0.8rem;">प्रति क्विंटल शुद्ध प्रभाव</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Recommendation Banner
        rec_bg = "#DCFCE7" if decision["is_profitable"] else "#FEE2E2"
        rec_border = "#22C55E" if decision["is_profitable"] else "#EF4444"
        rec_icon = "✅" if decision["is_profitable"] else "⚠️"

        st.markdown(
            f"""
            <div style="background-color: {rec_bg}; border: 1px solid {rec_border}; border-left: 6px solid {rec_border}; border-radius: 8px; padding: 1.1rem; margin-top: 1.25rem;">
                <h4 style="margin: 0 0 0.3rem 0; color: {'#14532D' if decision['is_profitable'] else '#991B1B'} !important;">
                    {rec_icon} सलाह (Advisory): {decision['advisory_hi']}
                </h4>
                <p style="margin: 0; color: #334155; font-size: 0.88rem;">
                    {decision['advisory_en']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# General Agricultural Disclaimer
st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "⚠️ **महत्वपूर्ण सूचना:** यह भविष्यवाणी ऐतिहासिक Agmarknet डेटा पर आधारित सांख्यिकीय अनुमान है। वास्तविक मंडी भाव स्थानीय मौसम, दैनिक ट्रक आवक, फसल की नमी और गुणवत्ता पर निर्भर करते हैं। कोई भी बड़ा वित्तीय निर्णय लेने से पहले अपनी स्थानीय कृषि उपज मंडी समिति (APMC) से संपर्क करें।"
)
