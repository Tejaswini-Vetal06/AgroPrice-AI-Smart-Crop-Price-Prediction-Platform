"""
AgroPrice AI — Crop Insights & Seasonal Patterns Page.
Visualizes multi-year historical price trajectories, harvest season dynamics,
and month-over-month percentage changes with objective explanations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    apply_custom_theme,
    get_plotly_layout,
    COLOR_FOREST_GREEN,
    COLOR_FRESH_GREEN,
    COLOR_SAGE_GREEN,
    COLOR_SOFT_MINT,
    COLOR_WHITE,
    SUPPORTED_CROPS
)
from src.data_loader import get_cached_agricultural_data
from src.market_data_service import MarketDataService


st.set_page_config(
    page_title="Crop Insights — AgroPrice AI",
    page_icon="📈",
    layout="wide"
)

apply_custom_theme()

st.markdown("## 📈 Crop Price Insights & Seasonal Dynamics")
st.markdown("Analyze multi-year historical trends, cyclical harvest seasons, and month-over-month price behavior across Indian mandis.")

# Data Initialization
if "active_data" in st.session_state:
    df = st.session_state.active_data
else:
    df = get_cached_agricultural_data()

service = MarketDataService(df)

# Crop & Market Selection
col_sel1, col_sel2 = st.columns([1, 2])

with col_sel1:
    crops = sorted(df["Commodity"].unique().tolist())
    selected_crop = st.selectbox("Select Crop to Analyze 🌾", options=crops, index=0)

with col_sel2:
    available_markets = ["All Mandis"] + sorted(df[df["Commodity"] == selected_crop]["Market"].unique().tolist())
    selected_market = st.selectbox("Filter Mandi (Optional) 🏪", options=available_markets, index=0)

target_market = None if selected_market == "All Mandis" else selected_market

# Fetch time series and patterns
ts_data = service.get_crop_time_series(crop=selected_crop, market=target_market)
seasonal_data = service.get_seasonal_monthly_pattern(crop=selected_crop)
mom_data = service.get_month_over_month_change(crop=selected_crop)

# Key Stats Overview
st.markdown(f"### 🔍 Historical Statistics for {selected_crop}")
stat1, stat2, stat3, stat4 = st.columns(4)

if not ts_data.empty:
    avg_price = ts_data["Modal_Price"].mean()
    min_recorded = ts_data["Min_Price"].min()
    max_recorded = ts_data["Max_Price"].max()
    volatility = ts_data["Modal_Price"].std()
else:
    avg_price = min_recorded = max_recorded = volatility = 0.0

with stat1:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">AVERAGE HISTORICAL PRICE</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.8rem; font-weight: 800;">₹ {avg_price:,.2f}</div>
            <div style="color: #64748B; font-size: 0.8rem;">Per quintal</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with stat2:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">LOWEST RECORDED MODAL</div>
            <div style="color: #0284C7; font-size: 1.8rem; font-weight: 800;">₹ {min_recorded:,.2f}</div>
            <div style="color: #64748B; font-size: 0.8rem;">Seasonal low</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with stat3:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">PEAK RECORDED MODAL</div>
            <div style="color: #16A34A; font-size: 1.8rem; font-weight: 800;">₹ {max_recorded:,.2f}</div>
            <div style="color: #64748B; font-size: 0.8rem;">Seasonal peak</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with stat4:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">PRICE VOLATILITY (STD DEV)</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.8rem; font-weight: 800;">± ₹ {volatility:,.2f}</div>
            <div style="color: #64748B; font-size: 0.8rem;">Historical spread</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# 1. Historical Time Series Chart
st.markdown("### 📊 Multi-Year Price Trajectory")
if not ts_data.empty:
    fig_ts = go.Figure()

    # Max-Min Band
    fig_ts.add_trace(go.Scatter(
        x=ts_data["Arrival_Date"],
        y=ts_data["Max_Price"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_data["Arrival_Date"],
        y=ts_data["Min_Price"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(134, 167, 137, 0.2)",
        name="Min-Max Price Spread"
    ))

    # Modal Price Line
    fig_ts.add_trace(go.Scatter(
        x=ts_data["Arrival_Date"],
        y=ts_data["Modal_Price"],
        mode="lines",
        line=dict(color=COLOR_FOREST_GREEN, width=2.5),
        name="Modal Price (₹/Quintal)"
    ))

    chart_title = f"{selected_crop} Historical Market Price Trajectory ({selected_market})"
    fig_ts.update_layout(get_plotly_layout(chart_title, height=380))
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("No time series data available for the chosen selection.")

# 2. Seasonal Patterns & Month-over-Month
col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown("### 🗓️ Seasonal Month-of-Year Cycle")
    st.markdown("Average price across all historical years, showing seasonal dips during harvest arrivals and peaks during off-season stock releases.")
    if not seasonal_data.empty:
        fig_season = px.bar(
            seasonal_data,
            x="Month_Name",
            y="Avg_Modal",
            labels={"Avg_Modal": "Average Price (₹/Quintal)", "Month_Name": "Month"},
            color_discrete_sequence=[COLOR_FRESH_GREEN]
        )
        fig_season.update_layout(get_plotly_layout(f"Average Monthly Pattern for {selected_crop}", height=320))
        st.plotly_chart(fig_season, use_container_width=True)

        # Contextual explanation
        min_mo = seasonal_data.loc[seasonal_data["Avg_Modal"].idxmin()]["Month_Name"]
        max_mo = seasonal_data.loc[seasonal_data["Avg_Modal"].idxmax()]["Month_Name"]
        st.caption(
            f"💡 **Observed Trend:** Historically, {selected_crop} prices tend to reach seasonal lows around **{min_mo}** (often coinciding with primary harvest mandi arrivals) and higher levels around **{max_mo}** (off-season storage offloading)."
        )

with col_p2:
    st.markdown("### 🔄 Month-over-Month Price Change (%)")
    st.markdown("Percentage change compared to preceding month across the timeline.")
    if not mom_data.empty:
        # Take recent 24 months
        recent_mom = mom_data.tail(24).copy()
        recent_mom["Color"] = recent_mom["MoM_Change_Pct"].apply(lambda x: COLOR_FRESH_GREEN if x >= 0 else "#EF4444")

        fig_mom = go.Figure(go.Bar(
            x=recent_mom["YearMonth_Str"],
            y=recent_mom["MoM_Change_Pct"],
            marker_color=recent_mom["Color"]
        ))
        fig_mom.update_layout(get_plotly_layout(f"MoM % Price Delta ({selected_crop})", height=320))
        fig_mom.update_yaxes(title_text="MoM Change (%)")
        st.plotly_chart(fig_mom, use_container_width=True)

        st.caption("ℹ️ Green bars indicate monthly price increases; red bars indicate price declines.")

# Download Filtered Data
st.markdown("### 📥 Download Historical Insights Dataset")
if not ts_data.empty:
    csv_bytes = ts_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {selected_crop} Historical Records (CSV)",
        data=csv_bytes,
        file_name=f"{selected_crop.lower()}_historical_prices.csv",
        mime="text/csv"
    )
