"""
AgroPrice AI — Compare Crops and Mandis Page.
Side-by-side mandi comparisons, price differences in Indian Rupees,
and critical economic reminders regarding freight and storage costs.
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
    COLOR_WHITE
)
from src.data_loader import get_cached_agricultural_data
from src.market_data_service import MarketDataService


st.set_page_config(
    page_title="Compare Mandis — AgroPrice AI",
    page_icon="⚖️",
    layout="wide"
)

apply_custom_theme()

st.markdown("## ⚖️ Compare Crops & Mandis")
st.markdown("Compare price differences across different agricultural trading centers in India to evaluate regional market variations.")

# Data Initialization
if "active_data" in st.session_state:
    df = st.session_state.active_data
else:
    df = get_cached_agricultural_data()

service = MarketDataService(df)

# Selection controls
col_c1, col_c2 = st.columns([1, 2])

with col_c1:
    crops = sorted(df["Commodity"].unique().tolist())
    selected_crop = st.selectbox("Select Commodity 🌾", options=crops, index=0)

with col_c2:
    all_mandis = sorted(df[df["Commodity"] == selected_crop]["Market"].unique().tolist())
    default_mandis = all_mandis[:3] if len(all_mandis) >= 3 else all_mandis
    selected_mandis = st.multiselect(
        "Select Mandis to Compare 🏪 (Choose 2 or more)",
        options=all_mandis,
        default=default_mandis
    )

if len(selected_mandis) < 2:
    st.info("💡 Please select at least two mandis from the list above to perform a side-by-side comparison.")
    st.stop()

# Comparison Data
comp_df = service.compare_markets_for_crop(selected_crop, selected_mandis)

if comp_df.empty:
    st.warning("No comparative records found for the selected mandis.")
    st.stop()

# Price Spread Highlights
st.markdown("### 🏷️ Latest Market Price Differences")

top_market = comp_df.iloc[0]["Market"]
top_price = comp_df.iloc[0]["Latest_Price"]
lowest_market = comp_df.iloc[-1]["Market"]
lowest_price = comp_df.iloc[-1]["Latest_Price"]
diff_rupees = round(top_price - lowest_price, 2)

col_diff1, col_diff2, col_diff3 = st.columns(3)

with col_diff1:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">HIGHEST QUOTED MANDI</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.6rem; font-weight: 800;">{top_market}</div>
            <div style="color: #16A34A; font-size: 0.95rem; font-weight: 700; margin-top: 0.2rem;">₹ {top_price:,.2f} / quintal</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_diff2:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">LOWEST QUOTED MANDI</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.6rem; font-weight: 800;">{lowest_market}</div>
            <div style="color: #0284C7; font-size: 0.95rem; font-weight: 700; margin-top: 0.2rem;">₹ {lowest_price:,.2f} / quintal</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_diff3:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">MAX REGIONAL DIFFERENCE</div>
            <div style="color: #D97706; font-size: 1.6rem; font-weight: 800;">₹ {diff_rupees:,.2f}</div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 0.2rem;">Gross difference per quintal</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Critical Transportation & Logistics Advisory
st.markdown(
    f"""
    <div style="background-color: #FEF3C7; border: 1px solid #FDE68A; border-left: 5px solid #D97706; padding: 1.1rem; border-radius: 8px; margin: 1.25rem 0;">
        <h4 style="color: #92400E !important; margin: 0 0 0.3rem 0; font-size: 0.95rem;">
            ⚠️ Essential Net Profitability Warning for Farmers & Traders
        </h4>
        <p style="color: #78350F; font-size: 0.88rem; margin: 0; line-height: 1.5;">
            <b>A higher mandi price does not automatically imply higher net profit!</b> Transporting produce to a distant market incurs diesel/freight costs, toll charges, bag packing fees, APMC mandi commission (arhat / cess 1-2%), loading/unloading charges, transit weight loss, and potential spoilage risks. 
            Always deduct the total logistics expenditure before deciding to transport produce outside your home district.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Side-by-Side Comparison Chart
st.markdown("### 📊 Side-by-Side Price Bar Chart")
fig_comp = px.bar(
    comp_df,
    x="Market",
    y="Latest_Price",
    color="State",
    text_auto="₹,.0f",
    labels={"Latest_Price": "Latest Price (₹/Quintal)", "Market": "Mandi Center"},
    color_discrete_sequence=[COLOR_FOREST_GREEN, COLOR_FRESH_GREEN, COLOR_SAGE_GREEN, "#34D399"]
)
fig_comp.update_layout(get_plotly_layout(f"Latest Recorded Modal Prices for {selected_crop}", height=350))
st.plotly_chart(fig_comp, use_container_width=True)

# Historical Multi-Market Overlay Chart
st.markdown("### 📈 Historical Price Trends Across Selected Mandis")
fig_ts = go.Figure()

for mkt in selected_mandis:
    m_ts = service.get_crop_time_series(crop=selected_crop, market=mkt)
    if not m_ts.empty:
        fig_ts.add_trace(go.Scatter(
            x=m_ts["Arrival_Date"],
            y=m_ts["Modal_Price"],
            mode="lines",
            name=mkt,
            line=dict(width=2)
        ))

fig_ts.update_layout(get_plotly_layout(f"Multi-Mandi Historical Trajectory ({selected_crop})", height=380))
st.plotly_chart(fig_ts, use_container_width=True)

# Comparison Table
st.markdown("### 📋 Mandi Comparison Breakdown")
comp_display = comp_df.rename(columns={
    "Market": "Mandi",
    "Latest_Date": "As of Date",
    "Latest_Price": "Latest Price (₹/Qtl)",
    "Avg_Price": "Historical Avg (₹/Qtl)",
    "Min_Observed": "Min Observed (₹)",
    "Max_Observed": "Max Observed (₹)"
})
st.dataframe(comp_display, use_container_width=True, hide_index=True)
