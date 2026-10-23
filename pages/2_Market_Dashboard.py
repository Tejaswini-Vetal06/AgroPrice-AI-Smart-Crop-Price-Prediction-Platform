"""
AgroPrice AI — Market Price Dashboard Page.
Displays latest available mandi prices, minimum/maximum/modal spreads,
state/district/commodity filters, and allows custom Agmarknet CSV uploads.
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
    COLOR_TEXT_MAIN,
    t,
    get_crop_name,
    render_language_selector
)
from src.data_loader import DataLoader, get_cached_agricultural_data
from src.market_data_service import MarketDataService


st.set_page_config(
    page_title="Market Price Dashboard — AgroPrice AI",
    page_icon="📊",
    layout="wide"
)

apply_custom_theme()

with st.sidebar:
    render_language_selector()
    st.markdown("---")

st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, #064E3B 0%, #065F46 45%, #047857 100%); padding: 1.5rem 1.8rem; border-radius: 14px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 15px -3px rgba(6, 78, 59, 0.25); border: 1px solid rgba(255, 255, 255, 0.15);">
        <span style="background-color: #FEF3C7; color: #92400E; font-size: 0.78rem; font-weight: 700; padding: 0.25rem 0.7rem; border-radius: 9999px; text-transform: uppercase; letter-spacing: 0.04em;">
            🏛️ AGMARKNET MANDI INTELLIGENCE
        </span>
        <h1 style="color: #FFFFFF !important; font-size: 2rem; font-weight: 800; margin: 0.6rem 0 0.25rem 0;">
            📊 <span style="color: #FBBF24;">{t('btn_dashboard')}</span>
        </h1>
        <p style="color: #ECFDF5; font-size: 0.98rem; margin: 0; line-height: 1.5;">
            Browse tracked Agmarknet mandi arrivals, minimum/maximum/modal price spreads, and regional trends across India.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Data Source & Session State Management
if "active_data" not in st.session_state:
    st.session_state.active_data = get_cached_agricultural_data()
    st.session_state.is_custom_upload = False
    st.session_state.source_name = "Agmarknet Historical Sample (Govt of India)"

# Upload Custom Dataset Expander
with st.expander("📤 Upload Custom Agricultural Price CSV Callset", expanded=False):
    st.markdown(
        """
        Upload an official Agmarknet or OGD India market prices CSV. The file must contain standard columns:
        `Arrival_Date`, `State`, `District`, `Market`, `Commodity`, `Modal_Price` (optional: `Min_Price`, `Max_Price`, `Variety`).
        """
    )
    uploaded_file = st.file_uploader("Select CSV file", type=["csv"], key="csv_uploader")
    if uploaded_file is not None:
        try:
            loader = DataLoader()
            custom_df, meta = loader.load_from_csv(uploaded_file)
            st.session_state.active_data = custom_df
            st.session_state.is_custom_upload = True
            st.session_state.source_name = f"Uploaded File: {uploaded_file.name}"
            st.success(f"✅ Successfully loaded {meta['record_count']:,} verified records for {meta['crop_count']} crops!")
        except Exception as e:
            st.error(f"❌ Failed to parse CSV: {str(e)}")

    if st.session_state.is_custom_upload:
        if st.button("🔄 Reset to Default Historical Dataset"):
            st.session_state.active_data = get_cached_agricultural_data()
            st.session_state.is_custom_upload = False
            st.session_state.source_name = "Agmarknet Historical Sample (Govt of India)"
            st.rerun()

df = st.session_state.active_data
service = MarketDataService(df)

# Source and Transparency Notice
source_badge_color = "#16A34A" if st.session_state.is_custom_upload else "#D97706"
st.markdown(
    f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <b>Data Source:</b> {st.session_state.source_name} &bull; 
            <b>Records:</b> {len(df):,} rows &bull; 
            <b>Date Span:</b> {df['Arrival_Date'].min().strftime('%d %b %Y')} to {df['Arrival_Date'].max().strftime('%d %b %Y')}
        </div>
        <div style="font-size: 0.8rem; color: #64748B;">
            ⚠️ <i>Prices reflect recorded market arrivals; not live real-time tickers.</i>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Filters Row
st.markdown("### 🔍 Filter Market Records")
col_f1, col_f2, col_f3, col_f4 = st.columns(4)

with col_f1:
    crops = ["All"] + sorted(df["Commodity"].unique().tolist())
    sel_crop = st.selectbox("Commodity", options=crops, index=0)

with col_f2:
    states = ["All"] + sorted(df["State"].unique().tolist())
    sel_state = st.selectbox("State", options=states, index=0)

with col_f3:
    if sel_state != "All":
        districts = ["All"] + sorted(df[df["State"] == sel_state]["District"].unique().tolist())
    else:
        districts = ["All"] + sorted(df["District"].unique().tolist())
    sel_district = st.selectbox("District", options=districts, index=0)

with col_f4:
    search_query = st.text_input("Search Mandi / Town", placeholder="e.g. Lasalgaon, Khanna...")

# Apply Filters
latest_df = service.get_latest_prices(
    crop=None if sel_crop == "All" else sel_crop,
    state=None if sel_state == "All" else sel_state,
    district=None if sel_district == "All" else sel_district
)

if search_query:
    latest_df = latest_df[
        latest_df["Market"].str.contains(search_query, case=False, na=False) |
        latest_df["District"].str.contains(search_query, case=False, na=False)
    ]

# Summary KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">MATCHING MANDIS</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.8rem; font-weight: 800;">{len(latest_df)} Centers</div>
            <div style="color: #64748B; font-size: 0.8rem;">Active price centers</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi2:
    avg_price = latest_df["Modal_Price"].mean() if not latest_df.empty else 0.0
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">AVERAGE MODAL PRICE</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.8rem; font-weight: 800;">₹ {avg_price:,.2f}</div>
            <div style="color: #64748B; font-size: 0.8rem;">Per quintal (100 kg)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi3:
    min_price = latest_df["Min_Price"].min() if not latest_df.empty else 0.0
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">LOWEST QUOTED PRICE</div>
            <div style="color: #0284C7; font-size: 1.8rem; font-weight: 800;">₹ {min_price:,.2f}</div>
            <div style="color: #64748B; font-size: 0.8rem;">Minimum market quote</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi4:
    max_price = latest_df["Max_Price"].max() if not latest_df.empty else 0.0
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">HIGHEST QUOTED PRICE</div>
            <div style="color: #16A34A; font-size: 1.8rem; font-weight: 800;">₹ {max_price:,.2f}</div>
            <div style="color: #64748B; font-size: 0.8rem;">Maximum market quote</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Charts Section: Crop-wise Price Spread & Mandi Comparison
st.markdown("### 📈 Price Distributions & Mandi Spreads")
tab_chart1, tab_chart2 = st.tabs(["Crop-wise Price Spreads (₹/Quintal)", "Top Mandis by Modal Price"])

with tab_chart1:
    if not latest_df.empty:
        fig_bar = px.bar(
            latest_df.groupby("Commodity")[["Min_Price", "Modal_Price", "Max_Price"]].mean().reset_index(),
            x="Commodity",
            y="Modal_Price",
            error_y=None,
            labels={"Modal_Price": "Average Modal Price (₹/Quintal)", "Commodity": "Crop"},
            color_discrete_sequence=[COLOR_FRESH_GREEN]
        )
        fig_bar.update_layout(get_plotly_layout("Average Mandi Modal Price by Commodity", height=350))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No records matching the selected filters.")

with tab_chart2:
    if not latest_df.empty:
        top_mandis = latest_df.sort_values(by="Modal_Price", ascending=True).tail(12)
        fig_mandi = px.bar(
            top_mandis,
            x="Modal_Price",
            y="Market",
            orientation="h",
            color="Commodity",
            labels={"Modal_Price": "Modal Price (₹/Quintal)", "Market": "Mandi Center"},
            color_discrete_sequence=[COLOR_FOREST_GREEN, COLOR_FRESH_GREEN, COLOR_SAGE_GREEN, "#34D399", "#059669"]
        )
        fig_mandi.update_layout(get_plotly_layout("Leading Mandi Quotations", height=380))
        st.plotly_chart(fig_mandi, use_container_width=True)
    else:
        st.info("No records to plot.")

# Data Table Section with Sorting
st.markdown("### 📋 Mandi Price Records Table")
if not latest_df.empty:
    display_df = latest_df[[
        "Commodity", "Market", "District", "State", "Arrival_Date",
        "Min_Price", "Modal_Price", "Max_Price", "Variety"
    ]].copy()

    display_df["Arrival_Date"] = pd.to_datetime(display_df["Arrival_Date"]).dt.strftime("%d %b %Y")
    display_df = display_df.rename(columns={
        "Arrival_Date": "Latest Arrival Date",
        "Min_Price": "Min Price (₹)",
        "Modal_Price": "Modal Price (₹)",
        "Max_Price": "Max Price (₹)"
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Download CSV
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Market Table as CSV",
        data=csv_bytes,
        file_name="agroprice_filtered_market_data.csv",
        mime="text/csv"
    )
else:
    st.warning("No records found for the selected combination.")

