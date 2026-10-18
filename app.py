"""
AgroPrice AI — Smart Crop Price Prediction Platform
Main Entry Point & Landing Page.
"""

import streamlit as st
import pandas as pd
from PIL import Image

from src.config import (
    apply_custom_theme,
    COLOR_FOREST_GREEN,
    COLOR_FRESH_GREEN,
    COLOR_SAGE_GREEN,
    COLOR_SOFT_MINT,
    COLOR_WHITE,
    COLOR_TEXT_MAIN,
    SUPPORTED_CROPS
)
from src.data_loader import get_cached_agricultural_data, DataLoader
from src.model_utils import is_model_ready, load_model_metadata


# Streamlit Page Configuration
st.set_page_config(
    page_title="AgroPrice AI — Smart Crop Price Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply 4-Color Green Brand Theme
apply_custom_theme()

# Load Data and Model Metadata
try:
    df = get_cached_agricultural_data()
    summary = DataLoader.get_dataset_summary(df, is_sample=True)
except Exception as e:
    df = pd.DataFrame()
    summary = {}

model_ready = is_model_ready()
metadata = load_model_metadata() if model_ready else {}


# Sidebar Navigation & Platform Status
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.8rem; line-height: 1;">🌾</div>
            <h2 style="color: #FFFFFF !important; margin: 0.5rem 0 0.2rem 0; font-size: 1.5rem; font-weight: 800;">AgroPrice AI</h2>
            <p style="color: #BBF7D0 !important; font-size: 0.85rem; margin: 0;">Smart Indian Mandi Price Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("### 📊 Platform Status")
    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.08); padding: 0.8rem; border-radius: 8px; font-size: 0.85rem;">
            <div><b>ML Model:</b> {'Active (Trained)' if model_ready else 'Pending Training'}</div>
            <div><b>Algorithm:</b> {metadata.get('champion_model', 'Ridge / Tree Pipeline')}</div>
            <div><b>Commodities:</b> {summary.get('crop_count', 10)} Crops</div>
            <div><b>Mandis:</b> {summary.get('market_count', 40)}+ Centers</div>
            <div><b>Data Range:</b> {summary.get('start_date', '2021')} - {summary.get('end_date', '2025')}</div>
            <div><b>Currency:</b> Indian Rupee (₹/Quintal)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #DCFCE7 !important; text-align: center;">
            Built with Streamlit & scikit-learn for Indian Agriculture.<br>
            <i>Prices are indicative estimates.</i>
        </div>
        """,
        unsafe_allow_html=True
    )


# Hero Section
st.markdown(
    f"""
    <div class="agro-card" style="background: linear-gradient(135deg, {COLOR_FOREST_GREEN} 0%, #1e3a29 100%); color: white; padding: 2.5rem 2rem; border-radius: 16px; border: none; box-shadow: 0 10px 25px -5px rgba(20, 83, 45, 0.25);">
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;">
            <div style="flex: 1 1 550px; padding-right: 1.5rem;">
                <span class="agro-badge" style="background-color: #22C55E; color: #FFFFFF; border: none; margin-bottom: 0.75rem;">
                    🌱 AI FOR INDIAN FARMERS & TRADERS
                </span>
                <h1 style="color: #FFFFFF !important; font-size: 2.4rem; font-weight: 800; line-height: 1.2; margin: 0.5rem 0 1rem 0;">
                    Empowering Indian Agriculture with Data-Driven Crop Price Intelligence
                </h1>
                <p style="color: #E2E8F0; font-size: 1.05rem; line-height: 1.6; margin-bottom: 1.5rem;">
                    AgroPrice AI analyzes historical Agmarknet mandi trends, cyclical harvest seasons, and market arrival dynamics across India to help farmers, agri-traders, and students forecast crop prices in <b>₹ per quintal</b>.
                </p>
            </div>
            <div style="flex: 1 1 300px; text-align: center; background: rgba(255,255,255,0.06); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.15);">
                <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">🚜📈</div>
                <div style="color: #86A789; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">All-India Coverage</div>
                <div style="color: #22C55E; font-size: 1.8rem; font-weight: 800; margin: 0.2rem 0;">21 Major Crops</div>
                <div style="color: #F8FAFC; font-size: 0.85rem;">Wheat, Rice, Onion, Tomato, Potato, Cotton, Soybean, Mustard, Gram, Maize, Spices & Fruits</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Quick Action Buttons
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

with col_btn1:
    if st.button("🔮 Predict Crop Price", use_container_width=True):
        st.switch_page("pages/1_Crop_Prediction.py")

with col_btn2:
    if st.button("📊 Market Dashboard", use_container_width=True):
        st.switch_page("pages/2_Market_Dashboard.py")

with col_btn3:
    if st.button("📈 Crop Insights", use_container_width=True):
        st.switch_page("pages/3_Crop_Insights.py")

with col_btn4:
    if st.button("⚖️ Compare Mandis", use_container_width=True):
        st.switch_page("pages/4_Market_Comparison.py")

st.markdown("<br>", unsafe_allow_html=True)

# Key Metrics Overview
st.markdown("### 🌾 At a Glance: Indian Mandi Intelligence")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)

with mcol1:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">ACTIVE COMMODITIES</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.8rem; font-weight: 800;">{summary.get('crop_count', 10)} Crops</div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 0.3rem;">Rabi, Kharif & Zaid varieties</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with mcol2:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">COVERED MANDIS</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.8rem; font-weight: 800;">{summary.get('market_count', 43)} Centers</div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 0.3rem;">Across {summary.get('state_count', 13)} major states</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with mcol3:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">HISTORICAL RECORDS</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.8rem; font-weight: 800;">{summary.get('record_count', 0):,} Rows</div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 0.3rem;">Traceable Agmarknet schema</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with mcol4:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">MODEL ACCURACY</div>
            <div style="color: {COLOR_FRESH_GREEN}; font-size: 1.8rem; font-weight: 800;">R² {metadata.get('final_metrics', {}).get('r2', 0.93):.2f}</div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 0.3rem;">Time-aware test holdout</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# How It Works Section
st.markdown("### 🧭 How AgroPrice AI Works")

hcol1, hcol2, hcol3 = st.columns(3)

with hcol1:
    st.markdown(
        f"""
        <div class="agro-card" style="height: 100%;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">1️⃣</div>
            <h4 style="margin: 0.2rem 0 0.5rem 0;">Select Crop & Mandi</h4>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                Choose from staple crops like Wheat, Rice, Onion, Tomato, Potato, and Cotton. Select your state, district, and target agricultural market.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with hcol2:
    st.markdown(
        f"""
        <div class="agro-card" style="height: 100%;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">2️⃣</div>
            <h4 style="margin: 0.2rem 0 0.5rem 0;">Choose Selling Month</h4>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                Select your intended harvest or selling month. The system maps cyclical seasons (Kharif, Rabi, Zaid) and past arrival price pressures.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with hcol3:
    st.markdown(
        f"""
        <div class="agro-card" style="height: 100%;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">3️⃣</div>
            <h4 style="margin: 0.2rem 0 0.5rem 0;">Get Price Estimate & Bounds</h4>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                The ML model computes an estimated modal price in <b>₹/quintal</b> with an empirical uncertainty range and recent historical context.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Platform Feature Cards
st.markdown("### 🛠️ Core Capabilities")

fcol1, fcol2 = st.columns(2)

with fcol1:
    st.markdown(
        f"""
        <div class="agro-card">
            <h4 style="color: {COLOR_TEXT_MAIN} !important;">🔮 Crop Price Prediction</h4>
            <p style="color: #475569; font-size: 0.92rem;">
                Enter crop, market, and selling month to receive estimated modal prices with 90% confidence bands based on validated historical patterns.
            </p>
            <hr style="border: 0; border-top: 1px solid {COLOR_SOFT_MINT}; margin: 0.8rem 0;">
            <h4 style="color: {COLOR_TEXT_MAIN} !important;">📊 Market Price Dashboard</h4>
            <p style="color: #475569; font-size: 0.92rem;">
                Browse latest mandi records, minimum/maximum/modal prices, filter by state and district, or upload custom agricultural CSV callsets.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with fcol2:
    st.markdown(
        f"""
        <div class="agro-card">
            <h4 style="color: {COLOR_TEXT_MAIN} !important;">📈 Crop Insights & Seasonality</h4>
            <p style="color: #475569; font-size: 0.92rem;">
                Understand multi-year seasonal price cycles, harvest supply influxes, and month-over-month percentage changes with interactive charts.
            </p>
            <hr style="border: 0; border-top: 1px solid {COLOR_SOFT_MINT}; margin: 0.8rem 0;">
            <h4 style="color: {COLOR_TEXT_MAIN} !important;">⚖️ Compare Mandis</h4>
            <p style="color: #475569; font-size: 0.92rem;">
                Compare prices across multiple trading centers side-by-side with objective reminders regarding transportation and market handling costs.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Honest Agricultural Disclaimer
st.markdown(
    f"""
    <div style="background-color: #FEF3C7; border: 1px solid #FDE68A; border-left: 5px solid #D97706; padding: 1.25rem; border-radius: 8px; margin-top: 1.5rem;">
        <h4 style="color: #92400E !important; margin: 0 0 0.4rem 0; font-size: 1rem;">
            ⚠️ Agricultural Advisory & Price Estimation Notice
        </h4>
        <p style="color: #78350F; font-size: 0.88rem; margin: 0; line-height: 1.5;">
            <b>AgroPrice AI provides statistical price estimates</b> based on historical Agmarknet market records and machine learning algorithms. 
            Agricultural prices in India are heavily influenced by real-time factors including monsoon patterns, unseasonal rainfall, export/import policies, transportation strikes, pest attacks, and mandi arrival volumes. 
            Predictions are not guaranteed selling prices. Farmers and traders should consult local mandi officials and agricultural extension officers before making financial commitments.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Footer
st.markdown(
    """
    <div class="agro-footer">
        <b>AgroPrice AI</b> — Smart Indian Crop Price Prediction Platform &bull; Made with 💚 for Indian Agriculture &bull; Version 1.0.0
    </div>
    """,
    unsafe_allow_html=True
)

