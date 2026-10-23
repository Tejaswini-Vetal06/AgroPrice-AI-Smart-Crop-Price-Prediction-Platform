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
    COLOR_GOLD_ACCENT,
    COLOR_GOLD_LIGHT,
    SUPPORTED_CROPS,
    t,
    get_crop_name,
    render_language_selector
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
    # Language Selector Card
    render_language_selector()
    st.markdown("---")

    st.markdown(
        f"""
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <div style="font-size: 2.8rem; line-height: 1;">🌾</div>
            <h2 style="color: #FFFFFF !important; margin: 0.4rem 0 0.1rem 0; font-size: 1.5rem; font-weight: 800;">
                AgroPrice <span style="color: #FBBF24;">AI</span>
            </h2>
            <p style="color: #BBF7D0 !important; font-size: 0.82rem; margin: 0;">{t('app_tagline')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown(f"### 📊 {t('platform_status')}")
    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.08); padding: 0.8rem; border-radius: 8px; font-size: 0.85rem;">
            <div><b>{t('status_model')}:</b> {'Active (Trained)' if model_ready else 'Pending Training'}</div>
            <div><b>Algorithm:</b> {metadata.get('champion_model', 'Ridge / Tree Pipeline')}</div>
            <div><b>{t('status_crops')}:</b> {summary.get('crop_count', 10)} Crops</div>
            <div><b>{t('status_mandis')}:</b> {summary.get('market_count', 40)}+ Centers</div>
            <div><b>Data Range:</b> {summary.get('start_date', '2021')} - {summary.get('end_date', '2025')}</div>
            <div><b>{t('status_currency')}</b></div>
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
    <div class="agro-hero-banner">
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1.5rem;">
            <div style="flex: 1 1 520px; padding-right: 1rem;">
                <span class="agro-badge-gold">
                    🇮🇳 {t('hero_badge')}
                </span>
                <h1 style="color: #FFFFFF !important; font-size: 2.3rem; font-weight: 800; line-height: 1.25; margin: 0.75rem 0 0.6rem 0;">
                    🌾 AgroPrice <span style="color: #FBBF24;">AI</span> — {t('hero_title')}
                </h1>
                <p style="color: #ECFDF5; font-size: 1.05rem; line-height: 1.6; margin-bottom: 1.25rem;">
                    {t('hero_desc')}
                </p>
                <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
                    <span style="background: rgba(255,255,255,0.15); padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.82rem; color: #FFFFFF; font-weight: 600;">🌾 21 Major Crops</span>
                    <span style="background: rgba(255,255,255,0.15); padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.82rem; color: #FFFFFF; font-weight: 600;">🏪 40+ APMC Mandis</span>
                    <span style="background: rgba(255,255,255,0.15); padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.82rem; color: #FFFFFF; font-weight: 600;">🌐 9 Indian Languages</span>
                </div>
            </div>
            <div style="flex: 1 1 280px; text-align: center; background: rgba(255,255,255,0.12); backdrop-filter: blur(8px); padding: 1.4rem; border-radius: 14px; border: 1px solid rgba(255,255,255,0.22); box-shadow: 0 8px 20px -4px rgba(0,0,0,0.15);">
                <div style="font-size: 3.2rem; line-height: 1; margin-bottom: 0.4rem;">🚜📈</div>
                <div style="color: #FDE68A; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">All-India Coverage</div>
                <div style="color: #FFFFFF; font-size: 1.7rem; font-weight: 800; margin: 0.2rem 0;">21 Major Crops</div>
                <div style="color: #E2E8F0; font-size: 0.82rem; line-height: 1.4;">Wheat, Rice, Onion, Tomato, Potato, Cotton, Soybean, Mustard, Gram & More</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Quick Action Buttons
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

with col_btn1:
    if st.button(t("btn_predict"), use_container_width=True):
        st.switch_page("pages/1_Crop_Prediction.py")

with col_btn2:
    if st.button(t("btn_dashboard"), use_container_width=True):
        st.switch_page("pages/2_Market_Dashboard.py")

with col_btn3:
    if st.button(t("btn_insights"), use_container_width=True):
        st.switch_page("pages/3_Crop_Insights.py")

with col_btn4:
    if st.button(t("btn_compare"), use_container_width=True):
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
st.markdown(f"### {t('how_it_works_title')}")

hcol1, hcol2, hcol3 = st.columns(3)

with hcol1:
    st.markdown(
        f"""
        <div class="agro-card" style="height: 100%;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">1️⃣</div>
            <h4 style="margin: 0.2rem 0 0.5rem 0; color: {COLOR_TEXT_MAIN} !important;">{t('step1_title')}</h4>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                {t('step1_desc')}
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
            <h4 style="margin: 0.2rem 0 0.5rem 0; color: {COLOR_TEXT_MAIN} !important;">{t('step2_title')}</h4>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                {t('step2_desc')}
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
            <h4 style="margin: 0.2rem 0 0.5rem 0; color: {COLOR_TEXT_MAIN} !important;">{t('step3_title')}</h4>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                {t('step3_desc')}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Platform Feature Cards
st.markdown(f"### 🛠️ {t('app_name')} — Features")

fcol1, fcol2 = st.columns(2)

with fcol1:
    st.markdown(
        f"""
        <div class="agro-card">
            <h4 style="color: {COLOR_TEXT_MAIN} !important;">{t('btn_predict')}</h4>
            <p style="color: #475569; font-size: 0.92rem;">
                Enter crop, market, and selling month to receive estimated modal prices with 90% confidence bands based on validated historical patterns.
            </p>
            <hr style="border: 0; border-top: 1px solid {COLOR_SOFT_MINT}; margin: 0.8rem 0;">
            <h4 style="color: {COLOR_TEXT_MAIN} !important;">{t('btn_dashboard')}</h4>
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
            <h4 style="color: {COLOR_TEXT_MAIN} !important;">{t('btn_insights')}</h4>
            <p style="color: #475569; font-size: 0.92rem;">
                Understand multi-year seasonal price cycles, harvest supply influxes, and month-over-month percentage changes with interactive charts.
            </p>
            <hr style="border: 0; border-top: 1px solid {COLOR_SOFT_MINT}; margin: 0.8rem 0;">
            <h4 style="color: {COLOR_TEXT_MAIN} !important;">{t('btn_compare')}</h4>
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
            {t('disclaimer_title')}
        </h4>
        <p style="color: #78350F; font-size: 0.88rem; margin: 0; line-height: 1.5;">
            {t('disclaimer_desc')}
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

