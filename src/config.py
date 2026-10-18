"""
Configuration module for AgroPrice AI.
Defines theme colors, paths, crop metadata, database configurations, and styling helpers.
"""

from pathlib import Path
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
DEFAULT_DATASET_PATH = SAMPLE_DATA_DIR / "agmarknet_crop_prices_sample.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = DATA_DIR / "database.sqlite"

DEFAULT_MODEL_PATH = MODELS_DIR / "crop_price_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVALUATION_REPORT_PATH = REPORTS_DIR / "model_evaluation_summary.json"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 4-Color Green Brand Theme
# ---------------------------------------------------------
COLOR_FOREST_GREEN = "#14532D"   # Sidebar, primary headings, nav, important sections
COLOR_FRESH_GREEN = "#22C55E"    # Primary buttons, active nav, important actions
COLOR_SAGE_GREEN = "#86A789"     # Secondary elements, chart highlights, supporting info
COLOR_SOFT_MINT = "#F0FDF4"      # Page backgrounds, dashboard sections, light surfaces

# Supporting neutrals (No extra brand accent colors)
COLOR_WHITE = "#FFFFFF"
COLOR_TEXT_MAIN = "#0F172A"
COLOR_TEXT_MUTED = "#475569"
COLOR_BORDER = "#DCFCE7"
COLOR_BORDER_SUBTLE = "#E2E8F0"
COLOR_DANGER = "#EF4444"

# ---------------------------------------------------------
# Domain Constants: All-India Agricultural Market
# ---------------------------------------------------------
SUPPORTED_CROPS = [
    "Wheat",
    "Rice",
    "Onion",
    "Tomato",
    "Potato",
    "Soybean",
    "Cotton",
    "Maize",
    "Turmeric",
    "Gram",
    "Mustard",
    "Groundnut",
    "Bajra",
    "Jowar",
    "Moong",
    "Arhar",
    "Chilli",
    "Garlic",
    "Ginger",
    "Apple",
    "Banana"
]

CROP_HINDI_NAMES = {
    "Wheat": "गेहूं (Wheat)",
    "Rice": "धान / चावल (Rice)",
    "Onion": "प्याज (Onion)",
    "Tomato": "टमाटर (Tomato)",
    "Potato": "आलू (Potato)",
    "Soybean": "सोयाबीन (Soybean)",
    "Cotton": "कपास (Cotton)",
    "Maize": "मक्का (Maize)",
    "Turmeric": "हल्दी (Turmeric)",
    "Gram": "चना (Gram)",
    "Mustard": "सरसों / राई (Mustard)",
    "Groundnut": "मूंगफली (Groundnut)",
    "Bajra": "बाजरा (Bajra)",
    "Jowar": "ज्वार (Jowar)",
    "Moong": "मूंग (Moong)",
    "Arhar": "अरहर / तुअर (Arhar)",
    "Chilli": "लाल मिर्च (Chilli)",
    "Garlic": "लहसुन (Garlic)",
    "Ginger": "अदरक (Ginger)",
    "Apple": "सेब (Apple)",
    "Banana": "केला (Banana)"
}

# Official Government of India Minimum Support Prices (MSP) in ₹ / Quintal
GOVT_MSP_BENCHMARKS = {
    "Wheat": 2425,
    "Rice": 2300,
    "Gram": 5650,
    "Mustard": 5950,
    "Soybean": 4892,
    "Cotton": 7121,
    "Maize": 2225,
    "Groundnut": 6783,
    "Bajra": 2625,
    "Jowar": 3371,
    "Moong": 8682,
    "Arhar": 7550
}

CROP_SEASONS = {
    "Wheat": "Rabi",
    "Rice": "Kharif",
    "Onion": "All Season",
    "Tomato": "All Season",
    "Potato": "Rabi",
    "Soybean": "Kharif",
    "Cotton": "Kharif",
    "Maize": "Kharif",
    "Turmeric": "Kharif",
    "Gram": "Rabi",
    "Mustard": "Rabi",
    "Groundnut": "Kharif",
    "Bajra": "Kharif",
    "Jowar": "Kharif",
    "Moong": "Kharif",
    "Arhar": "Kharif",
    "Chilli": "All Season",
    "Garlic": "Rabi",
    "Ginger": "Kharif",
    "Apple": "Kharif",
    "Banana": "All Season"
}

# Average warehouse & cold-storage holding cost per quintal per month (₹)
STORAGE_COST_ESTIMATES = {
    "Potato": 60,
    "Apple": 90,
    "Onion": 45,
    "Garlic": 40,
    "default": 30  # Standard grain / pulse / oilseed godown rent + handling
}

SEASONS = ["Kharif", "Rabi", "Zaid", "All Season"]

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Standard Agmarknet schema
STANDARD_COLUMNS = [
    "Arrival_Date",
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Min_Price",
    "Max_Price",
    "Modal_Price"
]

NUMERIC_PRICE_COLUMNS = ["Min_Price", "Max_Price", "Modal_Price"]

# ---------------------------------------------------------
# UI Theming and Custom CSS
# ---------------------------------------------------------
def apply_custom_theme():
    """Injects high-quality modern CSS adhering strictly to the 4-color green theme."""
    custom_css = f"""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {COLOR_TEXT_MAIN};
    }}

    .stApp {{
        background-color: {COLOR_SOFT_MINT};
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    /* Sidebar Styling (Dark background -> Crisp white opposite text) */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_FOREST_GREEN} !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    section[data-testid="stSidebar"] * {{
        color: #F8FAFC !important;
    }}
    
    section[data-testid="stSidebar"] a {{
        color: #F8FAFC !important;
        font-weight: 500;
    }}
    
    section[data-testid="stSidebar"] .st-emotion-cache-17lntkn {{
        color: #FFFFFF !important;
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {{
        color: #FFFFFF !important;
        font-weight: 700;
    }}

    /* Main Canvas Headings (Light mint background -> Deep dark slate opposite text) */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLOR_TEXT_MAIN} !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}
    
    h1 {{
        margin-bottom: 0.5rem;
    }}

    /* Dark banners & gradient cards (Dark background -> Crisp white opposite text) */
    .agro-card-dark,
    .agro-card-dark h1,
    .agro-card-dark h2,
    .agro-card-dark h3,
    .agro-card-dark h4,
    [style*="linear-gradient"] h1,
    [style*="linear-gradient"] h2,
    [style*="linear-gradient"] h3,
    [style*="linear-gradient"] h4 {{
        color: #FFFFFF !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {COLOR_FRESH_GREEN} !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 6px -1px rgba(34, 197, 94, 0.25) !important;
    }}

    .stButton > button:hover {{
        background-color: #16A34A !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 12px -2px rgba(34, 197, 94, 0.35) !important;
    }}

    .stButton > button:active {{
        transform: translateY(0px) !important;
    }}

    /* Cards & Containers */
    .agro-card {{
        background-color: {COLOR_WHITE};
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid {COLOR_BORDER};
        box-shadow: 0 4px 12px -2px rgba(20, 83, 45, 0.05);
        margin-bottom: 1.25rem;
    }}

    .agro-card-subtle {{
        background-color: #FAFCF9;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid {COLOR_BORDER_SUBTLE};
        margin-bottom: 1rem;
    }}

    .agro-metric-box {{
        background-color: {COLOR_WHITE};
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        border-left: 4px solid {COLOR_FRESH_GREEN};
        border-top: 1px solid {COLOR_BORDER};
        border-right: 1px solid {COLOR_BORDER};
        border-bottom: 1px solid {COLOR_BORDER};
        box-shadow: 0 2px 8px -2px rgba(0,0,0,0.04);
    }}

    /* Badge Tags */
    .agro-badge {{
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #DCFCE7;
        color: {COLOR_FOREST_GREEN};
        border: 1px solid #BBF7D0;
    }}

    .agro-badge-sage {{
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #F1F6F2;
        color: {COLOR_FOREST_GREEN};
        border: 1px solid {COLOR_SAGE_GREEN};
    }}

    /* Streamlit Metrics */
    div[data-testid="stMetricValue"] {{
        color: {COLOR_FOREST_GREEN} !important;
        font-weight: 700 !important;
    }}

    /* Inputs, Selectboxes, and Sliders */
    .stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label {{
        color: {COLOR_FOREST_GREEN} !important;
        font-weight: 600 !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: {COLOR_WHITE} !important;
        border-radius: 8px !important;
        border-color: {COLOR_BORDER_SUBTLE} !important;
    }}

    div[data-baseweb="input"] > div {{
        background-color: {COLOR_WHITE} !important;
        border-radius: 8px !important;
    }}

    /* Callout & Alerts */
    .stAlert {{
        border-radius: 8px !important;
    }}
    
    /* Footer & Watermark */
    .agro-footer {{
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid {COLOR_BORDER};
        text-align: center;
        color: {COLOR_TEXT_MUTED};
        font-size: 0.875rem;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def get_plotly_layout(title="", height=400):
    """Returns a standardized Plotly layout following the 4-color green theme."""
    return go.Layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=15, color=COLOR_FOREST_GREEN)
        ),
        paper_bgcolor=COLOR_WHITE,
        plot_bgcolor=COLOR_WHITE,
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color=COLOR_TEXT_MAIN),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor=COLOR_BORDER_SUBTLE,
            tickfont=dict(color=COLOR_TEXT_MUTED)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor=COLOR_BORDER_SUBTLE,
            tickfont=dict(color=COLOR_TEXT_MUTED)
        ),
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

