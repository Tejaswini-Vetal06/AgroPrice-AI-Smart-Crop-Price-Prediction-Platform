"""
AgroPrice AI — About, Documentation, and Help Page.
Comprehensive educational guide for farmers, students, and agricultural businesses.
"""

import streamlit as st

from src.config import (
    apply_custom_theme,
    COLOR_FOREST_GREEN,
    COLOR_FRESH_GREEN,
    COLOR_SAGE_GREEN,
    COLOR_SOFT_MINT,
    COLOR_WHITE
)


st.set_page_config(
    page_title="About & Help — AgroPrice AI",
    page_icon="📖",
    layout="wide"
)

apply_custom_theme()

st.markdown("## 📖 About AgroPrice AI & Help Center")
st.markdown("Everything you need to know about crop price forecasting, machine learning fundamentals, data origins, and agricultural decision-making.")

# Introduction Cards
col_a1, col_a2 = st.columns(2)

with col_a1:
    st.markdown(
        f"""
        <div class="agro-card" style="height: 100%;">
            <h3 style="color: {COLOR_FOREST_GREEN} !important; margin-top: 0;">🌱 What is Crop Price Prediction?</h3>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
                Crop price prediction is the use of statistical and machine learning algorithms to estimate future agricultural market (mandi) rates based on historical seasonal trends, calendar cycles, regional trading spreads, and arrival volume dynamics.
            </p>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
                In India, agricultural commodity prices fluctuate significantly between harvest peaks (glut supply) and off-season storage depletion (lean supply). AgroPrice AI maps these recurring cycles to provide benchmark estimates in <b>₹ per quintal (100 kg)</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_a2:
    st.markdown(
        f"""
        <div class="agro-card" style="height: 100%;">
            <h3 style="color: {COLOR_FOREST_GREEN} !important; margin-top: 0;">👥 Who is AgroPrice AI For?</h3>
            <ul style="color: #475569; font-size: 0.92rem; line-height: 1.7; padding-left: 1.2rem;">
                <li><b>Farmers:</b> Evaluate whether to sell produce immediately upon harvest or store in warehouses/cold-storage for future price recovery.</li>
                <li><b>Agri-Traders & Mandi Commission Agents:</b> Benchmark inter-state and inter-district price differentials to optimize procurement logistics.</li>
                <li><b>Agricultural Students & Researchers:</b> Study genuine time-aware regression pipelines, seasonal crop volatility, and model diagnostics.</li>
                <li><b>Agri-Businesses & Food Processors:</b> Plan procurement budgets and understand commodity price ceilings.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Step by Step Guide
st.markdown("### 📋 Required Inputs for Price Prediction")
st.markdown(
    f"""
    <div class="agro-card">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem;">
            <div>
                <b style="color: {COLOR_FOREST_GREEN};">1. Commodity Name</b>
                <p style="color: #64748B; font-size: 0.88rem; margin: 0.2rem 0 0 0;">
                    Select the target crop (Wheat, Rice, Onion, Tomato, Potato, Soybean, Cotton, Maize, Turmeric, Gram).
                </p>
            </div>
            <div>
                <b style="color: {COLOR_FOREST_GREEN};">2. State & Mandi Location</b>
                <p style="color: #64748B; font-size: 0.88rem; margin: 0.2rem 0 0 0;">
                    Pick the APMC trading market where you plan to auction your harvest.
                </p>
            </div>
            <div>
                <b style="color: {COLOR_FOREST_GREEN};">3. Selling Month & Year</b>
                <p style="color: #64748B; font-size: 0.88rem; margin: 0.2rem 0 0 0;">
                    The calendar month when produce will arrive at the market center.
                </p>
            </div>
            <div>
                <b style="color: {COLOR_FOREST_GREEN};">4. Variety & Grade</b>
                <p style="color: #64748B; font-size: 0.88rem; margin: 0.2rem 0 0 0;">
                    Grade classification (e.g. Lokwan for wheat, Red for onion, Yellow for soybean).
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Machine Learning Education
st.markdown("### 🧠 How the Machine Learning Model Learns")
st.markdown(
    f"""
    <div class="agro-card">
        <ol style="color: #475569; font-size: 0.92rem; line-height: 1.7; padding-left: 1.2rem; margin: 0;">
            <li><b>Chronological Agmarknet Data Ingestion:</b> The system ingests multi-year historical arrival and price records from India's official agricultural marketing portals.</li>
            <li><b>Cyclical Feature Engineering:</b> Calendar months are transformed into cyclical trigonometric features (<code>sin_month</code> and <code>cos_month</code>) so the model recognizes that December (month 12) connects smoothly to January (month 1).</li>
            <li><b>Agricultural Season Categorization:</b> Every record is tagged with its agricultural harvest cycle (Kharif, Rabi, Zaid).</li>
            <li><b>Leakage-Free Temporal Validation:</b> Instead of random shuffling, older records (2021-2024) train the model, while an out-of-time holdout (2025) verifies how accurately it forecasts unseen future prices.</li>
            <li><b>Empirical Prediction Intervals:</b> Rather than making overconfident single-number predictions, the model calculates residual standard deviations to present an honest 90% expected price band.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Frequently Asked Questions
st.markdown("### ❓ Frequently Asked Questions (FAQ)")

with st.expander("Q1: Are the predicted crop prices guaranteed?"):
    st.write(
        "**No.** Predicted prices are statistical estimates based on historical market trends and seasonality. "
        "Real-time mandi prices depend on instantaneous factors like daily truck arrival counts, local weather disruptions, "
        "produce moisture content, government export/import duties, and trader bidding behavior."
    )

with st.expander("Q2: Why does the prediction display a price range instead of a single number?"):
    st.write(
        "A single price point gives a false sense of certainty. In real agricultural markets, prices fluctuate across a spread. "
        "Our 90% confidence band is empirically calculated from the model's test residual errors, showing farmers the realistic "
        "range within which the modal price is expected to settle."
    )

with st.expander("Q3: Can I upload data from my local mandi?"):
    st.write(
        "Yes! In the **Market Price Dashboard** page, expand the 'Upload Custom Agricultural Price CSV Callset' box. "
        "You can upload any CSV containing standard Agmarknet columns (`Arrival_Date`, `State`, `District`, `Market`, `Commodity`, `Modal_Price`). "
        "The system will automatically validate the schema, clean the records, and update the dashboard visuals."
    )

with st.expander("Q4: Why does a higher mandi price in another state not guarantee higher profits?"):
    st.write(
        "Selling produce in distant mandis involves truck freight, diesel costs, toll gates, loading/unloading labor, "
        "APMC cess/market fees (1-2%), and weight loss during transit. Always subtract these logistics expenses before "
        "traveling to another market."
    )

with st.expander("Q5: What unit of measurement is used?"):
    st.write(
        "All prices in AgroPrice AI follow standard Indian mandi conventions: **Indian Rupees (₹) per quintal**. "
        "One quintal equals 100 kilograms (100 kg)."
    )

# Mandatory Disclaimer Notice
st.markdown(
    f"""
    <div style="background-color: #FEF3C7; border: 1px solid #FDE68A; border-left: 5px solid #D97706; padding: 1.25rem; border-radius: 8px; margin-top: 1.5rem;">
        <h4 style="color: #92400E !important; margin: 0 0 0.4rem 0; font-size: 1rem;">
            ⚠️ Legal & Financial Disclaimer
        </h4>
        <p style="color: #78350F; font-size: 0.88rem; margin: 0; line-height: 1.6;">
            The forecasts and analytics provided by <b>AgroPrice AI</b> are for educational, informational, and analytical purposes only. 
            They do not constitute binding financial, trading, or agricultural advisory. Neither the developers nor the platform accept liability 
            for financial losses incurred as a result of commercial decisions based on these estimates. Always cross-verify with your local 
            Agricultural Produce Market Committee (APMC) and Krishi Vigyan Kendra (KVK).
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="agro-footer">
        <b>AgroPrice AI</b> &bull; Built with Python, Streamlit, scikit-learn & Plotly &bull; Open Source for Indian Agriculture
    </div>
    """,
    unsafe_allow_html=True
)

