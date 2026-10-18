"""
AgroPrice AI — Prediction History Page.
Displays locally persisted predictions from SQLite, provides search and filter capabilities,
and supports record deletion and clearing with confirmation.
"""

import streamlit as st
import pandas as pd

from src.config import (
    apply_custom_theme,
    COLOR_FOREST_GREEN,
    COLOR_FRESH_GREEN,
    COLOR_SAGE_GREEN,
    COLOR_SOFT_MINT,
    COLOR_WHITE,
    COLOR_DANGER
)
from src.database import (
    get_predictions,
    delete_prediction,
    clear_all_predictions,
    get_prediction_count
)


st.set_page_config(
    page_title="Prediction History — AgroPrice AI",
    page_icon="📜",
    layout="wide"
)

apply_custom_theme()

st.markdown("## 📜 Prediction History")
st.markdown("Browse, search, and manage previous crop price predictions stored securely in your local SQLite database.")

# Filters and Search
col_s1, col_s2 = st.columns([1, 2])

with col_s1:
    crop_filter = st.selectbox(
        "Filter by Commodity",
        options=["All", "Wheat", "Rice", "Onion", "Tomato", "Potato", "Soybean", "Cotton", "Maize", "Turmeric", "Gram"],
        index=0
    )

with col_s2:
    search_keyword = st.text_input("Search (Mandi, State, District, Crop)", placeholder="e.g. Khanna, Nashik...")

# Retrieve Predictions
pred_df = get_predictions(
    search_query=search_keyword if search_keyword else None,
    crop_filter=crop_filter,
    limit=200
)

total_count = get_prediction_count()

# Metric summary
m1, m2 = st.columns(2)
with m1:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">SAVED PREDICTIONS</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.8rem; font-weight: 800;">{total_count} Records</div>
            <div style="color: #64748B; font-size: 0.8rem;">Persisted locally in SQLite</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">ACTIVE FILTERS</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.8rem; font-weight: 800;">{len(pred_df)} Displayed</div>
            <div style="color: #64748B; font-size: 0.8rem;">Filtered search results</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Display Table or Empty State
if pred_df.empty:
    st.markdown(
        f"""
        <div class="agro-card" style="text-align: center; padding: 3rem 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">📂🌱</div>
            <h3 style="color: {COLOR_FOREST_GREEN} !important;">No Prediction Records Found</h3>
            <p style="color: #64748B; font-size: 0.95rem;">
                You haven't generated any predictions matching this query yet.<br>
                Visit the <b>Crop Price Prediction</b> page to calculate your first mandi forecast.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Format columns for display
    display_df = pred_df[[
        "id", "created_at", "crop", "market", "district", "state",
        "month", "year", "season", "variety", "predicted_price",
        "price_range_low", "price_range_high", "model_version"
    ]].copy()

    display_df["Predicted (₹/Qtl)"] = display_df["predicted_price"].apply(lambda x: f"₹ {x:,.2f}")
    display_df["Expected Range"] = display_df.apply(
        lambda r: f"₹ {r['price_range_low']:,.0f} - ₹ {r['price_range_high']:,.0f}" if pd.notna(r['price_range_low']) else "N/A",
        axis=1
    )
    display_df["Target Schedule"] = display_df.apply(lambda r: f"Month {r['month']} / {r['year']}", axis=1)

    table_df = display_df[[
        "id", "created_at", "crop", "market", "state", "Target Schedule",
        "variety", "Predicted (₹/Qtl)", "Expected Range", "model_version"
    ]].rename(columns={
        "id": "ID",
        "created_at": "Timestamp",
        "crop": "Crop",
        "market": "Mandi",
        "state": "State",
        "variety": "Variety",
        "model_version": "Model"
    })

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Actions: Individual Delete and Clear All
    st.markdown("### ⚙️ History Management")
    col_del1, col_del2 = st.columns([1, 1])

    with col_del1:
        st.markdown("#### Delete Single Record")
        record_id_to_delete = st.selectbox(
            "Select Record ID to Delete",
            options=table_df["ID"].tolist(),
            key="del_record_id"
        )
        if st.button("🗑️ Delete Selected Record"):
            success = delete_prediction(int(record_id_to_delete))
            if success:
                st.success(f"Record #{record_id_to_delete} deleted successfully.")
                st.rerun()

    with col_del2:
        st.markdown("#### Clear All History")
        st.warning("⚠️ This will permanently remove all prediction records from the local SQLite database.")
        confirm_clear = st.checkbox("Yes, I understand and want to delete all prediction history.", key="confirm_clear")
        if st.button("🚨 Clear All History", disabled=not confirm_clear):
            clear_all_predictions()
            st.success("All prediction history cleared.")
            st.rerun()

