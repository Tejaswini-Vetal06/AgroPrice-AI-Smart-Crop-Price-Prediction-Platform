"""
AgroPrice AI — Model Insights & Machine Learning Pipeline Page.
Student-friendly explanation of time-aware validation, model candidate benchmarking,
error metrics, actual vs. predicted charts, and feature importance.
"""

import streamlit as st
import pandas as pd
import numpy as np
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
from src.model_utils import is_model_ready, load_model_metadata, load_trained_model
from src.data_loader import get_cached_agricultural_data
from src.feature_engineering import prepare_training_data
from src.train import split_time_aware


st.set_page_config(
    page_title="Model Insights — AgroPrice AI",
    page_icon="🤖",
    layout="wide"
)

apply_custom_theme()

st.markdown("## 🤖 Machine Learning Pipeline & Model Insights")
st.markdown("An educational, transparent breakdown of the agricultural price prediction architecture, candidate benchmarks, time-aware validation, and residual error calibration.")

if not is_model_ready():
    st.error("⚠️ Model metadata not found. Please execute the training pipeline first.")
    st.stop()

metadata = load_model_metadata()
champion_name = metadata.get("champion_model", "Selected Model")
final_metrics = metadata.get("final_metrics", {})
candidate_models = metadata.get("candidate_models", {})
train_period = metadata.get("training_period", {})
holdout_period = metadata.get("holdout_period", {})
crop_breakdown = metadata.get("crop_breakdown", {})
pred_interval = metadata.get("prediction_interval", {})
top_features = metadata.get("top_features", {})

# Top Metrics Highlights
st.markdown("### 🎯 Champion Model Evaluation on Out-of-Time Holdout")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">CHAMPION ARCHITECTURE</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.5rem; font-weight: 800;">{champion_name}</div>
            <div style="color: #64748B; font-size: 0.8rem;">Selected by lowest test MAE</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_m2:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">MEAN ABSOLUTE ERROR (MAE)</div>
            <div style="color: {COLOR_FRESH_GREEN}; font-size: 1.5rem; font-weight: 800;">₹ {final_metrics.get('mae', 0):,.2f}</div>
            <div style="color: #64748B; font-size: 0.8rem;">Avg error per quintal</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_m3:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">ROOT MEAN SQUARED ERROR</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.5rem; font-weight: 800;">₹ {final_metrics.get('rmse', 0):,.2f}</div>
            <div style="color: #64748B; font-size: 0.8rem;">Penalizes large deviations</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_m4:
    st.markdown(
        f"""
        <div class="agro-metric-box">
            <div style="color: {COLOR_SAGE_GREEN}; font-size: 0.85rem; font-weight: 600;">COEFFICIENT OF DETERMINATION (R²)</div>
            <div style="color: {COLOR_FOREST_GREEN}; font-size: 1.5rem; font-weight: 800;">{final_metrics.get('r2', 0):.4f}</div>
            <div style="color: #64748B; font-size: 0.8rem;">Explained variance (0 to 1)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Candidate Models Benchmark Table
st.markdown("### 🏆 Candidate Models Benchmark")
st.markdown("We trained and compared 4 distinct machine learning algorithms using strict chronological validation to avoid future price leakage:")

benchmark_rows = []
for model_key, scores in candidate_models.items():
    is_champ = (model_key == champion_name)
    benchmark_rows.append({
        "Model Algorithm": f"⭐ {model_key} (Champion)" if is_champ else model_key,
        "CV MAE (₹)": f"₹ {scores.get('cv_mae', 0):,.2f}",
        "Holdout Test MAE (₹)": f"₹ {scores.get('test_mae', 0):,.2f}",
        "Holdout RMSE (₹)": f"₹ {scores.get('test_rmse', 0):,.2f}",
        "R² Score": f"{scores.get('test_r2', 0):.4f}",
        "Test MAPE (%)": f"{scores.get('test_mape', 0):.2f}%"
    })

bench_df = pd.DataFrame(benchmark_rows)
st.dataframe(bench_df, use_container_width=True, hide_index=True)

# Student-Friendly Metric Explanations
with st.expander("📚 Understanding These Machine Learning Metrics (Student Guide)", expanded=False):
    st.markdown(
        """
        - **MAE (Mean Absolute Error):** The average difference between predicted and actual mandi prices in rupees. An MAE of ₹580 means our estimates differ from the actual market price by approximately ₹580 per quintal on average.
        - **RMSE (Root Mean Squared Error):** Gives higher penalty to large forecasting errors. If RMSE is close to MAE, large outlier errors are rare.
        - **R² Score (Coefficient of Determination):** Measures how much of the market price volatility is successfully explained by our crop, location, seasonality, and calendar features. An R² of 0.93 means 93% of price variation is captured.
        - **Time-Aware Splitting (No Data Leakage):** In time-series agricultural forecasting, we cannot use random k-fold splits because training on future dates to predict past dates causes severe data leakage. Our pipeline trains exclusively on older years (2021-2024) and tests on unseen future periods (2025).
        """
    )

st.markdown("<br>", unsafe_allow_html=True)

# Actual vs Predicted Scatter Chart on Test Set
st.markdown("### 📈 Actual vs. Predicted Price Validation")
st.markdown("Evaluated on the out-of-time holdout test set (unseen 2025 data points):")

try:
    data_full = get_cached_agricultural_data()
    train_split, test_split = split_time_aware(data_full, test_ratio=0.20)
    pipeline = load_trained_model()

    X_test, y_test = prepare_training_data(test_split)
    preds = pipeline.predict(X_test)

    eval_scatter_df = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": preds,
        "Commodity": test_split["Commodity"].values
    })

    # Sample for snappy scatter plot rendering
    sample_scatter = eval_scatter_df.sample(min(400, len(eval_scatter_df)), random_state=42)

    fig_scatter = px.scatter(
        sample_scatter,
        x="Actual Price",
        y="Predicted Price",
        color="Commodity",
        labels={"Actual Price": "Actual Agmarknet Price (₹/Quintal)", "Predicted Price": "Model Prediction (₹/Quintal)"},
        color_discrete_sequence=[COLOR_FOREST_GREEN, COLOR_FRESH_GREEN, COLOR_SAGE_GREEN, "#34D399", "#059669", "#0284C7", "#D97706", "#7C3AED", "#EC4899", "#EAB308"]
    )

    # Add 45-degree ideal fit line
    min_val = min(sample_scatter["Actual Price"].min(), sample_scatter["Predicted Price"].min())
    max_val = max(sample_scatter["Actual Price"].max(), sample_scatter["Predicted Price"].max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="#94A3B8", dash="dash", width=2),
        name="Ideal Perfect Prediction (y = x)"
    ))

    fig_scatter.update_layout(get_plotly_layout("Holdout Actual vs. Predicted Prices (45° Line = Perfect Accuracy)", height=450))
    st.plotly_chart(fig_scatter, use_container_width=True)

except Exception as e:
    st.info(f"Scatter plot generation notice: {e}")

# Performance Breakdown by Commodity
st.markdown("### 🌾 Crop-by-Crop Performance Breakdown")
st.markdown("Because commodity price bases differ significantly (e.g. Potato ~₹1,400 vs. Turmeric ~₹9,500), absolute errors vary naturally by crop:")

crop_rows = []
for crop, cmetrics in crop_breakdown.items():
    crop_rows.append({
        "Commodity": crop,
        "Test Sample Count": cmetrics.get("count", 0),
        "Mean Market Price": f"₹ {cmetrics.get('mean_price', 0):,.2f}",
        "Holdout MAE": f"₹ {cmetrics.get('mae', 0):,.2f}",
        "Holdout RMSE": f"₹ {cmetrics.get('rmse', 0):,.2f}",
        "R² Score": f"{cmetrics.get('r2', 0):.4f}",
        "MAPE (%)": f"{cmetrics.get('mape', 0):.2f}%"
    })

crop_perf_df = pd.DataFrame(crop_rows).sort_values(by="MAPE (%)")
st.dataframe(crop_perf_df, use_container_width=True, hide_index=True)

# Top Feature Importance
if top_features:
    st.markdown("### 🔑 Relative Feature Importance")
    st.markdown("The primary variables and categorical weights driving price variations:")

    feat_df = pd.DataFrame({
        "Feature": list(top_features.keys()),
        "Relative Importance": list(top_features.values())
    }).sort_values(by="Relative Importance", ascending=True)

    fig_feat = px.bar(
        feat_df,
        x="Relative Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=[COLOR_FOREST_GREEN]
    )
    fig_feat.update_layout(get_plotly_layout("Top Model Predictors", height=380))
    st.plotly_chart(fig_feat, use_container_width=True)
