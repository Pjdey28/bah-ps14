import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import glob
from utils import load_css

# --- Page Configuration ---
st.title("Model Predictions Explorer")
CSS_FILE = Path(__file__).parent.parent / "style.css"
load_css(CSS_FILE)

# --- Data Loading ---
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"
EVAL_DIR = ARTIFACTS / "evaluation"

# Find all available model prediction samples
sample_files = glob.glob(f"{EVAL_DIR}/*_sample.csv")
model_names = [Path(f).stem.replace("_sample", "") for f in sample_files]

if not model_names:
    st.error(
        "No prediction sample files found in 'artifacts/evaluation/'. "
        "Please run the evaluation pipeline to generate artifacts."
    )
    st.stop()

# --- Sidebar for Selections ---
st.sidebar.header("Select Model")
selected_model = st.sidebar.selectbox("Model", model_names, index=model_names.index("Ensemble") if "Ensemble" in model_names else 0)

# Load the selected model's data
file_path = EVAL_DIR / f"{selected_model}_sample.csv"
df = pd.read_csv(file_path)

st.markdown(f"### Showing Predictions for: **{selected_model}**")
st.markdown("This page allows you to explore and compare the actual vs. predicted values from the selected model's sample output.")

st.sidebar.divider()
st.sidebar.header("Select Horizon")
horizon = st.sidebar.selectbox("Prediction Horizon", ["30 Minutes", "6 Hours", "12 Hours"])

horizon_map = {
    "30 Minutes": {"pred": "Pred30", "actual": "Actual30"},
    "6 Hours": {"pred": "Pred6", "actual": "Actual6"},
    "12 Hours": {"pred": "Pred12", "actual": "Actual12"},
}
pred_col = horizon_map[horizon]["pred"]
actual_col = horizon_map[horizon]["actual"]

st.divider()

# --- Single Horizon Plot ---
st.subheader(f"Analysis for {horizon}")

c1, c2, c3 = st.columns(3, gap="large")
c1.metric("Maximum Prediction", f"{df[pred_col].max():.3f}")
c2.metric("Average Prediction", f"{df[pred_col].mean():.3f}")
c3.metric("Minimum Prediction", f"{df[pred_col].min():.3f}")

# Chart
fig = px.line(
    df,
    y=[actual_col, pred_col],
    title=f"Actual vs. Predicted - {horizon}",
    labels={"value": "Value", "variable": "Type", "index": "Sample Index"},
    height=500
)
fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Error Analysis ---
st.subheader(f"Error Analysis for {horizon}")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### Actual vs. Predicted Scatter Plot")
    fig_scatter = px.scatter(
        df,
        x=actual_col,
        y=pred_col,
        title="Actual vs. Predicted",
        labels={actual_col: "Actual Value", pred_col: "Predicted Value"},
        trendline="ols",
        trendline_color_override="red"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.markdown("#### Residuals (Predicted - Actual) Histogram")
    residuals = df[pred_col] - df[actual_col]
    fig_hist = px.histogram(
        residuals,
        nbins=50,
        title="Distribution of Residuals"
    )
    fig_hist.update_layout(showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# --- All Horizons Plot ---
st.subheader("Comparison Across All Horizons")
st.markdown("Visualizing actuals and predictions for all forecast horizons simultaneously.")

# Restructure data for Plotly
plot_df = df[['Actual30', 'Pred30', 'Actual6', 'Pred6', 'Actual12', 'Pred12']].head(100) # Subset for clarity

fig2 = px.line(
    plot_df,
    y=plot_df.columns,
    title="All Horizons: Actuals vs. Predictions",
    labels={"value": "Value", "variable": "Trace", "index": "Sample Index"},
    height=600,
)
fig2.update_layout(legend_title="Trace")
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Data Table and Download ---
st.subheader("Raw Prediction Data")
st.dataframe(df, use_container_width=True, height=450)

csv = df.to_csv(index=False).encode()
st.download_button(
    "Download Predictions as CSV",
    csv,
    f"{selected_model}_predictions.csv",
    "text/csv",
    key='download-csv'
)
