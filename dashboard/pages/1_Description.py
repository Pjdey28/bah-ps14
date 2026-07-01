import streamlit as st
import pandas as pd
from pathlib import Path
from utils import load_css

# Define project structure
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"
CSS_FILE = Path(__file__).parent.parent / "style.css"

# Load CSS
load_css(CSS_FILE)

st.title("Description of the Ensemble Model Predictions")

# Load data from the correct sample file
prediction_file = ARTIFACTS / "evaluation" / "Ensemble_sample.csv"

if not prediction_file.exists():
    st.error(
        f"Ensemble_sample.csv not found at {prediction_file}."
        "Run the evaluation pipeline first to generate artifacts."
    )
    st.stop()

df = pd.read_csv(prediction_file)

st.markdown("This page shows a sample of predictions from the final ensemble model, comparing the actual values to the predicted values for each forecast horizon.")

st.divider()

# --- Metrics ---
st.subheader("Sample Prediction Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Mean 30-min Prediction",
        f"{df['Pred30'].mean():.3f}",
        delta=f"{(df['Pred30'].mean() - df['Actual30'].mean()):.3f} vs Actual",
        delta_color="off"
    )

with col2:
    st.metric(
        "Mean 6-hr Prediction",
        f"{df['Pred6'].mean():.3f}",
        delta=f"{(df['Pred6'].mean() - df['Actual6'].mean()):.3f} vs Actual",
        delta_color="off"
    )

with col3:
    st.metric(
        "Mean 12-hr Prediction",
        f"{df['Pred12'].mean():.3f}",
        delta=f"{(df['Pred12'].mean() - df['Actual12'].mean()):.3f} vs Actual",
        delta_color="off"
    )

st.divider()

# --- Charts ---
st.subheader("Actual vs. Predicted Values")
chart_data = df.head(100) # Use a subset for cleaner visualization

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### 30-Minute Horizon")
    st.line_chart(chart_data[['Actual30', 'Pred30']])

with c2:
    st.markdown("#### 6-Hour Horizon")
    st.line_chart(chart_data[['Actual6', 'Pred6']])

with c3:
    st.markdown("#### 12-Hour Horizon")
    st.line_chart(chart_data[['Actual12', 'Pred12']])


st.divider()

# --- Raw Data ---
st.subheader("Prediction Sample Data")
st.dataframe(df.head(20), use_container_width=True)
