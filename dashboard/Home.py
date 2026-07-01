import streamlit as st
import pandas as pd
from pathlib import Path

from utils import load_css


# Set page configuration
st.set_page_config(
    page_title="Energetic Particle Radiation Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define project structure
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
CSS_FILE = Path(__file__).resolve().parent / "style.css"

# Inject CSS
load_css(CSS_FILE)

# Title
st.title("Energetic Particle Radiation Forecast Dashboard")

# Caption
st.caption(
    "Ensemble Machine Learning System for Multi-Horizon Energetic Particle Radiation Prediction"
)

st.divider()

# Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Models", "6")
with c2:
    st.metric("Forecast Horizons", "3")
with c3:
    st.metric("Years", "2020-2024")
with c4:
    st.metric("Stacking", "Ridge")

st.divider()

# Main content layout
left, right = st.columns([2, 1])

with left:
    st.subheader("Pipeline")
    st.graphviz_chart(
        """
        digraph {
            rankdir=TB;
            bgcolor="transparent";
            node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#69bff8", fontname="Helvetica", margin="0.12,0.06", width=2.2];
            edge [color="#94a3b8", penwidth=1.4, arrowsize=0.7];

            A [label="GOES Data"];
            B [label="WIND Data"];
            C [label="Decode & Clean"];
            D [label="Feature Engineering"];
            E [label="Feature Selection"];
            F [label="Sequence Samples"];
            G [label="Forecast Ensemble"];
            H [label="Forecast Output"];

            A -> C;
            B -> C;
            C -> D -> E -> F;
            F -> G -> H;
        }
        """
    )

with right:
    st.subheader("Quick Facts")
    st.info("200 Selected Features")
    st.info("72 Time Steps")
    st.info("30 min / 6 hr / 12 hr")
    st.info("SHAP Explainability")

st.divider()

# Model Performance
st.subheader("Model Performance")
metrics_path = ARTIFACTS / "evaluation" / "model_metrics.csv"
if metrics_path.exists():
    df = pd.read_csv(metrics_path)
    ranking = (
        df.groupby("Model")
        .agg({"RMSE": "mean", "MAE": "mean", "R2": "mean"})
        .sort_values("RMSE")
    )
    st.dataframe(ranking, use_container_width=True)
else:
    st.warning("Run evaluation first to display model metrics.")

st.divider()

# Prediction Sample section
st.subheader("Prediction Sample")
prediction_file = ARTIFACTS / "evaluation" / "Ensemble_sample.csv"
if prediction_file.exists():
    pred_df = pd.read_csv(prediction_file)
    st.dataframe(pred_df.head(20), use_container_width=True)
else:
    st.info("Could not find Ensemble_sample.csv to show predictions.")

st.divider()

# Dashboard Pages section
st.subheader("Dashboard Pages")
col1, col2, col3 = st.columns(3)
with col1:
    st.success("""
    **Data Overview**
    • Dataset statistics
    • Feature summary
    • Data preview
    """)
with col2:
    st.success("""
    **Predictions**
    • Forecast generation
    • Ensemble prediction
    • Download predictions
    """)
with col3:
    st.success("""
    **Model Comparison**
    • RMSE • MAE • R² • Rankings
    """)
col4, col5 = st.columns(2)
with col4:
    st.success("""
    **SHAP Explainability**
    • Feature importance
    • SHAP summary
    • Dependence plots
    """)
with col5:
    st.success("""
    **System Overview**
    • Pipeline • Architecture • Technology stack
    """)

st.divider()

# Footer
st.caption(
    "Energetic Particle Radiation Forecasting using Ensemble Machine Learning | Random Forest • LightGBM • XGBoost • LSTM • GRU • Ridge Stacking"
)