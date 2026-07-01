import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import glob
from utils import load_css

# --- Page Configuration ---
st.title("Model Comparison")
st.markdown("Interactively compare the performance metrics and evaluation plots of all models.")
CSS_FILE = Path(__file__).parent.parent / "style.css"
load_css(CSS_FILE)

# --- Data Loading ---
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"
EVAL_DIR = ARTIFACTS / "evaluation"

# Check for essential files
required_files = ["model_metrics.csv", "metrics_summary.csv", "model_ranking.csv"]
for f in required_files:
    if not (EVAL_DIR / f).exists():
        st.error(f"'{f}' not found in 'artifacts/evaluation/'. Please run the model comparison pipeline.")
        st.stop()

metrics_df = pd.read_csv(EVAL_DIR / "model_metrics.csv")
summary_df = pd.read_csv(EVAL_DIR / "metrics_summary.csv", index_col=0)
ranking_df = pd.read_csv(EVAL_DIR / "model_ranking.csv", index_col=0)

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["Metric Summary", "Performance Plots", "Per-Model Analysis"])

# --- Tab 1: Metric Summary ---
with tab1:
    st.subheader("Overall Model Performance")
    
    # Interactive bar chart for metrics
    metric_to_plot = st.selectbox("Select a Metric to Visualize", ["RMSE", "MAE", "R2"])
    
    fig = px.bar(
        metrics_df,
        x="Model",
        y=metric_to_plot,
        color="Horizon",
        barmode="group",
        title=f"Model Comparison for {metric_to_plot}",
        text_auto=".3f"
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # DataFrames
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average Metrics Across Horizons")
        st.dataframe(summary_df, use_container_width=True)
    with c2:
        st.subheader("Model Ranking (by average RMSE)")
        st.dataframe(ranking_df, use_container_width=True)

# --- Tab 2: Performance Plots ---
with tab2:
    st.subheader("Overall Metric Plots")
    st.markdown("Visual comparison of key performance indicators across all models.")
    
    # Load and display the main metric PNGs
    metric_plots = {"RMSE": "RMSE.png", "MAE": "MAE.png", "R2": "R2.png"}
    cols = st.columns(len(metric_plots))
    
    for i, (metric, file) in enumerate(metric_plots.items()):
        if (EVAL_DIR / file).exists():
            cols[i].image(str(EVAL_DIR / file), caption=f"{metric} Comparison", use_container_width=True)
        else:
            cols[i].warning(f"{file} not found.")

# --- Tab 3: Per-Model Analysis ---
with tab3:
    st.subheader("Detailed Analysis per Model")
    
    model_list = metrics_df["Model"].unique()
    selected_model = st.selectbox("Select Model for Detailed View", model_list)
    
    st.markdown(f"#### Showing plots for **{selected_model}**")
    
    # Find all plots for the selected model
    plot_types = {
        "Forecast vs. Actual": f"{selected_model}_*.png",
        "Scatter Plot": f"{selected_model}_Scatter_*.png",
        "Residuals": f"{selected_model}_Residual_*.png"
    }
    
    for plot_title, file_pattern in plot_types.items():
        st.markdown(f"##### {plot_title}")
        plot_files = glob.glob(str(EVAL_DIR / file_pattern))
        
        # Exclude scatter and residual from the main forecast plot search
        if plot_title == "Forecast vs. Actual":
            plot_files = [f for f in plot_files if "Scatter" not in f and "Residual" not in f]

        if plot_files:
            cols = st.columns(len(plot_files))
            for i, f in enumerate(sorted(plot_files)):
                with cols[i]:
                    st.image(f, use_container_width=True)
        else:
            st.info(f"No '{plot_title}' plots found for {selected_model}.")
        
        st.divider()
