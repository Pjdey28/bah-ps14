import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import glob
from utils import load_css

# --- Page Configuration ---
st.title("Model Explainability with SHAP")
st.markdown(
    "Explore how each feature contributes to the model's predictions using SHAP (SHapley Additive exPlanations)."
)
CSS_FILE = Path(__file__).parent.parent / "style.css"
load_css(CSS_FILE)

# --- Data Loading ---
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"
SHAP_DIR = ARTIFACTS / "shap"

# Check for essential files
required_plots = ["shap_summary.png", "shap_bar.png"]
for f in required_plots:
    if not (SHAP_DIR / f).exists():
        st.error(f"'{f}' not found in 'artifacts/shap/'. Please run the SHAP analysis pipeline.")
        st.stop()

# --- Main Tabs ---
tab1, tab2 = st.tabs(["SHAP Summary Plots", "Feature Importance Data"])

# --- Tab 1: SHAP Summary Plots ---
with tab1:
    st.subheader("Global Feature Explainability")
    st.markdown(
        """
        The following plots provide a global summary of feature impacts.
        - **SHAP Summary Plot**: Shows the distribution of SHAP values for each feature. Each point is a Shapley value for a feature and an instance.
        - **Global Feature Importance**: Shows the mean absolute SHAP value for each feature, ranked from most to least important.
        """
    )
    
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(SHAP_DIR / "shap_summary.png"), caption="SHAP Summary Plot (Beeswarm)", use_container_width=True)
    with c2:
        st.image(str(SHAP_DIR / "shap_bar.png"), caption="Global Feature Importance (Bar)", use_container_width=True)

# --- Tab 2: Feature Importance Data ---
with tab2:
    st.subheader("Feature Importance Rankings")
    st.markdown("Explore the raw data for feature importance from different models.")

    # Find all available importance CSVs
    importance_files = glob.glob(str(SHAP_DIR / "*_importance.csv"))
    if not importance_files:
        st.warning("No feature importance CSV files found in 'artifacts/shap/'.")
        st.stop()

    file_map = {Path(f).stem.replace("_importance", ""): f for f in importance_files}
    
    # Let user select which importance file to view
    selected_file_key = st.selectbox("Select Importance File", file_map.keys())
    
    # Load and display the selected dataframe
    df_importance = pd.read_csv(file_map[selected_file_key])
    
    st.markdown(f"#### Data for **{selected_file_key}**")
    st.dataframe(df_importance, height=400, use_container_width=True)
    
    # Create an interactive bar chart
    st.markdown("---")
    st.markdown(f"#### Top 20 Features for **{selected_file_key}**")
    
    # Ensure correct column name for features
    feature_col = df_importance.columns[0]
    value_col = df_importance.columns[1]

    # Take top 20 and sort for horizontal bar chart
    top_20_features = df_importance.nlargest(20, value_col).sort_values(value_col, ascending=True)

    fig = px.bar(
        top_20_features,
        x=value_col,
        y=feature_col,
        orientation='h',
        title=f"Top 20 Most Important Features ({selected_file_key})",
        labels={value_col: "Mean Absolute SHAP Value", feature_col: "Feature"},
        text_auto=".4f",
        height=600
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
