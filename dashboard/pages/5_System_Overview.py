import streamlit as st
from pathlib import Path
from utils import load_css

st.title("System Architecture & Overview")
st.markdown(
    "This page provides a high-level overview of the project's pipeline, models, and technology stack."
)
CSS_FILE = Path(__file__).parent.parent / "style.css"
load_css(CSS_FILE)

st.divider()

# --- Pipeline ---
with st.expander("Project Pipeline", expanded=True):
    st.markdown("The system processes satellite and solar wind data through a multi-stage pipeline to generate forecasts.")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            bgcolor="transparent";
            node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#69bff8", fontname="Helvetica", margin="0.12,0.06", width=2.3];
            edge [color="#6b7280", penwidth=1.4, arrowsize=0.7];

            A [label="Raw GOES/WIND"];
            B [label="Readers & Validation"];
            C [label="Preprocessing"];
            D [label="Feature Store"];
            E [label="Model Training"];
            F [label="Evaluation & SHAP"];
            G [label="Streamlit Pages"];

            A -> B -> C -> D -> E -> F -> G;
        }
        """
    )

st.divider()

# --- Columns for Details ---
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Models Used")
        st.info("Random Forest")
        st.info("LightGBM")
        st.info("XGBoost")
        st.info("LSTM (Long Short-Term Memory)")
        st.info("GRU (Gated Recurrent Unit)")
        st.success("**Final Model: Ridge Stacking Ensemble**")

with col2:
    with st.container(border=True):
        st.subheader("Forecast Details")
        st.metric("Sequence Length (Timesteps)", "72")
        st.markdown("**Forecast Horizons:**")
        st.success("30 Minutes")
        st.success("6 Hours")
        st.success("12 Hours")


st.divider()

# --- Technology Stack ---
with st.expander("Technology Stack"):
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    **Machine Learning**
    - Scikit-learn
    - PyTorch
    - LightGBM
    - XGBoost
    - SHAP
    """)
    c2.markdown("""
    **Core Libraries**
    - Pandas
    - NumPy
    - SciPy
    """)
    c3.markdown("""
    **App & Visualization**
    - Streamlit
    - Plotly Express
    - Matplotlib
    """)
