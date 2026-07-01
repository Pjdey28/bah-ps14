import streamlit as st
from pathlib import Path

def load_css(file_name: Path):
    """Loads a CSS file and injects it into the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
