import streamlit as st
import pandas as pd
from pages import page_summary, page_eda, page_explore, page_probability, page_regression

st.set_page_config(
    page_title="Insurlytics: Healthcare Cost Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/v1/raw/insurance.csv")

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Insurlytics")
st.sidebar.markdown("Healthcare Cost Analysis and Prediction")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Dataset Summary",
    "Exploratory Data Analysis",
    "Explore Patterns",
    "Probability and Distributions",
    "Regression and Prediction"
])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** Kaggle Medical Cost Personal Dataset  \n"
    "**Records:** 1,338  \n"
    "**Features:** 6 predictors + 1 target  \n"
    "**Model:** Multiple Linear Regression"
)

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "Dataset Summary":
    page_summary.show(df)
elif page == "Exploratory Data Analysis":
    page_eda.show(df)
elif page == "Explore Patterns":
    page_explore.show(df)
elif page == "Probability and Distributions":
    page_probability.show(df)
elif page == "Regression and Prediction":
    page_regression.show(df)
