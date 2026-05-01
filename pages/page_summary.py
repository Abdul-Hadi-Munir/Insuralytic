import streamlit as st
import pandas as pd
import numpy as np

def show(df):
    st.header("Dataset Summary and Descriptive Statistics")

    st.subheader("1.1 Dataset Overview")
    st.markdown(f"The dataset comprises **{df.shape[0]}** observations and **{df.shape[1]}** variables.")

    st.markdown("""
    **Data Dictionary**

    | Variable | Type | Description |
    |----------|------|-------------|
    | age | Numerical | Age of primary beneficiary (years) |
    | sex | Categorical | Biological sex of the policyholder |
    | bmi | Numerical | Body Mass Index (kg/m²) |
    | children | Discrete | Number of dependents covered |
    | smoker | Categorical | Whether the policyholder is a smoker |
    | region | Categorical | Residential region in the United States |
    | charges | Numerical | Annual medical insurance cost (USD) |
    """)

    st.subheader("1.2 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("1.3 Descriptive Statistics for Numerical Variables")
    num_cols = ["age", "bmi", "children", "charges"]
    stats = pd.DataFrame(index=num_cols)
    stats["Mean"] = df[num_cols].mean()
    stats["Median"] = df[num_cols].median()
    stats["Variance"] = df[num_cols].var()
    stats["Std Dev"] = df[num_cols].std()
    stats["Min"] = df[num_cols].min()
    stats["Q1 (25%)"] = df[num_cols].quantile(0.25)
    stats["Q2 (50%)"] = df[num_cols].quantile(0.50)
    stats["Q3 (75%)"] = df[num_cols].quantile(0.75)
    stats["Max"] = df[num_cols].max()
    st.dataframe(stats.style.format("{:,.2f}"), use_container_width=True)

    st.markdown(r"""
    **Formulas Used**

    - **Mean**: $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$
    - **Variance**: $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$
    - **Standard Deviation**: $s = \sqrt{s^2}$
    """)

    st.subheader("1.4 Grouped Summaries by Categorical Variables")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Average Charges by Smoker Status**")
        smoker_summary = df.groupby("smoker")["charges"].agg(["mean", "median", "count"])
        smoker_summary.columns = ["Mean Charges", "Median Charges", "Count"]
        st.dataframe(smoker_summary.style.format({"Mean Charges": "${:,.2f}", "Median Charges": "${:,.2f}", "Count": "{:,}"}), use_container_width=True)

    with col2:
        st.markdown("**Average Charges by Region**")
        region_summary = df.groupby("region")["charges"].agg(["mean", "median", "count"])
        region_summary.columns = ["Mean Charges", "Median Charges", "Count"]
        st.dataframe(region_summary.style.format({"Mean Charges": "${:,.2f}", "Median Charges": "${:,.2f}", "Count": "{:,}"}), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Average Charges by Sex**")
        sex_summary = df.groupby("sex")["charges"].agg(["mean", "median", "count"])
        sex_summary.columns = ["Mean Charges", "Median Charges", "Count"]
        st.dataframe(sex_summary.style.format({"Mean Charges": "${:,.2f}", "Median Charges": "${:,.2f}", "Count": "{:,}"}), use_container_width=True)

    with col4:
        st.markdown("**Average Charges by Number of Children**")
        child_summary = df.groupby("children")["charges"].agg(["mean", "median", "count"])
        child_summary.columns = ["Mean Charges", "Median Charges", "Count"]
        st.dataframe(child_summary.style.format({"Mean Charges": "${:,.2f}", "Median Charges": "${:,.2f}", "Count": "{:,}"}), use_container_width=True)
