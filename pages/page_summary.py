import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── consistent colour palette used across the whole app ──────────────────────
BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
GREY   = "#6B7280"
BG     = "#F8FAFC"

def _style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   BG,
        "axes.edgecolor":   "#CBD5E1",
        "axes.linewidth":   0.8,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "grid.color":       "#E2E8F0",
        "grid.linewidth":   0.6,
        "font.family":      "DejaVu Sans",
        "font.size":        10,
        "axes.titlesize":   12,
        "axes.titleweight": "bold",
        "axes.labelsize":   10,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
    })

def show(df):
    _style()

    # ── header ───────────────────────────────────────────────────────────────
    st.header("Dataset Summary and Descriptive Statistics")
    st.markdown(
        "This section provides a structured overview of the dataset used throughout this "
        "application. Understanding the data — its shape, variable types, distributions, "
        "and grouped aggregations — forms the essential foundation before any modelling "
        "or inferential analysis can be conducted."
    )

    # ── 1.1 dataset overview ─────────────────────────────────────────────────
    st.subheader("1.1 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{df.shape[0]:,}")
    c2.metric("Total Features", f"{df.shape[1]}")
    c3.metric("Numerical Variables", "4")
    c4.metric("Categorical Variables", "3")

    st.markdown(
        "The dataset originates from the **Medical Cost Personal Dataset** available on "
        "Kaggle and contains 1,338 anonymised insurance policy records from the United States. "
        "Each row represents one policyholder and includes six predictors along with the "
        "target variable — annual insurance charges in USD."
    )

    # data dictionary
    st.markdown("**Data Dictionary**")
    dict_df = pd.DataFrame({
        "Variable":    ["age",      "sex",               "bmi",                     "children",              "smoker",               "region",                      "charges"],
        "Type":        ["Numerical","Categorical",        "Numerical",               "Discrete",              "Categorical",          "Categorical",                 "Numerical (Target)"],
        "Range/Values":["18 – 64",  "male, female",      "15.96 – 53.13",           "0 – 5",                 "yes, no",              "northeast, northwest, southeast, southwest","$1,122 – $63,770"],
        "Description": [
            "Age of the primary beneficiary in years",
            "Biological sex of the policyholder",
            "Body Mass Index — weight in kg divided by height in metres squared",
            "Number of children or dependants covered by the insurance plan",
            "Whether the policyholder is a tobacco smoker",
            "Residential region of the beneficiary within the United States",
            "Individual annual medical costs billed by the insurer — the prediction target"
        ]
    })
    st.dataframe(dict_df, use_container_width=True, hide_index=True)

    # ── 1.2 data preview ─────────────────────────────────────────────────────
    st.subheader("1.2 Data Preview")
    st.markdown(
        "The table below shows the first ten records. No missing values or duplicate "
        "records were found in this dataset — it is clean and ready for analysis."
    )
    st.dataframe(df.head(10), use_container_width=True)

    col_miss, col_dup = st.columns(2)
    col_miss.success(f"Missing values: {df.isnull().sum().sum()}")
    col_dup.success(f"Duplicate rows: {df.duplicated().sum()}")

    # ── 1.3 numerical descriptive statistics ─────────────────────────────────
    st.subheader("1.3 Descriptive Statistics for Numerical Variables")
    st.markdown(
        "The table below summarises the central tendency, spread, and shape of each "
        "numerical feature. These statistics reveal, for example, that insurance charges "
        "are heavily right-skewed — the mean exceeds the median, indicating a long upper tail "
        "driven by high-cost policyholders (predominantly smokers with elevated BMI)."
    )

    num_cols = ["age", "bmi", "children", "charges"]
    stats = pd.DataFrame(index=num_cols)
    stats["Mean"]       = df[num_cols].mean()
    stats["Median"]     = df[num_cols].median()
    stats["Std Dev"]    = df[num_cols].std()
    stats["Variance"]   = df[num_cols].var()
    stats["Min"]        = df[num_cols].min()
    stats["Q1 (25%)"]   = df[num_cols].quantile(0.25)
    stats["Q3 (75%)"]   = df[num_cols].quantile(0.75)
    stats["IQR"]        = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)
    stats["Max"]        = df[num_cols].max()
    stats["Skewness"]   = df[num_cols].skew()
    st.dataframe(stats.style.format("{:,.3f}"), use_container_width=True)

    st.markdown(r"""
    **Key Formulas**

    | Statistic | Formula |
    |-----------|---------|
    | Mean | $\bar{x} = \dfrac{1}{n}\displaystyle\sum_{i=1}^{n} x_i$ |
    | Variance | $s^2 = \dfrac{1}{n-1}\displaystyle\sum_{i=1}^{n}(x_i - \bar{x})^2$ |
    | Standard Deviation | $s = \sqrt{s^2}$ |
    | IQR | $Q_3 - Q_1$ |
    | Skewness | $\gamma_1 = \dfrac{1}{n}\displaystyle\sum_{i=1}^{n}\left(\dfrac{x_i - \bar{x}}{s}\right)^{3}$ |
    """)
    skew_val = df["charges"].skew()
    st.markdown(
        "**Interpretation of skewness:** A value of 0 indicates perfect symmetry. "
        "Values > 0 indicate a right (positive) skew — a long upper tail. "
        f"Charges exhibit a skewness of **{skew_val:.2f}**, confirming a pronounced right tail."
    )

    # visual: four boxplots side by side
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors_bp = [BLUE, GREEN, ORANGE, PURPLE]
    for ax, col, color in zip(axes, num_cols, colors_bp):
        bp = ax.boxplot(df[col], patch_artist=True,
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=color, alpha=0.5),
                        medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(col.capitalize())
        ax.set_ylabel("Value")
    fig.suptitle("Boxplots of Numerical Variables", fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "**Reading a boxplot:** The box spans Q1 to Q3 (the interquartile range). "
        "The horizontal line inside is the median. Whiskers extend to 1.5 × IQR beyond the "
        "quartiles. Points beyond the whiskers are potential outliers. The charges boxplot "
        "shows numerous upper outliers — confirming the heavy right tail."
    )

    # ── 1.4 categorical distributions ───────────────────────────────────────
    st.subheader("1.4 Distribution of Categorical Variables")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cat_info = [
        ("sex",    [BLUE, RED],                   "Gender Distribution"),
        ("smoker", [GREEN, ORANGE],               "Smoker Distribution"),
        ("region", [BLUE, GREEN, ORANGE, PURPLE], "Regional Distribution"),
    ]
    for ax, (col, pal, title) in zip(axes, cat_info):
        counts = df[col].value_counts()
        bars = ax.bar(counts.index, counts.values, color=pal[:len(counts)],
                      edgecolor="black", linewidth=0.6, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                    f"{h:,} ({h/len(df)*100:.1f}%)", ha="center", fontsize=8.5, fontweight="bold")
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── 1.5 grouped summaries ────────────────────────────────────────────────
    st.subheader("1.5 Grouped Summaries by Categorical Variables")
    st.markdown(
        "Aggregating charges by categorical groups is one of the quickest ways to identify "
        "which variables have the strongest influence on cost. The differences seen here — "
        "especially for smoker status — are later confirmed statistically."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["By Smoker", "By Region", "By Sex", "By Children"])

    def grouped_table(group_col):
        g = df.groupby(group_col)["charges"].agg(
            Count="count",
            Mean="mean",
            Median="median",
            Std_Dev="std",
            Min="min",
            Max="max"
        ).reset_index()
        return g

    with tab1:
        g = grouped_table("smoker")
        st.dataframe(g.style.format({c: "${:,.2f}" for c in ["Mean","Median","Std_Dev","Min","Max"]}),
                     use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        g_plot = g.set_index("smoker")
        bars = ax.bar(["Non-Smoker" if s == "no" else "Smoker" for s in g_plot.index],
                      g_plot["Mean"], color=[GREEN, RED], edgecolor="black", alpha=0.85, width=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 300, f"${h:,.0f}",
                    ha="center", fontweight="bold", fontsize=11)
        ax.set_ylabel("Average Annual Charges ($)")
        ax.set_title("Average Charges: Smoker vs Non-Smoker")
        diff = g[g.smoker=="yes"]["Mean"].values[0] - g[g.smoker=="no"]["Mean"].values[0]
        ax.annotate(f"Difference: ${diff:,.0f}", xy=(0.5, 0.88), xycoords="axes fraction",
                    ha="center", fontsize=10, color=RED,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=RED, alpha=0.8))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info(f"Smokers pay on average **${diff:,.0f} more** per year than non-smokers — a difference of "
                f"{diff/g[g.smoker=='no']['Mean'].values[0]*100:.0f}%.")

    with tab2:
        g = grouped_table("region")
        st.dataframe(g.style.format({c: "${:,.2f}" for c in ["Mean","Median","Std_Dev","Min","Max"]}),
                     use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        g_sorted = g.sort_values("Mean", ascending=False)
        bars = ax.barh(g_sorted["region"], g_sorted["Mean"],
                       color=[BLUE, GREEN, ORANGE, PURPLE][:len(g_sorted)],
                       edgecolor="black", alpha=0.85)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 200, bar.get_y() + bar.get_height()/2,
                    f"${w:,.0f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("Average Annual Charges ($)")
        ax.set_title("Average Charges by Region")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info("Regional variation is modest compared to the smoking effect. The southeast has the "
                "highest average charges, likely due to a higher proportion of smokers and obese individuals.")

    with tab3:
        g = grouped_table("sex")
        st.dataframe(g.style.format({c: "${:,.2f}" for c in ["Mean","Median","Std_Dev","Min","Max"]}),
                     use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(g["sex"], g["Mean"], color=[BLUE, RED], edgecolor="black", alpha=0.85, width=0.4)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 100, f"${h:,.0f}",
                    ha="center", fontweight="bold")
        ax.set_ylabel("Average Annual Charges ($)")
        ax.set_title("Average Charges by Sex")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info("The difference in charges between males and females is small and may not be statistically "
                "significant after controlling for other factors such as smoking status.")

    with tab4:
        g = grouped_table("children")
        st.dataframe(g.style.format({c: "${:,.2f}" for c in ["Mean","Median","Std_Dev","Min","Max"]}),
                     use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(g["children"].astype(str), g["Mean"], color=BLUE,
                      edgecolor="black", alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 100, f"${h:,.0f}",
                    ha="center", fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Number of Children")
        ax.set_ylabel("Average Annual Charges ($)")
        ax.set_title("Average Charges by Number of Children")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info("There is no strong monotonic relationship between number of children and charges. "
                "Policyholders with no children actually exhibit higher average charges, likely due to "
                "other confounding factors such as age and BMI.")
