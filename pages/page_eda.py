import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})


def show(df):
    st.header("Exploratory Data Analysis")

    # --- Charges Distribution ---
    st.subheader("2.1 Distribution of Insurance Charges")
    st.markdown(
        "The histogram below illustrates the frequency distribution of annual insurance charges. "
        "A pronounced right skew indicates that the majority of policyholders incur moderate costs, "
        "while a smaller subset faces substantially higher charges."
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["charges"], bins=40, color="#3498db", edgecolor="black", alpha=0.75)
    axes[0].set_xlabel("Charges ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Insurance Charges")
    sns.boxplot(y=df["charges"], ax=axes[1], color="#3498db", flierprops=dict(marker="o", markersize=3))
    axes[1].set_ylabel("Charges ($)")
    axes[1].set_title("Boxplot of Insurance Charges")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Age Distribution ---
    st.subheader("2.2 Age Distribution and Charges")
    st.markdown(
        "The scatter plot reveals the relationship between age and insurance charges. "
        "Two distinct clusters emerge when colored by smoking status, confirming that smokers "
        "consistently incur higher costs across all age groups."
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["age"], bins=30, color="#2ecc71", edgecolor="black", alpha=0.75)
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Age Distribution")
    colors = df["smoker"].map({"yes": "#e74c3c", "no": "#2ecc71"})
    axes[1].scatter(df["age"], df["charges"], c=colors, alpha=0.5, edgecolors="black", linewidth=0.3, s=20)
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Charges ($)")
    axes[1].set_title("Age vs Charges (by Smoker Status)")
    import matplotlib.patches as mpatches
    axes[1].legend(handles=[
        mpatches.Patch(color="#e74c3c", label="Smoker"),
        mpatches.Patch(color="#2ecc71", label="Non-Smoker")
    ])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- BMI Distribution ---
    st.subheader("2.3 BMI Distribution and Charges")
    st.markdown(
        "BMI follows an approximately normal distribution centered around 30. "
        "The scatter plot demonstrates that elevated BMI combined with smoking status "
        "produces the highest insurance charges."
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["bmi"], bins=30, color="#9b59b6", edgecolor="black", alpha=0.75)
    axes[0].set_xlabel("BMI")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("BMI Distribution")
    axes[1].scatter(df["bmi"], df["charges"], c=colors, alpha=0.5, edgecolors="black", linewidth=0.3, s=20)
    axes[1].set_xlabel("BMI")
    axes[1].set_ylabel("Charges ($)")
    axes[1].set_title("BMI vs Charges (by Smoker Status)")
    axes[1].legend(handles=[
        mpatches.Patch(color="#e74c3c", label="Smoker"),
        mpatches.Patch(color="#2ecc71", label="Non-Smoker")
    ])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Smoker Impact ---
    st.subheader("2.4 Impact of Smoking on Insurance Charges")
    st.markdown(
        "Smoking status is the single most significant predictor of insurance charges. "
        "The bar chart quantifies the cost differential between smokers and non-smokers."
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    smoker_avg = df.groupby("smoker")["charges"].mean()
    bars = axes[0].bar(["Non-Smoker", "Smoker"], [smoker_avg["no"], smoker_avg["yes"]],
                       color=["#2ecc71", "#e74c3c"], edgecolor="black", alpha=0.8)
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., h + 300, f"${h:,.0f}",
                     ha="center", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Average Charges ($)")
    axes[0].set_title("Average Charges: Smoker vs Non-Smoker")
    sns.violinplot(x="smoker", y="charges", data=df, ax=axes[1],
                   palette={"no": "#2ecc71", "yes": "#e74c3c"}, inner="quartile")
    axes[1].set_xlabel("Smoker Status")
    axes[1].set_ylabel("Charges ($)")
    axes[1].set_title("Charge Distribution by Smoker Status")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Region ---
    st.subheader("2.5 Regional Analysis")
    st.markdown(
        "Regional differences in insurance charges are statistically modest. "
        "The southeast region exhibits the highest average cost, though the difference "
        "across regions is far smaller than the smoking effect."
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    region_avg = df.groupby("region")["charges"].mean().sort_values(ascending=False)
    bars = ax.bar(region_avg.index, region_avg.values, color="#3498db", edgecolor="black", alpha=0.8)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 200, f"${h:,.0f}",
                ha="center", fontweight="bold")
    ax.set_ylabel("Average Charges ($)")
    ax.set_title("Average Insurance Charges by Region")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Correlation Heatmap ---
    st.subheader("2.6 Correlation Heatmap")
    st.markdown(
        "The Pearson correlation matrix quantifies the linear relationships between numerical variables. "
        "Values close to +1 or -1 indicate strong positive or negative linear relationships, respectively."
    )
    st.markdown(r"""
    **Pearson Correlation Coefficient:**
    $$r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$
    """)
    df_encoded = df.copy()
    df_encoded["sex"] = df_encoded["sex"].map({"male": 1, "female": 0})
    df_encoded["smoker"] = df_encoded["smoker"].map({"yes": 1, "no": 0})
    df_encoded["region"] = df_encoded["region"].map({"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3})
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df_encoded.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=1, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "**Key Observations:** Smoker status exhibits the strongest positive correlation with charges "
        f"(r = {corr.loc['smoker', 'charges']:.2f}), followed by age "
        f"(r = {corr.loc['age', 'charges']:.2f}) and BMI "
        f"(r = {corr.loc['bmi', 'charges']:.2f})."
    )
