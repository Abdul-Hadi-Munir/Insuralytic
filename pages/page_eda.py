import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
BG     = "#F8FAFC"

def _style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor": "#CBD5E1", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#E2E8F0", "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
    })


def show(df):
    _style()
    st.header("Exploratory Data Analysis")
    st.markdown(
        "Exploratory Data Analysis (EDA) is the process of systematically investigating a "
        "dataset through statistical summaries and visualisations before applying any formal "
        "model. EDA reveals patterns, anomalies, and relationships that guide subsequent "
        "modelling decisions. Every chart below is accompanied by an interpretation to ensure "
        "findings are accessible to both technical and non-technical audiences."
    )

    # ── 2.1 Charges distribution ─────────────────────────────────────────────
    st.subheader("2.1 Distribution of Insurance Charges")
    st.markdown(
        "Understanding the distribution of the target variable — annual insurance charges — "
        "is the first step in any regression analysis. The shape of the distribution influences "
        "model assumptions, error characteristics, and the choice of evaluation metric."
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # histogram
    axes[0].hist(df["charges"], bins=45, color=BLUE, edgecolor="white",
                 linewidth=0.4, alpha=0.85)
    axes[0].axvline(df["charges"].mean(),   color=RED,    linestyle="--", linewidth=1.5, label=f"Mean ${df['charges'].mean():,.0f}")
    axes[0].axvline(df["charges"].median(), color=GREEN,  linestyle=":",  linewidth=1.5, label=f"Median ${df['charges'].median():,.0f}")
    axes[0].set_xlabel("Annual Charges ($)")
    axes[0].set_ylabel("Number of Policyholders")
    axes[0].set_title("Histogram of Insurance Charges")
    axes[0].legend(fontsize=8)

    # KDE
    kde_vals = df["charges"].values
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(kde_vals)
    x_kde = np.linspace(kde_vals.min(), kde_vals.max(), 300)
    axes[1].fill_between(x_kde, kde(x_kde), color=BLUE, alpha=0.35)
    axes[1].plot(x_kde, kde(x_kde), color=BLUE, linewidth=2)
    axes[1].axvline(df["charges"].mean(),   color=RED,   linestyle="--", linewidth=1.5, label="Mean")
    axes[1].axvline(df["charges"].median(), color=GREEN, linestyle=":",  linewidth=1.5, label="Median")
    axes[1].set_xlabel("Annual Charges ($)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Kernel Density Estimate")
    axes[1].legend(fontsize=8)

    # boxplot
    bp = axes[2].boxplot(df["charges"], vert=True, patch_artist=True,
                         flierprops=dict(marker="o", markersize=3, markerfacecolor=BLUE, alpha=0.4),
                         medianprops=dict(color=RED, linewidth=2))
    bp["boxes"][0].set_facecolor(BLUE)
    bp["boxes"][0].set_alpha(0.5)
    axes[2].set_ylabel("Annual Charges ($)")
    axes[2].set_title("Boxplot with Outlier Flags")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Charges",   f"${df['charges'].mean():,.2f}")
    col2.metric("Median Charges", f"${df['charges'].median():,.2f}")
    col3.metric("Skewness",       f"{df['charges'].skew():.3f}")

    st.markdown(
        "**Interpretation:** The distribution is strongly right-skewed (skewness = "
        f"{df['charges'].skew():.2f}). The mean (${df['charges'].mean():,.0f}) substantially "
        f"exceeds the median (${df['charges'].median():,.0f}), indicating that a small number "
        "of very high-cost policyholders pull the average upward. This is characteristic of "
        "insurance datasets where a minority of individuals — often smokers with high BMI — "
        "account for a disproportionate share of total expenditure."
    )

    # ── 2.2 Age analysis ──────────────────────────────────────────────────────
    st.subheader("2.2 Age: Distribution and Relationship with Charges")
    st.markdown(
        "Age is expected to be positively correlated with insurance charges because older "
        "individuals are statistically more likely to require medical treatment. The scatter "
        "plot below is coloured by smoking status to reveal how these two risk factors interact."
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(df["age"], bins=25, color=GREEN, edgecolor="white", linewidth=0.4, alpha=0.85)
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Age Distribution")

    colors_s = df["smoker"].map({"yes": RED, "no": BLUE})
    axes[1].scatter(df["age"], df["charges"], c=colors_s, alpha=0.45,
                    edgecolors="none", s=20)
    axes[1].set_xlabel("Age (years)")
    axes[1].set_ylabel("Charges ($)")
    axes[1].set_title("Age vs Charges (by Smoker Status)")
    axes[1].legend(handles=[
        mpatches.Patch(color=RED,  label="Smoker"),
        mpatches.Patch(color=BLUE, label="Non-Smoker")
    ], fontsize=8)

    for status, color, label in [("yes", RED, "Smoker"), ("no", BLUE, "Non-Smoker")]:
        sub = df[df["smoker"] == status]
        avg = sub.groupby("age")["charges"].mean()
        axes[2].plot(avg.index, avg.values, color=color, linewidth=2, label=label)
    axes[2].set_xlabel("Age (years)")
    axes[2].set_ylabel("Average Charges ($)")
    axes[2].set_title("Average Charges by Age Group")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "**Interpretation:** Three distinct charge bands are visible in the scatter plot. "
        "The lowest band (blue) represents young, healthy non-smokers. The middle band "
        "represents older non-smokers and young smokers. The highest band (red) represents "
        "older smokers — this group consistently incurs the highest charges. Within each band, "
        "charges rise gradually with age, confirming a positive age-charges relationship."
    )

    # ── 2.3 BMI analysis ─────────────────────────────────────────────────────
    st.subheader("2.3 BMI: Distribution and Relationship with Charges")
    st.markdown(
        "Body Mass Index (BMI) is a widely used proxy for overall health risk. "
        "A BMI above 30 is classified as obese by the World Health Organisation. "
        "Higher BMI is associated with greater risk of chronic diseases, which in turn "
        "drives higher insurance expenditure."
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(df["bmi"], bins=30, color=PURPLE, edgecolor="white", linewidth=0.4, alpha=0.85)
    axes[0].axvline(30, color=RED, linestyle="--", linewidth=1.5, label="Obese threshold (BMI=30)")
    axes[0].set_xlabel("BMI (kg/m²)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("BMI Distribution")
    axes[0].legend(fontsize=8)

    axes[1].scatter(df["bmi"], df["charges"], c=colors_s, alpha=0.45, edgecolors="none", s=20)
    axes[1].axvline(30, color="black", linestyle="--", linewidth=1, alpha=0.6, label="BMI = 30")
    axes[1].set_xlabel("BMI (kg/m²)")
    axes[1].set_ylabel("Charges ($)")
    axes[1].set_title("BMI vs Charges (by Smoker Status)")
    axes[1].legend(handles=[
        mpatches.Patch(color=RED,  label="Smoker"),
        mpatches.Patch(color=BLUE, label="Non-Smoker"),
        mpatches.Patch(color="black", label="BMI = 30 threshold")
    ], fontsize=7)

    # BMI bins
    df_tmp = df.copy()
    df_tmp["bmi_group"] = pd.cut(df_tmp["bmi"], bins=[0, 18.5, 25, 30, 35, 60],
                                  labels=["Underweight","Normal","Overweight","Obese I","Obese II+"])
    bmi_avg = df_tmp.groupby("bmi_group", observed=True)["charges"].mean()
    bars = axes[2].bar(bmi_avg.index, bmi_avg.values,
                       color=[GREEN, BLUE, ORANGE, RED, PURPLE],
                       edgecolor="black", linewidth=0.5, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, h + 200, f"${h:,.0f}",
                     ha="center", fontsize=8, fontweight="bold")
    axes[2].set_xlabel("BMI Category")
    axes[2].set_ylabel("Average Charges ($)")
    axes[2].set_title("Average Charges by BMI Category")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "**Interpretation:** Among non-smokers (blue), the relationship between BMI and charges "
        "is weak. However, for smokers (red), high BMI dramatically amplifies charges — the "
        "combined effect of smoking and obesity is more than additive. This interaction effect "
        "is important for the regression model but is not captured by simple linear terms alone."
    )

    # ── 2.4 Smoker impact ────────────────────────────────────────────────────
    st.subheader("2.4 Smoking Status: The Dominant Cost Driver")
    st.markdown(
        "Across all subgroups in this dataset, smoking status consistently emerges as the "
        "most influential predictor of insurance charges. The following charts quantify this "
        "effect from multiple perspectives."
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # bar: mean by smoker
    smoker_means = df.groupby("smoker")["charges"].mean()
    bars = axes[0].bar(["Non-Smoker", "Smoker"],
                       [smoker_means["no"], smoker_means["yes"]],
                       color=[GREEN, RED], edgecolor="black", linewidth=0.6, alpha=0.85, width=0.5)
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 400,
                     f"${h:,.0f}", ha="center", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Average Charges ($)")
    axes[0].set_title("Average Charges by Smoker Status")
    diff = smoker_means["yes"] - smoker_means["no"]
    axes[0].annotate(f"+${diff:,.0f} for smokers",
                     xy=(0.5, 0.88), xycoords="axes fraction", ha="center",
                     fontsize=9, color=RED,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=RED, alpha=0.85))

    # violin
    parts = axes[1].violinplot(
        [df[df["smoker"]=="no"]["charges"].values,
         df[df["smoker"]=="yes"]["charges"].values],
        positions=[1, 2], showmedians=True, showextrema=True
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor([GREEN, RED][i])
        body.set_alpha(0.6)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["Non-Smoker", "Smoker"])
    axes[1].set_ylabel("Charges ($)")
    axes[1].set_title("Charge Distribution (Violin Plot)")

    # proportion pie by smoker
    smoker_counts = df["smoker"].value_counts()
    axes[2].pie(smoker_counts, labels=["Non-Smoker", "Smoker"],
                autopct="%1.1f%%", colors=[GREEN, RED],
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[2].set_title("Proportion of Smokers in Dataset")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2, col3 = st.columns(3)
    col1.metric("Smoker Mean Charges",     f"${smoker_means['yes']:,.2f}")
    col2.metric("Non-Smoker Mean Charges", f"${smoker_means['no']:,.2f}")
    col3.metric("Charge Premium for Smoking", f"+${diff:,.0f}")

    st.markdown(
        f"**Interpretation:** Only {smoker_counts['yes']/len(df)*100:.1f}% of policyholders are "
        f"smokers, yet they incur charges that are on average **${diff:,.0f} higher** per year. "
        "The violin plot reveals that the entire distribution of charges shifts dramatically "
        "upward for smokers, with a much longer upper tail. This single variable is responsible "
        "for a large portion of the model's predictive accuracy."
    )

    # ── 2.5 Regional analysis ────────────────────────────────────────────────
    st.subheader("2.5 Geographic Region: A Secondary Factor")
    st.markdown(
        "The United States is divided into four census regions in this dataset: northeast, "
        "northwest, southeast, and southwest. Regional differences may reflect variations in "
        "local cost of living, access to healthcare, and the prevalence of lifestyle risk factors."
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    region_avg = df.groupby("region")["charges"].mean().sort_values(ascending=False)
    bars = axes[0].barh(region_avg.index, region_avg.values,
                        color=[BLUE, GREEN, ORANGE, PURPLE], edgecolor="black",
                        linewidth=0.5, alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        axes[0].text(w + 200, bar.get_y() + bar.get_height()/2,
                     f"${w:,.0f}", va="center", fontsize=9, fontweight="bold")
    axes[0].set_xlabel("Average Charges ($)")
    axes[0].set_title("Average Charges by Region")

    # stacked bar: smoker proportion by region
    region_smoker = df.groupby(["region","smoker"]).size().unstack()
    region_smoker_pct = region_smoker.div(region_smoker.sum(axis=1), axis=0) * 100
    region_smoker_pct.plot(kind="bar", ax=axes[1], color=[GREEN, RED],
                           edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Smoker Proportion by Region")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(["Non-Smoker","Smoker"], fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "**Interpretation:** The southeast has the highest average charges and also the highest "
        "proportion of smokers among the four regions. This partly explains the regional cost "
        "difference — it is a confounded effect. After controlling for smoking status, regional "
        "differences are substantially reduced, which is why the regression model assigns a "
        "relatively small coefficient to the region variable."
    )

    # ── 2.6 Correlation heatmap ──────────────────────────────────────────────
    st.subheader("2.6 Correlation Heatmap")
    st.markdown(
        "The Pearson correlation coefficient measures the strength and direction of the "
        "**linear** relationship between two numerical variables. It ranges from -1 (perfect "
        "negative correlation) to +1 (perfect positive correlation), with 0 indicating no "
        "linear relationship."
    )
    st.markdown(r"""
    $$r_{XY} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
    {\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\,\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}$$
    """)

    df_enc = df.copy()
    df_enc["sex"]    = df_enc["sex"].map({"male":1,"female":0})
    df_enc["smoker"] = df_enc["smoker"].map({"yes":1,"no":0})
    df_enc["region"] = df_enc["region"].map({"northeast":0,"northwest":1,"southeast":2,"southwest":3})

    corr = df_enc.corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                square=True, linewidths=0.8, ax=axes[0],
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 9})
    axes[0].set_title("Pearson Correlation Matrix")

    charges_corr = corr["charges"].drop("charges").sort_values()
    colors_bar = [RED if v < 0 else BLUE for v in charges_corr]
    axes[1].barh(charges_corr.index, charges_corr.values,
                 color=colors_bar, edgecolor="black", linewidth=0.5, alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=0.8)
    for i, v in enumerate(charges_corr.values):
        axes[1].text(v + (0.01 if v >= 0 else -0.01), i,
                     f"{v:+.3f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Correlation with Charges")
    axes[1].set_title("Feature Correlations with Charges")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    smoker_corr = corr.loc["smoker","charges"]
    age_corr    = corr.loc["age","charges"]
    bmi_corr    = corr.loc["bmi","charges"]

    st.markdown(
        f"**Interpretation:**\n\n"
        f"- **Smoker** has the strongest correlation with charges (r = {smoker_corr:.3f}), "
        "confirming it is by far the most important predictor.\n"
        f"- **Age** shows a moderate positive correlation (r = {age_corr:.3f}), consistent "
        "with the biological expectation that older individuals have higher medical costs.\n"
        f"- **BMI** has a weaker correlation (r = {bmi_corr:.3f}) when considered alone, "
        "but its effect becomes more pronounced in combination with smoking status.\n"
        "- **Sex**, **children**, and **region** have weak correlations, suggesting they "
        "contribute less to overall charge variation."
    )
