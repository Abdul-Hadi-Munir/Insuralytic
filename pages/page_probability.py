import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")


def show(df):
    st.header("Probability, Distributions and Confidence Intervals")

    # --- Distribution Analysis ---
    st.subheader("3.1 Distribution Shape Analysis")
    st.markdown(
        "The following plots compare the observed distributions of Age, BMI, and Charges "
        "against the theoretical normal distribution. A kernel density estimate (KDE) is "
        "overlaid on each histogram for visual comparison."
    )

    variables = [
        ("age", "#2ecc71", "Age"),
        ("bmi", "#9b59b6", "BMI"),
        ("charges", "#3498db", "Charges ($)")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (col, color, label) in enumerate(variables):
        data = df[col]
        axes[i].hist(data, bins=30, density=True, color=color, edgecolor="black", alpha=0.6, label="Observed")

        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 200)
        axes[i].plot(x, stats.norm.pdf(x, mu, sigma), "k--", linewidth=2, label="Normal Fit")

        axes[i].set_xlabel(label)
        axes[i].set_ylabel("Density")
        axes[i].set_title(f"Distribution of {label}")
        axes[i].legend(fontsize=8)

        skew = data.skew()
        kurt = data.kurtosis()
        axes[i].text(0.97, 0.95, f"Skew: {skew:.2f}\nKurt: {kurt:.2f}",
                     transform=axes[i].transAxes, ha="right", va="top",
                     fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(r"""
    **Formulas**

    - **Skewness** measures asymmetry: $\gamma_1 = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^3$
    - **Kurtosis** measures tail heaviness: $\gamma_2 = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^4 - 3$

    A skewness near 0 and kurtosis near 0 suggest the data approximates a normal distribution.
    """)

    col1, col2, col3 = st.columns(3)
    for c, (col, _, label) in zip([col1, col2, col3], variables):
        with c:
            data = df[col]
            st.markdown(f"**{label}**")
            skew = data.skew()
            if abs(skew) < 0.5:
                interpretation = "approximately symmetric"
            elif skew > 0:
                interpretation = "right-skewed (positively skewed)"
            else:
                interpretation = "left-skewed (negatively skewed)"
            st.markdown(f"- Skewness: {skew:.3f} -- {interpretation}")
            st.markdown(f"- Kurtosis: {data.kurtosis():.3f}")

    # --- Q-Q Plots ---
    st.subheader("3.2 Q-Q Plots (Normality Assessment)")
    st.markdown(
        "Quantile-Quantile plots compare observed quantiles against theoretical normal quantiles. "
        "Points falling along the diagonal red line indicate normality."
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (col, color, label) in enumerate(variables):
        stats.probplot(df[col], dist="norm", plot=axes[i])
        axes[i].set_title(f"Q-Q Plot: {label}")
        axes[i].get_lines()[0].set_color(color)
        axes[i].get_lines()[0].set_alpha(0.6)
        axes[i].get_lines()[1].set_color("#e74c3c")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- 95% CI for Overall Charges ---
    st.subheader("3.3 95% Confidence Interval for Average Insurance Charges")

    st.markdown(r"""
    **Formula (t-distribution based):**

    $$CI = \bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

    Where:
    - $\bar{x}$ is the sample mean
    - $t_{\alpha/2, n-1}$ is the critical value from the t-distribution
    - $s$ is the sample standard deviation
    - $n$ is the sample size
    """)

    n = len(df)
    mean_charges = df["charges"].mean()
    std_charges = df["charges"].std()
    se = std_charges / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_charges - t_crit * se
    ci_high = mean_charges + t_crit * se

    col1, col2, col3 = st.columns(3)
    col1.metric("Sample Mean", f"${mean_charges:,.2f}")
    col2.metric("95% CI Lower Bound", f"${ci_low:,.2f}")
    col3.metric("95% CI Upper Bound", f"${ci_high:,.2f}")

    st.markdown(
        f"**Interpretation:** We are 95% confident that the true population mean of annual "
        f"insurance charges lies between **${ci_low:,.2f}** and **${ci_high:,.2f}**. "
        f"The margin of error is **${t_crit * se:,.2f}**."
    )

    # --- 95% CI: Smokers vs Non-Smokers ---
    st.subheader("3.4 95% Confidence Intervals: Smokers vs Non-Smokers")

    st.markdown(
        "A comparison of confidence intervals between smokers and non-smokers reveals "
        "whether the difference in mean charges is statistically meaningful."
    )

    results = []
    for group in ["yes", "no"]:
        data = df[df["smoker"] == group]["charges"]
        n_g = len(data)
        mean_g = data.mean()
        std_g = data.std()
        se_g = std_g / np.sqrt(n_g)
        t_g = stats.t.ppf(0.975, df=n_g - 1)
        ci_l = mean_g - t_g * se_g
        ci_h = mean_g + t_g * se_g
        label = "Smoker" if group == "yes" else "Non-Smoker"
        results.append({"Group": label, "n": n_g, "Mean": mean_g, "Std Dev": std_g,
                        "CI Lower": ci_l, "CI Upper": ci_h})

    ci_df = pd.DataFrame(results)
    st.dataframe(
        ci_df.style.format({
            "Mean": "${:,.2f}", "Std Dev": "${:,.2f}",
            "CI Lower": "${:,.2f}", "CI Upper": "${:,.2f}", "n": "{:,}"
        }),
        use_container_width=True
    )

    # CI visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    colors_ci = ["#e74c3c", "#2ecc71"]
    for i, row in ci_df.iterrows():
        ax.errorbar(row["Mean"], i, xerr=[[row["Mean"] - row["CI Lower"]], [row["CI Upper"] - row["Mean"]]],
                     fmt="o", color=colors_ci[i], markersize=10, capsize=8, capthick=2, linewidth=2)
        ax.text(row["CI Upper"] + 500, i, f"${row['Mean']:,.0f}  [{row['CI Lower']:,.0f}, {row['CI Upper']:,.0f}]",
                va="center", fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(ci_df["Group"])
    ax.set_xlabel("Insurance Charges ($)")
    ax.set_title("95% Confidence Intervals: Smoker vs Non-Smoker")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "**Interpretation:** The confidence intervals for smokers and non-smokers do not overlap, "
        "providing strong statistical evidence that the difference in mean charges is significant. "
        f"Smokers pay approximately **${ci_df.iloc[0]['Mean'] - ci_df.iloc[1]['Mean']:,.0f}** more "
        "on average than non-smokers."
    )

    # --- Two-sample t-test ---
    st.subheader("3.5 Hypothesis Test: Two-Sample t-Test")
    st.markdown(r"""
    **Hypotheses:**
    - $H_0$: $\mu_{smoker} = \mu_{non\text{-}smoker}$ (no difference in mean charges)
    - $H_1$: $\mu_{smoker} \neq \mu_{non\text{-}smoker}$ (significant difference exists)

    **Test Statistic:**
    $$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
    """)

    smoker_charges = df[df["smoker"] == "yes"]["charges"]
    nonsmoker_charges = df[df["smoker"] == "no"]["charges"]
    t_stat, p_value = stats.ttest_ind(smoker_charges, nonsmoker_charges, equal_var=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("t-Statistic", f"{t_stat:.4f}")
    col2.metric("p-Value", f"{p_value:.2e}")
    col3.metric("Significance (alpha=0.05)", "Reject H0" if p_value < 0.05 else "Fail to Reject H0")

    st.markdown(
        f"**Conclusion:** With a p-value of {p_value:.2e}, which is far below the 0.05 significance "
        "threshold, we reject the null hypothesis. There is overwhelming statistical evidence that "
        "smokers and non-smokers have significantly different average insurance charges."
    )
